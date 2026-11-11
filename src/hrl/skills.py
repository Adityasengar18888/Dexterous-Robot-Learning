"""Goal-Conditioned Manipulation Skills & Learned Skill Termination Networks (Phase 16.1).

Defines 6 reusable goal-conditioned manipulation skills:
  0: REACH     - move with open gripper to pre-grasp clearance above target object
  1: ALIGN     - fine XY centering above target object at pre-grasp height
  2: GRASP     - descend vertically to object center & close parallel jaw gripper
  3: LIFT      - hold gripper closed & lift grasped object to safe transport altitude
  4: TRANSPORT - move grasped object toward target placement goal while maintaining grasp
  5: PLACE     - settle object at desired goal & open gripper once supported/within tolerance

Each skill pairs a neural low-level policy `SkillActorNet` and a neural termination
predictor `SkillTerminationNet` trained from PyBullet skill demonstrations and validated
independently of any hardcoded meta-state-machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


SKILL_NAMES: Tuple[str, ...] = ("REACH", "ALIGN", "GRASP", "LIFT", "TRANSPORT", "PLACE")
NUM_SKILLS: int = len(SKILL_NAMES)


def extract_skill_features(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Build unified 16D relative geometric state vector for skill & termination networks.

    Works uniformly with `PickPlace` (10D obs + 3D goal) and `SinglePreGrasp` (22D obs + 3D goal).
    Features:
      [0:3]   ee_pos (m)
      [3]     gripper_width (m)
      [4:7]   obj_pos (m)
      [7:10]  obj_vel (m/s)
      [10:13] rel_ee_to_obj = ee_pos - obj_pos (m)
      [13:16] rel_obj_to_goal = obj_pos - desired_goal (m)
    """
    raw = np.asarray(obs["observation"], dtype=np.float32).reshape(-1)
    ee_pos = raw[0:3]
    gw = raw[3:4]
    if raw.shape[0] >= 10 and raw.shape[0] < 20:
        # PickPlace: [ee_pos(3), gw(1), obj_pos(3), obj_vel(3)]
        obj_pos = raw[4:7]
        obj_vel = raw[7:10]
    else:
        # SinglePreGrasp / Sort (22D): [ee_pos(3), gw(1), cube_pos(3), cube_vel(3), ...]
        obj_pos = np.asarray(obs["achieved_goal"][:3], dtype=np.float32)
        obj_vel = raw[7:10] if raw.shape[0] >= 10 else np.zeros(3, dtype=np.float32)

    dg = np.asarray(obs["desired_goal"][:3], dtype=np.float32)
    rel_ee_obj = ee_pos - obj_pos
    rel_obj_goal = obj_pos - dg
    return np.concatenate([ee_pos, gw, obj_pos, obj_vel, rel_ee_obj, rel_obj_goal]).astype(np.float32)


class SkillActorNet(nn.Module):
    """Lightweight continuous-action MLP policy for an individual manipulation skill."""

    def __init__(self, in_dim: int = 16, act_dim: int = 4, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
            nn.Tanh(),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)


class SkillTerminationNet(nn.Module):
    """Learned binary termination probability beta_k(s) in (0, 1) for skill k."""

    def __init__(self, in_dim: int = 16, hidden: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.sigmoid(self.net(x))


class SkillLibrary(nn.Module):
    """Collection of 6 neural goal-conditioned skill actors and learned termination predictors."""

    def __init__(self, in_dim: int = 16) -> None:
        super().__init__()
        self.actors = nn.ModuleList([SkillActorNet(in_dim=in_dim) for _ in range(NUM_SKILLS)])
        self.terminations = nn.ModuleList([SkillTerminationNet(in_dim=in_dim) for _ in range(NUM_SKILLS)])

    @th.no_grad()
    def act(self, skill_id: int, feat: np.ndarray) -> Tuple[np.ndarray, float]:
        t = th.as_tensor(feat, dtype=th.float32).unsqueeze(0)
        action = self.actors[skill_id](t).squeeze(0).cpu().numpy().astype(np.float32)
        term_prob = float(self.terminations[skill_id](t).item())
        return action, term_prob

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        th.save(self.state_dict(), path)

    def load(self, path: str, device: str = "cpu") -> None:
        state = th.load(path, map_location=device, weights_only=False)
        self.load_state_dict(state)
        self.eval()


def compute_teacher_skill_target(skill_id: int, obs: Dict[str, np.ndarray], action_scale: float = 0.05) -> Tuple[np.ndarray, float]:
    """Reference geometric controller used solely to generate offline training samples for neural skills.

    Returns (target_action[4], target_termination_label in {0.0, 1.0}).
    """
    feat = extract_skill_features(obs)
    ee_pos = feat[0:3]
    gw = float(feat[3])
    obj_pos = feat[4:7]
    dg = obj_pos - feat[13:16]

    cmd = np.zeros(4, dtype=np.float32)
    done = 0.0

    if skill_id == 0:  # REACH: overhead pre-grasp (xy -> obj, z -> obj_z + 0.075, gripper open)
        tgt = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.075], dtype=np.float32)
        cmd[:3] = np.clip((tgt - ee_pos) / action_scale, -0.9, 0.9)
        cmd[3] = 1.0
        if np.linalg.norm(ee_pos - tgt) <= 0.025 and gw > 0.065:
            done = 1.0
    elif skill_id == 1:  # ALIGN: tight XY alignment at z = obj_z + 0.045, gripper open
        tgt = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.045], dtype=np.float32)
        cmd[:3] = np.clip((tgt - ee_pos) / action_scale, -0.7, 0.7)
        cmd[3] = 1.0
        if np.linalg.norm(ee_pos[:2] - obj_pos[:2]) <= 0.012 and abs(ee_pos[2] - tgt[2]) <= 0.015 and gw > 0.065:
            done = 1.0
    elif skill_id == 2:  # GRASP: descend to cube height & close
        tgt = np.array([obj_pos[0], obj_pos[1], max(0.018, obj_pos[2] - 0.004)], dtype=np.float32)
        delta = (tgt - ee_pos) / action_scale
        if np.linalg.norm(ee_pos[:2] - obj_pos[:2]) <= 0.015 and abs(ee_pos[2] - tgt[2]) <= 0.014:
            cmd[:3] = np.clip(delta, -0.2, 0.2)
            cmd[3] = -1.0
            if gw < 0.056:
                done = 1.0
        else:
            cmd[:3] = np.clip(delta, -0.5, 0.5)
            cmd[3] = 1.0
    elif skill_id == 3:  # LIFT: hold closed & raise vertically to >= 0.11 m
        lift_z = max(0.125, float(dg[2]) if dg[2] > 0.08 else 0.125)
        tgt = np.array([obj_pos[0], obj_pos[1], lift_z], dtype=np.float32)
        cmd[:3] = np.clip((tgt - ee_pos) / action_scale, -0.65, 0.65)
        cmd[3] = -1.0
        if obj_pos[2] >= 0.090:
            done = 1.0
    elif skill_id == 4:  # TRANSPORT: carry grasped object to target hover/approach point
        hover_z = float(dg[2]) if dg[2] >= 0.05 else max(0.08, float(dg[2]) + 0.04)
        tgt = np.array([dg[0], dg[1], hover_z], dtype=np.float32)
        delta = (tgt - obj_pos) / action_scale
        cmd[:3] = np.clip(delta * 0.85, -0.85, 0.85)
        cmd[3] = -1.0
        if np.linalg.norm(obj_pos - dg) <= 0.032 or np.linalg.norm(obj_pos - tgt) <= 0.025:
            done = 1.0
    else:  # PLACE (5): descend onto exact target & release if tabletop-supported, else hold at goal
        delta = (dg - obj_pos) / action_scale
        cmd[:3] = np.clip(delta * 0.75, -0.6, 0.6)
        dist = float(np.linalg.norm(obj_pos - dg))
        table_supported = bool(dg[2] <= 0.035)
        cmd[3] = 1.0 if (table_supported and dist <= 0.025) else -1.0
        if dist <= 0.025:
            done = 1.0

    return cmd, done


def train_skill_library_from_env_samples(
    samples_by_skill: Dict[int, List[Tuple[np.ndarray, np.ndarray, float]]],
    epochs: int = 80,
    lr: float = 2e-3,
    batch_size: int = 128,
    seed: int = 42,
) -> Tuple[SkillLibrary, Dict[str, Dict[str, float]]]:
    """Train the 6 neural SkillActorNets + 6 neural SkillTerminationNets on PyBullet samples."""
    th.manual_seed(seed)
    rng = np.random.default_rng(seed)
    lib = SkillLibrary(in_dim=16)
    metrics: Dict[str, Dict[str, float]] = {}

    for k in range(NUM_SKILLS):
        data = samples_by_skill[k]
        xs = np.stack([d[0] for d in data], axis=0).astype(np.float32)
        ys = np.stack([d[1] for d in data], axis=0).astype(np.float32)
        ts = np.asarray([d[2] for d in data], dtype=np.float32).reshape(-1, 1)

        x_t = th.from_numpy(xs)
        y_t = th.from_numpy(ys)
        t_t = th.from_numpy(ts)

        opt_a = th.optim.Adam(lib.actors[k].parameters(), lr=lr)
        opt_t = th.optim.Adam(lib.terminations[k].parameters(), lr=lr)
        n = x_t.shape[0]
        last_mse, last_bce = 0.0, 0.0

        for _ in range(epochs):
            perm = rng.permutation(n)
            for i in range(0, n, batch_size):
                idx = th.as_tensor(perm[i : i + batch_size], dtype=th.long)
                xb, yb, tb = x_t[idx], y_t[idx], t_t[idx]

                pred_a = lib.actors[k](xb)
                loss_a = F.mse_loss(pred_a, yb)
                opt_a.zero_grad()
                loss_a.backward()
                opt_a.step()

                pred_t = lib.terminations[k](xb)
                loss_t = F.binary_cross_entropy(th.clamp(pred_t, 1e-5, 1.0 - 1e-5), tb)
                opt_t.zero_grad()
                loss_t.backward()
                opt_t.step()

                last_mse = float(loss_a.item())
                last_bce = float(loss_t.item())

        with th.no_grad():
            full_pred_t = (lib.terminations[k](x_t) >= 0.5).float()
            term_acc = float((full_pred_t == t_t).float().mean().item())
            full_mse = float(F.mse_loss(lib.actors[k](x_t), y_t).item())
        metrics[SKILL_NAMES[k]] = {
            "n_samples": int(n),
            "actor_mse": round(full_mse, 6),
            "term_bce": round(last_bce, 6),
            "term_accuracy": round(term_acc, 4),
        }

    lib.eval()
    return lib, metrics

