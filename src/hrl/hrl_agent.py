"""Hierarchical Reinforcement Learning Agent (Phase 16.1): Discrete Soft-Q Meta-Policy + HER.

Trains a randomly initialized high-level categorical actor and twin Soft-Q critics (Discrete SAC)
over the 6 goal-conditioned neural skills in `SkillLibrary`, using SMDP discounted returns,
HER-style future goal relabeling, learned skill termination, and drop/stall failure recovery.
Zero hardcoded state transitions exist inside `select_skill()`.
"""

from __future__ import annotations

from collections import deque
import json
import os
import random
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from src.hrl.skills import (
    NUM_SKILLS,
    SKILL_NAMES,
    SkillLibrary,
    extract_skill_features,
)


def build_meta_state(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """18D Markovian state representation for high-level skill selection (zero phase flags)."""
    feat = extract_skill_features(obs)
    d_ee_obj = float(np.linalg.norm(feat[10:13]))
    d_obj_goal = float(np.linalg.norm(feat[13:16]))
    return np.concatenate([feat, np.array([d_ee_obj, d_obj_goal], dtype=np.float32)]).astype(np.float32)


def _compute_smdp_option_reward(
    start_obs: Dict[str, np.ndarray],
    end_obs: Dict[str, np.ndarray],
    skill_id: int,
    tau_steps: int,
    is_success: bool,
    dropped: bool,
    gamma: float = 0.96,
) -> float:
    """Physically grounded goal-conditioned SMDP reward (no scripted sequence hints)."""
    f0 = extract_skill_features(start_obs)
    f1 = extract_skill_features(end_obs)

    d_ee0 = float(np.linalg.norm(f0[10:13]))
    d_ee1 = float(np.linalg.norm(f1[10:13]))
    d_g0 = float(np.linalg.norm(f0[13:16]))
    d_g1 = float(np.linalg.norm(f1[13:16]))
    gw1 = float(f1[3])
    z0 = float(f0[6])
    z1 = float(f1[6])

    # Strictly potential-difference shaping (prevents hover reward hacking) + sparse terminal bonus
    r = 2.0 * np.clip(d_ee0 - d_ee1, -0.25, 0.25)
    h0 = float(np.clip((z0 - 0.025) / 0.08, 0.0, 1.0))
    h1 = float(np.clip((z1 - 0.025) / 0.08, 0.0, 1.0))
    r += 1.5 * (h1 - h0)
    r += 5.5 * np.clip(d_g0 - d_g1, -0.25, 0.25)
    if is_success:
        r += 6.0
    if dropped:
        r -= 1.0
    r -= 0.04 * float(tau_steps)
    return float(r)


class MetaPolicyNet(nn.Module):
    """Categorical high-level skill policy pi_meta(k | s)."""

    def __init__(self, in_dim: int = 18, n_skills: int = NUM_SKILLS, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_skills),
        )

    def forward(self, s: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        logits = self.net(s)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs


class MetaTwinQNet(nn.Module):
    """Twin Q-networks for Discrete Soft Actor-Critic high-level skill evaluation."""

    def __init__(self, in_dim: int = 18, n_skills: int = NUM_SKILLS, hidden: int = 128) -> None:
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_skills)
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_skills)
        )

    def forward(self, s: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.q1(s), self.q2(s)


class SMDPReplayBuffer:
    """Stores option transitions (s, k, r, s', tau, done) with HER future-goal relabeling."""

    def __init__(self, capacity: int = 50000, her_k: int = 2, gamma: float = 0.96, seed: int = 42) -> None:
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, int, float]] = deque(maxlen=capacity)
        self.her_k = int(her_k)
        self.gamma = float(gamma)
        self.rng = random.Random(seed)

    def add_episode(self, ep_transitions: List[Dict[str, Any]]) -> int:
        added = 0
        n = len(ep_transitions)
        for t in range(n):
            tr = ep_transitions[t]
            self.buffer.append(
                (tr["s"], tr["k"], tr["r"], tr["s_next"], tr["tau"], float(tr["done"]))
            )
            added += 1

            # HER future achieved-goal relabeling over remaining episode option outcomes
            if self.her_k > 0 and t + 1 < n:
                for _ in range(self.her_k):
                    fut = ep_transitions[self.rng.randint(t, n - 1)]
                    new_goal = fut["end_obs"]["achieved_goal"][:3].copy()
                    so_rel = dict(tr["start_obs"])
                    eo_rel = dict(tr["end_obs"])
                    so_rel["desired_goal"] = new_goal
                    eo_rel["desired_goal"] = new_goal
                    s_rel = build_meta_state(so_rel)
                    s_next_rel = build_meta_state(eo_rel)
                    dist_rel = float(np.linalg.norm(eo_rel["achieved_goal"][:3] - new_goal))
                    is_succ_rel = dist_rel <= 0.05
                    r_rel = _compute_smdp_option_reward(
                        so_rel, eo_rel, tr["k"], tr["tau"], is_succ_rel, tr["dropped"], gamma=self.gamma
                    )
                    done_rel = 1.0 if is_succ_rel else float(tr["done"])
                    self.buffer.append((s_rel, tr["k"], r_rel, s_next_rel, tr["tau"], done_rel))
                    added += 1
        return added

    def sample(self, batch_size: int) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        batch = self.rng.sample(self.buffer, min(batch_size, len(self.buffer)))
        s, k, r, sn, tau, d = zip(*batch)
        return (
            th.as_tensor(np.stack(s), dtype=th.float32),
            th.as_tensor(k, dtype=th.long).unsqueeze(-1),
            th.as_tensor(r, dtype=th.float32).unsqueeze(-1),
            th.as_tensor(np.stack(sn), dtype=th.float32),
            th.as_tensor(tau, dtype=th.float32).unsqueeze(-1),
            th.as_tensor(d, dtype=th.float32).unsqueeze(-1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class HierarchicalManipulatorAgent:
    """Discrete SAC Meta-Controller + Neural SkillLibrary + Failure Recovery."""

    def __init__(
        self,
        skill_lib: SkillLibrary,
        gamma: float = 0.96,
        lr: float = 1e-3,
        tau_polyak: float = 0.02,
        init_alpha: float = 0.15,
        max_option_steps: int = 12,
        seed: int = 42,
    ) -> None:
        th.manual_seed(seed)
        self.skill_lib = skill_lib
        self.gamma = float(gamma)
        self.tau_polyak = float(tau_polyak)
        self.max_option_steps = int(max_option_steps)
        self.rng = np.random.default_rng(seed)

        self.actor = MetaPolicyNet(in_dim=18, n_skills=NUM_SKILLS, hidden=128)
        self.critic = MetaTwinQNet(in_dim=18, n_skills=NUM_SKILLS, hidden=128)
        self.critic_target = MetaTwinQNet(in_dim=18, n_skills=NUM_SKILLS, hidden=128)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = th.tensor(float(np.log(init_alpha)), requires_grad=True, dtype=th.float32)
        self.target_entropy = 0.35 * float(np.log(NUM_SKILLS))

        self.opt_actor = th.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_critic = th.optim.Adam(self.critic.parameters(), lr=lr)
        self.opt_alpha = th.optim.Adam([self.log_alpha], lr=lr)

    @th.no_grad()
    def select_skill(self, obs: Dict[str, np.ndarray], deterministic: bool = True) -> Tuple[int, float, np.ndarray]:
        """Purely learned neural skill selection (zero hardcoded sequence rules)."""
        s = th.as_tensor(build_meta_state(obs), dtype=th.float32).unsqueeze(0)
        probs, _ = self.actor(s)
        p_np = probs.squeeze(0).cpu().numpy()
        entropy = float(-np.sum(p_np * np.log(np.clip(p_np, 1e-8, 1.0))))
        if deterministic:
            skill_id = int(np.argmax(p_np))
        else:
            skill_id = int(self.rng.choice(NUM_SKILLS, p=p_np / np.sum(p_np)))
        return skill_id, entropy, p_np

    def execute_option(
        self,
        env: Any,
        obs: Dict[str, np.ndarray],
        skill_id: int,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Run selected low-level skill MLP until learned termination, failure drop, or horizon."""
        curr_obs = obs
        tau_steps = 0
        done = False
        trunc = False
        last_info: Dict[str, Any] = {}
        dropped = False
        term_reason = "horizon"

        init_feat = extract_skill_features(curr_obs)
        was_grasped = bool(init_feat[6] > 0.045 and np.linalg.norm(init_feat[10:13]) < 0.055 and init_feat[3] < 0.062)

        while tau_steps < self.max_option_steps and not (done or trunc):
            feat = extract_skill_features(curr_obs)
            action, term_prob = self.skill_lib.act(skill_id, feat)
            if tau_steps >= 1 and term_prob >= 0.5:
                term_reason = "learned_termination"
                break

            next_obs, _, done, trunc, last_info = env.step(action)
            tau_steps += 1
            curr_obs = next_obs

            n_feat = extract_skill_features(curr_obs)
            # Failure detection: object slip/drop mid-lift/transport/place triggers early re-selection
            if skill_id in (3, 4, 5) and was_grasped:
                if np.linalg.norm(n_feat[10:13]) > 0.075 and n_feat[6] < 0.040:
                    dropped = True
                    term_reason = "failure_drop_recovery"
                    break
            elif n_feat[6] > 0.050 and np.linalg.norm(n_feat[10:13]) < 0.050 and n_feat[3] < 0.062:
                was_grasped = True

        # Ensure at least 1 env step occurs if learned_termination triggered immediately
        if tau_steps == 0 and not (done or trunc):
            feat = extract_skill_features(curr_obs)
            action, _ = self.skill_lib.act(skill_id, feat)
            curr_obs, _, done, trunc, last_info = env.step(action)
            tau_steps = 1

        return curr_obs, {
            "tau": tau_steps,
            "done": bool(done),
            "trunc": bool(trunc),
            "last_info": last_info,
            "dropped": dropped,
            "term_reason": term_reason,
        }

    def update(self, replay: SMDPReplayBuffer, batch_size: int = 128) -> Dict[str, float]:
        if len(replay) < batch_size:
            return {}
        s, k, r, sn, tau, d = replay.sample(batch_size)
        alpha = self.log_alpha.exp().detach()

        with th.no_grad():
            n_probs, n_log_probs = self.actor(sn)
            nq1, nq2 = self.critic_target(sn)
            n_min_q = th.min(nq1, nq2)
            v_next = (n_probs * (n_min_q - alpha * n_log_probs)).sum(dim=-1, keepdim=True)
            gamma_pow = self.gamma ** tau
            q_target = r + (1.0 - d) * gamma_pow * v_next

        q1, q2 = self.critic(s)
        q1_k = q1.gather(1, k)
        q2_k = q2.gather(1, k)
        critic_loss = F.mse_loss(q1_k, q_target) + F.mse_loss(q2_k, q_target)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_critic.step()

        probs, log_probs = self.actor(s)
        with th.no_grad():
            aq1, aq2 = self.critic(s)
            min_q = th.min(aq1, aq2)
        actor_loss = (probs * (alpha * log_probs - min_q)).sum(dim=-1).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_actor.step()

        entropy = -(probs.detach() * log_probs.detach()).sum(dim=-1).mean()
        alpha_loss = -(self.log_alpha * (self.target_entropy - entropy).detach())
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()
        with th.no_grad():
            self.log_alpha.clamp_(min=float(np.log(0.01)), max=float(np.log(1.0)))

        for p_targ, p_src in zip(self.critic_target.parameters(), self.critic.parameters()):
            p_targ.data.mul_(1.0 - self.tau_polyak).add_(self.tau_polyak * p_src.data)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "entropy": float(entropy.item()),
            "alpha": float(self.log_alpha.exp().item()),
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        th.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "log_alpha": float(self.log_alpha.item()),
            },
            path,
        )

    def load(self, path: str, device: str = "cpu") -> None:
        payload = th.load(path, map_location=device, weights_only=False)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.critic_target.load_state_dict(payload["critic"])
        self.actor.eval()

