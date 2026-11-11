"""Orientation-Aware Dense Reward & Telemetry Assembly Environment (Phase 4 / 15.3).

Preserves the original sparse ``PandaAssemblyEnv`` untouched while adding:
  - Quaternion geodesic angular error (rad/deg), normalization & double-cover sign invariance
  - Contact/jam telemetry (peg-vs-receptacle contact forces, robot-vs-table, finger bilateral grasp)
  - Insertion-depth tracking and completion verification
  - HER-recomputable stateless dense reward (approach, yaw align, open grasp, lift, align hole, insert)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import numpy as np
import pybullet as p

from src.environments.assembly_env import PandaAssemblyEnv


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n < 1e-8 or not np.isfinite(n):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def quaternion_geodesic_error(q1: np.ndarray, q2: np.ndarray) -> Tuple[float, float]:
    """Return (double_cover_loss = 1 - <q1,q2>^2, geodesic_angle_rad in [0, pi])."""
    u1 = normalize_quaternion(q1)
    u2 = normalize_quaternion(q2)
    dot = float(np.clip(abs(float(np.dot(u1, u2))), 0.0, 1.0))
    dc_loss = float(1.0 - dot * dot)
    angle_rad = float(2.0 * np.arccos(dot))
    return dc_loss, angle_rad


def yaw_from_quaternion(q: np.ndarray) -> float:
    euler = p.getEulerFromQuaternion(normalize_quaternion(q).tolist())
    return float(euler[2])


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class AssemblyRewardConfig:
    w_xy_prog: float = 4.0
    w_z_prog: float = 2.0
    w_ori_prog: float = 1.5
    w_grasp: float = 1.5
    w_lift: float = 1.5
    w_insert: float = 6.0
    w_assembled: float = 8.0
    w_jam_coll: float = 0.08
    w_open_comp: float = 0.05
    step_penalty: float = -0.02


class PandaAssemblyDenseEnv(PandaAssemblyEnv):
    """Orientation-aware peg-in-receptacle environment with dense HER reward and contact telemetry."""

    def __init__(
        self,
        render_mode: str | None = None,
        distance_threshold: float = 0.02,
        orientation_threshold: float = 0.1,
        max_episode_steps: int = 120,
        control_steps: int = 15,
        action_scale: float = 0.05,
        yaw_scale: float = 0.25,
        reward_config: AssemblyRewardConfig | None = None,
    ) -> None:
        self.asm_cfg = reward_config or AssemblyRewardConfig()
        self.yaw_scale = float(yaw_scale)
        super().__init__(
            render_mode=render_mode,
            distance_threshold=distance_threshold,
            orientation_threshold=orientation_threshold,
            max_episode_steps=max_episode_steps,
            control_steps=control_steps,
            action_scale=action_scale,
        )
        self._last_reward_components: Dict[str, float] = {}

    def _query_assembly_contacts(self) -> Dict[str, Any]:
        cid = self.sim.client_id
        rob_id = self.controller.robot.body_id
        peg_id = self.peg.body_id
        base_id = self.base.body_id
        table_id = self.sim.table_id

        peg_robot = p.getContactPoints(bodyA=rob_id, bodyB=peg_id, physicsClientId=cid)
        f9 = [c for c in peg_robot if c[3] == 9]
        f10 = [c for c in peg_robot if c[3] == 10]
        total_fn = float(sum(c[9] for c in f9) + sum(c[9] for c in f10))
        bilateral = bool(len(f9) > 0 and len(f10) > 0 and total_fn >= 5.0)

        peg_base_contacts = p.getContactPoints(bodyA=peg_id, bodyB=base_id, physicsClientId=cid)
        # High lateral wall/rim contact forces indicate insertion rubbing / jamming
        base_force = float(sum(c[9] for c in peg_base_contacts))
        wall_hits = [c for c in peg_base_contacts if c[4] in (0, 1, 2, 3)]  # link 0..3 are side walls
        wall_force = float(sum(c[9] for c in wall_hits))
        jam_detected = bool(wall_force >= 25.0)

        table_contacts = p.getContactPoints(bodyA=rob_id, bodyB=table_id, physicsClientId=cid) if table_id >= 0 else []
        return {
            "bilateral_grasp": bilateral,
            "finger_normal_force": total_fn,
            "peg_base_contact_count": len(peg_base_contacts),
            "peg_base_force": base_force,
            "wall_force": wall_force,
            "jam_detected": jam_detected,
            "table_collision": len(table_contacts) > 0,
        }

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        ct = self._query_assembly_contacts()
        peg_pos = self.peg.get_position().copy()
        base_pos = self.base.get_position().copy()
        dc_err, rad_err = quaternion_geodesic_error(obs["achieved_goal"][3:], obs["desired_goal"][3:])
        info.update(
            {
                "prev_ee_pos": obs["observation"][:3].copy(),
                "ee_pos": obs["observation"][:3].copy(),
                "prev_ag": obs["achieved_goal"].copy(),
                "prev_gw": float(obs["observation"][7]),
                "gripper_width": float(obs["observation"][7]),
                "prev_bilateral_contact": ct["bilateral_grasp"],
                "bilateral_contact": ct["bilateral_grasp"],
                "jam_detected": ct["jam_detected"],
                "table_collision": ct["table_collision"],
                "peg_base_force": ct["peg_base_force"],
                "geodesic_angle_rad": rad_err,
                "geodesic_angle_deg": float(np.degrees(rad_err)),
                "ori_double_cover_loss": dc_err,
                "insertion_depth_m": float(max(0.0, (base_pos[2] + 0.055) - (peg_pos[2] - 0.030))),
                "reward_components": {},
            }
        )
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        prev_ee_pos = self.controller.get_ee_position().copy().astype(np.float32)
        prev_gw = float(self.controller.get_gripper_state()["width"])
        prev_ag = np.concatenate([self.peg.get_position(), self.peg.get_orientation()]).astype(np.float32)
        prev_ct = self._query_assembly_contacts()

        action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        dx, dy, dz = action[:3] * self.action_scale
        dyaw = float(action[3]) * self.yaw_scale
        gripper_action = float(action[4])

        target_ee_pos = np.clip(
            prev_ee_pos + np.array([dx, dy, dz], dtype=np.float32),
            np.array([0.2, -0.4, 0.02], dtype=np.float32),
            np.array([0.8, 0.4, 0.50], dtype=np.float32),
        )
        self.current_yaw = wrap_angle(self.current_yaw + dyaw)
        target_ori = p.getQuaternionFromEuler([np.pi, 0.0, self.current_yaw])

        self.controller.move_ee(position=target_ee_pos, orientation=target_ori, steps=self.control_steps, tolerance=0.005)
        if gripper_action > 0.0:
            self.controller.open_gripper(steps=10)
        else:
            self.controller.close_gripper(steps=10)

        obs = self._get_obs()
        curr_ct = self._query_assembly_contacts()
        curr_ee_pos = obs["observation"][:3].copy().astype(np.float32)
        curr_gw = float(obs["observation"][7])
        curr_ag = obs["achieved_goal"].copy().astype(np.float32)

        if curr_ct["jam_detected"] or curr_ct["table_collision"]:
            self.collision_count += 1

        info = self._get_info(curr_ag, self.goal)
        dc_err, rad_err = quaternion_geodesic_error(curr_ag[3:], self.goal[3:])
        peg_pos = curr_ag[:3]
        base_pos = self.base.get_position()

        # Full completion check requires hole placement + orientation + inside receptacle rim (floor ~0.045)
        hole_xy = float(np.linalg.norm(peg_pos[:2] - base_pos[:2]))
        inserted_flag = bool(
            hole_xy <= self.distance_threshold
            and peg_pos[2] <= 0.052
            and dc_err <= self.orientation_threshold
        )
        info["is_success"] = inserted_flag
        if inserted_flag:
            info["assembly_state"] = "ASSEMBLED"

        info.update(
            {
                "prev_ee_pos": prev_ee_pos,
                "ee_pos": curr_ee_pos,
                "prev_ag": prev_ag,
                "prev_gw": prev_gw,
                "gripper_width": curr_gw,
                "prev_bilateral_contact": prev_ct["bilateral_grasp"],
                "bilateral_contact": curr_ct["bilateral_grasp"],
                "jam_detected": curr_ct["jam_detected"],
                "table_collision": curr_ct["table_collision"],
                "peg_base_force": curr_ct["peg_base_force"],
                "wall_force": curr_ct["wall_force"],
                "geodesic_angle_rad": rad_err,
                "geodesic_angle_deg": float(np.degrees(rad_err)),
                "ori_double_cover_loss": dc_err,
                "hole_xy_error_m": hole_xy,
                "insertion_depth_m": float(max(0.0, (base_pos[2] + 0.055) - (peg_pos[2] - 0.030))),
            }
        )

        reward = float(self.compute_reward(curr_ag, self.goal, info))
        info["reward_components"] = dict(self._last_reward_components)
        terminated = False
        truncated = self.current_step >= self.max_episode_steps
        return obs, reward, terminated, truncated, info

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Union[dict, list, tuple, np.ndarray],
    ) -> Union[float, np.ndarray]:
        is_single = achieved_goal.ndim == 1
        ag = np.asarray(achieved_goal, dtype=np.float32).reshape(-1, 7)
        dg = np.asarray(desired_goal, dtype=np.float32).reshape(-1, 7)
        B = ag.shape[0]
        cfg = self.asm_cfg

        req = ("prev_ee_pos", "ee_pos", "prev_ag", "gripper_width", "prev_bilateral_contact", "bilateral_contact", "jam_detected", "table_collision")
        if isinstance(info, dict) and all(k in info for k in req):
            prev_ee = np.asarray(info["prev_ee_pos"], dtype=np.float32).reshape(B, 3)
            ee = np.asarray(info["ee_pos"], dtype=np.float32).reshape(B, 3)
            prev_ag = np.asarray(info["prev_ag"], dtype=np.float32).reshape(B, 7)
            gw = np.asarray(info["gripper_width"], dtype=np.float32).reshape(B)
            prev_b = np.asarray(info["prev_bilateral_contact"], dtype=bool).reshape(B)
            curr_b = np.asarray(info["bilateral_contact"], dtype=bool).reshape(B)
            jam = np.asarray(info["jam_detected"], dtype=bool).reshape(B)
            tcoll = np.asarray(info["table_collision"], dtype=bool).reshape(B)
        elif isinstance(info, (list, tuple, np.ndarray)) and len(info) == B and all(isinstance(d, dict) and all(k in d for k in req) for d in info):
            prev_ee = np.stack([np.asarray(d["prev_ee_pos"], dtype=np.float32) for d in info])
            ee = np.stack([np.asarray(d["ee_pos"], dtype=np.float32) for d in info])
            prev_ag = np.stack([np.asarray(d["prev_ag"], dtype=np.float32) for d in info])
            gw = np.asarray([float(d["gripper_width"]) for d in info], dtype=np.float32)
            prev_b = np.asarray([bool(d["prev_bilateral_contact"]) for d in info], dtype=bool)
            curr_b = np.asarray([bool(d["bilateral_contact"]) for d in info], dtype=bool)
            jam = np.asarray([bool(d["jam_detected"]) for d in info], dtype=bool)
            tcoll = np.asarray([bool(d["table_collision"]) for d in info], dtype=bool)
        else:
            pos_d = np.linalg.norm(ag[:, :3] - dg[:, :3], axis=-1)
            dot = np.sum(ag[:, 3:] * dg[:, 3:], axis=-1)
            ori_d = 1.0 - dot * dot
            rew = (-0.5 * np.clip(pos_d, 0.0, 1.0) - 0.3 * ori_d + np.where((pos_d <= self.distance_threshold) & (ori_d <= self.orientation_threshold), 4.0, 0.0)).astype(np.float32)
            return float(rew[0]) if is_single else rew

        ungrasped = (~curr_b).astype(np.float32)
        held = (curr_b | prev_b).astype(np.float32)
        is_open = gw > 0.062

        xy_prev = np.linalg.norm(prev_ee[:, :2] - prev_ag[:, :2], axis=-1)
        xy_curr = np.linalg.norm(ee[:, :2] - ag[:, :2], axis=-1)
        r_xy = cfg.w_xy_prog * np.clip(xy_prev - xy_curr, -0.05, 0.05) * ungrasped

        z_prev = np.abs(prev_ee[:, 2] - prev_ag[:, 2])
        z_curr = np.abs(ee[:, 2] - ag[:, 2])
        r_z = cfg.w_z_prog * np.clip(z_prev - z_curr, -0.05, 0.05) * ungrasped * is_open.astype(np.float32)

        dot_prev = np.sum(prev_ag[:, 3:] * dg[:, 3:], axis=-1)
        dot_curr = np.sum(ag[:, 3:] * dg[:, 3:], axis=-1)
        ori_prev = 1.0 - dot_prev * dot_prev
        ori_curr = 1.0 - dot_curr * dot_curr
        r_ori = cfg.w_ori_prog * np.clip(ori_prev - ori_curr, -0.5, 0.5) * held

        r_grasp = cfg.w_grasp * (curr_b.astype(np.float32) - prev_b.astype(np.float32))
        r_lift = cfg.w_lift * np.clip((ag[:, 2] - prev_ag[:, 2]) / 0.09, -0.5, 0.5) * held

        g_prev = np.linalg.norm(prev_ag[:, :3] - dg[:, :3], axis=-1)
        g_curr = np.linalg.norm(ag[:, :3] - dg[:, :3], axis=-1)
        r_ins = cfg.w_insert * np.clip(g_prev - g_curr, -0.05, 0.05) * held

        succ_prev = (g_prev <= self.distance_threshold) & (ori_prev <= self.orientation_threshold)
        succ_curr = (g_curr <= self.distance_threshold) & (ori_curr <= self.orientation_threshold)
        r_asm = cfg.w_assembled * ((~succ_prev) & succ_curr).astype(np.float32)

        r_coll = -cfg.w_jam_coll * (jam | tcoll).astype(np.float32)
        prem_close = (~curr_b) & (xy_curr > 0.025) & (~is_open)
        r_open = -cfg.w_open_comp * prem_close.astype(np.float32)
        r_step = np.where(succ_curr, 0.0, cfg.step_penalty).astype(np.float32)

        total = np.nan_to_num(r_xy + r_z + r_ori + r_grasp + r_lift + r_ins + r_asm + r_coll + r_open + r_step, nan=0.0, posinf=20.0, neginf=-20.0).astype(np.float32)
        if is_single:
            self._last_reward_components = {
                "r_xy": float(r_xy[0]),
                "r_z": float(r_z[0]),
                "r_ori": float(r_ori[0]),
                "r_grasp": float(r_grasp[0]),
                "r_lift": float(r_lift[0]),
                "r_insert": float(r_ins[0]),
                "r_assembled": float(r_asm[0]),
                "r_coll": float(r_coll[0]),
                "r_open_comp": float(r_open[0]),
                "r_step": float(r_step[0]),
                "total_reward": float(total[0]),
            }
            return float(total[0])
        return total

