"""Single-object Pre-Grasp & Lift Acquisition environment for Phase 15.2.5B.

Inherits all robot, cube, and PyBullet physics directly from ``PandaMultiSortEnv``
with ``num_objects=1`` (22D obs, 3D goal, 4D action) and reuses the Phase 15.2.5A
collision-aware pre-grasp reward structure with stateless HER recomputation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pybullet as p

from src.environments.multi_sort_env import PandaMultiSortEnv
from src.environments.pregrasp_multi_sort_env import PreGraspRewardConfig


@dataclass
class SinglePreGraspRewardConfig(PreGraspRewardConfig):
    """Single-object lift-and-hold adaptation of PreGraspRewardConfig."""

    target_lift_z: float = 0.140       # Desired lifted altitude above spawned cube XY
    hold_z_min: float = 0.095          # Minimum altitude with bilateral contact to verify lifted hold
    w_hold_goal: float = 4.0           # Milestone bonus on reaching lifted target with bilateral hold
    w_drop: float = 4.5                # Penalty if lifted object is lost/dropped before episode end
    release_width_threshold: float = 0.065

    # Phase 15.2.5B.1 Continuous Vertical Potential Shaping (opt-in to preserve Phase B baseline)
    use_vertical_potential: bool = False
    w_vertical_potential: float = 2.0
    grasp_z_offset: float = 0.000
    vert_stop_tol_m: float = 0.008


class PandaSinglePreGraspEnv(PandaMultiSortEnv):
    """Single-object pre-grasp, bilateral grasp, and lift-hold training environment."""

    def __init__(
        self,
        render_mode: str | None = None,
        distance_threshold: float = 0.05,
        max_episode_steps: int = 80,
        control_steps: int = 15,
        action_scale: float = 0.05,
        reward_config: Optional[SinglePreGraspRewardConfig] = None,
    ) -> None:
        self.pg_cfg = reward_config or SinglePreGraspRewardConfig()
        super().__init__(
            render_mode=render_mode,
            num_objects=1,
            distance_threshold=distance_threshold,
            max_episode_steps=max_episode_steps,
            control_steps=control_steps,
            action_scale=action_scale,
        )
        self._last_reward_components: Dict[str, float] = {}

    def _query_physics_flags(self, floor_scrape: bool = False) -> Tuple[bool, bool, bool]:
        """Return (bilateral_contact, orn_ok, table_collision) from PyBullet."""
        cid = self.sim.client_id
        robot_id = self.controller.robot.body_id
        table_id = self.sim.table_id
        cube_id = self.cubes[0].body_id
        if robot_id < 0 or cube_id < 0:
            return False, True, False

        contacts = p.getContactPoints(bodyA=robot_id, bodyB=cube_id, physicsClientId=cid)
        f9 = [c for c in contacts if c[3] == 9]
        f10 = [c for c in contacts if c[3] == 10]
        bilateral = False
        if len(f9) > 0 and len(f10) > 0:
            total_fn = float(sum(c[9] for c in f9) + sum(c[9] for c in f10))
            if total_fn >= self.pg_cfg.min_bilateral_force_n:
                bilateral = True

        ee_orn = np.asarray(self.controller.get_ee_orientation(), dtype=np.float32)
        default_orn = np.asarray(self.controller.robot.default_ee_orientation, dtype=np.float32)
        orn_ok = bool(abs(float(np.dot(ee_orn, default_orn))) >= self.pg_cfg.min_orn_dot)

        tc_count = len(p.getContactPoints(bodyA=robot_id, bodyB=table_id, physicsClientId=cid)) if table_id >= 0 else 0
        table_coll = bool((tc_count > 0) or floor_scrape)
        return bilateral, orn_ok, table_coll

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        obs, _ = super().reset(seed=seed, options=options)
        cube_pos = obs["achieved_goal"][:3].copy()
        self.goal = np.array([cube_pos[0], cube_pos[1], self.pg_cfg.target_lift_z], dtype=np.float32)
        obs["desired_goal"] = self.goal.copy()

        bilat, orn_ok, tcoll = self._query_physics_flags(floor_scrape=False)
        ee_pos = obs["observation"][:3].copy().astype(np.float32)
        gw = float(obs["observation"][3])

        info = self._get_info(obs["achieved_goal"], self.goal)
        info.update(
            {
                "prev_ee_pos": ee_pos,
                "ee_pos": ee_pos,
                "prev_ag": cube_pos.astype(np.float32),
                "prev_gw": gw,
                "gripper_width": gw,
                "prev_bilateral_contact": bilat,
                "bilateral_contact": bilat,
                "prev_orn_ok": orn_ok,
                "orn_ok": orn_ok,
                "prev_table_collision": tcoll,
                "table_collision": tcoll,
                "reward_components": {},
            }
        )
        return obs, info

    def _get_info(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Dict[str, Any]:
        cfg = self.pg_cfg
        is_batched = achieved_goal.ndim > 1
        if is_batched:
            ag_r = achieved_goal.reshape(-1, 3)
            dg_r = desired_goal.reshape(-1, 3)
            dists = np.linalg.norm(ag_r - dg_r, axis=-1)
            is_succ = (dists <= self.distance_threshold) & (ag_r[:, 2] >= cfg.hold_z_min)
            return {"is_success": is_succ, "distance": dists}

        ag = achieved_goal.reshape(3)
        dg = desired_goal.reshape(3)
        dist = float(np.linalg.norm(ag - dg))
        bilat, _, _ = self._query_physics_flags(floor_scrape=False)
        is_lifted_hold = bool(dist <= self.distance_threshold and ag[2] >= cfg.hold_z_min and bilat)
        if self.object_last_z[0] > 0.07 and ag[2] < 0.04 and not bilat:
            self.dropped_objects += 1
        self.object_last_z[0] = float(ag[2])

        return {
            "is_success": is_lifted_hold,
            "distance": dist,
            "episode_length": self.current_step,
            "dropped_objects": self.dropped_objects,
            "collisions": self.collision_count,
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        prev_ee_pos = self.controller.get_ee_position().copy().astype(np.float32)
        prev_gw = float(self.controller.get_gripper_state()["width"])
        prev_ag = self.cubes[0].get_position().copy().astype(np.float32)
        prev_bilat, prev_orn_ok, prev_tcoll = self._query_physics_flags(floor_scrape=False)

        action = np.clip(action, self.action_space.low, self.action_space.high)
        dx, dy, dz = action[:3] * self.action_scale
        gripper_action = action[3]

        target_ee_pos = prev_ee_pos + np.array([dx, dy, dz], dtype=np.float32)
        arm_workspace_min = np.array([0.2, -0.4, 0.0], dtype=np.float32)
        arm_workspace_max = np.array([0.8, 0.4, 0.5], dtype=np.float32)
        target_ee_pos = np.clip(target_ee_pos, arm_workspace_min, arm_workspace_max)

        floor_scrape = bool(target_ee_pos[2] <= 0.01 and prev_ee_pos[2] <= 0.02 and dz < -0.01)
        if floor_scrape:
            self.collision_count += 1

        self.controller.move_ee(position=target_ee_pos, steps=self.control_steps, tolerance=0.005)
        if gripper_action > 0.0:
            self.controller.open_gripper(steps=10)
        else:
            self.controller.close_gripper(steps=10)

        obs = self._get_obs()
        curr_ee_pos = obs["observation"][:3].copy().astype(np.float32)
        curr_gw = float(obs["observation"][3])
        curr_ag = obs["achieved_goal"].copy().astype(np.float32)
        curr_bilat, curr_orn_ok, curr_tcoll = self._query_physics_flags(floor_scrape=floor_scrape)

        info = self._get_info(curr_ag, self.goal)
        info["prev_ee_pos"] = prev_ee_pos
        info["ee_pos"] = curr_ee_pos
        info["prev_ag"] = prev_ag
        info["prev_gw"] = prev_gw
        info["gripper_width"] = curr_gw
        info["prev_bilateral_contact"] = prev_bilat
        info["bilateral_contact"] = curr_bilat
        info["prev_orn_ok"] = prev_orn_ok
        info["orn_ok"] = curr_orn_ok
        info["prev_table_collision"] = prev_tcoll
        info["table_collision"] = curr_tcoll

        reward = float(self.compute_reward(curr_ag, self.goal, info))
        info["reward_components"] = dict(self._last_reward_components)

        terminated = False
        truncated = self.current_step >= self.max_episode_steps
        return obs, reward, terminated, truncated, info

    def _extract_single_batched_info(
        self, info: Union[dict, list, tuple, np.ndarray], batch_size: int
    ) -> Optional[Dict[str, np.ndarray]]:
        req = (
            "prev_ee_pos", "ee_pos", "prev_ag", "prev_gw", "gripper_width",
            "prev_bilateral_contact", "bilateral_contact",
            "prev_orn_ok", "orn_ok", "prev_table_collision", "table_collision",
        )
        if isinstance(info, dict):
            if not all(k in info for k in req):
                return None
            return {
                "prev_ee_pos": np.asarray(info["prev_ee_pos"], dtype=np.float32).reshape(batch_size, 3),
                "ee_pos": np.asarray(info["ee_pos"], dtype=np.float32).reshape(batch_size, 3),
                "prev_ag": np.asarray(info["prev_ag"], dtype=np.float32).reshape(batch_size, 3),
                "prev_gw": np.asarray(info["prev_gw"], dtype=np.float32).reshape(batch_size),
                "gripper_width": np.asarray(info["gripper_width"], dtype=np.float32).reshape(batch_size),
                "prev_bilat": np.asarray(info["prev_bilateral_contact"], dtype=bool).reshape(batch_size),
                "curr_bilat": np.asarray(info["bilateral_contact"], dtype=bool).reshape(batch_size),
                "prev_orn": np.asarray(info["prev_orn_ok"], dtype=bool).reshape(batch_size),
                "curr_orn": np.asarray(info["orn_ok"], dtype=bool).reshape(batch_size),
                "prev_tcoll": np.asarray(info["prev_table_collision"], dtype=bool).reshape(batch_size),
                "curr_tcoll": np.asarray(info["table_collision"], dtype=bool).reshape(batch_size),
            }
        if isinstance(info, (list, tuple, np.ndarray)) and len(info) == batch_size:
            if not all(isinstance(d, dict) and all(k in d for k in req) for d in info):
                return None
            return {
                "prev_ee_pos": np.stack([np.asarray(d["prev_ee_pos"], dtype=np.float32) for d in info]),
                "ee_pos": np.stack([np.asarray(d["ee_pos"], dtype=np.float32) for d in info]),
                "prev_ag": np.stack([np.asarray(d["prev_ag"], dtype=np.float32).reshape(3) for d in info]),
                "prev_gw": np.asarray([float(d["prev_gw"]) for d in info], dtype=np.float32),
                "gripper_width": np.asarray([float(d["gripper_width"]) for d in info], dtype=np.float32),
                "prev_bilat": np.asarray([bool(d["prev_bilateral_contact"]) for d in info], dtype=bool),
                "curr_bilat": np.asarray([bool(d["bilateral_contact"]) for d in info], dtype=bool),
                "prev_orn": np.asarray([bool(d["prev_orn_ok"]) for d in info], dtype=bool),
                "curr_orn": np.asarray([bool(d["orn_ok"]) for d in info], dtype=bool),
                "prev_tcoll": np.asarray([bool(d["prev_table_collision"]) for d in info], dtype=bool),
                "curr_tcoll": np.asarray([bool(d["table_collision"]) for d in info], dtype=bool),
            }
        return None

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Union[dict, list, tuple, np.ndarray],
    ) -> Union[float, np.ndarray]:
        is_single = achieved_goal.ndim == 1
        ag = np.asarray(achieved_goal, dtype=np.float32).reshape(-1, 3)
        dg = np.asarray(desired_goal, dtype=np.float32).reshape(-1, 3)
        batch_size = ag.shape[0]
        cfg = self.pg_cfg

        ctx = self._extract_single_batched_info(info, batch_size)
        if ctx is None:
            dists = np.linalg.norm(ag - dg, axis=-1)
            rew = (-0.5 * np.clip(dists, 0.0, 1.0) + np.where(dists <= self.distance_threshold, 2.0, 0.0)).astype(np.float32)
            rew = np.nan_to_num(rew, nan=-1.0, posinf=5.0, neginf=-5.0)
            return float(rew[0]) if is_single else rew

        prev_ag = ctx["prev_ag"]
        ee_pos = ctx["ee_pos"]
        prev_ee_pos = ctx["prev_ee_pos"]
        prev_bilat = ctx["prev_bilat"]
        curr_bilat = ctx["curr_bilat"]
        prev_open = ctx["prev_gw"] > cfg.release_width_threshold
        curr_open = ctx["gripper_width"] > cfg.release_width_threshold

        ungrasped_curr = (~curr_bilat).astype(np.float32)
        ungrasped_both = ((~prev_bilat) & (~curr_bilat)).astype(np.float32)

        # A. Pre-grasp alignment
        xy_d_prev = np.linalg.norm(prev_ee_pos[:, :2] - prev_ag[:, :2], axis=-1)
        xy_d_curr = np.linalg.norm(ee_pos[:, :2] - ag[:, :2], axis=-1)
        r_align_prog = cfg.w_align_prog * np.clip(xy_d_prev - xy_d_curr, -cfg.step_delta_clip_m, cfg.step_delta_clip_m) * ungrasped_curr
        r_align_prox = -cfg.w_align_prox * np.clip(xy_d_curr, 0.0, 1.0) * ungrasped_curr
        r_align = r_align_prog + r_align_prox

        # B. Safe vertical approach (Phase B.1 continuous potential OR Phase B XY-gated delta)
        target_z = np.maximum(cfg.min_safe_ee_z, ag[:, 2] + cfg.grasp_z_offset)
        prev_target_z = np.maximum(cfg.min_safe_ee_z, prev_ag[:, 2] + cfg.grasp_z_offset)
        z_err_prev = np.abs(prev_ee_pos[:, 2] - prev_target_z)
        z_err_curr = np.abs(ee_pos[:, 2] - target_z)

        if cfg.use_vertical_potential:
            phi_z_prev = -cfg.w_vertical_potential * np.clip(z_err_prev, 0.0, 1.0)
            phi_z_curr = -cfg.w_vertical_potential * np.clip(z_err_curr, 0.0, 1.0)
            raw_z_prog = (phi_z_curr - phi_z_prev) * ungrasped_both
            # Stop positive vertical approach shaping once valid grasp height (above floor) is reached
            at_valid_grasp_height = (
                (z_err_prev <= cfg.vert_stop_tol_m)
                & (z_err_curr <= cfg.vert_stop_tol_m)
                & (ee_pos[:, 2] >= target_z - 0.002)
            )
            raw_z_prog = np.where(at_valid_grasp_height, 0.0, raw_z_prog)
            # Never reward positive vertical progress on table collision, closed gripper, or excessive downward dip below target_z
            safe_for_pos_descent = curr_open & (~ctx["curr_tcoll"]) & ctx["curr_orn"] & (ee_pos[:, 2] >= target_z - 0.002)
            r_vert_progress = np.where(raw_z_prog > 0.0, raw_z_prog * safe_for_pos_descent.astype(np.float32), raw_z_prog).astype(np.float32)
            r_vert_potential = (phi_z_curr * ungrasped_curr).astype(np.float32)
            r_vert = r_vert_progress
        else:
            safe_gate = ((xy_d_curr <= cfg.safe_vert_xy_tol) & curr_open & (~ctx["curr_tcoll"]) & ctx["curr_orn"] & (~curr_bilat)).astype(np.float32)
            r_vert = cfg.w_vert_approach * np.clip(z_err_prev - z_err_curr, -cfg.step_delta_clip_m, cfg.step_delta_clip_m) * safe_gate
            r_vert_progress = r_vert
            r_vert_potential = np.zeros_like(r_vert)

        # Grasp authorization predicate
        auth_prev = (xy_d_prev <= cfg.auth_xy_tol) & (np.abs(prev_ee_pos[:, 2] - np.maximum(cfg.min_safe_ee_z, prev_ag[:, 2])) <= cfg.auth_z_tol) & ctx["prev_orn"] & (~ctx["prev_tcoll"])
        auth_curr = (xy_d_curr <= cfg.auth_xy_tol) & (z_err_curr <= cfg.auth_z_tol) & ctx["curr_orn"] & (~ctx["curr_tcoll"])

        # C. Open-gripper compliance penalty
        premature = (~curr_bilat) & (~auth_curr) & (~curr_open)
        r_open_comp = -cfg.w_premature_close * premature.astype(np.float32)

        # D. Collision penalty
        r_coll = -cfg.w_table_coll * ctx["curr_tcoll"].astype(np.float32)

        # E. Ungrasped object displacement penalty
        obj_disp_xy = np.linalg.norm(ag[:, :2] - prev_ag[:, :2], axis=-1)
        r_disp = -cfg.w_disp * np.clip(obj_disp_xy - cfg.disp_deadzone_m, 0.0, cfg.step_delta_clip_m) * ungrasped_both

        # F. Grasp authorization milestone
        valid_prev = (auth_prev & prev_open & (~prev_bilat)).astype(np.float32)
        valid_curr = (auth_curr & curr_open & (~curr_bilat)).astype(np.float32)
        r_auth = cfg.w_auth * (valid_curr - valid_prev) * ungrasped_curr

        # G. Verified bilateral grasp transition
        r_grasp = cfg.w_grasp * (curr_bilat.astype(np.float32) - prev_bilat.astype(np.float32))

        # H. Bounded lift + lift-goal progress while held
        lift_span = max(cfg.lift_z_target - cfg.lift_z_base, 1e-4)
        h_prev = np.clip((prev_ag[:, 2] - cfg.lift_z_base) / lift_span, 0.0, 1.0)
        h_curr = np.clip((ag[:, 2] - cfg.lift_z_base) / lift_span, 0.0, 1.0)
        held_any = (curr_bilat | prev_bilat).astype(np.float32)
        r_lift = cfg.w_lift * (h_curr - h_prev) * held_any

        g_prev = np.linalg.norm(prev_ag - dg, axis=-1)
        g_curr = np.linalg.norm(ag - dg, axis=-1)
        r_goal_lift = cfg.w_transport * np.clip(g_prev - g_curr, -cfg.step_delta_clip_m, cfg.step_delta_clip_m) * held_any

        held_goal_prev = (g_prev <= self.distance_threshold) & (prev_ag[:, 2] >= cfg.hold_z_min) & prev_bilat
        held_goal_curr = (g_curr <= self.distance_threshold) & (ag[:, 2] >= cfg.hold_z_min) & curr_bilat
        r_hold = cfg.w_hold_goal * ((~held_goal_prev) & held_goal_curr).astype(np.float32) - cfg.w_drop * (held_goal_prev & (~held_goal_curr)).astype(np.float32)

        r_step = np.where(held_goal_curr, 0.0, cfg.step_penalty).astype(np.float32)
        total = r_align + r_vert + r_open_comp + r_coll + r_disp + r_auth + r_grasp + r_lift + r_goal_lift + r_hold + r_step
        total = np.nan_to_num(total, nan=0.0, posinf=20.0, neginf=-20.0).astype(np.float32)

        if is_single:
            self._last_reward_components = {
                "r_align": float(r_align[0]),
                "r_align_prog": float(r_align_prog[0]),
                "r_align_prox": float(r_align_prox[0]),
                "r_vert_approach": float(r_vert[0]),
                "r_vert_potential": float(r_vert_potential[0]),
                "r_vert_progress": float(r_vert_progress[0]),
                "r_open_comp": float(r_open_comp[0]),
                "r_collision": float(r_coll[0]),
                "r_disp": float(r_disp[0]),
                "r_auth": float(r_auth[0]),
                "r_grasp": float(r_grasp[0]),
                "r_lift": float(r_lift[0]),
                "r_goal_lift": float(r_goal_lift[0]),
                "r_hold": float(r_hold[0]),
                "r_step": float(r_step[0]),
                "total_reward": float(total[0]),
            }
            return float(total[0])
        return total

