"""Collision-aware Pre-Grasp Dense Reward Environment for Phase 15.2.5A.

Preserves original ``PandaMultiSortEnv`` and Phase 15.2.2/15.2.4 environments untouched.
Adds 8 HER-recomputable pre-grasp and collision-aware reward components (A-H) using
goal-agnostic per-step transition telemetry stored in ``info`` (for ``copy_info_dict=True``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pybullet as p

from src.environments.dense_multi_sort_env import (
    MultiSortRewardConfig,
    PandaMultiSortDenseRewardEnv,
)


@dataclass
class PreGraspRewardConfig(MultiSortRewardConfig):
    """Phase 15.2.5A collision-aware pre-grasp & grasp-authorization reward configuration."""

    # Stage A: Horizontal pre-grasp alignment + proximity
    w_align_prog: float = 4.0         # Progress potential on XY error to active ungrasped object
    w_align_prox: float = 0.25        # Continuous XY distance shaping (-0.25 * clip(xy_dist, 0, 1))
    # Stage B: Safe vertical descent (gated on XY align + open gripper + zero table collision + orn)
    w_vert_approach: float = 2.0      # Progress potential on |ee_z - target_grasp_z|
    safe_vert_xy_tol: float = 0.030   # Must align within 3cm XY before vertical descent is rewarded
    # Stage C: Open-gripper compliance
    w_premature_close: float = 0.06   # Per-step penalty for closing gripper before grasp authorization
    # Stage D: Collision penalties
    w_table_coll: float = 0.10        # Per-step penalty for table contact / floor scrape
    w_other_obj_coll: float = 0.08    # Per-step penalty for unintended contact with inactive cube
    # Stage E: Ungrasped active-object displacement penalty
    w_disp: float = 6.0               # Penalty * clip(disp_xy - deadzone, 0, 0.05) on knocking cube
    disp_deadzone_m: float = 0.003
    # Stage F: Grasp authorization transition bonus
    w_auth: float = 0.50              # Potential bonus on entering valid open-gripper pre-grasp pose
    auth_xy_tol: float = 0.018
    auth_z_tol: float = 0.015
    min_orn_dot: float = 0.985
    min_safe_ee_z: float = 0.0205
    # Stage G & H: Bilateral grasp & lift
    w_grasp: float = 1.5
    w_lift: float = 1.5


class PandaMultiSortPreGraspEnv(PandaMultiSortDenseRewardEnv):
    """Multi-object sorting env with collision-aware pre-grasp rewards and HER auditability."""

    def __init__(
        self,
        render_mode: str | None = None,
        num_objects: int = 2,
        distance_threshold: float = 0.05,
        max_episode_steps: int = 300,
        control_steps: int = 15,
        action_scale: float = 0.05,
        reward_config: Optional[PreGraspRewardConfig] = None,
    ) -> None:
        self.pg_cfg = reward_config or PreGraspRewardConfig()
        super().__init__(
            render_mode=render_mode,
            num_objects=num_objects,
            distance_threshold=distance_threshold,
            max_episode_steps=max_episode_steps,
            control_steps=control_steps,
            action_scale=action_scale,
            reward_config=self.pg_cfg,
        )

    def _query_safety_flags(self, floor_scrape: bool = False) -> Tuple[bool, bool, np.ndarray]:
        """Return (orn_ok, table_collision, any_cube_contact[2]) from PyBullet."""
        cid = self.sim.client_id
        robot_id = self.controller.robot.body_id
        table_id = self.sim.table_id
        if robot_id < 0:
            return True, False, np.zeros(2, dtype=bool)

        ee_orn = np.asarray(self.controller.get_ee_orientation(), dtype=np.float32)
        default_orn = np.asarray(self.controller.robot.default_ee_orientation, dtype=np.float32)
        orn_ok = bool(abs(float(np.dot(ee_orn, default_orn))) >= self.pg_cfg.min_orn_dot)

        tc_count = len(p.getContactPoints(bodyA=robot_id, bodyB=table_id, physicsClientId=cid)) if table_id >= 0 else 0
        table_collision = bool((tc_count > 0) or floor_scrape)

        any_cube_contact = np.zeros(self.num_objects, dtype=bool)
        for idx, cube in enumerate(self.cubes):
            if cube.body_id >= 0:
                if len(p.getContactPoints(bodyA=robot_id, bodyB=cube.body_id, physicsClientId=cid)) > 0:
                    any_cube_contact[idx] = True
        return orn_ok, table_collision, any_cube_contact

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        orn_ok, table_coll, any_contact = self._query_safety_flags(floor_scrape=False)
        info["prev_orn_ok"] = orn_ok
        info["orn_ok"] = orn_ok
        info["prev_table_collision"] = table_coll
        info["table_collision"] = table_coll
        info["prev_any_cube_contact"] = any_contact.copy()
        info["any_cube_contact"] = any_contact.copy()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        prev_ee_pos = self.controller.get_ee_position().copy().astype(np.float32)
        prev_gw = float(self.controller.get_gripper_state()["width"])
        prev_ag = np.concatenate([c.get_position() for c in self.cubes]).astype(np.float32)
        prev_bilateral = self._get_bilateral_contacts()
        prev_orn_ok, prev_table_coll, prev_any_contact = self._query_safety_flags(floor_scrape=False)

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
        curr_bilateral = self._get_bilateral_contacts()
        curr_orn_ok, curr_table_coll, curr_any_contact = self._query_safety_flags(floor_scrape=floor_scrape)

        info = self._get_info(curr_ag, self.goal)
        stage = self._update_sorting_stage(curr_ee_pos, curr_ag, self.goal, curr_gw, curr_bilateral)

        # Goal-agnostic physical transition telemetry for stateless HER recomputation
        info["prev_ee_pos"] = prev_ee_pos
        info["ee_pos"] = curr_ee_pos
        info["prev_ag"] = prev_ag
        info["prev_gw"] = prev_gw
        info["gripper_width"] = curr_gw
        info["prev_bilateral_contact"] = prev_bilateral
        info["bilateral_contact"] = curr_bilateral
        info["prev_orn_ok"] = prev_orn_ok
        info["orn_ok"] = curr_orn_ok
        info["prev_table_collision"] = prev_table_coll
        info["table_collision"] = curr_table_coll
        info["prev_any_cube_contact"] = prev_any_contact
        info["any_cube_contact"] = curr_any_contact
        info["bin_positions"] = self.bin_positions.copy().astype(np.float32)
        info["active_color_indices"] = np.array(self.active_color_indices, dtype=np.int32)
        info["sorting_stage"] = stage.value
        info["obj1_verified"] = bool(self.obj1_verified)
        info["obj2_verified"] = bool(self.obj2_verified)

        reward = float(self.compute_reward(curr_ag, self.goal, info))
        info["reward_components"] = dict(self._last_reward_components)

        terminated = False
        truncated = self.current_step >= self.max_episode_steps
        return obs, reward, terminated, truncated, info

    def _extract_pg_batched_info(
        self, info: Union[dict, list, tuple, np.ndarray], batch_size: int
    ) -> Optional[Dict[str, np.ndarray]]:
        base = self._extract_batched_info(info, batch_size)
        if base is None:
            return None

        pg_req = ("orn_ok", "prev_orn_ok", "table_collision", "prev_table_collision", "any_cube_contact")
        if isinstance(info, dict):
            if not all(k in info for k in pg_req):
                return None
            base["prev_orn_ok"] = np.asarray(info["prev_orn_ok"], dtype=bool).reshape(batch_size)
            base["orn_ok"] = np.asarray(info["orn_ok"], dtype=bool).reshape(batch_size)
            base["prev_table_collision"] = np.asarray(info["prev_table_collision"], dtype=bool).reshape(batch_size)
            base["table_collision"] = np.asarray(info["table_collision"], dtype=bool).reshape(batch_size)
            base["any_cube_contact"] = np.asarray(info["any_cube_contact"], dtype=bool).reshape(batch_size, 2)
            return base

        if isinstance(info, (list, tuple, np.ndarray)) and len(info) == batch_size:
            if not all(isinstance(d, dict) and all(k in d for k in pg_req) for d in info):
                return None
            base["prev_orn_ok"] = np.asarray([bool(d["prev_orn_ok"]) for d in info], dtype=bool)
            base["orn_ok"] = np.asarray([bool(d["orn_ok"]) for d in info], dtype=bool)
            base["prev_table_collision"] = np.asarray([bool(d["prev_table_collision"]) for d in info], dtype=bool)
            base["table_collision"] = np.asarray([bool(d["table_collision"]) for d in info], dtype=bool)
            base["any_cube_contact"] = np.stack([np.asarray(d["any_cube_contact"], dtype=bool) for d in info])
            return base
        return None

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Union[dict, list, tuple, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Compute bounded, HER-recomputable collision-aware pre-grasp + sorting reward."""
        is_single = achieved_goal.ndim == 1
        ag = np.asarray(achieved_goal, dtype=np.float32)
        dg = np.asarray(desired_goal, dtype=np.float32)
        if is_single:
            ag = ag.reshape(1, 6)
            dg = dg.reshape(1, 6)

        batch_size = ag.shape[0]
        ctx = self._extract_pg_batched_info(info, batch_size)
        if ctx is None:
            rewards = self._compute_pure_goal_fallback(ag, dg)
            return float(rewards[0]) if is_single else rewards

        cfg = self.pg_cfg
        ag_r = ag.reshape(batch_size, 2, 3)
        dg_r = dg.reshape(batch_size, 2, 3)
        prev_ag = ctx["prev_ag"]
        ee_pos = ctx["ee_pos"]
        prev_ee_pos = ctx["prev_ee_pos"]
        prev_bilat = ctx["prev_bilateral_contact"]
        curr_bilat = ctx["bilateral_contact"]

        prev_rel_global = ctx["prev_gw"] > cfg.release_width_threshold
        curr_rel_global = ctx["gripper_width"] > cfg.release_width_threshold
        prev_ee_obj_dists = np.linalg.norm(prev_ag - prev_ee_pos[:, None, :], axis=-1)
        curr_ee_obj_dists = np.linalg.norm(ag_r - ee_pos[:, None, :], axis=-1)
        prev_rel = prev_rel_global[:, None] | ((~prev_bilat) & (prev_ee_obj_dists > cfg.release_width_threshold))
        curr_rel = curr_rel_global[:, None] | ((~curr_bilat) & (curr_ee_obj_dists > cfg.release_width_threshold))

        d_prev = np.linalg.norm(prev_ag - dg_r, axis=-1)
        d_curr = np.linalg.norm(ag_r - dg_r, axis=-1)
        placed_prev = (d_prev <= self.distance_threshold) & (prev_ag[..., 2] < cfg.placed_z_max) & prev_rel
        placed_curr = (d_curr <= self.distance_threshold) & (ag_r[..., 2] < cfg.placed_z_max) & curr_rel

        use_obj1 = placed_prev[:, 0] & (~placed_prev[:, 1])
        both_prev = placed_prev[:, 0] & placed_prev[:, 1]
        both_curr = placed_curr[:, 0] & placed_curr[:, 1]
        active_idx = np.where(use_obj1, 1, 0)
        other_idx = 1 - active_idx
        active_mask = (~both_prev).astype(np.float32)
        incomplete_curr = (~both_curr).astype(np.float32)

        b_arange = np.arange(batch_size)
        act_prev_pos = prev_ag[b_arange, active_idx]
        act_curr_pos = ag_r[b_arange, active_idx]
        act_goal = dg_r[b_arange, active_idx]
        act_bilat_prev = prev_bilat[b_arange, active_idx]
        act_bilat_curr = curr_bilat[b_arange, active_idx]
        act_placed_curr = placed_curr[b_arange, active_idx]

        ungrasped_curr = (~act_bilat_curr).astype(np.float32)
        ungrasped_both = ((~act_bilat_prev) & (~act_bilat_curr)).astype(np.float32)

        # A. Pre-grasp horizontal alignment progress + proximity
        xy_d_prev = np.linalg.norm(prev_ee_pos[:, :2] - act_prev_pos[:, :2], axis=-1)
        xy_d_curr = np.linalg.norm(ee_pos[:, :2] - act_curr_pos[:, :2], axis=-1)
        delta_xy = np.clip(xy_d_prev - xy_d_curr, -cfg.step_delta_clip_m, cfg.step_delta_clip_m)
        r_align_prog = cfg.w_align_prog * delta_xy * ungrasped_curr * incomplete_curr * active_mask
        r_align_prox = -cfg.w_align_prox * np.clip(xy_d_curr, 0.0, 1.0) * ungrasped_curr * incomplete_curr * active_mask
        r_align = r_align_prog + r_align_prox

        # B. Safe vertical approach toward grasp height (gated on XY align + open + no table coll + orn)
        target_z = np.maximum(cfg.min_safe_ee_z, act_curr_pos[:, 2])
        z_err_prev = np.abs(prev_ee_pos[:, 2] - target_z)
        z_err_curr = np.abs(ee_pos[:, 2] - target_z)
        delta_z = np.clip(z_err_prev - z_err_curr, -cfg.step_delta_clip_m, cfg.step_delta_clip_m)
        safe_vert_gate = (
            (xy_d_curr <= cfg.safe_vert_xy_tol)
            & curr_rel_global
            & (~ctx["table_collision"])
            & ctx["orn_ok"]
            & (~act_bilat_curr)
            & (~both_curr)
            & (~both_prev)
        ).astype(np.float32)
        r_vert_approach = cfg.w_vert_approach * delta_z * safe_vert_gate

        # Grasp authorization geometry predicate (prev and curr)
        auth_prev = (
            (xy_d_prev <= cfg.auth_xy_tol)
            & (np.abs(prev_ee_pos[:, 2] - np.maximum(cfg.min_safe_ee_z, act_prev_pos[:, 2])) <= cfg.auth_z_tol)
            & ctx["prev_orn_ok"]
            & (~ctx["prev_table_collision"])
        )
        auth_curr = (
            (xy_d_curr <= cfg.auth_xy_tol)
            & (z_err_curr <= cfg.auth_z_tol)
            & ctx["orn_ok"]
            & (~ctx["table_collision"])
        )

        # C. Open-gripper compliance: penalize premature closure outside grasp authorization
        premature_close = (~act_bilat_curr) & (~auth_curr) & (~curr_rel_global) & (~both_curr) & (~both_prev)
        r_open_comp = -cfg.w_premature_close * premature_close.astype(np.float32)

        # D. Collision penalty (table contact / floor scrape + unintended contact with inactive cube)
        other_hit = ctx["any_cube_contact"][b_arange, other_idx] & (~both_curr) & (~both_prev)
        r_collision = (
            -cfg.w_table_coll * ctx["table_collision"].astype(np.float32)
            - cfg.w_other_obj_coll * other_hit.astype(np.float32)
        )

        # E. Active object unintended displacement penalty during ungrasped pre-grasp
        obj_disp_xy = np.linalg.norm(act_curr_pos[:, :2] - act_prev_pos[:, :2], axis=-1)
        excess_disp = np.clip(obj_disp_xy - cfg.disp_deadzone_m, 0.0, cfg.step_delta_clip_m)
        r_disp = -cfg.w_disp * excess_disp * ungrasped_both * (~act_placed_curr).astype(np.float32) * incomplete_curr * active_mask

        # F. Grasp authorization milestone (zeroed on bilateral grasp step so +w_auth isn't undone on grasp)
        valid_pregrasp_prev = (auth_prev & prev_rel_global & (~act_bilat_prev) & (~both_prev)).astype(np.float32)
        valid_pregrasp_curr = (auth_curr & curr_rel_global & (~act_bilat_curr) & (~both_curr)).astype(np.float32)
        r_auth = cfg.w_auth * (valid_pregrasp_curr - valid_pregrasp_prev) * ungrasped_curr * incomplete_curr * active_mask

        # G. Verified bilateral grasp transition
        r_grasp = cfg.w_grasp * (act_bilat_curr.astype(np.float32) - act_bilat_prev.astype(np.float32)) * active_mask

        # H. Bounded lift potential
        lift_span = max(cfg.lift_z_target - cfg.lift_z_base, 1e-4)
        h_prev = np.clip((act_prev_pos[:, 2] - cfg.lift_z_base) / lift_span, 0.0, 1.0)
        h_curr = np.clip((act_curr_pos[:, 2] - cfg.lift_z_base) / lift_span, 0.0, 1.0)
        xy_goal_curr = np.linalg.norm(act_curr_pos[:, :2] - act_goal[:, :2], axis=-1)
        outside_bin = (xy_goal_curr > cfg.bin_xy_gate_m).astype(np.float32)
        held_any = (act_bilat_curr | act_bilat_prev).astype(np.float32)
        r_lift = cfg.w_lift * (h_curr - h_prev) * outside_bin * held_any * active_mask

        # Transport, Wrong-Bin, Placement, Complete, Step (preserved cleanly from Phase 15.2.2)
        g_d_prev = d_prev[b_arange, active_idx]
        g_d_curr = d_curr[b_arange, active_idx]
        delta_trans = np.clip(g_d_prev - g_d_curr, -cfg.step_delta_clip_m, cfg.step_delta_clip_m)
        r_transport = cfg.w_transport * delta_trans * active_mask

        r_wrong_bin = np.zeros(batch_size, dtype=np.float32)
        bins = ctx["bin_positions"]
        c_idx = ctx["active_color_indices"]
        if np.any(np.abs(bins) > 1e-6):
            for k in range(2):
                for b in range(3):
                    is_wrong_color = (c_idx[:, k] != b)
                    not_relabeled_to_b = np.linalg.norm(dg_r[:, k, :2] - bins[:, b, :2], axis=-1) > self.distance_threshold
                    in_b_prev = (
                        (np.linalg.norm(prev_ag[:, k, :2] - bins[:, b, :2], axis=-1) <= self.distance_threshold)
                        & (prev_ag[:, k, 2] < cfg.placed_z_max)
                        & prev_rel[:, k]
                    )
                    in_b_curr = (
                        (np.linalg.norm(ag_r[:, k, :2] - bins[:, b, :2], axis=-1) <= self.distance_threshold)
                        & (ag_r[:, k, 2] < cfg.placed_z_max)
                        & curr_rel[:, k]
                    )
                    entered_wrong = (~in_b_prev) & in_b_curr & is_wrong_color & not_relabeled_to_b
                    r_wrong_bin -= cfg.w_wrong_bin * entered_wrong.astype(np.float32)

        newly_placed = (~placed_prev) & placed_curr
        newly_disturbed = placed_prev & (~placed_curr)
        r_place = cfg.w_place * np.sum(newly_placed.astype(np.float32), axis=-1) - cfg.w_disturb * np.sum(
            newly_disturbed.astype(np.float32), axis=-1
        )
        r_complete = cfg.w_complete * (
            ((~both_prev) & both_curr).astype(np.float32) - (both_prev & (~both_curr)).astype(np.float32)
        )
        r_step = np.where(both_curr, 0.0, cfg.step_penalty).astype(np.float32)

        total = (
            r_align
            + r_vert_approach
            + r_open_comp
            + r_collision
            + r_disp
            + r_auth
            + r_grasp
            + r_lift
            + r_transport
            + r_wrong_bin
            + r_place
            + r_complete
            + r_step
        )
        total = np.nan_to_num(total, nan=0.0, posinf=20.0, neginf=-20.0).astype(np.float32)

        if is_single:
            self._last_reward_components = {
                "r_align": float(r_align[0]),
                "r_align_prog": float(r_align_prog[0]),
                "r_align_prox": float(r_align_prox[0]),
                "r_vert_approach": float(r_vert_approach[0]),
                "r_open_comp": float(r_open_comp[0]),
                "r_collision": float(r_collision[0]),
                "r_disp": float(r_disp[0]),
                "r_auth": float(r_auth[0]),
                "r_grasp": float(r_grasp[0]),
                "r_lift": float(r_lift[0]),
                "r_transport": float(r_transport[0]),
                "r_wrong_bin": float(r_wrong_bin[0]),
                "r_place": float(r_place[0]),
                "r_complete": float(r_complete[0]),
                "r_step": float(r_step[0]),
                "total_reward": float(total[0]),
                "active_obj_idx": int(active_idx[0]),
            }
            return float(total[0])
        return total

