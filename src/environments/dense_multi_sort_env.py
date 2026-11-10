"""Dense staged reward and corrected release-verification wrapper for PandaMultiSortEnv.

Phase 15.2.2: Preserves the original ``PandaMultiSortEnv`` benchmark, sparse reward,
and Phase 14 controllers untouched while providing bounded sequential dense rewards
and strict HER compatibility (``copy_info_dict=True`` transition mode + pure
``(ag, dg)`` fallback mode).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pybullet as p

from src.environments.multi_sort_env import PandaMultiSortEnv


class SortingStage(str, Enum):
    """Explicit sequential state machine stages for two-object sorting."""

    IDLE = "IDLE"
    SELECT_OBJECT_1 = "SELECT_OBJECT_1"
    REACH_OBJECT_1 = "REACH_OBJECT_1"
    GRASP_OBJECT_1 = "GRASP_OBJECT_1"
    LIFT_OBJECT_1 = "LIFT_OBJECT_1"
    TRANSPORT_OBJECT_1 = "TRANSPORT_OBJECT_1"
    PLACE_OBJECT_1 = "PLACE_OBJECT_1"
    VERIFY_OBJECT_1 = "VERIFY_OBJECT_1"
    SELECT_OBJECT_2 = "SELECT_OBJECT_2"
    REACH_OBJECT_2 = "REACH_OBJECT_2"
    GRASP_OBJECT_2 = "GRASP_OBJECT_2"
    LIFT_OBJECT_2 = "LIFT_OBJECT_2"
    TRANSPORT_OBJECT_2 = "TRANSPORT_OBJECT_2"
    PLACE_OBJECT_2 = "PLACE_OBJECT_2"
    VERIFY_OBJECT_2 = "VERIFY_OBJECT_2"
    COMPLETE = "COMPLETE"


@dataclass
class MultiSortRewardConfig:
    """Bounded weights and physical thresholds for ``PandaMultiSortDenseRewardEnv``."""

    w_reach: float = 4.0              # Potential weight for EE-to-active-object distance delta (per-step clip +-0.05m)
    w_proximity: float = 0.0          # Continuous proximity penalty (-w_proximity * clip(d_ee_obj, 0, 1)) when ungrasped
    w_grasp: float = 1.0              # Transition bonus/penalty for verified PyBullet bilateral finger contact
    w_lift: float = 1.5               # Potential weight for normalized elevation from 0.025m to 0.120m outside bin
    w_transport: float = 6.0          # Potential weight for active-object-to-assigned-bin distance delta
    w_place: float = 4.0              # One-time milestone bonus per object verified placed & released in its bin
    w_disturb: float = 4.5            # Asymmetric penalty if an already placed object is disturbed out of its bin
    w_wrong_bin: float = 2.0          # One-time transition penalty on releasing object into a wrong-colored bin
    w_complete: float = 8.0           # Terminal milestone bonus only when BOTH objects are simultaneously placed
    step_penalty: float = -0.02       # Small time penalty before both objects are placed

    step_delta_clip_m: float = 0.05   # Matches action_scale to keep step potential differences strictly bounded
    lift_z_base: float = 0.025        # Cube resting center height
    lift_z_target: float = 0.120      # Clearance height for bounded lift normalization
    bin_xy_gate_m: float = 0.060      # Inside bin XY radius, lift descent into bin is not penalized
    release_width_threshold: float = 0.065  # > 0.0506m pinched cube width, < 0.080m fully open aperture
    placed_z_max: float = 0.050       # Maximum cube center height to count as table/bin supported
    min_bilateral_force_n: float = 5.0      # Minimum sum of normal forces across f9 & f10 for real grasp
    reach_tol_m: float = 0.045        # EE-object reach threshold for state-machine reporting


class PandaMultiSortDenseRewardEnv(PandaMultiSortEnv):
    """Sequential two-object sorting environment with bounded HER-compatible dense rewards.

    Retains ``PandaMultiSortEnv`` physics, 31D observation space, 6D goal space, and 4D
    action space while fixing release detection (``width > 0.065``), aligning single/batched
    ``_get_info`` height checks, tracking a 16-state sequential sorting machine, and exposing
    goal-agnostic transition telemetry in ``info`` so HER ``future`` relabeling computes exact
    counterfactual rewards when ``copy_info_dict=True``.
    """

    def __init__(
        self,
        render_mode: str | None = None,
        num_objects: int = 2,
        distance_threshold: float = 0.05,
        max_episode_steps: int = 300,
        control_steps: int = 15,
        action_scale: float = 0.05,
        reward_config: Optional[MultiSortRewardConfig] = None,
    ) -> None:
        if num_objects != 2:
            raise ValueError("PandaMultiSortDenseRewardEnv currently supports strictly num_objects=2.")
        super().__init__(
            render_mode=render_mode,
            num_objects=2,
            distance_threshold=distance_threshold,
            max_episode_steps=max_episode_steps,
            control_steps=control_steps,
            action_scale=action_scale,
        )
        self.reward_cfg = reward_config or MultiSortRewardConfig()
        self.sorting_stage: SortingStage = SortingStage.IDLE
        self.obj1_verified: bool = False
        self.obj2_verified: bool = False
        self._last_reward_components: Dict[str, float] = {}

    def _get_bilateral_contacts(self) -> np.ndarray:
        """Query PyBullet finger contacts (links 9 and 10) for both cubes."""
        cid = self.sim.client_id
        robot_id = self.controller.robot.body_id
        if robot_id < 0:
            return np.zeros(self.num_objects, dtype=bool)
        min_f = self.reward_cfg.min_bilateral_force_n
        bilateral = np.zeros(self.num_objects, dtype=bool)
        for idx, cube in enumerate(self.cubes):
            if cube.body_id < 0:
                continue
            contacts = p.getContactPoints(bodyA=robot_id, bodyB=cube.body_id, physicsClientId=cid)
            f9 = [c for c in contacts if c[3] == 9]
            f10 = [c for c in contacts if c[3] == 10]
            if len(f9) > 0 and len(f10) > 0:
                total_fn = float(sum(c[9] for c in f9) + sum(c[9] for c in f10))
                if total_fn >= min_f:
                    bilateral[idx] = True
        return bilateral

    def _update_sorting_stage(
        self,
        ee_pos: np.ndarray,
        ag: np.ndarray,
        dg: np.ndarray,
        gw: float,
        bilateral: np.ndarray,
    ) -> SortingStage:
        """Update sequential two-object state machine cleanly without advancing on invalid placements."""
        cfg = self.reward_cfg
        ag_r = ag.reshape(2, 3)
        dg_r = dg.reshape(2, 3)
        dists = np.linalg.norm(ag_r - dg_r, axis=-1)
        ee_dists = np.linalg.norm(ag_r - ee_pos[None, :], axis=-1)
        is_rel_global = gw > cfg.release_width_threshold
        obj_released = np.array(
            [is_rel_global or ((not bool(bilateral[k])) and float(ee_dists[k]) > cfg.release_width_threshold) for k in range(2)],
            dtype=bool,
        )

        # Object 1 must remain placed; otherwise verification revokes cleanly
        p0_ok = bool(dists[0] <= self.distance_threshold and ag_r[0, 2] < cfg.placed_z_max)
        p1_ok = bool(dists[1] <= self.distance_threshold and ag_r[1, 2] < cfg.placed_z_max)

        if p0_ok and bool(obj_released[0]):
            self.obj1_verified = True
        elif not p0_ok:
            # Object 1 drifted or was knocked outside bin target
            self.obj1_verified = False

        if self.obj1_verified and p1_ok and bool(obj_released[1]):
            self.obj2_verified = True
        elif not p1_ok:
            self.obj2_verified = False

        if self.obj1_verified and self.obj2_verified:
            self.sorting_stage = SortingStage.COMPLETE
            return self.sorting_stage

        if not self.obj1_verified:
            pos = ag_r[0]
            goal = dg_r[0]
            d_goal = float(dists[0])
            d_ee = float(np.linalg.norm(ee_pos - pos))
            if p0_ok and not bool(obj_released[0]):
                self.sorting_stage = SortingStage.PLACE_OBJECT_1
            elif bool(bilateral[0]) and float(np.linalg.norm(pos[:2] - goal[:2])) <= cfg.bin_xy_gate_m:
                self.sorting_stage = SortingStage.PLACE_OBJECT_1
            elif bool(bilateral[0]) and pos[2] > 0.055:
                self.sorting_stage = SortingStage.TRANSPORT_OBJECT_1
            elif bool(bilateral[0]) and pos[2] > 0.032:
                self.sorting_stage = SortingStage.LIFT_OBJECT_1
            elif bool(bilateral[0]):
                self.sorting_stage = SortingStage.GRASP_OBJECT_1
            elif d_ee <= cfg.reach_tol_m:
                self.sorting_stage = SortingStage.REACH_OBJECT_1
            else:
                self.sorting_stage = SortingStage.SELECT_OBJECT_1 if self.current_step == 0 else SortingStage.REACH_OBJECT_1
            return self.sorting_stage

        # Object 1 verified -> Object 2 active
        pos = ag_r[1]
        goal = dg_r[1]
        d_ee = float(np.linalg.norm(ee_pos - pos))
        if p1_ok and not bool(obj_released[1]):
            self.sorting_stage = SortingStage.PLACE_OBJECT_2
        elif bool(bilateral[1]) and float(np.linalg.norm(pos[:2] - goal[:2])) <= cfg.bin_xy_gate_m:
            self.sorting_stage = SortingStage.PLACE_OBJECT_2
        elif bool(bilateral[1]) and pos[2] > 0.055:
            self.sorting_stage = SortingStage.TRANSPORT_OBJECT_2
        elif bool(bilateral[1]) and pos[2] > 0.032:
            self.sorting_stage = SortingStage.LIFT_OBJECT_2
        elif bool(bilateral[1]):
            self.sorting_stage = SortingStage.GRASP_OBJECT_2
        elif d_ee <= cfg.reach_tol_m:
            self.sorting_stage = SortingStage.REACH_OBJECT_2
        else:
            self.sorting_stage = SortingStage.SELECT_OBJECT_2 if d_ee > 0.12 and not bilateral[1] else SortingStage.REACH_OBJECT_2
        return self.sorting_stage

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        obs, base_info = super().reset(seed=seed, options=options)
        self.obj1_verified = False
        self.obj2_verified = False
        self.sorting_stage = SortingStage.IDLE
        ee_pos = obs["observation"][:3].copy()
        gw = float(obs["observation"][3])
        ag = obs["achieved_goal"].copy()
        bilateral = self._get_bilateral_contacts()

        transition_info: Dict[str, Any] = {
            "prev_ee_pos": ee_pos.astype(np.float32),
            "ee_pos": ee_pos.astype(np.float32),
            "prev_ag": ag.astype(np.float32),
            "prev_gw": float(gw),
            "gripper_width": float(gw),
            "prev_bilateral_contact": bilateral.astype(bool),
            "bilateral_contact": bilateral.astype(bool),
            "bin_positions": self.bin_positions.copy().astype(np.float32),
            "active_color_indices": np.array(self.active_color_indices, dtype=np.int32),
        }
        info = self._get_info(ag, self.goal)
        info.update(transition_info)
        info["sorting_stage"] = self.sorting_stage.value
        info["obj1_verified"] = False
        info["obj2_verified"] = False
        info["reward_components"] = {}
        return obs, info

    def _get_info(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        gripper_widths: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Consistent single & batched success checks with corrected width > 0.065m release gate."""
        cfg = self.reward_cfg
        is_batched = achieved_goal.ndim > 1

        if is_batched:
            batch_size = achieved_goal.shape[0]
            ag_r = achieved_goal.reshape(batch_size, self.num_objects, 3)
            dg_r = desired_goal.reshape(batch_size, self.num_objects, 3)
            distances = np.linalg.norm(ag_r - dg_r, axis=-1)
            height_ok = ag_r[..., 2] < cfg.placed_z_max
            if gripper_widths is not None:
                rel_ok = (np.asarray(gripper_widths, dtype=np.float32).reshape(-1, 1) > cfg.release_width_threshold)
            else:
                rel_ok = np.ones((batch_size, 1), dtype=bool)
            is_success_per_obj = (distances <= self.distance_threshold) & height_ok & rel_ok
            is_success = np.all(is_success_per_obj, axis=-1)
            mean_dist = np.mean(distances, axis=-1)
            return {
                "is_success": is_success,
                "distance": mean_dist,
                "is_success_per_obj": is_success_per_obj,
            }

        ag_r = achieved_goal.reshape(self.num_objects, 3)
        dg_r = desired_goal.reshape(self.num_objects, 3)
        distances = np.linalg.norm(ag_r - dg_r, axis=-1)
        gripper_state = self.controller.get_gripper_state()
        gw = float(gripper_state["width"])
        ee_pos = self.controller.get_ee_position()
        bilateral = self._get_bilateral_contacts()
        is_rel_global = gw > cfg.release_width_threshold

        objects_placed_correctly = 0
        objects_in_wrong_bin = 0
        per_obj_placed: List[bool] = []

        for idx in range(self.num_objects):
            pos = ag_r[idx]
            dist_ok = bool(distances[idx] <= self.distance_threshold)
            height_ok = bool(pos[2] < cfg.placed_z_max)
            ee_dist = float(np.linalg.norm(ee_pos - pos))
            obj_released = is_rel_global or ((not bool(bilateral[idx])) and ee_dist > cfg.release_width_threshold)
            placed_i = dist_ok and obj_released and height_ok
            per_obj_placed.append(placed_i)
            if placed_i:
                objects_placed_correctly += 1

            # Wrong bin or drop tracking (evaluated safely across episode)
            if self.object_last_z[idx] > 0.05 and pos[2] < 0.04 and not dist_ok:
                in_wrong = False
                for b_idx in range(3):
                    if b_idx != self.active_color_indices[idx]:
                        b_pos = self.bin_positions[b_idx]
                        if np.linalg.norm(pos[:2] - b_pos[:2]) < self.distance_threshold:
                            in_wrong = True
                            break
                if in_wrong:
                    objects_in_wrong_bin += 1
                else:
                    self.dropped_objects += 1

            self.object_last_z[idx] = pos[2]

        episode_success = (objects_placed_correctly == self.num_objects)
        return {
            "is_success": bool(episode_success),
            "distance": float(np.mean(distances)),
            "episode_length": self.current_step,
            "objects_placed": objects_placed_correctly,
            "wrong_bin_placements": objects_in_wrong_bin,
            "dropped_objects": self.dropped_objects,
            "collisions": self.collision_count,
            "is_success_per_obj": per_obj_placed,
            "is_released": bool(is_rel_global),
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        prev_ee_pos = self.controller.get_ee_position().copy().astype(np.float32)
        prev_gw = float(self.controller.get_gripper_state()["width"])
        prev_ag = np.concatenate([c.get_position() for c in self.cubes]).astype(np.float32)
        prev_bilateral = self._get_bilateral_contacts()

        action = np.clip(action, self.action_space.low, self.action_space.high)
        dx, dy, dz = action[:3] * self.action_scale
        gripper_action = action[3]

        target_ee_pos = prev_ee_pos + np.array([dx, dy, dz], dtype=np.float32)
        arm_workspace_min = np.array([0.2, -0.4, 0.0], dtype=np.float32)
        arm_workspace_max = np.array([0.8, 0.4, 0.5], dtype=np.float32)
        target_ee_pos = np.clip(target_ee_pos, arm_workspace_min, arm_workspace_max)

        if target_ee_pos[2] <= 0.01 and prev_ee_pos[2] <= 0.02 and dz < -0.01:
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

        info = self._get_info(curr_ag, self.goal)
        stage = self._update_sorting_stage(curr_ee_pos, curr_ag, self.goal, curr_gw, curr_bilateral)

        # Store goal-agnostic transition context for HER-compatible reward calculation
        info["prev_ee_pos"] = prev_ee_pos
        info["ee_pos"] = curr_ee_pos
        info["prev_ag"] = prev_ag
        info["prev_gw"] = prev_gw
        info["gripper_width"] = curr_gw
        info["prev_bilateral_contact"] = prev_bilateral
        info["bilateral_contact"] = curr_bilateral
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

    def _compute_pure_goal_fallback(self, ag: np.ndarray, dg: np.ndarray) -> np.ndarray:
        """Bounded, smooth (ag, dg)-only fallback when HER/callers pass empty info={}."""
        cfg = self.reward_cfg
        ag_r = ag.reshape(-1, 2, 3)
        dg_r = dg.reshape(-1, 2, 3)
        dists = np.linalg.norm(ag_r - dg_r, axis=-1)
        height_ok = ag_r[..., 2] < cfg.placed_z_max
        placed = (dists <= self.distance_threshold) & height_ok

        # Smooth distance shaping bounded in [-2.0, 0.0] + milestone bonuses
        dist_shaping = -0.5 * np.sum(np.clip(dists, 0.0, 1.0), axis=-1)
        lift_bonus = 0.25 * np.sum((ag_r[..., 2] > 0.05) & (dists > self.distance_threshold), axis=-1)
        per_obj_bonus = 2.0 * np.sum(placed.astype(np.float32), axis=-1)
        both_bonus = 5.0 * np.all(placed, axis=-1).astype(np.float32)
        step_pen = np.where(np.all(placed, axis=-1), 0.0, cfg.step_penalty)
        rewards = (dist_shaping + lift_bonus + per_obj_bonus + both_bonus + step_pen).astype(np.float32)
        return np.nan_to_num(rewards, nan=-2.0, posinf=10.0, neginf=-10.0)

    def _extract_batched_info(
        self, info: Union[dict, list, tuple, np.ndarray], batch_size: int
    ) -> Optional[Dict[str, np.ndarray]]:
        """Extract batched transition arrays from SB3 list-of-dicts or single-step dict info."""
        required = ("prev_ee_pos", "ee_pos", "prev_ag", "prev_gw", "gripper_width", "prev_bilateral_contact", "bilateral_contact")
        if isinstance(info, dict):
            if not all(k in info for k in required):
                return None
            return {
                "prev_ee_pos": np.asarray(info["prev_ee_pos"], dtype=np.float32).reshape(batch_size, 3),
                "ee_pos": np.asarray(info["ee_pos"], dtype=np.float32).reshape(batch_size, 3),
                "prev_ag": np.asarray(info["prev_ag"], dtype=np.float32).reshape(batch_size, 2, 3),
                "prev_gw": np.asarray(info["prev_gw"], dtype=np.float32).reshape(batch_size),
                "gripper_width": np.asarray(info["gripper_width"], dtype=np.float32).reshape(batch_size),
                "prev_bilateral_contact": np.asarray(info["prev_bilateral_contact"], dtype=bool).reshape(batch_size, 2),
                "bilateral_contact": np.asarray(info["bilateral_contact"], dtype=bool).reshape(batch_size, 2),
                "bin_positions": np.asarray(
                    info.get("bin_positions", np.zeros((3, 3), dtype=np.float32)), dtype=np.float32
                ).reshape(batch_size, 3, 3),
                "active_color_indices": np.asarray(
                    info.get("active_color_indices", np.array([0, 1], dtype=np.int32)), dtype=np.int32
                ).reshape(batch_size, 2),
            }

        if isinstance(info, (list, tuple, np.ndarray)) and len(info) == batch_size:
            if not all(isinstance(d, dict) and all(k in d for k in required) for d in info):
                return None
            return {
                "prev_ee_pos": np.stack([np.asarray(d["prev_ee_pos"], dtype=np.float32) for d in info]),
                "ee_pos": np.stack([np.asarray(d["ee_pos"], dtype=np.float32) for d in info]),
                "prev_ag": np.stack([np.asarray(d["prev_ag"], dtype=np.float32).reshape(2, 3) for d in info]),
                "prev_gw": np.asarray([float(d["prev_gw"]) for d in info], dtype=np.float32),
                "gripper_width": np.asarray([float(d["gripper_width"]) for d in info], dtype=np.float32),
                "prev_bilateral_contact": np.stack([np.asarray(d["prev_bilateral_contact"], dtype=bool) for d in info]),
                "bilateral_contact": np.stack([np.asarray(d["bilateral_contact"], dtype=bool) for d in info]),
                "bin_positions": np.stack([
                    np.asarray(d.get("bin_positions", np.zeros((3, 3), dtype=np.float32)), dtype=np.float32)
                    for d in info
                ]),
                "active_color_indices": np.stack([
                    np.asarray(d.get("active_color_indices", np.array([0, 1], dtype=np.int32)), dtype=np.int32)
                    for d in info
                ]),
            }
        return None

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Union[dict, list, tuple, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Compute bounded, HER-compatible staged sequential sorting reward.

        Supports:
          1. Exact counterfactual evaluation using goal-agnostic transition context in ``info``
             (triggered during ``env.step()`` and ``HerReplayBuffer(..., copy_info_dict=True)``).
          2. Pure ``(achieved_goal, desired_goal)`` bounded fallback when ``info`` is empty.
        """
        is_single = achieved_goal.ndim == 1
        ag = np.asarray(achieved_goal, dtype=np.float32)
        dg = np.asarray(desired_goal, dtype=np.float32)
        if is_single:
            ag = ag.reshape(1, 6)
            dg = dg.reshape(1, 6)

        batch_size = ag.shape[0]
        ctx = self._extract_batched_info(info, batch_size)
        if ctx is None:
            rewards = self._compute_pure_goal_fallback(ag, dg)
            return float(rewards[0]) if is_single else rewards

        cfg = self.reward_cfg
        ag_r = ag.reshape(batch_size, 2, 3)
        dg_r = dg.reshape(batch_size, 2, 3)
        prev_ag = ctx["prev_ag"]
        ee_pos = ctx["ee_pos"]
        prev_ee_pos = ctx["prev_ee_pos"]
        prev_bilat = ctx["prev_bilateral_contact"]
        curr_bilat = ctx["bilateral_contact"]

        prev_rel_global = ctx["prev_gw"] > cfg.release_width_threshold
        curr_rel_global = ctx["gripper_width"] > cfg.release_width_threshold

        # Per-object release: either gripper is globally open (> 0.065m), or EE moved clear (> 0.065m) without contact
        prev_ee_obj_dists = np.linalg.norm(prev_ag - prev_ee_pos[:, None, :], axis=-1)  # (B, 2)
        curr_ee_obj_dists = np.linalg.norm(ag_r - ee_pos[:, None, :], axis=-1)          # (B, 2)
        prev_rel = prev_rel_global[:, None] | ((~prev_bilat) & (prev_ee_obj_dists > cfg.release_width_threshold))
        curr_rel = curr_rel_global[:, None] | ((~curr_bilat) & (curr_ee_obj_dists > cfg.release_width_threshold))

        d_prev = np.linalg.norm(prev_ag - dg_r, axis=-1)  # (B, 2)
        d_curr = np.linalg.norm(ag_r - dg_r, axis=-1)     # (B, 2)

        placed_prev = (d_prev <= self.distance_threshold) & (prev_ag[..., 2] < cfg.placed_z_max) & prev_rel
        placed_curr = (d_curr <= self.distance_threshold) & (ag_r[..., 2] < cfg.placed_z_max) & curr_rel

        # Select active object k dynamically from (prev_ag, dg) — never hidden stateful counters
        # k=0 until obj 0 placed; then k=1 until obj 1 placed
        use_obj1 = placed_prev[:, 0] & (~placed_prev[:, 1])
        both_prev = placed_prev[:, 0] & placed_prev[:, 1]
        both_curr = placed_curr[:, 0] & placed_curr[:, 1]
        active_idx = np.where(use_obj1, 1, 0)  # (B,)
        active_mask = (~both_prev).astype(np.float32)

        b_arange = np.arange(batch_size)
        act_prev_pos = prev_ag[b_arange, active_idx]      # (B, 3)
        act_curr_pos = ag_r[b_arange, active_idx]         # (B, 3)
        act_goal = dg_r[b_arange, active_idx]             # (B, 3)
        act_bilat_prev = prev_bilat[b_arange, active_idx] # (B,)
        act_bilat_curr = curr_bilat[b_arange, active_idx] # (B,)

        # Stage A: Reach potential + optional continuous proximity (when active object is ungrasped & incomplete)
        ee_d_prev = np.linalg.norm(prev_ee_pos - act_prev_pos, axis=-1)
        ee_d_curr = np.linalg.norm(ee_pos - act_curr_pos, axis=-1)
        delta_reach = np.clip(ee_d_prev - ee_d_curr, -cfg.step_delta_clip_m, cfg.step_delta_clip_m)
        ungrasped = (~act_bilat_curr).astype(np.float32)
        incomplete_curr = (~both_curr).astype(np.float32)
        r_reach_progress = cfg.w_reach * delta_reach * ungrasped * active_mask
        r_proximity = -cfg.w_proximity * np.clip(ee_d_curr, 0.0, 1.0) * ungrasped * incomplete_curr * active_mask
        r_reach = r_reach_progress + r_proximity

        # Stage B: Verified bilateral grasp acquisition / loss transition
        r_grasp = cfg.w_grasp * (act_bilat_curr.astype(np.float32) - act_bilat_prev.astype(np.float32)) * active_mask

        # Stage C: Bounded lift potential (gated on grasp contact & outside bin XY drop zone)
        lift_span = max(cfg.lift_z_target - cfg.lift_z_base, 1e-4)
        h_prev = np.clip((act_prev_pos[:, 2] - cfg.lift_z_base) / lift_span, 0.0, 1.0)
        h_curr = np.clip((act_curr_pos[:, 2] - cfg.lift_z_base) / lift_span, 0.0, 1.0)
        xy_goal_curr = np.linalg.norm(act_curr_pos[:, :2] - act_goal[:, :2], axis=-1)
        outside_bin = (xy_goal_curr > cfg.bin_xy_gate_m).astype(np.float32)
        held_any = (act_bilat_curr | act_bilat_prev).astype(np.float32)
        r_lift = cfg.w_lift * (h_curr - h_prev) * outside_bin * held_any * active_mask

        # Stage D: Transport potential strictly toward assigned bin dg[active_idx]
        g_d_prev = d_prev[b_arange, active_idx]
        g_d_curr = d_curr[b_arange, active_idx]
        delta_trans = np.clip(g_d_prev - g_d_curr, -cfg.step_delta_clip_m, cfg.step_delta_clip_m)
        r_transport = cfg.w_transport * delta_trans * active_mask

        # Wrong-bin deterrence (only when active/non-matching bin positions are distinct from desired_goal)
        r_wrong_bin = np.zeros(batch_size, dtype=np.float32)
        bins = ctx["bin_positions"]  # (B, 3, 3)
        c_idx = ctx["active_color_indices"]  # (B, 2)
        if np.any(np.abs(bins) > 1e-6):
            for k in range(2):
                for b in range(3):
                    is_wrong_color = (c_idx[:, k] != b)
                    # Also ensure HER virtual goal wasn't relabeled ONTO bin b
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

        # Stage E: Per-object milestone placement / disturbance penalty
        newly_placed = (~placed_prev) & placed_curr
        newly_disturbed = placed_prev & (~placed_curr)
        r_place = cfg.w_place * np.sum(newly_placed.astype(np.float32), axis=-1) - cfg.w_disturb * np.sum(
            newly_disturbed.astype(np.float32), axis=-1
        )

        # Stage F: Simultaneous two-object terminal completion transition
        newly_complete = (~both_prev) & both_curr
        lost_complete = both_prev & (~both_curr)
        r_complete = cfg.w_complete * (newly_complete.astype(np.float32) - lost_complete.astype(np.float32))

        r_step = np.where(both_curr, 0.0, cfg.step_penalty).astype(np.float32)

        total = r_reach + r_grasp + r_lift + r_transport + r_wrong_bin + r_place + r_complete + r_step
        total = np.nan_to_num(total, nan=0.0, posinf=20.0, neginf=-20.0).astype(np.float32)

        if is_single:
            self._last_reward_components = {
                "r_reach": float(r_reach[0]),
                "r_reach_progress": float(r_reach_progress[0]),
                "r_proximity": float(r_proximity[0]),
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


@dataclass
class MultiSortProximityRewardConfig(MultiSortRewardConfig):
    """Phase 15.2.4 reward configuration adding continuous EE-to-active-object proximity."""

    w_proximity: float = 0.25


class PandaMultiSortDenseV2Env(PandaMultiSortDenseRewardEnv):
    """Versioned Phase 15.2.4 dense sorting environment with continuous reach proximity."""

    def __init__(
        self,
        render_mode: str | None = None,
        num_objects: int = 2,
        distance_threshold: float = 0.05,
        max_episode_steps: int = 300,
        control_steps: int = 15,
        action_scale: float = 0.05,
        reward_config: Optional[MultiSortRewardConfig] = None,
    ) -> None:
        super().__init__(
            render_mode=render_mode,
            num_objects=num_objects,
            distance_threshold=distance_threshold,
            max_episode_steps=max_episode_steps,
            control_steps=control_steps,
            action_scale=action_scale,
            reward_config=reward_config or MultiSortProximityRewardConfig(),
        )
