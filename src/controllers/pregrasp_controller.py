"""Modular Pre-Grasp Acquisition State Machine (Phase 15.2.5A).

Implements a deterministic, safety-checked 6-stage pre-grasp controller:
  APPROACH -> ALIGNMENT -> CONTROLLED_DESCENT -> GRASP_AUTHORIZATION -> GRASP -> LIFT_VERIFICATION

Monitors real PyBullet table/object contacts, gripper orientation, XY/Z centering,
and unintended object displacement. Leaves Phase 14 controllers untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pybullet as p


class PreGraspStage(str, Enum):
    APPROACH = "APPROACH"
    ALIGNMENT = "ALIGNMENT"
    CONTROLLED_DESCENT = "CONTROLLED_DESCENT"
    GRASP_AUTHORIZATION = "GRASP_AUTHORIZATION"
    GRASP = "GRASP"
    LIFT_VERIFICATION = "LIFT_VERIFICATION"
    ACQUIRED = "ACQUIRED"


@dataclass
class PreGraspConfig:
    """Configurable thresholds and safety tolerances for PreGraspController."""

    approach_clearance_z: float = 0.115    # Overhead approach clearance above cube center (~0.140m)
    align_clearance_z: float = 0.055       # Fine pre-descent alignment hover height above cube (~0.080m)
    grasp_z_offset: float = 0.000          # Target grasp EE Z relative to cube center (0.025m cube mid-height)
    min_safe_ee_z: float = 0.0205          # Strictly above ws_z minimum (0.020m) to avoid floor push

    approach_xy_tol: float = 0.020
    approach_z_tol: float = 0.025
    align_xy_tol: float = 0.011            # Tight horizontal centering prior to descent
    align_z_tol: float = 0.018
    grasp_xy_tol: float = 0.015
    grasp_z_tol: float = 0.012
    min_orn_dot: float = 0.985             # |<ee_quat, default_downward_quat>| >= 0.985

    max_approach_cmd: float = 0.90
    max_align_cmd: float = 0.55
    max_descent_dz: float = 0.32           # Small controlled vertical increment (<= 0.016m/step)
    max_descent_xy: float = 0.35

    open_gripper_cmd: float = 1.0
    close_gripper_cmd: float = -1.0
    open_width_min: float = 0.065          # Must exceed pinched 0.05m cube width
    min_bilateral_force_n: float = 10.0    # Verified bilateral contact threshold across f9 & f10
    max_pregrasp_disp_m: float = 0.020     # Active object XY displacement tolerance during approach/descent
    max_grasp_close_steps: int = 8

    # Phase 14-Lite C_SLIP_AWARE lift parameters reused for LIFT_VERIFICATION
    lift_target_z: float = 0.130
    lift_verify_z_min: float = 0.075
    accel_limit_z: float = 0.45
    jerk_limit_z: float = 0.30
    slip_rel_dist_delta: float = 0.0025
    slip_accel_limit_z: float = 0.20
    max_retries: int = 2


def query_pybullet_telemetry(env, active_obj_idx: int) -> Dict[str, Any]:
    """Query ground-truth PyBullet kinematics and contact telemetry for pre-grasp safety."""
    cid = env.sim.client_id
    robot_id = env.controller.robot.body_id
    table_id = env.sim.table_id
    cube = env.cubes[active_obj_idx]
    cube_id = cube.body_id

    ee_pos = np.asarray(env.controller.get_ee_position(), dtype=np.float32)
    ee_orn = np.asarray(env.controller.get_ee_orientation(), dtype=np.float32)
    default_orn = np.asarray(env.controller.robot.default_ee_orientation, dtype=np.float32)
    orn_dot = abs(float(np.dot(ee_orn, default_orn)))
    gw = float(env.controller.get_gripper_state()["width"])

    obj_pos = np.asarray(cube.get_position(), dtype=np.float32)
    obj_vel = np.asarray(cube.get_velocity(), dtype=np.float32)

    contacts_active = p.getContactPoints(bodyA=robot_id, bodyB=cube_id, physicsClientId=cid)
    f9 = [c for c in contacts_active if c[3] == 9]
    f10 = [c for c in contacts_active if c[3] == 10]
    f9_force = float(sum(c[9] for c in f9))
    f10_force = float(sum(c[9] for c in f10))
    bilateral_contact = bool(len(f9) > 0 and len(f10) > 0)

    table_contacts = p.getContactPoints(bodyA=robot_id, bodyB=table_id, physicsClientId=cid) if table_id >= 0 else []

    other_idx = 1 - active_obj_idx if len(env.cubes) > 1 else -1
    other_contacts = (
        p.getContactPoints(bodyA=robot_id, bodyB=env.cubes[other_idx].body_id, physicsClientId=cid)
        if other_idx >= 0
        else []
    )

    return {
        "ee_pos": ee_pos,
        "orn_dot": orn_dot,
        "gripper_width": gw,
        "obj_pos": obj_pos,
        "obj_vel": obj_vel,
        "active_contact_count": len(contacts_active),
        "bilateral_contact": bilateral_contact,
        "total_normal_force": f9_force + f10_force,
        "table_contact_count": len(table_contacts),
        "other_obj_contact_count": len(other_contacts),
    }


class PreGraspController:
    """Modular 6-stage pre-grasp state machine with explicit transition logging."""

    def __init__(self, config: Optional[PreGraspConfig] = None, action_scale: float = 0.05) -> None:
        self.cfg = config or PreGraspConfig()
        self.action_scale = float(action_scale)
        self.reset(active_obj_idx=0)

    def reset(self, active_obj_idx: int = 0, anchor_obj_pos: Optional[np.ndarray] = None) -> None:
        self.active_obj_idx = int(active_obj_idx)
        self.stage: PreGraspStage = PreGraspStage.APPROACH
        self.anchor_xy: Optional[np.ndarray] = (
            np.asarray(anchor_obj_pos[:2], dtype=np.float32).copy() if anchor_obj_pos is not None else None
        )
        self.grasp_steps: int = 0
        self.retries: int = 0
        self.prev_az: float = 0.0
        self.prev_daz: float = 0.0
        self.prev_rel_dist: Optional[float] = None
        self.transition_log: List[Dict[str, Any]] = []
        self.stage_steps: Dict[str, int] = {s.value: 0 for s in PreGraspStage}
        self.metrics: Dict[str, Any] = {
            "approach_reached": False,
            "alignment_reached": False,
            "safe_descent_reached": False,
            "grasp_authorized": False,
            "bilateral_grasped": False,
            "lift_verified": False,
            "premature_closure": False,
            "table_collision_steps": 0,
            "pregrasp_obj_collision_steps": 0,
            "max_pregrasp_disp_m": 0.0,
        }

    def _log_transition(self, step: int, new_stage: PreGraspStage, reason: str, telem: Dict[str, Any]) -> None:
        if new_stage == self.stage:
            return
        self.transition_log.append(
            {
                "step": int(step),
                "active_obj_idx": self.active_obj_idx,
                "from": self.stage.value,
                "to": new_stage.value,
                "reason": reason,
                "ee_z": round(float(telem["ee_pos"][2]), 4),
                "gw": round(float(telem["gripper_width"]), 4),
            }
        )
        self.stage = new_stage

    def check_grasp_authorization(self, telem: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Explicitly verify all 5 safety & alignment conditions before authorizing closure."""
        cfg = self.cfg
        ee_pos = telem["ee_pos"]
        obj_pos = telem["obj_pos"]
        target_grasp_z = max(cfg.min_safe_ee_z, float(obj_pos[2]) + cfg.grasp_z_offset)
        xy_err = float(np.linalg.norm(ee_pos[:2] - obj_pos[:2]))
        z_err = abs(float(ee_pos[2]) - target_grasp_z)
        disp = (
            float(np.linalg.norm(obj_pos[:2] - self.anchor_xy))
            if self.anchor_xy is not None
            else 0.0
        )

        failures: List[str] = []
        if xy_err > cfg.grasp_xy_tol:
            failures.append(f"xy_err={xy_err:.4f}>{cfg.grasp_xy_tol}")
        if z_err > cfg.grasp_z_tol or float(ee_pos[2]) < cfg.min_safe_ee_z - 0.002:
            failures.append(f"z={ee_pos[2]:.4f},err={z_err:.4f}")
        if float(telem["orn_dot"]) < cfg.min_orn_dot:
            failures.append(f"orn_dot={telem['orn_dot']:.4f}")
        if int(telem["table_contact_count"]) > 0:
            failures.append(f"table_contacts={telem['table_contact_count']}")
        if disp > cfg.max_pregrasp_disp_m:
            failures.append(f"obj_disp={disp:.4f}>{cfg.max_pregrasp_disp_m}")
        if float(telem["gripper_width"]) <= cfg.open_width_min:
            failures.append(f"gw_not_open={telem['gripper_width']:.4f}")

        return (len(failures) == 0), failures

    def step(self, step_idx: int, telem: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute safe 4D normalized action [dx, dy, dz, gripper] for current pre-grasp stage."""
        cfg = self.cfg
        ee_pos = telem["ee_pos"]
        obj_pos = telem["obj_pos"]
        gw = float(telem["gripper_width"])
        orn_dot = float(telem["orn_dot"])
        bilat_ok = bool(telem["bilateral_contact"] and telem["total_normal_force"] >= cfg.min_bilateral_force_n)
        table_hit = int(telem["table_contact_count"]) > 0

        if self.anchor_xy is None:
            self.anchor_xy = obj_pos[:2].copy()

        disp = float(np.linalg.norm(obj_pos[:2] - self.anchor_xy))
        if self.stage in (PreGraspStage.APPROACH, PreGraspStage.ALIGNMENT, PreGraspStage.CONTROLLED_DESCENT, PreGraspStage.GRASP_AUTHORIZATION):
            self.metrics["max_pregrasp_disp_m"] = max(self.metrics["max_pregrasp_disp_m"], disp)
            if table_hit:
                self.metrics["table_collision_steps"] += 1
            if int(telem["active_contact_count"]) > 0 or int(telem["other_obj_contact_count"]) > 0:
                self.metrics["pregrasp_obj_collision_steps"] += 1

        xy_err = float(np.linalg.norm(ee_pos[:2] - obj_pos[:2]))
        rel_dist = float(np.linalg.norm(ee_pos - obj_pos))
        self.stage_steps[self.stage.value] += 1

        cmd = np.array([0.0, 0.0, 0.0, cfg.open_gripper_cmd], dtype=np.float32)

        if self.stage == PreGraspStage.APPROACH:
            target = np.array([obj_pos[0], obj_pos[1], float(obj_pos[2]) + cfg.approach_clearance_z], dtype=np.float32)
            delta = target - ee_pos
            cmd[:3] = np.clip(delta / self.action_scale, -cfg.max_approach_cmd, cfg.max_approach_cmd)
            cmd[3] = cfg.open_gripper_cmd
            if float(np.linalg.norm(delta[:2])) <= cfg.approach_xy_tol and abs(float(delta[2])) <= cfg.approach_z_tol:
                self.metrics["approach_reached"] = True
                self.anchor_xy = obj_pos[:2].copy()
                self._log_transition(step_idx, PreGraspStage.ALIGNMENT, "overhead_approach_reached", telem)

        elif self.stage == PreGraspStage.ALIGNMENT:
            target = np.array([obj_pos[0], obj_pos[1], float(obj_pos[2]) + cfg.align_clearance_z], dtype=np.float32)
            delta = target - ee_pos
            cmd[:3] = np.clip(delta / self.action_scale, -cfg.max_align_cmd, cfg.max_align_cmd)
            cmd[3] = cfg.open_gripper_cmd
            if xy_err <= cfg.align_xy_tol and abs(float(delta[2])) <= cfg.align_z_tol and orn_dot >= cfg.min_orn_dot and gw > cfg.open_width_min:
                self.metrics["alignment_reached"] = True
                self.anchor_xy = obj_pos[:2].copy()
                self._log_transition(step_idx, PreGraspStage.CONTROLLED_DESCENT, "xy_and_orn_aligned", telem)

        elif self.stage == PreGraspStage.CONTROLLED_DESCENT:
            target_grasp_z = max(cfg.min_safe_ee_z, float(obj_pos[2]) + cfg.grasp_z_offset)
            # Re-center cleanly if unsafe displacement or table collision detected
            if table_hit or disp > cfg.max_pregrasp_disp_m or orn_dot < cfg.min_orn_dot:
                self.anchor_xy = obj_pos[:2].copy()
                self.retries += 1
                cmd[:3] = np.array([0.0, 0.0, 0.8], dtype=np.float32)
                cmd[3] = cfg.open_gripper_cmd
                self._log_transition(step_idx, PreGraspStage.APPROACH, "descent_safety_abort", telem)
            else:
                delta_xy = (obj_pos[:2] - ee_pos[:2]) / self.action_scale
                delta_z = (target_grasp_z - float(ee_pos[2])) / self.action_scale
                cmd[:2] = np.clip(delta_xy, -cfg.max_descent_xy, cfg.max_descent_xy)
                cmd[2] = float(np.clip(delta_z, -cfg.max_descent_dz, 0.15))
                cmd[3] = cfg.open_gripper_cmd
                if abs(float(ee_pos[2]) - target_grasp_z) <= cfg.grasp_z_tol and xy_err <= cfg.grasp_xy_tol:
                    self.metrics["safe_descent_reached"] = True
                    self._log_transition(step_idx, PreGraspStage.GRASP_AUTHORIZATION, "grasp_height_reached", telem)

        elif self.stage == PreGraspStage.GRASP_AUTHORIZATION:
            auth_ok, reasons = self.check_grasp_authorization(telem)
            if auth_ok:
                self.metrics["grasp_authorized"] = True
                self.grasp_steps = 0
                # Immediately issue closure command once authorized
                cmd[:2] = np.clip((obj_pos[:2] - ee_pos[:2]) / self.action_scale, -0.20, 0.20)
                cmd[2] = 0.0
                cmd[3] = cfg.close_gripper_cmd
                self._log_transition(step_idx, PreGraspStage.GRASP, "all_5_checks_passed", telem)
            else:
                self.retries += 1
                cmd[3] = cfg.open_gripper_cmd
                if self.retries <= cfg.max_retries:
                    self._log_transition(step_idx, PreGraspStage.ALIGNMENT, f"auth_rejected:{reasons[0]}", telem)
                else:
                    self._log_transition(step_idx, PreGraspStage.APPROACH, f"auth_reacquire:{reasons[0]}", telem)

        elif self.stage == PreGraspStage.GRASP:
            target_grasp_z = max(cfg.min_safe_ee_z, float(obj_pos[2]) + cfg.grasp_z_offset)
            cmd[:2] = np.clip((obj_pos[:2] - ee_pos[:2]) / self.action_scale, -0.25, 0.25)
            cmd[2] = float(np.clip((target_grasp_z - float(ee_pos[2])) / self.action_scale, -0.15, 0.15))
            cmd[3] = cfg.close_gripper_cmd
            self.grasp_steps += 1
            if bilat_ok and gw >= 0.042:
                self.metrics["bilateral_grasped"] = True
                self.prev_az = 0.0
                self.prev_daz = 0.0
                self.prev_rel_dist = rel_dist
                self._log_transition(step_idx, PreGraspStage.LIFT_VERIFICATION, f"bilateral_force={telem['total_normal_force']:.1f}N", telem)
            elif self.grasp_steps >= cfg.max_grasp_close_steps:
                self.retries += 1
                self.anchor_xy = obj_pos[:2].copy()
                cmd[3] = cfg.open_gripper_cmd
                self._log_transition(step_idx, PreGraspStage.APPROACH, "bilateral_contact_timeout", telem)

        elif self.stage == PreGraspStage.LIFT_VERIFICATION:
            raw_z = 0.90
            delta_rel = 0.0 if self.prev_rel_dist is None else (rel_dist - self.prev_rel_dist)
            slip_detected = (delta_rel > cfg.slip_rel_dist_delta) and (float(obj_pos[2]) < 0.08)
            eff_prev = max(0.0, self.prev_az)
            if slip_detected:
                daz = float(np.clip(raw_z - eff_prev, 0.0, cfg.slip_accel_limit_z))
                az = float(np.clip(eff_prev + daz, 0.0, 0.35))
            else:
                eff_daz = max(0.0, self.prev_daz) if self.prev_az >= 0.0 else 0.0
                raw_jerk = (raw_z - eff_prev) - eff_daz
                clamped_jerk = float(np.clip(raw_jerk, -cfg.jerk_limit_z, cfg.jerk_limit_z))
                daz = float(np.clip(eff_daz + clamped_jerk, -cfg.accel_limit_z, cfg.accel_limit_z))
                az = float(np.clip(eff_prev + daz, -1.0, 1.0))

            cmd[:2] = np.clip((obj_pos[:2] - ee_pos[:2]) / self.action_scale, -0.25, 0.25)
            cmd[2] = az
            cmd[3] = cfg.close_gripper_cmd
            self.prev_daz = az - self.prev_az
            self.prev_az = az
            self.prev_rel_dist = rel_dist

            if float(obj_pos[2]) >= cfg.lift_verify_z_min and bilat_ok:
                self.metrics["lift_verified"] = True
                self._log_transition(step_idx, PreGraspStage.ACQUIRED, f"lifted_z={obj_pos[2]:.4f}", telem)
            elif (not bilat_ok and rel_dist > 0.065) or (float(obj_pos[2]) < 0.032 and self.stage_steps[PreGraspStage.LIFT_VERIFICATION.value] > 10):
                self.retries += 1
                self.anchor_xy = obj_pos[:2].copy()
                cmd[3] = cfg.open_gripper_cmd
                self._log_transition(step_idx, PreGraspStage.APPROACH, "lift_drop_or_miss", telem)

        else:  # ACQUIRED
            cmd = np.array([0.0, 0.0, 0.0, cfg.close_gripper_cmd], dtype=np.float32)

        # Track premature closure safety invariant across all pre-grasp stages
        if self.stage in (PreGraspStage.APPROACH, PreGraspStage.ALIGNMENT, PreGraspStage.CONTROLLED_DESCENT):
            if float(cmd[3]) <= 0.0:
                self.metrics["premature_closure"] = True

        return cmd, {"stage": self.stage.value, "bilateral_ok": bilat_ok, "xy_err": xy_err, "disp_m": disp}

