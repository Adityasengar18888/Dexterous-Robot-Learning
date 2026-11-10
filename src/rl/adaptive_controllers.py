"""Lightweight adaptive lift stabilization controllers for Phase 14-Lite.

Preserves existing SAC policies, HER buffers, reward functions, and environments.
Wraps deterministic policy actions at inference/control time only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
import pybullet as p


def get_step_telemetry(env) -> Dict[str, Any]:
    """Extract ground-truth PyBullet contact, force, and kinematic state.

    Never fabricates or invents force measurements; sums only contacts
    explicitly reported by ``pybullet.getContactPoints``.
    """
    cid = env.unwrapped.sim.client_id
    robot_id = env.unwrapped.controller.robot.body_id
    cube_id = env.unwrapped.cube.body_id

    contacts = p.getContactPoints(bodyA=robot_id, bodyB=cube_id, physicsClientId=cid)
    f9_contacts = [c for c in contacts if c[3] == 9]
    f10_contacts = [c for c in contacts if c[3] == 10]

    f9_force = float(sum(c[9] for c in f9_contacts))
    f10_force = float(sum(c[9] for c in f10_contacts))
    total_force = f9_force + f10_force

    f9_lateral = float(sum(c[10] + c[12] for c in f9_contacts))
    f10_lateral = float(sum(c[10] + c[12] for c in f10_contacts))
    total_lateral = f9_lateral + f10_lateral

    bilateral_contact = bool(len(f9_contacts) > 0 and len(f10_contacts) > 0)
    has_contact = bool(len(contacts) > 0)

    cube_pos, cube_orn = p.getBasePositionAndOrientation(cube_id, physicsClientId=cid)
    cube_vel, cube_ang_vel = p.getBaseVelocity(cube_id, physicsClientId=cid)
    ee_pos = env.unwrapped.controller.get_ee_position()
    gripper_width = float(env.unwrapped.controller.robot.get_gripper_width())

    rel_vec = np.array(ee_pos, dtype=np.float32) - np.array(cube_pos, dtype=np.float32)
    rel_dist = float(np.linalg.norm(rel_vec))
    xy_dist = float(np.linalg.norm(rel_vec[:2]))

    return {
        "ee_pos": np.array(ee_pos, dtype=np.float32),
        "cube_pos": np.array(cube_pos, dtype=np.float32),
        "cube_orn": np.array(cube_orn, dtype=np.float32),
        "cube_vel": np.array(cube_vel, dtype=np.float32),
        "cube_ang_vel": np.array(cube_ang_vel, dtype=np.float32),
        "gripper_width": gripper_width,
        "contact_count": len(contacts),
        "f9_contact_count": len(f9_contacts),
        "f10_contact_count": len(f10_contacts),
        "f9_normal_force": f9_force,
        "f10_normal_force": f10_force,
        "total_normal_force": total_force,
        "total_lateral_force": total_lateral,
        "contact_symmetry": bilateral_contact,
        "has_contact": has_contact,
        "rel_dist": rel_dist,
        "xy_dist": xy_dist,
        "cube_z": float(cube_pos[2]),
        "cube_vz": float(cube_vel[2]),
        "ee_z": float(ee_pos[2]),
        "desired_goal": np.array(getattr(env.unwrapped, "goal", np.zeros(3, dtype=np.float32)), dtype=np.float32),
    }


@dataclass
class ControllerConfig:
    """Configurable parameters for Phase 14-Lite controllers."""
    # Controller B: Acceleration Smoothing during lift initiation
    accel_limit_z: float = 0.45          # Max upward delta action[2] per step during lift init
    jerk_limit_z: float = 0.30           # Max delta-accel of action[2] per step during lift init
    lift_init_z_max: float = 0.06        # Lift initiation altitude envelope (cube_z < threshold)
    lift_init_rel_dist_max: float = 0.055

    # Controller C: Slip-Aware Lifting
    slip_rel_dist_delta: float = 0.0025  # Relative EE-object separation indicating slip (m/step)
    min_stable_force: float = 15.0       # Minimum PyBullet bilateral normal force for lift resume (N)
    slip_accel_limit_z: float = 0.20     # Reduced vertical acceleration during slip recovery
    stable_width_min: float = 0.042      # Minimum valid finger width pinching 0.05m cube (m)
    max_slip_hold_steps: int = 3         # Cap consecutive hold steps per attempt
    safe_closure_cmd: float = -1.0       # Physical gripper closure action within [-1, 1]
    shear_damp_factor: float = 0.25      # Reduce sideways drag ONLY while pausing upward slip

    # Controller D (Phase 14.3): Spatial Grasp Acquisition + Dual-Stage Adaptive Control
    train_ws_x_min: float = 0.40         # Nominal DR training workspace X lower bound
    train_ws_x_max: float = 0.60         # Nominal DR training workspace X upper bound
    train_ws_y_min: float = -0.20        # Nominal DR training workspace Y lower bound
    train_ws_y_max: float = 0.20         # Nominal DR training workspace Y upper bound
    ws_edge_margin: float = 0.02         # Trigger spatial edge assist near/beyond DR boundary
    action_scale: float = 0.05           # Environment action step scale (m)
    xy_align_tol: float = 0.014          # Lateral alignment tolerance to cube center (m)
    grasp_z_offset: float = 0.019        # Desired EE height above cube center at grasp (m)
    pregrasp_z_offset: float = 0.045     # Safe overhead clearance height while aligning XY (m)
    min_acquire_force: float = 10.0      # Minimum bilateral PyBullet normal force to verify contact (N)

    # Controller E (Phase 14.4): Adaptive Post-Lift Transport & Target Alignment
    transport_lift_z_thresh: float = 0.050  # Lift threshold required before transport activates (m)
    align_cos_thresh: float = 0.75          # Minimum directional alignment cosine before correction
    fine_align_radius: float = 0.085        # Radius inside which overshoot-damped alignment engages (m)
    target_settle_radius: float = 0.015     # Target center settle tolerance well inside 0.05m (m)
    max_transport_cmd: float = 0.90         # Bounded transport displacement step magnitude
    overshoot_brake_radius: float = 0.060   # Linear velocity taper inside this distance to goal (m)
    kp_transport: float = 0.85              # Proportional gain on (desired_goal - cube_pos) / action_scale
    table_release_z_max: float = 0.028      # Max target Z considered physically table-supported (m)


class BaseLiftController:
    """Interface for evaluation-time action filters."""

    name: str = "BASE"

    def reset(self) -> None:
        pass

    def step_action(self, raw_action: np.ndarray, telemetry: Dict[str, Any]) -> tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError


class BaselineController(BaseLiftController):
    """Variant A: Original unmodified SAC deterministic policy."""

    name: str = "A_BASELINE"

    def step_action(self, raw_action: np.ndarray, telemetry: Dict[str, Any]) -> tuple[np.ndarray, Dict[str, Any]]:
        cmd = np.clip(np.asarray(raw_action, dtype=np.float32), -1.0, 1.0)
        return cmd, {"slip_detected": False, "controller_mode": "baseline"}


class AccelerationSmoothingController(BaseLiftController):
    """Variant B: Configurable vertical acceleration and jerk limits during lift initiation.

    Applies vertical acceleration and jerk limits ONLY when the robot is actively
    grasping the cube (gripper_width >= 0.042m and contact/closure active) and
    commanding upward lift near the table. Missed/empty re-lifts pass straight through.
    """

    name: str = "B_ACCEL_SMOOTHING"

    def __init__(self, config: Optional[ControllerConfig] = None) -> None:
        self.cfg = config or ControllerConfig()
        self.reset()

    def reset(self) -> None:
        self.prev_az: float = 0.0
        self.prev_daz: float = 0.0
        self.prev_gripper_cmd: float = 1.0

    def step_action(self, raw_action: np.ndarray, telemetry: Dict[str, Any]) -> tuple[np.ndarray, Dict[str, Any]]:
        cmd = np.clip(np.asarray(raw_action, dtype=np.float32).copy(), -1.0, 1.0)
        raw_az = float(cmd[2])
        raw_grip = float(cmd[3])

        cube_z = telemetry["cube_z"]
        rel_dist = telemetry["rel_dist"]
        gw = telemetry["gripper_width"]
        any_contact = bool(telemetry["has_contact"])

        has_object = (gw >= self.cfg.stable_width_min) and (any_contact or (raw_grip <= 0.0 and rel_dist < 0.045))
        in_lift_init = (
            (cube_z < self.cfg.lift_init_z_max)
            and (rel_dist < self.cfg.lift_init_rel_dist_max)
            and has_object
            and (raw_az > 0.0)
        )

        mode = "passthrough"
        if in_lift_init:
            eff_prev_az = max(0.0, self.prev_az)
            eff_prev_daz = max(0.0, self.prev_daz) if self.prev_az >= 0.0 else 0.0

            desired_daz = raw_az - eff_prev_az
            if desired_daz > 0.0:
                raw_jerk = desired_daz - eff_prev_daz
                clamped_jerk = float(np.clip(raw_jerk, -self.cfg.jerk_limit_z, self.cfg.jerk_limit_z))
                daz = eff_prev_daz + clamped_jerk
                daz = float(np.clip(daz, -self.cfg.accel_limit_z, self.cfg.accel_limit_z))
                cmd[2] = float(np.clip(eff_prev_az + daz, -1.0, 1.0))
                self.prev_daz = float(cmd[2]) - eff_prev_az
                self.prev_az = float(cmd[2])
                mode = "accel_jerk_limited"
            else:
                self.prev_daz = float(cmd[2]) - self.prev_az
                self.prev_az = float(cmd[2])
        else:
            self.prev_daz = float(cmd[2]) - self.prev_az
            self.prev_az = float(cmd[2])

        self.prev_gripper_cmd = raw_grip
        return cmd, {"slip_detected": False, "controller_mode": mode}


class SlipAwareLiftController(BaseLiftController):
    """Variant C: Slip-aware adaptive lifting using real displacement & contact feedback.

    Strictly operates ONLY when a real grasp/contact on the 0.05m cube exists
    (gripper_width >= 0.042m and physical contact present). Never interferes with
    downward centering (raw_az <= 0) or empty/missed re-approaches.
    """

    name: str = "C_SLIP_AWARE"

    def __init__(self, config: Optional[ControllerConfig] = None) -> None:
        self.cfg = config or ControllerConfig()
        self.reset()

    def reset(self) -> None:
        self.prev_az: float = 0.0
        self.prev_daz: float = 0.0
        self.prev_rel_dist: Optional[float] = None
        self.prev_bilateral: bool = False
        self.prev_gripper_cmd: float = 1.0
        self.slip_hold_steps: int = 0
        self.cooldown_steps: int = 0

    def step_action(self, raw_action: np.ndarray, telemetry: Dict[str, Any]) -> tuple[np.ndarray, Dict[str, Any]]:
        cmd = np.clip(np.asarray(raw_action, dtype=np.float32).copy(), -1.0, 1.0)
        raw_az = float(cmd[2])
        raw_grip = float(cmd[3])

        rel_dist = telemetry["rel_dist"]
        delta_rel = 0.0 if self.prev_rel_dist is None else (rel_dist - self.prev_rel_dist)
        gw = telemetry["gripper_width"]
        cube_z = telemetry["cube_z"]
        bilateral = bool(telemetry["contact_symmetry"])
        any_contact = bool(telemetry["has_contact"])
        norm_force = float(telemetry["total_normal_force"])

        contact_stable = (
            bilateral
            and (norm_force >= self.cfg.min_stable_force)
            and (abs(delta_rel) <= self.cfg.slip_rel_dist_delta)
            and (gw >= self.cfg.stable_width_min)
        )

        if self.cooldown_steps > 0:
            self.cooldown_steps -= 1

        slip_detected = False
        mode = "passthrough"

        # Gate strictly on real physical contact + valid cube-width aperture
        in_lift_zone = (cube_z < self.cfg.lift_init_z_max) and (rel_dist < 0.055)
        real_grasp_contact = (gw >= self.cfg.stable_width_min) and (any_contact or self.prev_bilateral)

        if in_lift_zone and real_grasp_contact and (self.cooldown_steps == 0):
            slip_expanding = (delta_rel > self.cfg.slip_rel_dist_delta) and (self.prev_bilateral or bilateral or norm_force >= 10.0)
            contact_unseated = (raw_az > 0.05) and (
                (not bilateral or norm_force < self.cfg.min_stable_force)
                and (self.prev_gripper_cmd <= 0.0 or raw_grip <= 0.0)
            )
            # Detect strong lateral shear drag on table after bilateral grip forms (before upward lift starts)
            table_drag_slip = (
                (cube_z < 0.01)
                and bilateral
                and (norm_force >= self.cfg.min_stable_force)
                and (raw_az <= 0.05)
                and (np.hypot(cmd[0], cmd[1]) > 0.40)
                and (raw_grip <= 0.10)
            )
            if slip_expanding or contact_unseated or table_drag_slip:
                slip_detected = True

        if slip_detected and self.slip_hold_steps < self.cfg.max_slip_hold_steps:
            self.slip_hold_steps += 1
            mode = "slip_mitigation"

            # 1. Maintain safe gripper closure ONLY if cube was genuinely grasped/lifting
            # (prevents premature ungrasp chatter while letting empty grazes re-open)
            if (rel_dist < 0.045) and (gw >= self.cfg.stable_width_min) and (bilateral or self.prev_bilateral or norm_force >= 10.0):
                cmd[3] = self.cfg.safe_closure_cmd

            # 2. Temporarily reduce vertical acceleration & lateral shear while stabilizing contact
            eff_prev_az = max(0.0, self.prev_az)
            if raw_az > 0.0:
                if (not bilateral or norm_force < 5.0) and raw_grip <= 0.0:
                    daz = float(np.clip(raw_az - eff_prev_az, 0.0, 0.10))
                    cmd[2] = float(np.clip(eff_prev_az + daz, 0.0, 0.15))
                else:
                    daz = float(np.clip(raw_az - eff_prev_az, 0.0, self.cfg.slip_accel_limit_z))
                    cmd[2] = float(np.clip(eff_prev_az + daz, 0.0, 0.45))
                scale_xy = float(np.clip(cmd[2] / max(raw_az, 1e-4), 0.35, 1.0))
                cmd[0] *= scale_xy
                cmd[1] *= scale_xy
            else:
                # Damp excessive horizontal drag on tabletop while bilateral contact is settling
                cmd[0] *= self.cfg.shear_damp_factor
                cmd[1] *= self.cfg.shear_damp_factor

            self.prev_daz = float(cmd[2]) - eff_prev_az
            self.prev_az = float(cmd[2])
        else:
            if self.slip_hold_steps >= self.cfg.max_slip_hold_steps:
                self.slip_hold_steps = 0
                self.cooldown_steps = 4
            elif contact_stable:
                self.slip_hold_steps = 0

            # Smooth lift initiation/resumption ONLY when real grasp contact exists & lifting upward
            if in_lift_zone and real_grasp_contact and raw_az > 0.0:
                eff_prev_az = max(0.0, self.prev_az)
                eff_prev_daz = max(0.0, self.prev_daz) if self.prev_az >= 0.0 else 0.0
                desired_daz = raw_az - eff_prev_az
                if desired_daz > 0.0:
                    raw_jerk = desired_daz - eff_prev_daz
                    clamped_jerk = float(np.clip(raw_jerk, -self.cfg.jerk_limit_z, self.cfg.jerk_limit_z))
                    daz = float(np.clip(eff_prev_daz + clamped_jerk, -self.cfg.accel_limit_z, self.cfg.accel_limit_z))
                    cmd[2] = float(np.clip(eff_prev_az + daz, -1.0, 1.0))
                    if cube_z < 0.03 and raw_az > 0.15:
                        ratio = float(np.clip(cmd[2] / raw_az, 0.40, 1.0))
                        cmd[0] *= ratio
                        cmd[1] *= ratio
                    self.prev_daz = float(cmd[2]) - eff_prev_az
                    self.prev_az = float(cmd[2])
                    mode = "smooth_lift_resume" if contact_stable else "smooth_lift"
                else:
                    self.prev_daz = float(cmd[2]) - self.prev_az
                    self.prev_az = float(cmd[2])
            else:
                self.prev_daz = float(cmd[2]) - self.prev_az
                self.prev_az = float(cmd[2])

        self.prev_rel_dist = rel_dist
        self.prev_bilateral = bilateral
        self.prev_gripper_cmd = float(cmd[3])

        return cmd, {
            "slip_detected": slip_detected,
            "contact_stable": contact_stable,
            "delta_rel_dist": float(delta_rel),
            "controller_mode": mode,
        }


class DualStageGraspLiftController(BaseLiftController):
    """Variant D (Phase 14.3): Spatial Grasp Acquisition + Slip-Aware Lifting.

    Stage 1 (Spatial Grasp Acquisition):
      - Detects objects near/outside nominal training workspace boundaries.
      - Corrects misaligned XY/Z approach trajectories before closing so the
        parallel-jaw gripper aligns cleanly over the cube without lateral knock.
      - Prevents premature upward lift commands until PyBullet confirms valid
        bilateral finger contact (link 9 & link 10) and non-zero normal force.

    Stage 2 (Slip-Aware Lifting):
      - Strictly reuses Phase 14-Lite ``SlipAwareLiftController`` once bilateral
        grasp acquisition succeeds, preserving all physical action limits [-1, 1].
    """

    name: str = "D_DUAL_STAGE"

    def __init__(self, config: Optional[ControllerConfig] = None) -> None:
        self.cfg = config or ControllerConfig()
        self.lift_controller = SlipAwareLiftController(self.cfg)
        self.reset()

    def reset(self) -> None:
        self.lift_controller.reset()
        self.grasp_acquired: bool = False
        self.stage1_active_steps: int = 0

    def _is_near_workspace_edge(self, pos: np.ndarray) -> bool:
        x, y = float(pos[0]), float(pos[1])
        m = self.cfg.ws_edge_margin
        return (
            x <= (self.cfg.train_ws_x_min + m)
            or x >= (self.cfg.train_ws_x_max - m)
            or y <= (self.cfg.train_ws_y_min + m)
            or y >= (self.cfg.train_ws_y_max - m)
        )

    def step_action(self, raw_action: np.ndarray, telemetry: Dict[str, Any]) -> tuple[np.ndarray, Dict[str, Any]]:
        cmd = np.clip(np.asarray(raw_action, dtype=np.float32).copy(), -1.0, 1.0)
        ee_pos = telemetry["ee_pos"]
        cube_pos = telemetry["cube_pos"]
        cube_z = float(telemetry["cube_z"])
        rel_dist = float(telemetry["rel_dist"])
        gw = float(telemetry["gripper_width"])
        bilateral = bool(telemetry["contact_symmetry"])
        any_contact = bool(telemetry["has_contact"])
        norm_force = float(telemetry["total_normal_force"])

        # True bilateral grasp verification from PyBullet contacts
        valid_bilateral_grasp = (
            bilateral
            and (norm_force >= self.cfg.min_acquire_force)
            and (gw >= self.cfg.stable_width_min)
        )

        # Update stage latch cleanly
        if valid_bilateral_grasp or (cube_z > 0.045 and rel_dist < 0.05):
            self.grasp_acquired = True
        elif cube_z < 0.025 and (not bilateral) and (not any_contact) and (rel_dist > 0.045):
            self.grasp_acquired = False

        # Stage 2: Reuse existing C_SLIP_AWARE controller once grasp is acquired
        if self.grasp_acquired:
            cmd_out, lift_info = self.lift_controller.step_action(cmd, telemetry)
            lift_info["stage"] = "stage2_slip_aware_lift"
            lift_info["edge_correction"] = False
            lift_info["lift_blocked_pre_contact"] = False
            return cmd_out, lift_info

        # Stage 1: Spatial Grasp Acquisition (pre-contact, cube on/near table)
        near_edge = self._is_near_workspace_edge(cube_pos)
        err_vec = cube_pos - ee_pos
        err_xy = err_vec[:2]
        xy_dist = float(np.linalg.norm(err_xy))
        desired_grasp_z = cube_z + self.cfg.grasp_z_offset

        edge_correction = False
        lift_blocked = False

        # Trigger spatial acquisition correction when object is near workspace edges
        # and robot is in the approach/grasp funnel (or oscillating short of object)
        in_edge_funnel = near_edge and (cube_z < 0.035) and (rel_dist < 0.16 or ee_pos[2] < 0.12)

        if in_edge_funnel:
            # Check if SAC policy is already cleanly approaching and aligned, or needs guidance
            raw_xy = cmd[:2]
            dot_align = float(np.dot(raw_xy, err_xy)) / (float(np.linalg.norm(raw_xy)) * max(xy_dist, 1e-6) + 1e-8)
            needs_xy_correction = (xy_dist > self.cfg.xy_align_tol) and (
                (dot_align < 0.70)
                or (ee_pos[2] < cube_z + 0.055)
                or (float(cmd[3]) <= 0.0)
            )
            premature_close = (float(cmd[3]) <= 0.0) and (
                (xy_dist > self.cfg.xy_align_tol + 0.004) or (ee_pos[2] > desired_grasp_z + 0.014)
            )
            premature_lift = (float(cmd[2]) > 0.05) and (rel_dist < 0.075)

            if (xy_dist <= self.cfg.xy_align_tol + 0.004) and (abs(float(ee_pos[2]) - desired_grasp_z) <= 0.012):
                # Centered in XY & at grasp height: stabilize position and close gripper cleanly
                edge_correction = True
                cmd[0] = float(np.clip(err_xy[0] / self.cfg.action_scale, -0.35, 0.35))
                cmd[1] = float(np.clip(err_xy[1] / self.cfg.action_scale, -0.35, 0.35))
                cmd[2] = float(np.clip((desired_grasp_z - float(ee_pos[2])) / self.cfg.action_scale, -0.30, 0.0))
                cmd[3] = -1.0
                if float(raw_action[2]) > 0.05:
                    lift_blocked = True
            elif needs_xy_correction or premature_close or premature_lift:
                edge_correction = True
                self.stage1_active_steps += 1
                # Keep gripper open until aligned over the cube
                cmd[3] = 1.0
                # Steer XY smoothly toward cube center
                cmd[0] = float(np.clip(err_xy[0] / self.cfg.action_scale, -0.95, 0.95))
                cmd[1] = float(np.clip(err_xy[1] / self.cfg.action_scale, -0.95, 0.95))
                if xy_dist > 0.022:
                    # Maintain safe overhead clearance while translating horizontally so cube isn't bumped
                    target_z = cube_z + self.cfg.pregrasp_z_offset
                else:
                    # Aligned in XY funnel -> descend straight down onto grasp plane
                    target_z = desired_grasp_z
                cmd[2] = float(np.clip((target_z - float(ee_pos[2])) / self.cfg.action_scale, -0.95, 0.60))
                if float(raw_action[2]) > 0.05 and float(cmd[2]) <= 0.0:
                    lift_blocked = True

            # Hard physical guard: never lift upward near cube without verified bilateral contact
            if (rel_dist < 0.055) and (not bilateral) and (float(cmd[2]) > 0.0) and (float(ee_pos[2]) < cube_z + 0.038):
                cmd[2] = 0.0
                lift_blocked = True

        # Keep lift_controller history synchronized for smooth handoff
        self.lift_controller.prev_az = float(cmd[2])
        self.lift_controller.prev_daz = 0.0
        self.lift_controller.prev_rel_dist = rel_dist
        self.lift_controller.prev_bilateral = bilateral
        self.lift_controller.prev_gripper_cmd = float(cmd[3])

        return cmd, {
            "slip_detected": False,
            "contact_stable": False,
            "delta_rel_dist": 0.0,
            "controller_mode": "stage1_edge_acquire" if edge_correction else "stage1_passthrough",
            "stage": "stage1_spatial_acquisition",
            "edge_correction": edge_correction,
            "lift_blocked_pre_contact": lift_blocked,
        }


class AdaptiveTransportController(BaseLiftController):
    """Variant E (Phase 14.4): Complete 5-Stage Dual-Stage + Adaptive Post-Lift Transport.

    Pipeline:
      1. Spatial Grasp Acquisition (reused from DualStageGraspLiftController Stage 1)
      2. Bilateral Contact Verification (reused from DualStageGraspLiftController)
      3. Slip-Aware Lifting (reused from Phase 14-Lite SlipAwareLiftController)
      4. Adaptive Post-Lift Transport & Target Alignment (activated ONLY after verified
         cube_z > 0.05m and validated grasp)
      5. Placement Stabilization (overshoot-damped alignment & safe closure / table release)
    """

    name: str = "E_ADAPTIVE_TRANSPORT"

    def __init__(self, config: Optional[ControllerConfig] = None) -> None:
        self.cfg = config or ControllerConfig()
        self.dual_stage = DualStageGraspLiftController(self.cfg)
        self.reset()

    def reset(self) -> None:
        self.dual_stage.reset()
        self.transport_active: bool = False
        self.placed_released: bool = False
        self.stable_target_steps: int = 0

    def step_action(self, raw_action: np.ndarray, telemetry: Dict[str, Any]) -> tuple[np.ndarray, Dict[str, Any]]:
        # Execute Stages 1-3 via Phase 14.3 DualStageGraspLiftController
        cmd, info = self.dual_stage.step_action(raw_action, telemetry)

        ee_pos = telemetry["ee_pos"]
        cube_pos = telemetry["cube_pos"]
        cube_z = float(telemetry["cube_z"])
        cube_vel = telemetry["cube_vel"]
        rel_dist = float(telemetry["rel_dist"])
        gw = float(telemetry["gripper_width"])
        bilateral = bool(telemetry["contact_symmetry"])
        any_contact = bool(telemetry["has_contact"])
        norm_force = float(telemetry["total_normal_force"])
        desired_goal = telemetry.get("desired_goal", np.zeros(3, dtype=np.float32))

        goal_vec = np.asarray(desired_goal, dtype=np.float32) - np.asarray(cube_pos, dtype=np.float32)
        goal_dist = float(np.linalg.norm(goal_vec))
        goal_dir = goal_vec / max(goal_dist, 1e-6)

        # Stage 1 (Transport Activation): activate strictly after verified lift & validated grasp
        has_valid_grasp = self.dual_stage.grasp_acquired and (bilateral or (any_contact and gw >= self.cfg.stable_width_min))
        if has_valid_grasp and (cube_z > self.cfg.transport_lift_z_thresh):
            self.transport_active = True
        elif (not self.dual_stage.grasp_acquired) or (not any_contact and rel_dist > 0.065):
            self.transport_active = False
            self.stable_target_steps = 0

        if self.placed_released:
            # Stay motionless above released cube so placement remains undisturbed
            info["stage"] = "stage5_placed"
            info["transport_active"] = False
            info["transport_corrected"] = False
            info["goal_dist"] = goal_dist
            info["dir_alignment"] = 1.0
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), info

        # Prior stages remain strictly in control if lift/grasp incomplete or slip mitigation active
        if (not self.transport_active) or info.get("slip_detected", False):
            info["transport_active"] = self.transport_active
            info["transport_corrected"] = False
            info["goal_dist"] = goal_dist
            raw_xyz = np.asarray(raw_action[:3], dtype=np.float32)
            rn = float(np.linalg.norm(raw_xyz))
            info["dir_alignment"] = float(np.dot(raw_xyz / max(rn, 1e-6), goal_dir)) if rn > 1e-4 else 0.0
            return cmd, info

        # Stage 2 & 3: Adaptive Target-Direction & Overshoot-Damped Alignment Control
        raw_xyz = np.asarray(cmd[:3], dtype=np.float32)
        raw_norm = float(np.linalg.norm(raw_xyz))
        dir_align = float(np.dot(raw_xyz / max(raw_norm, 1e-6), goal_dir)) if raw_norm > 1e-4 else 0.0

        transport_corrected = False
        # Preserve safe gripper closure during elevated transport
        cmd[3] = self.cfg.safe_closure_cmd

        # Desired object-to-target displacement in normalized action space (compensated for EE-object offset)
        target_cmd = (goal_vec / self.cfg.action_scale) * self.cfg.kp_transport

        # Overshoot brake: smoothly taper max displacement norm as object approaches goal
        if goal_dist < self.cfg.overshoot_brake_radius:
            speed_cap = float(np.clip(goal_dist / self.cfg.overshoot_brake_radius, 0.08, 1.0)) * self.cfg.max_transport_cmd
        else:
            speed_cap = self.cfg.max_transport_cmd

        tc_norm = float(np.linalg.norm(target_cmd))
        if tc_norm > speed_cap and tc_norm > 1e-6:
            target_cmd = target_cmd * (speed_cap / tc_norm)
        target_cmd = np.clip(target_cmd, -self.cfg.max_transport_cmd, self.cfg.max_transport_cmd)

        # Intervene smoothly if inside fine alignment radius OR policy diverges from goal vector OR risks overshoot
        needs_align = (
            (goal_dist <= self.cfg.fine_align_radius)
            or (dir_align < self.cfg.align_cos_thresh)
            or (raw_norm * self.cfg.action_scale > goal_dist * 1.15)
        )

        if needs_align:
            transport_corrected = True
            # Progressive blend toward closed-loop object-to-target vector (full precision inside 0.085m)
            blend = 1.0 if (goal_dist <= self.cfg.fine_align_radius or dir_align < 0.4) else 0.65
            blended_xyz = (1.0 - blend) * raw_xyz + blend * target_cmd

            # Respect valid physical arm workspace boundaries [0.20, 0.70] x [-0.30, 0.30] x [0.02, 0.48]
            next_ee = ee_pos + blended_xyz * self.cfg.action_scale
            clipped_ee = np.clip(next_ee, [0.20, -0.30, 0.02], [0.70, 0.30, 0.48])
            blended_xyz = (clipped_ee - ee_pos) / self.cfg.action_scale
            cmd[:3] = np.clip(blended_xyz, -1.0, 1.0)

        # Stage 4: Target Alignment & Placement Verification
        cube_speed = float(np.linalg.norm(cube_vel))
        if (goal_dist <= self.cfg.target_settle_radius) and (cube_speed < 0.08):
            self.stable_target_steps += 1
            # Only open gripper if target is physically resting near tabletop surface
            if (float(desired_goal[2]) <= self.cfg.table_release_z_max) and (cube_z <= 0.028) and (self.stable_target_steps >= 3):
                cmd[3] = 1.0
                cmd[:3] = 0.0
                self.placed_released = True
                info["stage"] = "stage5_placed"
            else:
                info["stage"] = "stage5_target_hold"
        else:
            self.stable_target_steps = 0
            info["stage"] = "stage4_adaptive_transport"

        info["transport_active"] = True
        info["transport_corrected"] = transport_corrected
        info["goal_dist"] = goal_dist
        info["dir_alignment"] = dir_align
        info["controller_mode"] = "transport_align" if transport_corrected else "transport_passthrough"
        return cmd, info
