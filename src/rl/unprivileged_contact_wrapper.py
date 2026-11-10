"""Unprivileged-contact wrapper & kinematic fallback controller for Phase 15.1 (Experiment C).

Strictly strips PyBullet-only contact point coordinates and normal/lateral force
measurements from telemetry, exposing only ordinary robot proprioception and
object-pose perception signals. Never fabricates physical force measurements.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.rl.adaptive_controllers import BaseLiftController, ControllerConfig


PRIVILEGED_TELEMETRY_KEYS: Tuple[str, ...] = (
    "contact_count",
    "f9_contact_count",
    "f10_contact_count",
    "f9_normal_force",
    "f10_normal_force",
    "total_normal_force",
    "total_lateral_force",
    "contact_symmetry",
    "has_contact",
)

UNPRIVILEGED_AVAILABLE_KEYS: Tuple[str, ...] = (
    "ee_pos",
    "ee_z",
    "gripper_width",
    "cube_pos",
    "cube_orn",
    "cube_vel",
    "cube_z",
    "cube_vz",
    "rel_dist",
    "xy_dist",
    "desired_goal",
)


def strip_privileged_telemetry(telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """Return a clean telemetry dictionary with all simulator-only contact/force keys removed."""
    filtered: Dict[str, Any] = {}
    for k in UNPRIVILEGED_AVAILABLE_KEYS:
        if k in telemetry:
            val = telemetry[k]
            filtered[k] = np.array(val, dtype=np.float32).copy() if isinstance(val, np.ndarray) else val
    filtered["privileged_contacts_stripped"] = True
    return filtered


class UnprivilegedAdaptiveController(BaseLiftController):
    """Conservative kinematics-only Phase 14 fallback for C1 (Unprivileged Contact).

    Consumes ONLY filtered observations from ``strip_privileged_telemetry``:
      - ``ee_pos``, ``gripper_width`` (robot proprioception)
      - ``cube_pos``, ``cube_vel``, ``desired_goal`` (perception / goal state)

    Replaces PyBullet force/bilateral-contact checks with:
      1. Spatial edge acquisition identical to Phase 14.3 Stage 1.
      2. Conservative dwell verification: keeps gripper closing at ``gw in [0.042, 0.066]m``
         for ``settle_closure_steps`` (default 2) before permitting upward lift.
      3. Kinematic slip monitoring via ``delta_rel_dist`` and jaw width collapse.
      4. Phase 14.4 post-lift adaptive transport & placement once ``cube_z > 0.05m``.
    """

    name: str = "C1_UNPRIVILEGED"

    def __init__(
        self,
        config: Optional[ControllerConfig] = None,
        settle_closure_steps: int = 2,
        jaw_cube_width_min: float = 0.042,
        jaw_cube_width_max: float = 0.066,
    ) -> None:
        self.cfg = config or ControllerConfig()
        self.settle_closure_steps = int(settle_closure_steps)
        self.jaw_cube_width_min = float(jaw_cube_width_min)
        self.jaw_cube_width_max = float(jaw_cube_width_max)
        self.reset()

    def reset(self) -> None:
        self.grasp_acquired: bool = False
        self.transport_active: bool = False
        self.placed_released: bool = False
        self.closure_dwell: int = 0
        self.stable_target_steps: int = 0
        self.slip_hold_steps: int = 0
        self.prev_az: float = 0.0
        self.prev_daz: float = 0.0
        self.prev_rel_dist: Optional[float] = None
        self.prev_gw: Optional[float] = None

    def _is_near_workspace_edge(self, pos: np.ndarray) -> bool:
        x, y = float(pos[0]), float(pos[1])
        m = self.cfg.ws_edge_margin
        return (
            x <= (self.cfg.train_ws_x_min + m)
            or x >= (self.cfg.train_ws_x_max - m)
            or y <= (self.cfg.train_ws_y_min + m)
            or y >= (self.cfg.train_ws_y_max - m)
        )

    def step_action(self, raw_action: np.ndarray, telemetry: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Explicitly verify no privileged force keys are used
        for forbidden in PRIVILEGED_TELEMETRY_KEYS:
            if forbidden in telemetry:
                raise AssertionError(f"Privileged key '{forbidden}' leaked into UnprivilegedAdaptiveController!")

        cmd = np.clip(np.asarray(raw_action, dtype=np.float32).copy(), -1.0, 1.0)
        ee_pos = telemetry["ee_pos"]
        cube_pos = telemetry["cube_pos"]
        cube_vel = telemetry["cube_vel"]
        cube_z = float(telemetry["cube_z"])
        rel_dist = float(telemetry["rel_dist"])
        xy_dist = float(telemetry["xy_dist"])
        gw = float(telemetry["gripper_width"])
        desired_goal = telemetry.get("desired_goal", np.zeros(3, dtype=np.float32))

        delta_rel = 0.0 if self.prev_rel_dist is None else (rel_dist - self.prev_rel_dist)
        gw_stalled_on_object = (self.jaw_cube_width_min <= gw <= self.jaw_cube_width_max) and (rel_dist < 0.048)

        # Kinematic closure settle counter (no force sensor available)
        if gw_stalled_on_object and (self.prev_gw is not None and abs(gw - self.prev_gw) < 0.010):
            self.closure_dwell += 1
        elif gw < self.jaw_cube_width_min or rel_dist >= 0.055:
            self.closure_dwell = 0

        kinematic_grasp_ok = (self.closure_dwell >= self.settle_closure_steps) and gw_stalled_on_object
        if kinematic_grasp_ok or (cube_z > 0.045 and rel_dist < 0.045 and gw >= self.jaw_cube_width_min):
            self.grasp_acquired = True
        elif cube_z < 0.025 and (gw < self.jaw_cube_width_min or rel_dist > 0.055):
            self.grasp_acquired = False
            self.transport_active = False

        if self.placed_released:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), {
                "slip_detected": False,
                "controller_mode": "c1_placed",
                "stage": "stage5_placed",
                "inferred_kinematic_grasp": True,
            }

        # Stage 1: Spatial grasp acquisition & conservative closure dwell (unprivileged)
        if not self.grasp_acquired:
            near_edge = self._is_near_workspace_edge(cube_pos)
            err_xy = cube_pos[:2] - ee_pos[:2]
            desired_grasp_z = cube_z + self.cfg.grasp_z_offset
            mode = "c1_passthrough"

            if near_edge and (cube_z < 0.035) and (rel_dist < 0.16 or ee_pos[2] < 0.12):
                raw_xy = cmd[:2]
                dot_align = float(np.dot(raw_xy, err_xy)) / (float(np.linalg.norm(raw_xy)) * max(xy_dist, 1e-6) + 1e-8)
                needs_xy = (xy_dist > self.cfg.xy_align_tol) and (
                    (dot_align < 0.70) or (ee_pos[2] < cube_z + 0.055) or (float(cmd[3]) <= 0.0)
                )
                if (xy_dist <= self.cfg.xy_align_tol + 0.004) and (abs(float(ee_pos[2]) - desired_grasp_z) <= 0.012):
                    cmd[0] = float(np.clip(err_xy[0] / self.cfg.action_scale, -0.35, 0.35))
                    cmd[1] = float(np.clip(err_xy[1] / self.cfg.action_scale, -0.35, 0.35))
                    cmd[2] = float(np.clip((desired_grasp_z - float(ee_pos[2])) / self.cfg.action_scale, -0.30, 0.0))
                    cmd[3] = -1.0
                    mode = "c1_close_dwell"
                elif needs_xy or (float(cmd[3]) <= 0.0) or (float(cmd[2]) > 0.05 and rel_dist < 0.075):
                    cmd[3] = 1.0
                    cmd[0] = float(np.clip(err_xy[0] / self.cfg.action_scale, -0.95, 0.95))
                    cmd[1] = float(np.clip(err_xy[1] / self.cfg.action_scale, -0.95, 0.95))
                    target_z = (cube_z + self.cfg.pregrasp_z_offset) if xy_dist > 0.022 else desired_grasp_z
                    cmd[2] = float(np.clip((target_z - float(ee_pos[2])) / self.cfg.action_scale, -0.95, 0.60))
                    mode = "c1_edge_acquire"

                if (rel_dist < 0.055) and (float(cmd[2]) > 0.0) and (float(ee_pos[2]) < cube_z + 0.038):
                    cmd[2] = 0.0

            self.prev_az = float(cmd[2])
            self.prev_rel_dist = rel_dist
            self.prev_gw = gw
            return cmd, {
                "slip_detected": False,
                "controller_mode": mode,
                "stage": "stage1_unprivileged_acquire",
                "inferred_kinematic_grasp": kinematic_grasp_ok,
            }

        # Stage 2 & 3: Kinematic slip mitigation & smooth lift
        if cube_z > self.cfg.transport_lift_z_thresh and self.grasp_acquired:
            self.transport_active = True

        slip_detected = (delta_rel > self.cfg.slip_rel_dist_delta) and (cube_z < self.cfg.lift_init_z_max)
        if not self.transport_active:
            if slip_detected and self.slip_hold_steps < self.cfg.max_slip_hold_steps:
                self.slip_hold_steps += 1
                cmd[3] = self.cfg.safe_closure_cmd
                eff_prev = max(0.0, self.prev_az)
                daz = float(np.clip(float(cmd[2]) - eff_prev, 0.0, self.cfg.slip_accel_limit_z))
                cmd[2] = float(np.clip(eff_prev + daz, 0.0, 0.35))
                if cube_z < 0.03 and float(raw_action[2]) > 0.10:
                    ratio = float(np.clip(cmd[2] / max(float(raw_action[2]), 1e-4), 0.35, 1.0))
                    cmd[0] *= ratio
                    cmd[1] *= ratio
            else:
                self.slip_hold_steps = 0
                if float(cmd[2]) > 0.0:
                    eff_prev = max(0.0, self.prev_az)
                    eff_daz = max(0.0, self.prev_daz) if self.prev_az >= 0.0 else 0.0
                    raw_jerk = (float(cmd[2]) - eff_prev) - eff_daz
                    clamped_jerk = float(np.clip(raw_jerk, -self.cfg.jerk_limit_z, self.cfg.jerk_limit_z))
                    daz = float(np.clip(eff_daz + clamped_jerk, -self.cfg.accel_limit_z, self.cfg.accel_limit_z))
                    cmd[2] = float(np.clip(eff_prev + daz, -1.0, 1.0))
                cmd[3] = self.cfg.safe_closure_cmd

            self.prev_daz = float(cmd[2]) - self.prev_az
            self.prev_az = float(cmd[2])
            self.prev_rel_dist = rel_dist
            self.prev_gw = gw
            return cmd, {
                "slip_detected": slip_detected,
                "controller_mode": "c1_kinematic_lift",
                "stage": "stage2_unprivileged_lift",
                "inferred_kinematic_grasp": True,
            }

        # Stage 4 & 5: Adaptive post-lift transport (kinematic/perception signals only)
        goal_vec = np.asarray(desired_goal, dtype=np.float32) - np.asarray(cube_pos, dtype=np.float32)
        goal_dist = float(np.linalg.norm(goal_vec))
        goal_dir = goal_vec / max(goal_dist, 1e-6)
        raw_xyz = np.asarray(cmd[:3], dtype=np.float32)
        rn = float(np.linalg.norm(raw_xyz))
        dir_align = float(np.dot(raw_xyz / max(rn, 1e-6), goal_dir)) if rn > 1e-4 else 0.0

        cmd[3] = self.cfg.safe_closure_cmd
        target_cmd = (goal_vec / self.cfg.action_scale) * self.cfg.kp_transport
        speed_cap = (
            float(np.clip(goal_dist / self.cfg.overshoot_brake_radius, 0.08, 1.0)) * self.cfg.max_transport_cmd
            if goal_dist < self.cfg.overshoot_brake_radius
            else self.cfg.max_transport_cmd
        )
        tc_norm = float(np.linalg.norm(target_cmd))
        if tc_norm > speed_cap and tc_norm > 1e-6:
            target_cmd = target_cmd * (speed_cap / tc_norm)
        target_cmd = np.clip(target_cmd, -self.cfg.max_transport_cmd, self.cfg.max_transport_cmd)

        if (goal_dist <= self.cfg.fine_align_radius) or (dir_align < self.cfg.align_cos_thresh) or (rn * self.cfg.action_scale > goal_dist * 1.15):
            blend = 1.0 if (goal_dist <= self.cfg.fine_align_radius or dir_align < 0.4) else 0.65
            blended_xyz = (1.0 - blend) * raw_xyz + blend * target_cmd
            next_ee = ee_pos + blended_xyz * self.cfg.action_scale
            clipped_ee = np.clip(next_ee, [0.20, -0.30, 0.02], [0.70, 0.30, 0.48])
            cmd[:3] = np.clip((clipped_ee - ee_pos) / self.cfg.action_scale, -1.0, 1.0)

        if (goal_dist <= self.cfg.target_settle_radius) and (float(np.linalg.norm(cube_vel)) < 0.08):
            self.stable_target_steps += 1
            if (float(desired_goal[2]) <= self.cfg.table_release_z_max) and (cube_z <= 0.028) and (self.stable_target_steps >= 3):
                cmd = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                self.placed_released = True
        else:
            self.stable_target_steps = 0

        self.prev_az = float(cmd[2])
        self.prev_rel_dist = rel_dist
        self.prev_gw = gw
        return cmd, {
            "slip_detected": slip_detected,
            "controller_mode": "c1_transport_align",
            "stage": "stage4_unprivileged_transport",
            "inferred_kinematic_grasp": True,
        }

