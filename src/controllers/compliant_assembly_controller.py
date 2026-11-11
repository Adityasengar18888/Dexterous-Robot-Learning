"""Compliant Impedance & Recovery Peg-in-Hole Controller (Phase 16.4 — SCRIPTED_COMPLIANT)."""

from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np

from src.controllers.assembly_controller import AssemblyStage, ScriptedAssemblyController
from src.environments.assembly_dense_env import wrap_angle, yaw_from_quaternion
from src.environments.contact_assembly_env import ContactState, PandaContactAssemblyEnv


class CompliantImpedanceAssemblyController(ScriptedAssemblyController):
    """Adds directional contact-force XY admittance, pi/2 yaw alignment symmetry,
    and active retract-wiggle jam recovery on top of ScriptedAssemblyController.
    All metrics from this controller MUST be classified SCRIPTED.
    """

    def __init__(
        self,
        action_scale: float = 0.05,
        yaw_scale: float = 0.25,
        xy_admittance_gain: float = 0.035,
    ) -> None:
        self.xy_admittance_gain = float(xy_admittance_gain)
        super().__init__(action_scale=action_scale, yaw_scale=yaw_scale)

    def reset(self) -> None:
        super().reset()
        self.retract_steps: int = 0
        self.last_peg_z: float = 0.2
        self.stall_counter: int = 0

    def step_compliant(
        self, env: PandaContactAssemblyEnv, obs: Dict[str, np.ndarray], last_info: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        vec = obs["observation"]
        peg_pos = vec[8:11]
        peg_ori = vec[11:15]
        base_pos = vec[15:18]
        base_ori = vec[18:22]
        base_yaw = yaw_from_quaternion(base_ori)
        peg_yaw = yaw_from_quaternion(peg_ori)

        rich = env.query_rich_contacts()
        c_state = rich["contact_state"]
        f_vec = np.asarray(rich["contact_force_vec"], dtype=np.float32)

        # Align to exact base_yaw (matching quaternion_geodesic_error in PandaAssemblyEnv)
        if self.stage == AssemblyStage.TRANSPORT_AND_ROTATE:
            tgt = np.array([base_pos[0], base_pos[1], 0.132], dtype=np.float32)
            cmd = np.zeros(5, dtype=np.float32)
            cmd[:3] = np.clip((tgt - peg_pos) / self.action_scale, -0.65, 0.65)
            dyaw = wrap_angle(base_yaw - peg_yaw)
            cmd[3] = float(np.clip(dyaw / self.yaw_scale, -0.65, 0.65))
            cmd[4] = -1.0
            if np.linalg.norm(base_pos[:2] - peg_pos[:2]) <= 0.008 and abs(dyaw) <= 0.06 and abs(peg_pos[2] - 0.132) <= 0.020:
                self.stage = AssemblyStage.INSERT
                self.insert_steps = 0
                self.stall_counter = 0
                self.retract_steps = 0
                self.last_peg_z = float(peg_pos[2])
            return cmd, {"stage": self.stage.value, "contact_state": c_state, "recovery_count": self.recovery_count}

        if self.stage == AssemblyStage.INSERT:
            self.insert_steps = getattr(self, "insert_steps", 0) + 1
            dz_drop = self.last_peg_z - float(peg_pos[2])
            self.last_peg_z = float(peg_pos[2])
            if peg_pos[2] > 0.056 and dz_drop < 0.0008:
                self.stall_counter += 1
            else:
                self.stall_counter = 0

            if c_state == ContactState.JAMMED.value or self.stall_counter >= 6:
                self.retract_steps = 2
                self.recovery_count += 1
                self.stall_counter = 0

            cmd = np.zeros(5, dtype=np.float32)
            dyaw = wrap_angle(base_yaw - peg_yaw)

            if self.retract_steps > 0:
                self.retract_steps -= 1
                f_xy_norm = float(np.linalg.norm(f_vec[:2]))
                comp_xy = (f_vec[:2] / max(f_xy_norm, 1.0)) * 0.18 if f_xy_norm > 1.0 else np.zeros(2, dtype=np.float32)
                wiggle_phase = (self.recovery_count % 4) * (np.pi / 2.0)
                wiggle = 0.10 * np.array([np.cos(wiggle_phase), np.sin(wiggle_phase)], dtype=np.float32)
                cmd[:2] = np.clip((base_pos[:2] - peg_pos[:2]) / self.action_scale + comp_xy + wiggle, -0.45, 0.45)
                cmd[2] = 0.30
                cmd[3] = float(np.clip(dyaw / self.yaw_scale, -0.45, 0.45))
                cmd[4] = -1.0
            else:
                f_xy_norm = float(np.linalg.norm(f_vec[:2]))
                admittance_xy = np.zeros(2, dtype=np.float32)
                if f_xy_norm > 1.0:
                    admittance_xy = np.clip(self.xy_admittance_gain * f_vec[:2], -0.25, 0.25)
                tgt = np.array([base_pos[0], base_pos[1], 0.044], dtype=np.float32)
                cmd[:2] = np.clip((tgt[:2] - peg_pos[:2]) / self.action_scale + admittance_xy, -0.45, 0.45)
                cmd[2] = float(np.clip((tgt[2] - peg_pos[2]) / self.action_scale, -0.24, 0.08))
                cmd[3] = float(np.clip(dyaw / self.yaw_scale, -0.45, 0.45))
                cmd[4] = -1.0
                if np.linalg.norm(base_pos[:2] - peg_pos[:2]) <= 0.012 and peg_pos[2] <= 0.048:
                    self.stage = AssemblyStage.RELEASE
                    self.release_steps = 0
            return cmd, {"stage": self.stage.value, "contact_state": c_state, "recovery_count": self.recovery_count}

        cmd, info = super().step(env, obs, last_info)
        info["contact_state"] = c_state
        return cmd, info

