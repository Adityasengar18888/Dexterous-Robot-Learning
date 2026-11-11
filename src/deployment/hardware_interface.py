"""Optional Hardware Integration Architecture (Phase 17.3 — Simulation Interface Stubs Only).

WARNING: This repository is simulation-only and no physical Franka Panda hardware is attached.
All hardware execution calls raise HardwareUnavailableError unless a verified hardware bridge
and hardware safety interlock are explicitly connected. Physical integration status: NOT_ATTEMPTED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple
import numpy as np


HARDWARE_AVAILABLE: bool = False


class HardwareUnavailableError(RuntimeError):
    """Raised whenever live robot actuation is requested without verified hardware connection."""


@dataclass(frozen=True)
class ROS2TopicContract:
    """ROS 2 Humble/Iron message and topic specification for Franka + RealSense."""

    joint_states_topic: str = "/franka_robot_state_broadcaster/robot_state"
    joint_states_type: str = "franka_msgs/msg/FrankaRobotState"
    cartesian_vel_cmd_topic: str = "/cartesian_velocity_controller/command"
    cartesian_vel_cmd_type: str = "geometry_msgs/msg/TwistStamped"
    gripper_action_topic: str = "/panda_gripper/grasp"
    gripper_action_type: str = "franka_msgs/action/Grasp"
    rgbd_color_topic: str = "/camera/color/image_raw"
    rgbd_depth_topic: str = "/camera/aligned_depth_to_color/image_raw"
    camera_info_topic: str = "/camera/color/camera_info"
    estop_service: str = "/safety_supervisor/emergency_stop"
    control_rate_hz: float = 20.0  # Policy outer loop (low-level libfranka runs at 1000 Hz)


@dataclass
class FrankaState:
    """Immutable snapshot of robot arm, gripper, and force telemetry."""

    q: np.ndarray              # (7,) joint positions (rad)
    dq: np.ndarray             # (7,) joint velocities (rad/s)
    ee_pos: np.ndarray         # (3,) base-frame EE xyz (m)
    ee_quat: np.ndarray        # (4,) base-frame EE xyzw
    gripper_width: float       # scalar (m), [0.0, 0.08]
    ext_wrench: np.ndarray     # (6,) estimated external wrench [Fx,Fy,Fz,Tx,Ty,Tz]
    timestamp_s: float = field(default_factory=time.time)
    robot_mode_ok: bool = True


@dataclass
class CartesianDeltaCommand:
    """Bounded Cartesian delta + gripper target command issued by policy inference."""

    dpos_m: np.ndarray         # (3,) desired Cartesian delta (m) over 1 policy tick
    dyaw_rad: float = 0.0
    gripper_cmd: float = 1.0   # > 0 open, <= 0 close
    timestamp_s: float = field(default_factory=time.time)


class FrankaDriverInterface(Protocol):
    """Abstract robot state/command contract decoupled from simulator or libfranka."""

    def read_state(self) -> FrankaState: ...
    def send_cartesian_delta(self, cmd: CartesianDeltaCommand) -> bool: ...
    def emergency_stop(self, reason: str) -> None: ...


class CameraStreamInterface(Protocol):
    """Abstract RGB-D camera stream contract matching Phase 16.2 vision pipeline."""

    def read_rgbd(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...  # rgb, depth_m, K(3x3)


@dataclass
class HardwareSafetyConfig:
    """Strict physical envelope guards enforced by SafetySupervisor before any command."""

    workspace_min: Tuple[float, float, float] = (0.25, -0.30, 0.015)
    workspace_max: Tuple[float, float, float] = (0.72, 0.30, 0.48)
    max_step_translation_m: float = 0.03
    max_external_force_n: float = 40.0
    max_state_age_s: float = 0.15


class SafetySupervisor:
    """Hardware-ready command gate and emergency-stop state machine."""

    def __init__(self, cfg: Optional[HardwareSafetyConfig] = None) -> None:
        self.cfg = cfg or HardwareSafetyConfig()
        self.estop_latched: bool = False
        self.estop_reason: str = ""
        self.blocked_commands: int = 0

    def trigger_estop(self, reason: str) -> None:
        self.estop_latched = True
        self.estop_reason = str(reason)

    def reset_estop(self, operator_ack: bool = False) -> bool:
        if not operator_ack:
            return False
        self.estop_latched = False
        self.estop_reason = ""
        return True

    def validate_command(self, state: FrankaState, cmd: CartesianDeltaCommand, now_s: Optional[float] = None) -> Tuple[bool, str, CartesianDeltaCommand]:
        now = time.time() if now_s is None else float(now_s)
        if self.estop_latched:
            self.blocked_commands += 1
            return False, f"estop_latched:{self.estop_reason}", CartesianDeltaCommand(np.zeros(3, dtype=np.float32), 0.0, cmd.gripper_cmd)
        if not state.robot_mode_ok:
            self.trigger_estop("robot_mode_fault")
            self.blocked_commands += 1
            return False, self.estop_reason, CartesianDeltaCommand(np.zeros(3, dtype=np.float32), 0.0, cmd.gripper_cmd)
        if (now - state.timestamp_s) > self.cfg.max_state_age_s:
            self.trigger_estop("stale_state_timeout")
            self.blocked_commands += 1
            return False, self.estop_reason, CartesianDeltaCommand(np.zeros(3, dtype=np.float32), 0.0, cmd.gripper_cmd)
        if not (np.all(np.isfinite(state.ee_pos)) and np.all(np.isfinite(cmd.dpos_m))):
            self.trigger_estop("non_finite_telemetry_or_command")
            self.blocked_commands += 1
            return False, self.estop_reason, CartesianDeltaCommand(np.zeros(3, dtype=np.float32), 0.0, cmd.gripper_cmd)
        force_mag = float(np.linalg.norm(state.ext_wrench[:3]))
        if force_mag > self.cfg.max_external_force_n:
            self.trigger_estop(f"force_limit_exceeded:{force_mag:.1f}N")
            self.blocked_commands += 1
            return False, self.estop_reason, CartesianDeltaCommand(np.zeros(3, dtype=np.float32), 0.0, cmd.gripper_cmd)

        dpos = np.asarray(cmd.dpos_m, dtype=np.float32).copy()
        dnorm = float(np.linalg.norm(dpos))
        if dnorm > self.cfg.max_step_translation_m:
            dpos = dpos * (self.cfg.max_step_translation_m / dnorm)
        next_pos = state.ee_pos[:3] + dpos
        ws_min = np.asarray(self.cfg.workspace_min, dtype=np.float32)
        ws_max = np.asarray(self.cfg.workspace_max, dtype=np.float32)
        clipped = np.clip(next_pos, ws_min, ws_max)
        safe_dpos = (clipped - state.ee_pos[:3]).astype(np.float32)
        return True, "ok", CartesianDeltaCommand(dpos_m=safe_dpos, dyaw_rad=float(cmd.dyaw_rad), gripper_cmd=float(cmd.gripper_cmd))


class StubLiveFrankaDriver:
    """Safety stub refusing physical actuation because no live Franka hardware exists."""

    def read_state(self) -> FrankaState:
        raise HardwareUnavailableError("Physical Franka Panda driver is not connected (simulation-only repository).")

    def send_cartesian_delta(self, cmd: CartesianDeltaCommand) -> bool:
        raise HardwareUnavailableError("Refusing live hardware command: physical Franka Panda is not available.")

    def emergency_stop(self, reason: str) -> None:
        raise HardwareUnavailableError(f"Physical E-stop unavailable (no hardware): {reason}")


def get_hardware_interface_documentation() -> Dict[str, Any]:
    """Structured hardware interface manifest for Phase 17.3 reporting."""
    topics = ROS2TopicContract()
    safety = HardwareSafetyConfig()
    return {
        "hardware_available": HARDWARE_AVAILABLE,
        "physical_integration_status": "NOT_ATTEMPTED",
        "ros2_topics": topics.__dict__,
        "safety_limits": safety.__dict__,
        "supported_abstractions": [
            "FrankaDriverInterface",
            "CameraStreamInterface",
            "SafetySupervisor",
            "StubLiveFrankaDriver",
        ],
    }

