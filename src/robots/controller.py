"""High-level motion and kinematics controller for the Franka Emika Panda robot."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple
import numpy as np
import pybullet as p

from src.robots.panda import FrankaPanda
from src.utils.config import ConfigDict, load_config
from src.utils.simulation import PyBulletSimulation


class PandaController:
    """
    High-level robot controller for the Franka Panda manipulator.

    Provides Cartesian end-effector positioning via Inverse Kinematics (IK),
    gripper actuation, joint telemetry, and numerical sanity checks (no NaN/Inf).
    """

    def __init__(
        self,
        simulation: PyBulletSimulation,
        config: ConfigDict | Dict[str, Any] | None = None,
        config_path: str = "configs/robot/panda.yaml",
    ) -> None:
        """
        Initialize controller with a PyBullet simulation instance.

        Args:
            simulation: Active or inactive PyBulletSimulation instance.
            config: Optional pre-loaded configuration dictionary.
            config_path: Path to robot configuration YAML if config is None.
        """
        self.sim = simulation
        if config is None:
            self.config = load_config(config_path)
        else:
            self.config = config

        self.robot: FrankaPanda | None = None

    def load(self) -> None:
        """Load the Franka Panda robot into the simulation."""
        if not self.sim.is_connected():
            self.sim.connect()

        self.robot = FrankaPanda(
            physics_client_id=self.sim.client_id,
            config=self.config,
        )

    def reset(self, joint_positions: Sequence[float] | None = None) -> None:
        """
        Reset arm joints to neutral pose (or specified joint values) and zero velocities.
        """
        if self.robot is None:
            self.load()
        assert self.robot is not None
        self.robot.reset(joint_positions=joint_positions)
        self.sim.step(10)

    def move_ee(
        self,
        position: Sequence[float] | np.ndarray,
        orientation: Sequence[float] | np.ndarray | None = None,
        steps: int = 150,
        tolerance: float = 0.005,
    ) -> np.ndarray:
        """
        Move the end-effector to a target 3D Cartesian position using inverse kinematics.

        Args:
            position: Target Cartesian position [x, y, z].
            orientation: Optional target quaternion [x, y, z, w]. If None, uses default downward pose.
            steps: Maximum number of simulation steps to allow motors to reach target.
            tolerance: Position error tolerance in meters to allow early completion.

        Returns:
            Actual reached end-effector position [x, y, z].
        """
        if self.robot is None:
            self.load()
        assert self.robot is not None

        pos = np.array(position, dtype=np.float64)
        self._check_finite(pos, "target position")

        orn = None
        if orientation is not None:
            orn = np.array(orientation, dtype=np.float64)
            self._check_finite(orn, "target orientation")

        # Solve IK and send target angles to motor controllers
        self.robot.set_ee_pose(pos, orn)

        # Step simulation to execute motion with adaptive early exit
        chunk = 10
        total_stepped = 0
        while total_stepped < steps:
            step_count = min(chunk, steps - total_stepped)
            self.sim.step(step_count)
            total_stepped += step_count
            current_pos = self.get_ee_position()
            if np.linalg.norm(current_pos - pos) <= tolerance:
                break

        actual_pos = self.get_ee_position()
        self._check_finite(actual_pos, "achieved ee position")
        return actual_pos

    def get_ee_position(self) -> np.ndarray:
        """Return the current 3D Cartesian position of the end-effector."""
        if self.robot is None:
            self.load()
        assert self.robot is not None
        pos = self.robot.get_ee_position()
        self._check_finite(pos, "ee position")
        return pos

    def get_ee_orientation(self) -> np.ndarray:
        """Return the current orientation quaternion [x, y, z, w] of the end-effector."""
        if self.robot is None:
            self.load()
        assert self.robot is not None
        orn = self.robot.get_ee_orientation()
        self._check_finite(orn, "ee orientation")
        return orn

    def get_joint_positions(self) -> np.ndarray:
        """Return array of 7 arm joint positions in radians."""
        if self.robot is None:
            self.load()
        assert self.robot is not None
        joints = self.robot.get_joint_positions()
        self._check_finite(joints, "joint positions")
        return joints

    def get_joint_velocities(self) -> np.ndarray:
        """Return array of 7 arm joint velocities in rad/s."""
        if self.robot is None:
            self.load()
        assert self.robot is not None
        vels = self.robot.get_joint_velocities()
        self._check_finite(vels, "joint velocities")
        return vels

    def open_gripper(self, steps: int = 60) -> float:
        """
        Open the parallel-jaw gripper to maximum aperture (~0.08m).

        Args:
            steps: Number of simulation steps to complete opening.

        Returns:
            Current gripper width after actuation.
        """
        if self.robot is None:
            self.load()
        assert self.robot is not None
        self.robot.set_gripper_width(self.robot.finger_max_pos * 2.0)
        self.sim.step(steps)
        return self.get_gripper_state()["width"]

    def close_gripper(self, steps: int = 60) -> float:
        """
        Close the parallel-jaw gripper to minimum aperture (~0.0m).

        Args:
            steps: Number of simulation steps to complete closing.

        Returns:
            Current gripper width after actuation.
        """
        if self.robot is None:
            self.load()
        assert self.robot is not None
        self.robot.set_gripper_width(0.0)
        self.sim.step(steps)
        return self.get_gripper_state()["width"]

    def get_gripper_state(self) -> Dict[str, Any]:
        """
        Return the current state of the gripper.

        Returns:
            Dictionary with width (float), is_open (bool), and finger positions.
        """
        if self.robot is None:
            self.load()
        assert self.robot is not None
        width = self.robot.get_gripper_width()
        self._check_finite(np.array([width]), "gripper width")
        return {
            "width": float(width),
            "is_open": bool(width > 0.04),
            "finger_positions": (float(width / 2.0), float(width / 2.0)),
        }

    @staticmethod
    def _check_finite(arr: np.ndarray, name: str) -> None:
        """Verify that an array does not contain NaN or Inf values."""
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Numerical instability detected: {name} contains NaN or Inf: {arr}")
