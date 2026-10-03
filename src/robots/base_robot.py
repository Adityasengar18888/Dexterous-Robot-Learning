"""Abstract base class for robot manipulators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseRobot(ABC):
    """
    Abstract interface for robotic manipulators.
    
    Decouples robot physics, kinematic actuation, and gripper controls
    from environment task logic, allowing interchangeable backends (PyBullet, MuJoCo).
    """

    @abstractmethod
    def reset(self, joint_positions: np.ndarray | None = None) -> None:
        """Reset robot joints to an initial state."""
        pass

    @abstractmethod
    def get_ee_position(self) -> np.ndarray:
        """Return the current 3D Cartesian position of the end-effector [x, y, z]."""
        pass

    @abstractmethod
    def get_ee_orientation(self) -> np.ndarray:
        """Return the current orientation quaternion of the end-effector [x, y, z, w]."""
        pass

    @abstractmethod
    def get_ee_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return linear velocity (3,) and angular velocity (3,) of the end-effector."""
        pass

    @abstractmethod
    def get_joint_positions(self) -> np.ndarray:
        """Return the positions of all actuated joints."""
        pass

    @abstractmethod
    def get_joint_velocities(self) -> np.ndarray:
        """Return the velocities of all actuated joints."""
        pass

    @abstractmethod
    def get_gripper_width(self) -> float:
        """Return the current gripper aperture/width in meters."""
        pass

    @abstractmethod
    def apply_action(self, action: np.ndarray) -> None:
        """
        Apply a normalized Cartesian action [dx, dy, dz, gripper_action].
        
        Args:
            action: Array of shape (4,), entries in [-1, 1].
        """
        pass

    @abstractmethod
    def set_ee_pose(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray | None = None,
    ) -> None:
        """Command the end-effector to a Cartesian target pose via inverse kinematics."""
        pass

    @abstractmethod
    def set_gripper_width(self, width: float) -> None:
        """Set the target gripper width in meters."""
        pass
