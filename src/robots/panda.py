"""PyBullet implementation of the Franka Emika Panda manipulator."""

from __future__ import annotations

from typing import Tuple, Sequence, Dict, Any
import numpy as np
import pybullet as p
import pybullet_data

from src.robots.base_robot import BaseRobot
from src.utils.config import ConfigDict


class FrankaPanda(BaseRobot):
    """
    Franka Emika Panda 7-DOF manipulator with 2-finger parallel jaw gripper in PyBullet.

    Provides Cartesian end-effector control [dx, dy, dz, gripper], IK resolution,
    joint state observation, and workspace bounding box clipping.
    """

    def __init__(
        self,
        physics_client_id: int,
        config: ConfigDict | Dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the Franka Panda robot inside a PyBullet simulation client.

        Args:
            physics_client_id: PyBullet physics server connection ID.
            config: Robot configuration dictionary or ConfigDict.
        """
        self.cid = physics_client_id
        if config is None:
            # Fallback default configuration if not provided
            config = {}
        robot_cfg = config.get("robot", config)

        self.urdf_path = robot_cfg.get("urdf_path", "franka_panda/panda.urdf")
        self.base_position = list(robot_cfg.get("base_position", [0.0, 0.0, 0.0]))
        self.base_orientation = list(robot_cfg.get("base_orientation", [0.0, 0.0, 0.0, 1.0]))
        self.use_fixed_base = bool(robot_cfg.get("use_fixed_base", True))

        # Joint & link layout for franka_panda/panda.urdf
        self.arm_joint_indices = list(robot_cfg.get("arm_joint_indices", [0, 1, 2, 3, 4, 5, 6]))
        self.finger_joint_indices = list(robot_cfg.get("finger_joint_indices", [9, 10]))
        self.ee_link_idx = int(robot_cfg.get("ee_link_index", 11)) # panda_grasptarget
        self.hand_link_idx = int(robot_cfg.get("hand_link_index", 8))

        # Kinematic limits for null-space projection
        self.joint_lower_limits = list(
            robot_cfg.get("joint_lower_limits", [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        )
        self.joint_upper_limits = list(
            robot_cfg.get("joint_upper_limits", [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        )
        self.joint_ranges = list(
            robot_cfg.get("joint_ranges", [5.7946, 3.5256, 5.7946, 3.002, 5.7946, 3.77, 5.7946])
        )
        self.neutral_joint_positions = list(
            robot_cfg.get("neutral_joint_positions", [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        )
        self.default_ee_orientation = list(
            robot_cfg.get("default_ee_orientation", [1.0, 0.0, 0.0, 0.0])
        )

        # Cartesian control parameters
        cart_cfg = robot_cfg.get("cartesian_control", {})
        self.max_displacement = float(cart_cfg.get("max_displacement", 0.05))
        ws_bounds = cart_cfg.get("workspace_bounds", {})
        self.ws_x = list(ws_bounds.get("x", [0.25, 0.75]))
        self.ws_y = list(ws_bounds.get("y", [-0.35, 0.35]))
        self.ws_z = list(ws_bounds.get("z", [0.02, 0.50]))

        # IK solver parameters
        ik_cfg = robot_cfg.get("ik", {})
        self.ik_max_iterations = int(ik_cfg.get("max_iterations", 100))
        self.ik_residual_threshold = float(ik_cfg.get("residual_threshold", 1e-4))
        self.ik_damping = float(ik_cfg.get("damping", 0.05))

        # Gripper parameters
        grip_cfg = robot_cfg.get("gripper", {})
        self.finger_min_pos = float(grip_cfg.get("finger_min_position", 0.0))
        self.finger_max_pos = float(grip_cfg.get("finger_max_position", 0.04))
        self.gripper_max_force = float(grip_cfg.get("max_force", 25.0))
        self.gripper_open_pos = float(grip_cfg.get("open_position", 0.04))
        self.gripper_closed_pos = float(grip_cfg.get("closed_position", 0.0))

        # Max motor forces
        self.arm_max_forces = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]

        # Load the robot into PyBullet
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        self.body_id = p.loadURDF(
            self.urdf_path,
            basePosition=self.base_position,
            baseOrientation=self.base_orientation,
            useFixedBase=self.use_fixed_base,
            physicsClientId=self.cid,
        )

        # Set joint collision and damping properties
        for joint_idx in self.arm_joint_indices:
            p.changeDynamics(
                self.body_id,
                joint_idx,
                linearDamping=0.04,
                angularDamping=0.04,
                physicsClientId=self.cid,
            )

        self.reset()

    def reset(self, joint_positions: Sequence[float] | None = None) -> None:
        """
        Reset arm joints to specified positions (or neutral home pose) and zero velocities.
        """
        target_joints = self.neutral_joint_positions if joint_positions is None else list(joint_positions)
        
        for idx, pos in zip(self.arm_joint_indices, target_joints):
            p.resetJointState(
                self.body_id,
                idx,
                targetValue=float(pos),
                targetVelocity=0.0,
                physicsClientId=self.cid,
            )

        # Reset gripper to fully open state
        for finger_idx in self.finger_joint_indices:
            p.resetJointState(
                self.body_id,
                finger_idx,
                targetValue=self.gripper_open_pos,
                targetVelocity=0.0,
                physicsClientId=self.cid,
            )

        # Enforce position control holding current targets
        self._command_arm_joints(target_joints)
        self.set_gripper_width(self.gripper_open_pos * 2.0)

    def get_ee_position(self) -> np.ndarray:
        """Return 3D Cartesian position [x, y, z] of the end-effector grasp target."""
        state = p.getLinkState(self.body_id, self.ee_link_idx, physicsClientId=self.cid)
        return np.array(state[0], dtype=np.float32)

    def get_ee_orientation(self) -> np.ndarray:
        """Return quaternion [x, y, z, w] orientation of the end-effector."""
        state = p.getLinkState(self.body_id, self.ee_link_idx, physicsClientId=self.cid)
        return np.array(state[1], dtype=np.float32)

    def get_ee_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (linear_velocity [3], angular_velocity [3]) of the end-effector."""
        state = p.getLinkState(
            self.body_id,
            self.ee_link_idx,
            computeLinkVelocity=1,
            physicsClientId=self.cid,
        )
        linear_vel = np.array(state[6], dtype=np.float32)
        angular_vel = np.array(state[7], dtype=np.float32)
        return linear_vel, angular_vel

    def get_joint_positions(self) -> np.ndarray:
        """Return array of 7 arm joint positions in radians."""
        states = p.getJointStates(self.body_id, self.arm_joint_indices, physicsClientId=self.cid)
        return np.array([s[0] for s in states], dtype=np.float32)

    def get_joint_velocities(self) -> np.ndarray:
        """Return array of 7 arm joint velocities in rad/s."""
        states = p.getJointStates(self.body_id, self.arm_joint_indices, physicsClientId=self.cid)
        return np.array([s[1] for s in states], dtype=np.float32)

    def get_gripper_width(self) -> float:
        """Return current distance between gripper finger pads in meters."""
        states = p.getJointStates(self.body_id, self.finger_joint_indices, physicsClientId=self.cid)
        return float(states[0][0] + states[1][0])

    def clip_to_workspace(self, position: np.ndarray | Sequence[float]) -> np.ndarray:
        """Clip a 3D position vector strictly to the robot's configured workspace boundaries."""
        pos = np.array(position, dtype=np.float64)
        pos[0] = np.clip(pos[0], self.ws_x[0], self.ws_x[1])
        pos[1] = np.clip(pos[1], self.ws_y[0], self.ws_y[1])
        pos[2] = np.clip(pos[2], self.ws_z[0], self.ws_z[1])
        return pos

    def calculate_ik(
        self,
        target_position: np.ndarray | Sequence[float],
        target_orientation: np.ndarray | Sequence[float] | None = None,
    ) -> np.ndarray:
        """
        Compute inverse kinematics for the 7 arm joints using PyBullet damped least squares.
        
        Args:
            target_position: 3D coordinates [x, y, z].
            target_orientation: Quaternion [x, y, z, w]. If None, uses default downward orientation.

        Returns:
            Array of 7 joint angles.
        """
        clipped_pos = self.clip_to_workspace(target_position)
        orn = self.default_ee_orientation if target_orientation is None else list(target_orientation)

        # PyBullet returns joint angles for all joints (including fixed and finger joints)
        all_ik_angles = p.calculateInverseKinematics(
            bodyUniqueId=self.body_id,
            endEffectorLinkIndex=self.ee_link_idx,
            targetPosition=list(clipped_pos),
            targetOrientation=orn,
            lowerLimits=self.joint_lower_limits,
            upperLimits=self.joint_upper_limits,
            jointRanges=self.joint_ranges,
            restPoses=self.neutral_joint_positions,
            maxNumIterations=self.ik_max_iterations,
            residualThreshold=self.ik_residual_threshold,
            physicsClientId=self.cid,
        )

        # Map to arm joint indices (first 7 arm joints)
        arm_angles = np.array(all_ik_angles[:7], dtype=np.float64)
        # Enforce joint limit clipping for kinematic safety
        for i in range(7):
            arm_angles[i] = np.clip(
                arm_angles[i],
                self.joint_lower_limits[i],
                self.joint_upper_limits[i],
            )
        return arm_angles

    def set_ee_pose(
        self,
        target_position: np.ndarray | Sequence[float],
        target_orientation: np.ndarray | Sequence[float] | None = None,
    ) -> None:
        """Solve IK and command arm joints to reach target Cartesian pose."""
        joint_targets = self.calculate_ik(target_position, target_orientation)
        self._command_arm_joints(joint_targets)

    def set_gripper_width(self, width: float) -> None:
        """
        Command gripper fingers to reach a target opening width in meters.

        Args:
            width: Target total opening width in meters (clamped to [0.0, 0.08]).
        """
        finger_pos = np.clip(width / 2.0, self.finger_min_pos, self.finger_max_pos)
        p.setJointMotorControlArray(
            self.body_id,
            self.finger_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[finger_pos, finger_pos],
            forces=[self.gripper_max_force, self.gripper_max_force],
            physicsClientId=self.cid,
        )

    def apply_action(self, action: np.ndarray | Sequence[float]) -> None:
        """
        Apply normalized Cartesian action [dx, dy, dz, gripper_action] in [-1.0, 1.0].
        
        Args:
            action: Array of shape (4,). First 3 elements correspond to normalized
                    Cartesian displacement. 4th element controls gripper aperture
                    (> 0 opens, <= 0 closes).
        """
        act = np.clip(np.array(action, dtype=np.float64), -1.0, 1.0)
        
        # Scale displacement by max_displacement
        delta_xyz = act[:3] * self.max_displacement
        current_pos = self.get_ee_position()
        target_pos = self.clip_to_workspace(current_pos + delta_xyz)

        self.set_ee_pose(target_pos, self.default_ee_orientation)

        # Gripper action: continuous mapping from [-1, 1] to [0.0, 0.08] m
        target_width = float((act[3] + 1.0) / 2.0 * (self.finger_max_pos * 2.0))
        self.set_gripper_width(target_width)

    def _command_arm_joints(self, target_positions: Sequence[float]) -> None:
        """Send position control commands to the 7 arm joints."""
        p.setJointMotorControlArray(
            self.body_id,
            self.arm_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=list(target_positions),
            forces=self.arm_max_forces,
            physicsClientId=self.cid,
        )
