"""Unit and smoke tests for Franka Panda robot kinematics and Cartesian controller."""

import pytest
import numpy as np
import pybullet as p
from pathlib import Path

from src.robots.panda import FrankaPanda
from src.utils.config import load_config


@pytest.fixture
def sim_client():
    """Create and tear down a headless PyBullet physics client."""
    client_id = p.connect(p.DIRECT)
    p.resetSimulation(physicsClientId=client_id)
    p.setGravity(0, 0, -9.81, physicsClientId=client_id)
    p.setTimeStep(1.0 / 240.0, physicsClientId=client_id)
    yield client_id
    p.disconnect(physicsClientId=client_id)


@pytest.fixture
def robot_config():
    """Load the default Franka Panda configuration."""
    cfg_path = Path("configs/robot/panda.yaml")
    return load_config(cfg_path)


def test_panda_initialization(sim_client, robot_config):
    """Test loading Franka Panda into PyBullet and verifying joint configurations."""
    robot = FrankaPanda(physics_client_id=sim_client, config=robot_config)
    assert robot.body_id >= 0
    assert len(robot.arm_joint_indices) == 7
    assert len(robot.finger_joint_indices) == 2
    assert robot.ee_link_idx == 11

    # End-effector should initialize within valid workspace
    ee_pos = robot.get_ee_position()
    assert len(ee_pos) == 3
    assert robot.ws_x[0] <= ee_pos[0] <= robot.ws_x[1]
    assert robot.ws_y[0] <= ee_pos[1] <= robot.ws_y[1]
    assert robot.ws_z[0] <= ee_pos[2] <= robot.ws_z[1]


def test_panda_reset(sim_client, robot_config):
    """Test that resetting the robot puts arm joints at neutral positions with zero velocities."""
    robot = FrankaPanda(physics_client_id=sim_client, config=robot_config)
    
    # Disturb joints
    disturbed_positions = [0.2, -0.5, 0.1, -1.8, 0.1, 1.2, 0.4]
    robot.reset(joint_positions=disturbed_positions)
    np.testing.assert_allclose(robot.get_joint_positions(), disturbed_positions, atol=1e-3)

    # Reset back to default neutral pose
    robot.reset()
    np.testing.assert_allclose(robot.get_joint_positions(), robot.neutral_joint_positions, atol=1e-3)
    np.testing.assert_allclose(robot.get_joint_velocities(), np.zeros(7), atol=1e-3)


def test_workspace_clipping(sim_client, robot_config):
    """Test safety clipping to the bounding box."""
    robot = FrankaPanda(physics_client_id=sim_client, config=robot_config)

    # Point far outside workspace
    far_point = np.array([2.0, -5.0, 10.0])
    clipped = robot.clip_to_workspace(far_point)
    assert clipped[0] == robot.ws_x[1]
    assert clipped[1] == robot.ws_y[0]
    assert clipped[2] == robot.ws_z[1]

    # Point inside workspace should remain unchanged
    inside_point = np.array([0.5, 0.0, 0.25])
    np.testing.assert_allclose(robot.clip_to_workspace(inside_point), inside_point)


def test_ik_solver_accuracy(sim_client, robot_config):
    """Test that commanding a target position via IK moves the EE close to target."""
    robot = FrankaPanda(physics_client_id=sim_client, config=robot_config)
    target_pos = np.array([0.55, 0.10, 0.20])

    # Command EE pose
    robot.set_ee_pose(target_pos)

    # Step simulation to let motors settle (240 steps = 1 sec at 240 Hz)
    for _ in range(240):
        p.stepSimulation(physicsClientId=sim_client)

    actual_pos = robot.get_ee_position()
    # Should reach target within 10mm (0.010m) tolerance
    distance_error = np.linalg.norm(actual_pos - target_pos)
    assert distance_error < 0.010, f"IK error too large: {distance_error:.4f} m"


def test_cartesian_delta_motion(sim_client, robot_config):
    """Test that applying Cartesian delta actions [dx, dy, dz, 0] moves EE in expected direction."""
    robot = FrankaPanda(physics_client_id=sim_client, config=robot_config)
    initial_pos = robot.get_ee_position().copy()

    # Move along +X direction
    action = np.array([1.0, 0.0, 0.0, 0.0])
    robot.apply_action(action)

    for _ in range(60):
        p.stepSimulation(physicsClientId=sim_client)

    new_pos = robot.get_ee_position()
    assert new_pos[0] > initial_pos[0], "EE did not move in positive X direction"


def test_gripper_control(sim_client, robot_config):
    """Test gripper opening and closing actuation."""
    robot = FrankaPanda(physics_client_id=sim_client, config=robot_config)

    # Close gripper (action[3] = -1.0)
    robot.apply_action(np.array([0.0, 0.0, 0.0, -1.0]))
    for _ in range(120):
        p.stepSimulation(physicsClientId=sim_client)
    closed_width = robot.get_gripper_width()
    assert closed_width < 0.015, f"Gripper did not close properly: width={closed_width}"

    # Open gripper (action[3] = 1.0)
    robot.apply_action(np.array([0.0, 0.0, 0.0, 1.0]))
    for _ in range(120):
        p.stepSimulation(physicsClientId=sim_client)
    open_width = robot.get_gripper_width()
    assert open_width > 0.065, f"Gripper did not open properly: width={open_width}"


def test_determinism_and_reproducibility(robot_config):
    """Verify bitwise identical outcomes for identical action sequences under reset."""
    actions = [
        np.array([0.5, -0.2, 0.1, 0.0]),
        np.array([-0.3, 0.4, -0.2, 1.0]),
        np.array([0.2, 0.1, 0.3, -1.0]),
    ]

    def run_trajectory():
        cid = p.connect(p.DIRECT)
        p.resetSimulation(physicsClientId=cid)
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        p.setTimeStep(1.0 / 240.0, physicsClientId=cid)
        bot = FrankaPanda(physics_client_id=cid, config=robot_config)
        
        trajectory = []
        for act in actions:
            bot.apply_action(act)
            for _ in range(20):
                p.stepSimulation(physicsClientId=cid)
            trajectory.append(bot.get_ee_position())
        p.disconnect(physicsClientId=cid)
        return np.array(trajectory)

    traj1 = run_trajectory()
    traj2 = run_trajectory()

    np.testing.assert_array_equal(traj1, traj2)
