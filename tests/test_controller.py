"""Unit tests for the PandaController class."""

import pytest
import numpy as np

from src.utils.simulation import PyBulletSimulation
from src.robots.controller import PandaController


@pytest.fixture
def sim():
    """Fixture providing a connected headless PyBullet simulation."""
    simulation = PyBulletSimulation(gui=False)
    simulation.connect()
    yield simulation
    simulation.disconnect()


def test_panda_controller_load(sim):
    """Verify PandaController loads Panda robot into simulation."""
    controller = PandaController(simulation=sim)
    assert controller.robot is None
    controller.load()
    assert controller.robot is not None
    assert controller.robot.body_id >= 0


def test_panda_joints_accessible(sim):
    """Verify that arm joint positions and velocities can be read and are correct shapes."""
    controller = PandaController(simulation=sim)
    controller.load()

    positions = controller.get_joint_positions()
    velocities = controller.get_joint_velocities()

    assert isinstance(positions, np.ndarray)
    assert positions.shape == (7,)
    assert isinstance(velocities, np.ndarray)
    assert velocities.shape == (7,)


def test_ee_position_can_be_read(sim):
    """Verify that end-effector position and orientation are readable vectors."""
    controller = PandaController(simulation=sim)
    controller.load()

    ee_pos = controller.get_ee_position()
    ee_orn = controller.get_ee_orientation()

    assert isinstance(ee_pos, np.ndarray)
    assert ee_pos.shape == (3,)
    assert isinstance(ee_orn, np.ndarray)
    assert ee_orn.shape == (4,)


def test_controller_does_not_produce_nan_or_inf(sim):
    """Verify that all state queries and trajectories contain only finite real numbers."""
    controller = PandaController(simulation=sim)
    controller.load()

    assert np.all(np.isfinite(controller.get_ee_position()))
    assert np.all(np.isfinite(controller.get_ee_orientation()))
    assert np.all(np.isfinite(controller.get_joint_positions()))
    assert np.all(np.isfinite(controller.get_joint_velocities()))

    grip_state = controller.get_gripper_state()
    assert np.isfinite(grip_state["width"])


def test_move_ee_cartesian_positioning(sim):
    """Verify moving end-effector reaches target without producing NaN/Inf."""
    controller = PandaController(simulation=sim)
    controller.load()

    target = np.array([0.52, 0.05, 0.25])
    reached = controller.move_ee(target)

    assert np.all(np.isfinite(reached))
    error = np.linalg.norm(reached - target)
    assert error < 0.015, f"IK error exceeded tolerance: {error:.4f} m"


def test_gripper_open_and_close(sim):
    """Verify open_gripper and close_gripper actuate properly."""
    controller = PandaController(simulation=sim)
    controller.load()

    # Close gripper
    closed_width = controller.close_gripper(steps=80)
    assert np.isfinite(closed_width)
    assert closed_width < 0.015
    assert not controller.get_gripper_state()["is_open"]

    # Open gripper
    open_width = controller.open_gripper(steps=80)
    assert np.isfinite(open_width)
    assert open_width > 0.065
    assert controller.get_gripper_state()["is_open"]


def test_controller_reset_works(sim):
    """Verify reset restores joints and sets velocities to zero."""
    controller = PandaController(simulation=sim)
    controller.load()

    # Move to new pose
    controller.move_ee([0.60, -0.10, 0.20], steps=60)

    # Reset
    controller.reset()
    joints = controller.get_joint_positions()
    vels = controller.get_joint_velocities()

    np.testing.assert_allclose(joints, controller.robot.neutral_joint_positions, atol=1e-3)
    np.testing.assert_allclose(vels, np.zeros(7), atol=1e-2)
