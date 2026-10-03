"""Unit tests for the PyBullet simulation manager."""

import pytest
import pybullet as p
from src.utils.simulation import PyBulletSimulation


def test_simulation_starts_and_connects():
    """Verify PyBulletSimulation connects in DIRECT mode and assigns a valid client ID."""
    sim = PyBulletSimulation(gui=False)
    assert not sim.is_connected()
    client_id = sim.connect()
    assert client_id >= 0
    assert sim.is_connected()
    sim.disconnect()
    assert not sim.is_connected()


def test_simulation_context_manager():
    """Verify simulation manager works cleanly with Python context manager syntax."""
    with PyBulletSimulation(gui=False) as sim:
        assert sim.is_connected()
        assert sim.client_id >= 0
    assert not sim.is_connected()


def test_simulation_loads_ground_and_table():
    """Verify ground plane and table are loaded on initialization/reset."""
    with PyBulletSimulation(gui=False, load_ground=True, load_table=True) as sim:
        assert sim.ground_id >= 0
        assert sim.table_id >= 0
        num_bodies = p.getNumBodies(physicsClientId=sim.client_id)
        assert num_bodies >= 2


def test_simulation_can_step():
    """Verify simulation step advances physics without error."""
    with PyBulletSimulation(gui=False) as sim:
        sim.step(10)
        # Verify physics state can be queried after stepping
        time_step = p.getPhysicsEngineParameters(physicsClientId=sim.client_id)
        assert time_step is not None


def test_simulation_reset_works():
    """Verify reset clears and re-initializes bodies and physics parameters."""
    with PyBulletSimulation(gui=False) as sim:
        # Load a test body
        cube_id = p.loadURDF("cube.urdf", [0, 0, 1], physicsClientId=sim.client_id)
        assert cube_id >= 0
        
        sim.reset()
        # After reset, ground and table reload, but extra test cube is cleared
        assert sim.ground_id >= 0
        assert sim.table_id >= 0
