"""Unit tests for the configuration system."""

import pytest
from pathlib import Path
from src.utils.config import ConfigDict, load_config, deep_update


def test_config_dict_attribute_access():
    """Verify dot notation and nested dict access in ConfigDict."""
    cfg = ConfigDict({"a": 1, "b": {"c": 2, "d": {"e": "test"}}})
    assert cfg.a == 1
    assert cfg.b.c == 2
    assert cfg.b.d.e == "test"
    assert cfg["b"]["c"] == 2

    # Setting attributes dynamically
    cfg.b.new_val = 42
    assert cfg.b.new_val == 42


def test_deep_update():
    """Verify deep dictionary merging without overriding unaffected keys."""
    base = {"sim": {"timestep": 0.001, "gravity": -9.81}, "task": "reach"}
    update = {"sim": {"timestep": 0.004}, "new_key": 123}
    merged = deep_update(base, update)

    assert merged["sim"]["timestep"] == 0.004
    assert merged["sim"]["gravity"] == -9.81
    assert merged["task"] == "reach"
    assert merged["new_key"] == 123


def test_load_default_config():
    """Verify loading configs/default.yaml."""
    cfg_path = Path("configs/default.yaml")
    assert cfg_path.exists()
    cfg = load_config(cfg_path)

    assert cfg.sim.timestep > 0
    assert cfg.sim.control_frequency > 0
    assert cfg.robot.type == "panda"
    assert cfg.task.name == "reach"


def test_load_panda_robot_config():
    """Verify loading configs/robot/panda.yaml."""
    cfg_path = Path("configs/robot/panda.yaml")
    assert cfg_path.exists()
    cfg = load_config(cfg_path)

    assert cfg.robot.name == "franka_panda"
    assert len(cfg.robot.arm_joint_indices) == 7
    assert len(cfg.robot.finger_joint_indices) == 2
    assert cfg.robot.ee_link_index == 11
    assert len(cfg.robot.neutral_joint_positions) == 7
    assert cfg.robot.cartesian_control.max_displacement > 0


def test_missing_config_raises_file_not_found():
    """Verify FileNotFoundError on nonexistent configuration path."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.yaml")
