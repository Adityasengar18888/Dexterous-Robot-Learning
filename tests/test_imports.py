"""Smoke test to verify that all core robotic and RL dependencies import cleanly."""

import importlib
import pytest


@pytest.mark.parametrize(
    "package_name",
    [
        "pybullet",
        "gymnasium",
        "stable_baselines3",
        "torch",
        "numpy",
        "scipy",
        "pandas",
        "yaml",
        "tensorboard",
        "pytest",
    ],
)
def test_core_dependency_imports(package_name: str):
    """Verify that each required core library can be imported without errors."""
    module = importlib.import_module(package_name)
    assert module is not None


def test_stable_baselines3_her_sac():
    """Verify SAC and HerReplayBuffer are available in Stable-Baselines3."""
    from stable_baselines3 import SAC, HerReplayBuffer
    assert SAC is not None
    assert HerReplayBuffer is not None


def test_gymnasium_goal_env():
    """Verify gymnasium.Env and GoalEnv spaces compatibility."""
    import gymnasium as gym
    from gymnasium import spaces

    assert hasattr(gym, "Env")
    # Verify Dict space for goal-conditioned format
    obs_space = spaces.Dict(
        {
            "observation": spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=float),
            "achieved_goal": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=float),
            "desired_goal": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=float),
        }
    )
    sample = obs_space.sample()
    assert "observation" in sample
    assert "achieved_goal" in sample
    assert "desired_goal" in sample


def test_pybullet_connection():
    """Verify pybullet can initialize in DIRECT headless mode and load standard data assets."""
    import pybullet as p
    import pybullet_data

    cid = p.connect(p.DIRECT)
    try:
        assert cid >= 0
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        plane_id = p.loadURDF("plane.urdf", physicsClientId=cid)
        assert plane_id >= 0
    finally:
        p.disconnect(physicsClientId=cid)
