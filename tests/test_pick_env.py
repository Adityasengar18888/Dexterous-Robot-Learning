"""Tests for the Gymnasium Pick task."""

import gymnasium as gym
import numpy as np
import pytest

from src.environments.pick_env import PandaPickEnv

def test_pick_env_initialization():
    env = PandaPickEnv(render_mode=None)
    obs, info = env.reset(seed=42)
    
    assert "observation" in obs
    assert "achieved_goal" in obs
    assert "desired_goal" in obs
    assert obs["observation"].shape == (10,)
    assert obs["achieved_goal"].shape == (3,)
    assert obs["desired_goal"].shape == (3,)
    
    # Cube should spawn near z=0.025
    assert np.isclose(obs["achieved_goal"][2], 0.025, atol=0.01)
    
    # Desired goal should be target_altitude higher than achieved_goal's x,y
    assert np.isclose(obs["desired_goal"][0], obs["achieved_goal"][0], atol=0.01)
    assert np.isclose(obs["desired_goal"][1], obs["achieved_goal"][1], atol=0.01)
    assert np.isclose(obs["desired_goal"][2], obs["achieved_goal"][2] + 0.2, atol=0.01)
    
    env.close()

def test_pick_env_randomization():
    env = PandaPickEnv(render_mode=None)
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=100)
    
    # Check that positions are randomized
    assert not np.allclose(obs1["achieved_goal"], obs2["achieved_goal"])
    
    # Check that seeding works
    obs3, _ = env.reset(seed=42)
    assert np.allclose(obs1["achieved_goal"], obs3["achieved_goal"])
    
    env.close()
