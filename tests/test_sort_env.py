"""Tests for the Sorting environment."""

import pytest
import numpy as np
from src.environments.sort_env import PandaSortEnv

def test_sort_env_initialization():
    env = PandaSortEnv(render_mode=None)
    assert env.observation_space["observation"].shape == (22,)
    assert env.observation_space["achieved_goal"].shape == (3,)
    assert env.observation_space["desired_goal"].shape == (3,)
    env.close()

def test_sort_env_reset():
    env = PandaSortEnv(render_mode=None)
    obs, info = env.reset(seed=42)
    
    assert "observation" in obs
    assert "achieved_goal" in obs
    assert "desired_goal" in obs
    
    assert info["is_success"] is False
    assert info["episode_length"] == 0
    env.close()

def test_sort_env_step():
    env = PandaSortEnv(render_mode=None)
    env.reset(seed=42)
    
    # take a dummy action
    action = np.zeros(4, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    
    assert "is_success" in info
    assert "sorting_success" in info
    env.close()
