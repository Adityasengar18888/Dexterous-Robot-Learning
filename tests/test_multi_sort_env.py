"""Tests for the Multi-Sorting environment."""

import pytest
import numpy as np
from src.environments.multi_sort_env import PandaMultiSortEnv

def test_multi_sort_env_initialization():
    env = PandaMultiSortEnv(render_mode=None, num_objects=2)
    # obs dim: 4 + (2 * 9) + 9 = 31
    assert env.observation_space["observation"].shape == (31,)
    assert env.observation_space["achieved_goal"].shape == (6,)
    assert env.observation_space["desired_goal"].shape == (6,)
    env.close()

def test_multi_sort_env_reset():
    env = PandaMultiSortEnv(render_mode=None, num_objects=3)
    obs, info = env.reset(seed=42)
    
    # obs dim: 4 + (3 * 9) + 9 = 40
    assert obs["observation"].shape == (40,)
    assert obs["achieved_goal"].shape == (9,)
    assert obs["desired_goal"].shape == (9,)
    
    assert info["is_success"] is False
    assert info["episode_length"] == 0
    env.close()

def test_multi_sort_env_step():
    env = PandaMultiSortEnv(render_mode=None, num_objects=2)
    env.reset(seed=42)
    
    action = np.zeros(4, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(reward, float)
    assert reward <= 0
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    
    assert "is_success" in info
    assert "objects_placed" in info
    assert "wrong_bin_placements" in info
    env.close()
