"""Tests for the Assembly environment."""

import pytest
import numpy as np
from src.environments.assembly_env import PandaAssemblyEnv

def test_assembly_env_initialization():
    env = PandaAssemblyEnv(render_mode=None)
    # obs dim: 3+4+1 + 3+4 + 3+4 = 8 + 7 + 7 = 22
    assert env.observation_space["observation"].shape == (22,)
    assert env.observation_space["achieved_goal"].shape == (7,)
    assert env.observation_space["desired_goal"].shape == (7,)
    env.close()

def test_assembly_env_reset():
    env = PandaAssemblyEnv(render_mode=None)
    obs, info = env.reset(seed=42)
    
    assert obs["observation"].shape == (22,)
    assert obs["achieved_goal"].shape == (7,)
    assert obs["desired_goal"].shape == (7,)
    
    assert info["is_success"] is False
    assert info["episode_length"] == 0
    assert info["assembly_state"] == "NOT_ASSEMBLED"
    env.close()

def test_assembly_env_step():
    env = PandaAssemblyEnv(render_mode=None)
    env.reset(seed=42)
    
    action = np.zeros(5, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(reward, float)
    assert reward <= 0
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    
    assert "is_success" in info
    assert "assembly_state" in info
    env.close()
