"""Tests for the Gymnasium environment foundation."""

import gymnasium as gym
import numpy as np
import pytest

from src.environments.reach_env import PandaReachEnv

def test_reach_env_initialization():
    """Verify environment initialization and spaces."""
    env = PandaReachEnv(render_mode=None)
    
    assert isinstance(env.observation_space, gym.spaces.Dict)
    assert isinstance(env.action_space, gym.spaces.Box)
    assert env.action_space.shape == (4,)
    
    obs, info = env.reset(seed=42)
    assert "observation" in obs
    assert "achieved_goal" in obs
    assert "desired_goal" in obs
    
    assert "is_success" in info
    assert "distance" in info
    env.close()

def test_reach_env_step():
    """Verify environment stepping, truncation, and return types."""
    env = PandaReachEnv(render_mode=None, max_episode_steps=5)
    obs, info = env.reset(seed=42)
    
    # Take a zero action (no movement)
    action = np.zeros(4, dtype=np.float32)
    obs_next, reward, terminated, truncated, info_next = env.step(action)
    
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert not truncated
    assert not terminated
    
    # Take 5 steps to check truncation
    for _ in range(5):
        obs_next, reward, terminated, truncated, info_next = env.step(action)
        
    assert truncated
    env.close()

def test_reach_env_compute_reward():
    """Verify batched and unbatched sparse reward computation."""
    env = PandaReachEnv(render_mode=None, distance_threshold=0.05)
    
    # Test unbatched (single instance)
    achieved = np.array([0.5, 0.0, 0.2])
    desired_success = np.array([0.51, 0.0, 0.2])  # distance 0.01 <= 0.05
    desired_fail = np.array([0.6, 0.0, 0.2])     # distance 0.1 > 0.05
    
    rew_success = env.compute_reward(achieved, desired_success, {})
    rew_fail = env.compute_reward(achieved, desired_fail, {})
    
    assert rew_success == 0.0
    assert rew_fail == -1.0
    
    # Test batched (e.g. from SB3 HER)
    achieved_batch = np.array([[0.5, 0.0, 0.2], [0.5, 0.0, 0.2]])
    desired_batch = np.array([[0.51, 0.0, 0.2], [0.6, 0.0, 0.2]])
    
    rew_batch = env.compute_reward(achieved_batch, desired_batch, {})
    assert np.allclose(rew_batch, [0.0, -1.0])
    
    env.close()

def test_reach_env_reproducibility():
    """Verify seeding produces consistent trajectories."""
    env1 = PandaReachEnv(render_mode=None)
    env2 = PandaReachEnv(render_mode=None)
    
    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)
    
    assert np.allclose(obs1["desired_goal"], obs2["desired_goal"])
    
    # Take deterministic step
    action = np.array([0.5, -0.5, 0.1, 1.0], dtype=np.float32)
    step1 = env1.step(action)
    step2 = env2.step(action)
    
    assert np.allclose(step1[0]["observation"], step2[0]["observation"])
    
    env1.close()
    env2.close()
