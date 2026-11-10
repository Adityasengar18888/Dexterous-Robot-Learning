"""Smoke test: Random Policy Rollout for Reach Task."""

import gymnasium as gym
import numpy as np

from src.environments.reach_env import PandaReachEnv


def run_random_rollout():
    """Execute a purely random policy inside the Reach environment."""
    env = PandaReachEnv(render_mode="human", max_episode_steps=50)
    
    obs, info = env.reset(seed=42)
    print("=== Random Policy Rollout ===")
    print(f"Goal generated at: {obs['desired_goal']}")
    
    for step in range(50):
        # Sample uniformly from the continuous action space
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished after {info['episode_length']} steps")
            print(f"Final distance: {info['distance']:.4f}")
            print(f"Success: {info['is_success']}")
            break

    env.close()

if __name__ == "__main__":
    run_random_rollout()
