"""Smoke test: Scripted Controller Baseline for Reach Task."""

import numpy as np

from src.environments.reach_env import PandaReachEnv


def run_scripted_rollout():
    """Execute a deterministic proportional controller to validate environment logic."""
    env = PandaReachEnv(render_mode="human", max_episode_steps=50)
    
    obs, info = env.reset(seed=42)
    desired_goal = obs["desired_goal"]
    
    print("=== Scripted Policy Rollout ===")
    print(f"Goal generated at: {desired_goal}")
    
    for step in range(50):
        achieved_goal = obs["achieved_goal"]
        
        # Simple Proportional Control (P-controller)
        delta = desired_goal - achieved_goal
        
        # The environment action scales delta by `action_scale` (e.g. 0.05).
        # Calculate raw action needed to traverse the delta precisely.
        action_xyz = np.clip(delta / env.action_scale, -1.0, 1.0)
        action = np.zeros(4, dtype=np.float32)
        action[:3] = action_xyz  # Keep gripper stationary
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if info["is_success"]:
            print(f"Goal achieved early at step {info['episode_length']}!")
            print(f"Final distance: {info['distance']:.4f}")
            print(f"Success: {info['is_success']}")
            break
            
        if terminated or truncated:
            print(f"Episode finished without success after {info['episode_length']} steps")
            print(f"Final distance: {info['distance']:.4f}")
            print(f"Success: {info['is_success']}")
            break

    env.close()

if __name__ == "__main__":
    run_scripted_rollout()
