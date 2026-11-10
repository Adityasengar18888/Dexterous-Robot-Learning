"""Scripted policy for testing Pick Environment."""

import numpy as np
from src.environments.pick_env import PandaPickEnv

def run_scripted_pick():
    env = PandaPickEnv(render_mode="human", max_episode_steps=150)
    obs, info = env.reset(seed=42)
    
    print("=== Scripted Pick Policy Rollout ===")
    
    # Phases:
    # 1. Move to directly above the object
    # 2. Move down to grasp
    # 3. Close gripper
    # 4. Move up to desired goal
    
    phase = 1
    
    for step in range(150):
        ee_pos = obs["observation"][:3]
        obj_pos = obs["achieved_goal"]
        desired_goal = obs["desired_goal"]
        
        action = np.zeros(4, dtype=np.float32)
        
        if phase == 1:
            target = obj_pos.copy()
            target[2] += 0.15 # 15cm above object
            delta = target - ee_pos
            action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
            action[3] = 1.0 # open gripper
            if np.linalg.norm(delta) < 0.02:
                phase = 2
                
        elif phase == 2:
            target = obj_pos.copy()
            target[2] += 0.02 # just above object center (fingers overlap cube)
            delta = target - ee_pos
            action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
            action[3] = 1.0 # open gripper
            if np.linalg.norm(delta) < 0.02:
                phase = 3
                
        elif phase == 3:
            action[3] = -1.0 # close gripper
            # Phase transitions by time
            if step > 60:
                phase = 4
                
        elif phase == 4:
            target = desired_goal
            delta = target - ee_pos
            action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
            action[3] = -1.0 # keep closed
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        if info["is_success"]:
            print(f"Goal achieved at step {info['episode_length']}!")
            print(f"Final distance: {info['distance']:.4f}")
            break
            
    if not info["is_success"]:
        print(f"Episode finished without success after {info['episode_length']} steps")
        print(f"Final distance: {info['distance']:.4f}")

    env.close()

if __name__ == "__main__":
    run_scripted_pick()
