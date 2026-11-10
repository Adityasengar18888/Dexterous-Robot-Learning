"""Scripted policy for testing Sorting Environment."""

import numpy as np
from src.environments.sort_env import PandaSortEnv

def run_scripted_sort():
    env = PandaSortEnv(render_mode="human", max_episode_steps=200)
    obs, info = env.reset(seed=42)
    
    print("=== Scripted Sorting Policy Rollout ===")
    
    phase = 1
    initial_obj_pos = obs["achieved_goal"].copy()
    
    for step in range(200):
        ee_pos = obs["observation"][:3]
        achieved_goal = obs["achieved_goal"]
        desired_goal = obs["desired_goal"] # This is the matching bin pos
        
        action = np.zeros(4, dtype=np.float32)
        
        if phase == 1: # Move above object
            target = initial_obj_pos.copy()
            target[2] += 0.15
            delta = target - ee_pos
            action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
            action[3] = 1.0 # open gripper
            if np.linalg.norm(delta) < 0.02:
                phase = 2
                
        elif phase == 2: # Move down to object
            target = initial_obj_pos.copy()
            target[2] = 0.015 # Grasp lower around the middle of the cube
            delta = target - ee_pos
            action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
            action[3] = 1.0 # open gripper
            if np.linalg.norm(delta) < 0.02:
                phase = 3
                
        elif phase == 3: # Close gripper
            action[3] = -1.0
            if step > 60: # Wait for gripper to close
                phase = 4
                
        elif phase == 4: # Lift object
            target = initial_obj_pos.copy()
            target[2] += 0.2
            delta = target - ee_pos
            action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
            action[3] = -1.0 # keep closed
            if np.linalg.norm(delta) < 0.02:
                phase = 5
                
        elif phase == 5: # Move to above target bin
            target = desired_goal.copy()
            target[2] += 0.15
            delta = target - ee_pos
            action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
            action[3] = -1.0
            if np.linalg.norm(delta) < 0.02:
                phase = 6
                
        elif phase == 6: # Move down into target bin
            target = desired_goal.copy()
            target[2] = 0.04 # Hover slightly over the bin floor
            delta = target - ee_pos
            action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
            action[3] = -1.0
            if np.linalg.norm(delta) < 0.02:
                phase = 7
                
        elif phase == 7: # Open gripper to drop
            action[3] = 1.0
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        if phase == 7 and info["sorting_success"]:
            print(f"Goal achieved and sorted at step {info['episode_length']}!")
            print(f"Final distance: {info['distance']:.4f}")
            break
            
    if not info["sorting_success"]:
        print(f"Episode finished without sorting success after {info['episode_length']} steps")
        print(f"Final distance: {info['distance']:.4f}")
        print(f"Target pos: {desired_goal}")
        print(f"Object pos: {achieved_goal}")
        print(f"Wrong Bin: {info['wrong_bin_placement']}")

    env.close()

if __name__ == "__main__":
    run_scripted_sort()
