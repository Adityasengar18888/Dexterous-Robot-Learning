"""Scripted policy for testing Multi-Sorting Environment."""

import numpy as np
from src.environments.multi_sort_env import PandaMultiSortEnv

def run_scripted_multi_sort():
    num_objects = 2
    env = PandaMultiSortEnv(render_mode="human", num_objects=num_objects, max_episode_steps=600)
    obs, info = env.reset(seed=42)
    
    print(f"=== Scripted Multi-Sorting Policy Rollout (Objects: {num_objects}) ===")
    
    step = 0
    
    for obj_idx in range(num_objects):
        phase = 1
        
        while phase <= 7 and step < 600:
            ee_pos = obs["observation"][:3]
            
            # The achieved goal holds all object positions [N * 3]
            obj_pos = obs["achieved_goal"][obj_idx*3:(obj_idx+1)*3]
            desired_goal = obs["desired_goal"][obj_idx*3:(obj_idx+1)*3]
            
            action = np.zeros(4, dtype=np.float32)
            
            if phase == 1: # Move above object
                target = obj_pos.copy()
                target[2] = 0.15
                delta = target - ee_pos
                action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
                action[3] = 1.0 # open gripper
                if np.linalg.norm(delta) < 0.02:
                    phase = 2
                    
            elif phase == 2: # Move down to object
                target = obj_pos.copy()
                target[2] = 0.015
                delta = target - ee_pos
                action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
                action[3] = 1.0
                if np.linalg.norm(delta) < 0.02:
                    phase = 3
                    wait_step = step
                    
            elif phase == 3: # Close gripper
                action[3] = -1.0
                if step - wait_step > 60:
                    phase = 4
                    
            elif phase == 4: # Lift object
                target = obj_pos.copy()
                target[2] = 0.2
                delta = target - ee_pos
                action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
                action[3] = -1.0
                if np.linalg.norm(delta) < 0.02:
                    phase = 5
                    
            elif phase == 5: # Move above target
                target = desired_goal.copy()
                target[2] = 0.15
                delta = target - ee_pos
                action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
                action[3] = -1.0
                if np.linalg.norm(delta) < 0.02:
                    phase = 6
                    
            elif phase == 6: # Move into target
                target = desired_goal.copy()
                target[2] = 0.04
                delta = target - ee_pos
                action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
                action[3] = -1.0
                if np.linalg.norm(delta) < 0.02:
                    phase = 7
                    wait_step = step
                    
            elif phase == 7: # Open gripper to drop
                action[3] = 1.0
                if step - wait_step > 30:
                    break # Finished with this object
                
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            
            if info["is_success"]:
                break
                
        if info["is_success"]:
            print(f"Goal achieved and sorted at step {step}!")
            break
            
    if not info["is_success"]:
        print(f"Episode finished without full sorting success after {step} steps")
        print(f"Final Info: {info}")

    env.close()

if __name__ == "__main__":
    run_scripted_multi_sort()
