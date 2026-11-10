"""Scripted policy for testing Assembly Environment."""

import numpy as np
import pybullet as p
from src.environments.assembly_env import PandaAssemblyEnv

def run_scripted_assembly():
    env = PandaAssemblyEnv(render_mode="human", max_episode_steps=600)
    obs, info = env.reset(seed=42)
    
    print("=== Scripted Assembly Policy Rollout ===")
    
    phase = 1
    step = 0
    
    while phase <= 7 and step < 600:
        ee_pos = obs["observation"][:3]
        ee_ori = obs["observation"][3:7] # this is the current rotation
        
        peg_pos = obs["observation"][8:11]
        peg_ori = obs["observation"][11:15]
        
        base_pos = obs["observation"][15:18]
        base_ori = obs["observation"][18:22]
        
        # Calculate yaw target from base orientation
        _, _, base_yaw = p.getEulerFromQuaternion(base_ori)
        # End effector's current yaw (assuming downward pose where Z is roughly aligned but inverted)
        _, _, ee_yaw = p.getEulerFromQuaternion(ee_ori)
        
        # Peg's current yaw
        _, _, peg_yaw = p.getEulerFromQuaternion(peg_ori)
        
        action = np.zeros(5, dtype=np.float32)
        
        if phase == 1: # Move above peg
            target = peg_pos.copy()
            target[2] = 0.15
            delta = target - ee_pos
            action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
            
            # Align yaw to peg
            dyaw = peg_yaw - ee_yaw
            dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
            action[3] = np.clip(dyaw / env.action_scale, -1.0, 1.0)
            
            action[4] = 1.0 # open gripper
            if np.linalg.norm(delta) < 0.02 and abs(dyaw) < 0.05:
                phase = 2
                
        elif phase == 2: # Move down to peg
            target = peg_pos.copy()
            target[2] = 0.03
            delta = target - ee_pos
            action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
            action[4] = 1.0
            if np.linalg.norm(delta) < 0.02:
                phase = 3
                wait_step = step
                
        elif phase == 3: # Close gripper
            action[4] = -1.0
            if step - wait_step > 60:
                phase = 4
                
        elif phase == 4: # Lift peg
            target = peg_pos.copy()
            target[2] = 0.25
            delta = target - ee_pos
            action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
            action[4] = -1.0
            if np.linalg.norm(delta) < 0.02:
                phase = 5
                
        elif phase == 5: # Move above base and align yaw
            target = base_pos.copy()
            target[2] = 0.25
            delta = target - ee_pos
            action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
            
            # Align yaw to base
            dyaw = base_yaw - peg_yaw
            dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
            action[3] = np.clip(dyaw / env.action_scale, -1.0, 1.0)
            
            action[4] = -1.0
            
            if np.linalg.norm(delta) < 0.02 and abs(dyaw) < 0.05:
                phase = 6
                
        elif phase == 6: # Move into receptacle
            target = base_pos.copy()
            target[2] = 0.04 # inside depth
            delta = target - ee_pos
            action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
            action[4] = -1.0
            if np.linalg.norm(delta) < 0.01:
                phase = 7
                wait_step = step
                
        elif phase == 7: # Open gripper to drop
            action[4] = 1.0
            if step - wait_step > 30:
                break
            
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        
        if info["is_success"]:
            print(f"Goal achieved and assembled at step {step}!")
            break
            
    if not info["is_success"]:
        print(f"Episode finished without assembly success after {step} steps")
        
    print(f"Final Info: {info}")
    env.close()

if __name__ == "__main__":
    run_scripted_assembly()
