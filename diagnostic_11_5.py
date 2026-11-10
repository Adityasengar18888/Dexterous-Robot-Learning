import os
import glob
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from src.rl.evaluate import make_env
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def analyze_actions(actions):
    actions = np.array(actions)
    return {
        "mean": np.mean(actions, axis=0).tolist(),
        "std": np.std(actions, axis=0).tolist(),
        "min": np.min(actions, axis=0).tolist(),
        "max": np.max(actions, axis=0).tolist(),
        "saturated": np.mean(np.abs(actions) > 0.95, axis=0).tolist(),
        "near_zero": np.mean(np.abs(actions) < 0.05, axis=0).tolist()
    }

def run_subthreshold_diagnostic(model_path, config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)["env"]
        
    env = make_env(cfg, render_mode=None)
    model = SAC.load(model_path, env=env)
    
    print("--- HER Audit ---")
    print(f"Replay Buffer: {model.replay_buffer_class}")
    
    # Extract env state
    num_episodes = 5
    all_actions = []
    
    results = []
    
    for ep in range(num_episodes):
        obs, info = env.reset(seed=42+ep)
        done = False
        
        initial_obj_pos = obs["achieved_goal"]
        initial_ee_pos = obs["observation"][:3]
        initial_ee_obj_dist = np.linalg.norm(initial_ee_pos - initial_obj_pos)
        
        min_ee_obj_dist = initial_ee_obj_dist
        final_ee_obj_dist = initial_ee_obj_dist
        min_obj_target_dist = np.linalg.norm(initial_obj_pos - obs["desired_goal"])
        
        gripper_near_obj_closed = 0
        rewards = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            all_actions.append(action)
            obs, rew, term, trunc, info = env.step(action)
            done = term or trunc
            rewards.append(rew)
            
            ee_pos = obs["observation"][:3]
            obj_pos = obs["achieved_goal"]
            target_pos = obs["desired_goal"]
            
            ee_obj_dist = np.linalg.norm(ee_pos - obj_pos)
            obj_target_dist = np.linalg.norm(obj_pos - target_pos)
            
            if ee_obj_dist < min_ee_obj_dist:
                min_ee_obj_dist = ee_obj_dist
                
            if obj_target_dist < min_obj_target_dist:
                min_obj_target_dist = obj_target_dist
                
            # If gripper closed and near obj
            gripper_action = action[-1]
            if gripper_action < 0 and ee_obj_dist < 0.05:
                gripper_near_obj_closed += 1
                
        results.append({
            "initial_ee_obj_dist": initial_ee_obj_dist,
            "min_ee_obj_dist": min_ee_obj_dist,
            "min_obj_target_dist": min_obj_target_dist,
            "gripper_near_obj_closed_steps": gripper_near_obj_closed,
            "reward_nonzero": np.sum(np.array(rewards) != -1.0)
        })
        
    env.close()
    
    action_stats = analyze_actions(all_actions)
    
    return results, action_stats

def main():
    print("Running Diagnostics on pnp_dr0_seed42...")
    res, act_stats = run_subthreshold_diagnostic(
        "experiments/pnp_dr0_seed42/models/final_model.zip",
        "configs/pick_place_baseline.yaml"
    )
    
    print("\n--- Sub-threshold Learning ---")
    for idx, r in enumerate(res):
        print(f"Ep {idx+1}: Init EE-Obj: {r['initial_ee_obj_dist']:.3f}, Min EE-Obj: {r['min_ee_obj_dist']:.3f}, Min Obj-Target: {r['min_obj_target_dist']:.3f}, Gripper Closed Near: {r['gripper_near_obj_closed_steps']}, Reward NonZero: {r['reward_nonzero']}")
        
    print("\n--- Action Effectiveness ---")
    print(f"Mean: {act_stats['mean']}")
    print(f"Std:  {act_stats['std']}")
    print(f"Min:  {act_stats['min']}")
    print(f"Max:  {act_stats['max']}")
    print(f"Saturated (>0.95): {act_stats['saturated']}")
    print(f"Near Zero (<0.05): {act_stats['near_zero']}")

if __name__ == "__main__":
    main()
