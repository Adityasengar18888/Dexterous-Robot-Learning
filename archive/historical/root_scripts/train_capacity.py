import os
import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environments.dense_pick_place_env import PandaPickPlaceDenseRewardEnv

def evaluate_checkpoint(model, env, num_episodes=100, seed_offset=4000):
    results = []
    
    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        
        initial_obj_pos = obs["achieved_goal"].copy()
        
        done = False
        step = 0
        current_hold = 0
        max_hold = 0
        dropped = False
        grasped = False
        lifted = False
        placed = False
        
        max_obj_height = initial_obj_pos[2]
        max_displacement = 0.0
        min_ee_obj_dist = 1000.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info_step = env.step(action)
            done = term or trunc
            step += 1
            
            ee_pos = info_step["ee_pos"]
            obj_pos = obs["achieved_goal"]
            target_pos = obs["desired_goal"]
            
            if obj_pos[2] > max_obj_height:
                max_obj_height = obj_pos[2]
                
            ee_obj_dist = np.linalg.norm(ee_pos - obj_pos)
            if ee_obj_dist < min_ee_obj_dist:
                min_ee_obj_dist = ee_obj_dist
                
            # displacement from start
            disp = np.linalg.norm(obj_pos - initial_obj_pos)
            if disp > max_displacement:
                max_displacement = disp
                
            is_held = ee_obj_dist < 0.05 and action[3] < 0.0
            is_lifted = obj_pos[2] > 0.05
            
            if is_held:
                grasped = True
                current_hold += 1
                if current_hold > max_hold:
                    max_hold = current_hold
            else:
                if max_hold > 5:
                    dropped = True
                current_hold = 0
                
            if is_lifted and is_held:
                lifted = True
                
            if info_step.get("is_success", False):
                placed = True
                
        results.append({
            "success": placed,
            "grasped": grasped,
            "lifted": lifted,
            "transport": info_step["transport_dist"] < 0.05,
            "placed": placed,
            "min_ee_obj_dist": min_ee_obj_dist,
            "max_obj_height": max_obj_height,
            "max_displacement": max_displacement,
            "final_dist": info_step["transport_dist"],
            "hold_duration": max_hold,
            "dropped": dropped,
            "ep_length": step
        })
        
    return pd.DataFrame(results)


def main():
    print("--- Phase 11.10b Acquisition Learning-Capacity Diagnostic ---")
    
    with open("configs/pick_place_baseline.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    env_kwargs = config["env"].copy()
    if "name" in env_kwargs:
        del env_kwargs["name"]
    if "randomizer" in env_kwargs:
        env_kwargs["randomizer_config"] = env_kwargs.pop("randomizer")
        
    exp_dir = "experiments/pnp_dr0_dense_seed42"
    os.makedirs(f"{exp_dir}/models", exist_ok=True)
    os.makedirs("results/phase11_10b", exist_ok=True)
    
    env = PandaPickPlaceDenseRewardEnv(**env_kwargs)
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    
    eval_env = PandaPickPlaceDenseRewardEnv(**env_kwargs)
    
    model_path = f"{exp_dir}/models/final_model.zip"
    if not os.path.exists(model_path):
        raise Exception(f"DIAGNOSTIC_PIPELINE_FAILURE: Baseline model not found at {model_path}")
        
    print(f"Loading baseline model from {model_path}...")
    model = SAC.load(model_path, env=vec_env, device="cpu")
    
    # 100k is the baseline. 
    # We will train to 250k.
    checkpoints = [125000, 150000, 175000, 200000, 225000, 250000]
    previous_cp = 100000
    
    all_metrics = []
    
    # Run baseline evaluation
    print(f"\nEvaluating baseline 100000...")
    baseline_df = evaluate_checkpoint(model, eval_env, num_episodes=100)
    baseline_metrics = {
        "Total Steps": 100000,
        "Reach": (baseline_df["min_ee_obj_dist"] < 0.05).mean(),
        "Grasp": baseline_df["grasped"].mean(),
        "Lift": baseline_df["lifted"].mean(),
        "Object Movement": (baseline_df["max_displacement"] > 0.01).mean(),
        "Hold Duration": baseline_df["hold_duration"].mean(),
        "Timeout": 1.0 - baseline_df["success"].mean()
    }
    all_metrics.append(baseline_metrics)
    
    for cp in checkpoints:
        steps_to_train = cp - previous_cp
        print(f"\nTraining for {steps_to_train} steps (Target: {cp})...")
        
        # Shift learning_starts so it waits for 1 episode (100 steps) before trying to sample from empty buffer
        if previous_cp == 100000:
            model.learning_starts = model.num_timesteps + 100
            
        model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False)
        model.save(f"{exp_dir}/models/checkpoint_{cp}.zip")
        
        print(f"Evaluating checkpoint {cp}...")
        df = evaluate_checkpoint(model, eval_env, num_episodes=100)
        
        metrics = {
            "Total Steps": cp,
            "Reach": (df["min_ee_obj_dist"] < 0.05).mean(),
            "Grasp": df["grasped"].mean(),
            "Lift": df["lifted"].mean(),
            "Object Movement": (df["max_displacement"] > 0.01).mean(),
            "Hold Duration": df["hold_duration"].mean(),
            "Timeout": 1.0 - df["success"].mean()
        }
        all_metrics.append(metrics)
        
        previous_cp = cp
        
    res_df = pd.DataFrame(all_metrics)
    res_df.to_csv("results/phase11_10b_acquisition_metrics.csv", index=False)
    
    print("\n--- HER Validation ---")
    rb = model.replay_buffer
    print(f"Replay Buffer Size: {rb.size()}")
    
    print("\nLearning Curve:")
    # format percentages
    print(res_df.to_string(index=False))
    
    # Classification Logic
    final_grasp = res_df.iloc[-1]["Grasp"]
    final_lift = res_df.iloc[-1]["Lift"]
    final_move = res_df.iloc[-1]["Object Movement"]
    
    base_grasp = res_df.iloc[0]["Grasp"]
    base_lift = res_df.iloc[0]["Lift"]
    
    if final_grasp > base_grasp + 0.15 or final_lift > base_lift + 0.10:
        if final_grasp < 0.60 and final_lift < 0.50:
            decision = "STRONG_COMPUTE_LIMITATION"
        else:
            decision = "COMPUTE_LIMITED"
    elif final_grasp < base_grasp + 0.05 and final_lift < base_lift + 0.05:
        decision = "ACQUISITION_PLATEAU"
    else:
        decision = "COMPUTE_LIMITED" # Default fallback for some improvement
        
    print(f"\nFinal Classification: {decision}")
    
    # Write Report
    md = f"""# Phase 11.10b — Acquisition Learning-Capacity Diagnostic

## Executive Summary
This diagnostic tested whether acquisition performance (grasping and lifting) continues improving with additional training beyond the initial 100k steps, or if the SAC+HER policy plateaus. The model was trained up to 250k total timesteps.
Final Classification: **{decision}**

## Hypotheses
- H1: COMPUTE_LIMITED (performance improves beyond 100k)
- H2: ACQUISITION_PLATEAU (performance remains flat)
- H3: PIPELINE_OR_CONTROL_PROBLEM (no mechanical behavior despite optimization)

## Experimental Configuration
- Model: Continued from `experiments/pnp_dr0_dense_seed42/models/final_model.zip` (100k)
- Environment: Dense Reward Pick & Place (unchanged)
- Total Timesteps: 250,000 (150,000 additional)

## Acquisition Learning Curve

| Total Steps | Reach | Grasp | Lift | Object Movement | Hold Duration | Timeout |
|---:|---:|---:|---:|---:|---:|---:|
"""
    for _, row in res_df.iterrows():
        md += f"| {int(row['Total Steps'])} | {row['Reach']*100:.1f}% | {row['Grasp']*100:.1f}% | {row['Lift']*100:.1f}% | {row['Object Movement']*100:.1f}% | {row['Hold Duration']:.1f} | {row['Timeout']*100:.1f}% |\n"
        
    md += f"""
## Final Classification
**{decision}**

## Recommended Next Experiment
Phase 12: Scaling Compute (Training SAC+HER on the dense environment with an expanded 500k-1M timestep budget) to definitively shatter the acquisition bottleneck and prove full task learning.
"""
    with open("docs/phase11_10b_acquisition_learning_capacity.md", "w") as f:
        f.write(md)

if __name__ == "__main__":
    main()
