import os
import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environments.curriculum_env import PandaCurriculumEnv

def evaluate_distribution(model, env, distribution_type, num_episodes=100, seed_offset=3000):
    """Evaluate 100 deterministic episodes forcing a specific initial state."""
    
    # We temporarily set the env distribution to 100% of the requested type
    # just for the evaluation loop.
    old_dist = dict(zip(env.envs[0].env.states, env.envs[0].env.probs))
    env.envs[0].env.set_distribution({distribution_type: 1.0})
    
    results = []
    
    for ep in range(num_episodes):
        obs, info = env.envs[0].env.reset(seed=seed_offset + ep)
        
        if not info["state_initialization_valid"]:
            continue
            
        initial_dist = info["transport_dist"]
        min_dist = initial_dist
        max_z = obs["achieved_goal"][2]
        
        done = False
        step = 0
        current_hold = 0
        max_hold = 0
        dropped = False
        grasped = False
        lifted = False
        placed = False
        
        cos_sims = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info_step = env.envs[0].env.step(action)
            done = term or trunc
            step += 1
            
            ee_pos = info_step["ee_pos"]
            obj_pos = obs["achieved_goal"]
            target_pos = obs["desired_goal"]
            dist = info_step["transport_dist"]
            
            if dist < min_dist:
                min_dist = dist
            if obj_pos[2] > max_z:
                max_z = obj_pos[2]
                
            ee_obj_dist = np.linalg.norm(ee_pos - obj_pos)
            is_held = ee_obj_dist < 0.05 and action[3] < 0.0
            is_lifted = obj_pos[2] > 0.05
            
            if is_held:
                grasped = True
                current_hold += 1
                if current_hold > max_hold:
                    max_hold = current_hold
            else:
                if max_hold > 10:
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
            "initial_dist": initial_dist,
            "min_dist": min_dist,
            "final_dist": info_step["transport_dist"],
            "progress": initial_dist - min_dist,
            "hold_duration": max_hold,
            "dropped": dropped
        })
        
    # Restore original distribution
    env.envs[0].env.set_distribution(old_dist)
    
    if not results:
        return None
        
    return pd.DataFrame(results)

def write_metrics(stage, timesteps, eval_type, df, all_metrics):
    if df is None or df.empty:
        return
        
    metrics = {
        "stage": stage,
        "timesteps": timesteps,
        "initial_state_type": eval_type,
        "success_rate": df["success"].mean(),
        "grasp_rate": df["grasped"].mean(),
        "lift_rate": df["lifted"].mean(),
        "transport_rate": df["transport"].mean(),
        "placement_rate": df["placed"].mean(),
        "mean_final_distance": df["final_dist"].mean(),
        "median_final_distance": df["final_dist"].median(),
        "mean_transport_progress": df["progress"].mean(),
        "mean_hold_duration": df["hold_duration"].mean(),
        "drop_rate": df["dropped"].mean(),
        "timeout_rate": 1.0 - df["success"].mean()
    }
    all_metrics.append(metrics)
    
    # Save append
    pd.DataFrame([metrics]).to_csv("results/phase11_10_curriculum_metrics.csv", mode='a', header=not os.path.exists("results/phase11_10_curriculum_metrics.csv"), index=False)

def check_gate(stage, eval_results):
    if stage == 1:
        # lift success >= 50% AND grasp success >= 60%
        df = eval_results.get("NORMAL_START")
        if df is None: return False
        return df["lifted"].mean() >= 0.50 and df["grasped"].mean() >= 0.60
        
    elif stage == 2:
        # LIFTED_AND_HELD transport success >= 70% AND mean object-target distance <= 0.05m
        df = eval_results.get("LIFTED_AND_HELD")
        if df is None: return False
        return df["transport"].mean() >= 0.70 and df["final_dist"].mean() <= 0.05
        
    elif stage == 3:
        # NEAR_TARGET_AND_HELD placement success >= 70% AND LIFTED_AND_HELD transport success >= 70%
        df_near = eval_results.get("NEAR_TARGET_AND_HELD")
        df_lift = eval_results.get("LIFTED_AND_HELD")
        if df_near is None or df_lift is None: return False
        return df_near["placed"].mean() >= 0.70 and df_lift["transport"].mean() >= 0.70
        
    elif stage == 4:
        # Just return True to finish
        return True
        
    return False

def main():
    print("--- Phase 11.10 Curriculum Pick & Place Diagnostic ---")
    
    with open("configs/pick_place_baseline.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    env_kwargs = config["env"].copy()
    if "name" in env_kwargs:
        del env_kwargs["name"]
    if "randomizer" in env_kwargs:
        env_kwargs["randomizer_config"] = env_kwargs.pop("randomizer")
        
    exp_dir = "experiments/pnp_curriculum_seed42"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/models", exist_ok=True)
    os.makedirs("results/phase11_10", exist_ok=True)
    
    if os.path.exists("results/phase11_10_curriculum_metrics.csv"):
        os.remove("results/phase11_10_curriculum_metrics.csv")
    if os.path.exists("results/phase11_10_curriculum_distribution.csv"):
        os.remove("results/phase11_10_curriculum_distribution.csv")
        
    dist_records = []
    all_metrics = []
    
    stages = [
        {
            "id": 1,
            "dist": {"NORMAL_START": 1.0},
            "evals": ["NORMAL_START"]
        },
        {
            "id": 2,
            "dist": {"NORMAL_START": 0.5, "LIFTED_AND_HELD": 0.5},
            "evals": ["NORMAL_START", "LIFTED_AND_HELD"]
        },
        {
            "id": 3,
            "dist": {"NORMAL_START": 0.33, "LIFTED_AND_HELD": 0.33, "NEAR_TARGET_AND_HELD": 0.34},
            "evals": ["NORMAL_START", "LIFTED_AND_HELD", "NEAR_TARGET_AND_HELD"]
        },
        {
            "id": 4,
            "dist": {"NORMAL_START": 1.0},
            "evals": ["NORMAL_START"]
        }
    ]
    
    # Create single env to be reused
    base_env = PandaCurriculumEnv(**env_kwargs)
    mon_env = Monitor(base_env)
    vec_env = DummyVecEnv([lambda: mon_env])
    
    eval_base = PandaCurriculumEnv(**env_kwargs)
    eval_mon = Monitor(eval_base)
    eval_env = DummyVecEnv([lambda: eval_mon])
    
    model = SAC(
        "MultiInputPolicy",
        vec_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        learning_rate=config["rl"].get("learning_rate", 0.001),
        learning_starts=config["rl"].get("learning_starts", 100),
        buffer_size=config["rl"].get("buffer_size", 100000),
        batch_size=config["rl"].get("batch_size", 256),
        tau=config["rl"].get("tau", 0.05),
        gamma=config["rl"].get("gamma", 0.95),
        tensorboard_log=f"{exp_dir}/logs",
        verbose=1,
        seed=42
    )
    
    total_timesteps = 0
    final_classification = ""
    
    for stage_info in stages:
        stage_id = stage_info["id"]
        dist = stage_info["dist"]
        evals = stage_info["evals"]
        
        print(f"\n======================================")
        print(f"      STAGE {stage_id} ")
        print(f"======================================")
        
        # Set distribution
        vec_env.envs[0].env.set_distribution(dist)
        
        dist_records.append({
            "stage": stage_id,
            "NORMAL_START": dist.get("NORMAL_START", 0.0),
            "LIFTED_AND_HELD": dist.get("LIFTED_AND_HELD", 0.0),
            "NEAR_TARGET_AND_HELD": dist.get("NEAR_TARGET_AND_HELD", 0.0)
        })
        pd.DataFrame(dist_records).to_csv("results/phase11_10_curriculum_distribution.csv", index=False)
        
        checkpoints = [10000, 25000, 50000]
        stage_timesteps = 0
        
        gate_passed = False
        
        for cp in checkpoints:
            steps_to_train = cp - stage_timesteps
            print(f"\n[Stage {stage_id}] Training {steps_to_train} steps (Target: {cp})...")
            
            model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False)
            total_timesteps += steps_to_train
            stage_timesteps = cp
            
            model.save(f"{exp_dir}/models/model_s{stage_id}_{cp}.zip")
            
            print(f"[Stage {stage_id}] Evaluating checkpoint {cp}...")
            
            eval_results = {}
            for e_type in evals:
                df = evaluate_distribution(model, eval_env, e_type, num_episodes=100)
                eval_results[e_type] = df
                write_metrics(stage_id, total_timesteps, e_type, df, all_metrics)
                
                if df is not None:
                    print(f"  {e_type} -> Grasp: {df['grasped'].mean():.2f} | Lift: {df['lifted'].mean():.2f} | Trans: {df['transport'].mean():.2f} | Place: {df['placed'].mean():.2f}")
            
            if cp == 50000:
                gate_passed = check_gate(stage_id, eval_results)
                
        if not gate_passed:
            final_classification = f"CURRICULUM_STAGE_{stage_id}_FAILURE"
            print(f"\n!!! STAGE {stage_id} GATE FAILED !!!")
            break
        else:
            print(f"\n>>> STAGE {stage_id} GATE PASSED <<<")
            if stage_id == 4:
                # Need to check if full task success is strong
                df = eval_results.get("NORMAL_START")
                if df is not None and df["placed"].mean() > 0.5:
                    final_classification = "CURRICULUM_SUCCESS"
                else:
                    final_classification = "CURRICULUM_COMPOSITION_FAILURE"
                    
    print(f"\n--- EXPERIMENT COMPLETE ---")
    print(f"Classification: {final_classification}")
    
    # Write basic markdown report
    md = f"""# Phase 11.10 — Curriculum Pick & Place Diagnostic

## Executive Summary
This diagnostic tested whether Curriculum Learning (progressively harder initial state distributions) could allow SAC+HER to master the end-to-end Pick & Place task.
Final Result: **{final_classification}**

## Results
Please refer to `results/phase11_10_curriculum_metrics.csv` for detailed metrics across all checkpoints and stages.

## Limitations
This diagnostic relied on a single seed (Seed 42).

## Recommended Next Experiment
Depending on the failure/success stage, we recommend either evaluating hierarchical RL or scaling the compute further.
"""
    with open("docs/phase11_10_curriculum_report.md", "w") as f:
        f.write(md)

if __name__ == "__main__":
    main()
