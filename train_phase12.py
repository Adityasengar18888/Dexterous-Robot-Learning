import os
import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import shutil

from src.environments.dense_pick_place_env import PandaPickPlaceDenseRewardEnv

def evaluate_checkpoint(model, env, num_episodes=100, seed_offset=5000):
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
        
        # for transport dir
        positive_align_count = 0
        held_steps = 0
        
        total_rew = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info_step = env.step(action)
            done = term or trunc
            step += 1
            total_rew += reward
            
            ee_pos = info_step["ee_pos"]
            obj_pos = obs["achieved_goal"]
            target_pos = obs["desired_goal"]
            
            if obj_pos[2] > max_obj_height:
                max_obj_height = obj_pos[2]
                
            ee_obj_dist = np.linalg.norm(ee_pos - obj_pos)
            if ee_obj_dist < min_ee_obj_dist:
                min_ee_obj_dist = ee_obj_dist
                
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
                    
                # calculate direction
                target_dir = target_pos - obj_pos
                if np.linalg.norm(target_dir) > 1e-4:
                    target_dir = target_dir / np.linalg.norm(target_dir)
                    align = np.dot(action[:3], target_dir)
                    if align > 0:
                        positive_align_count += 1
                held_steps += 1
            else:
                if max_hold > 5:
                    dropped = True
                current_hold = 0
                
            if is_lifted and is_held:
                lifted = True
                
            if info_step.get("is_success", False):
                placed = True
                
        # Failure taxonomy
        failure_type = "SUCCESS"
        if not placed:
            if not grasped and min_ee_obj_dist > 0.05:
                failure_type = "NO_REACH"
            elif not grasped and min_ee_obj_dist <= 0.05:
                failure_type = "REACH_BUT_NO_GRASP"
            elif grasped and not lifted:
                failure_type = "GRASP_BUT_NO_LIFT"
            elif lifted and dropped:
                failure_type = "LIFT_BUT_DROPPED"
            elif lifted and not dropped:
                # did it transport?
                if info_step["transport_dist"] > 0.05:
                    failure_type = "LIFT_BUT_WRONG_TRANSPORT"
                else:
                    failure_type = "REACHED_TARGET_BUT_NO_PLACE"

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
            "ep_length": step,
            "failure_type": failure_type,
            "reward": total_rew
        })
        
    return pd.DataFrame(results)


def main():
    print("--- Phase 12 1M-Step Dense SAC+HER Scaling Experiment ---")
    
    with open("configs/pick_place_baseline.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    env_kwargs = config["env"].copy()
    if "name" in env_kwargs:
        del env_kwargs["name"]
    if "randomizer" in env_kwargs:
        env_kwargs["randomizer_config"] = env_kwargs.pop("randomizer")
        
    exp_dir = "experiments/pnp_dr0_dense_seed42_1m"
    os.makedirs(f"{exp_dir}/models", exist_ok=True)
    os.makedirs("results/phase12_1m", exist_ok=True)
    
    # 1. Select the 225k checkpoint
    source_model_path = "experiments/pnp_dr0_dense_seed42/models/checkpoint_225000.zip"
    if not os.path.exists(source_model_path):
        raise Exception(f"DIAGNOSTIC_PIPELINE_FAILURE: Starting checkpoint not found at {source_model_path}")
        
    # Copy baseline model so we have a record
    shutil.copy(source_model_path, f"{exp_dir}/models/checkpoint_225k.zip")
    
    env = PandaPickPlaceDenseRewardEnv(**env_kwargs)
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    
    eval_env = PandaPickPlaceDenseRewardEnv(**env_kwargs)
    
    print(f"Loading baseline model from {source_model_path}...")
    model = SAC.load(source_model_path, env=vec_env, device="cpu")
    
    checkpoints = [300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
    previous_cp = 225000
    
    all_metrics = []
    
    for cp in checkpoints:
        steps_to_train = cp - previous_cp
        print(f"\nTraining for {steps_to_train} steps (Target: {cp})...")
        
        # We need to shift learning starts if the buffer is empty
        if previous_cp == 225000:
            model.learning_starts = model.num_timesteps + 100
            
        model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False)
        
        cp_path = f"{exp_dir}/models/checkpoint_{int(cp/1000)}k.zip"
        model.save(cp_path)
        
        print(f"Evaluating checkpoint {cp}...")
        df = evaluate_checkpoint(model, eval_env, num_episodes=100)
        
        metrics = {
            "Total Steps": cp,
            "Reach": (df["min_ee_obj_dist"] < 0.05).mean(),
            "Grasp": df["grasped"].mean(),
            "Lift": df["lifted"].mean(),
            "Transport": df["transport"].mean(),
            "Placement": df["placed"].mean(),
            "End-to-End Success": df["success"].mean(),
            "Object Movement": (df["max_displacement"] > 0.01).mean(),
            "Mean Final Dist": df["final_dist"].mean(),
            "Hold Duration": df["hold_duration"].mean(),
            "Timeout Rate": 1.0 - df["success"].mean()
        }
        all_metrics.append(metrics)
        
        # Save running metrics
        res_df = pd.DataFrame(all_metrics)
        res_df.to_csv("results/phase12_1m/checkpoint_metrics.csv", index=False)
        
        # Save failure taxonomy for this checkpoint
        taxonomy_df = df["failure_type"].value_counts().reset_index()
        taxonomy_df.columns = ["Failure Type", "Count"]
        taxonomy_df["Checkpoint"] = cp
        
        tax_path = "results/phase12_1m/failure_taxonomy.csv"
        taxonomy_df.to_csv(tax_path, mode='a', header=not os.path.exists(tax_path), index=False)
        
        print(f"  Reach: {metrics['Reach']:.2f} | Grasp: {metrics['Grasp']:.2f} | Lift: {metrics['Lift']:.2f} | Success: {metrics['End-to-End Success']:.2f}")
        
        previous_cp = cp
        
    print("\n--- HER Validation ---")
    rb = model.replay_buffer
    print(f"Replay Buffer Size: {rb.size()}")
    
    # Classification Logic
    final_success = res_df.iloc[-1]["End-to-End Success"]
    final_transport = res_df.iloc[-1]["Transport"]
    final_lift = res_df.iloc[-1]["Lift"]
    final_grasp = res_df.iloc[-1]["Grasp"]
    
    if final_success >= 0.5:
        decision = "END_TO_END_LEARNING"
    elif final_transport >= 0.5:
        decision = "PLACEMENT_LIMITED"
    elif final_lift >= 0.5:
        decision = "TRANSPORT_LIMITED"
    elif final_grasp >= 0.5:
        decision = "ACQUISITION_LIMITED"
    elif final_success > 0.05 or final_lift > 0.05 or final_transport > 0.05:
        decision = "STRONG_PARTIAL_LEARNING"
    else:
        decision = "PLATEAU"
        
    print(f"\nFinal Classification: {decision}")

if __name__ == "__main__":
    main()
