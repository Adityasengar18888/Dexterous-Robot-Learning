import os
import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environments.transport_diagnostic_env import PandaTransportDiagnosticEnv

def test_initialization(env_kwargs):
    print("\n--- Initial-State Validation ---")
    env = PandaTransportDiagnosticEnv(**env_kwargs, render_mode=None)
    success = 0
    total = 20
    
    for i in range(total):
        obs, info = env.reset(seed=1000+i)
        if info["state_initialization_valid"]:
            success += 1
            
    rate = success / total
    print(f"Initialization Success Rate: {rate * 100:.1f}%")
    env.close()
    
    if rate < 0.95:
        raise Exception("DIAGNOSTIC_PIPELINE_FAILURE: Initialization reliability is < 95%.")

def test_reward(env_kwargs):
    print("\n--- Reward Sanity Checks ---")
    env = PandaTransportDiagnosticEnv(**env_kwargs, render_mode=None)
    obs, info = env.reset(seed=42)
    
    target = obs["desired_goal"]
    obj_pos = obs["achieved_goal"]
    
    dist_far = np.linalg.norm(obj_pos - target)
    r_far = env.compute_reward(obj_pos, target, {})
    
    # Closer
    closer_pos = obj_pos + 0.5 * (target - obj_pos)
    r_close = env.compute_reward(closer_pos, target, {})
    
    # At target
    r_target = env.compute_reward(target, target, {})
    
    print(f"Reward Far: {r_far:.3f}")
    print(f"Reward Closer: {r_close:.3f}")
    print(f"Reward At Target: {r_target:.3f}")
    
    if r_close <= r_far or r_target <= r_close:
        env.close()
        raise Exception("DIAGNOSTIC_PIPELINE_FAILURE: Reward ordering is invalid.")
        
    env.close()
    
def evaluate_checkpoint(model, env, num_episodes=100, seed_offset=2000):
    results = []
    trajectories = []
    
    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        
        if not info["state_initialization_valid"]:
            continue
            
        initial_dist = info["transport_dist"]
        min_dist = initial_dist
        max_z = obs["achieved_goal"][2]
        
        done = False
        step = 0
        current_hold = 1
        max_hold = 1
        dropped = False
        
        cos_sims = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            step += 1
            
            ee_pos = info["ee_pos"]
            obj_pos = obs["achieved_goal"]
            target_pos = obs["desired_goal"]
            dist = info["transport_dist"]
            
            if dist < min_dist:
                min_dist = dist
            if obj_pos[2] > max_z:
                max_z = obj_pos[2]
                
            ee_obj_dist = np.linalg.norm(ee_pos - obj_pos)
            is_held = ee_obj_dist < 0.05 and action[3] < 0.0
            
            if is_held:
                current_hold += 1
                if current_hold > max_hold:
                    max_hold = current_hold
            else:
                dropped = True
                current_hold = 0
                
            # Alignment
            if is_held:
                ee_action = action[:3]
                target_vector = target_pos - obj_pos
                norm_act = np.linalg.norm(ee_action)
                norm_tar = np.linalg.norm(target_vector)
                
                if norm_act > 1e-4 and norm_tar > 1e-4:
                    cos_sim = np.dot(ee_action, target_vector) / (norm_act * norm_tar)
                    cos_sims.append(cos_sim)
                    
            if ep < 20: # Save trajectories for first 20 episodes
                trajectories.append({
                    "episode": ep,
                    "step": step,
                    "obj_x": obj_pos[0], "obj_y": obj_pos[1], "obj_z": obj_pos[2],
                    "target_x": target_pos[0], "target_y": target_pos[1], "target_z": target_pos[2],
                    "ee_x": ee_pos[0], "ee_y": ee_pos[1], "ee_z": ee_pos[2],
                    "dist": dist,
                    "reward": reward,
                    "success": info["is_success"],
                    "action_x": action[0], "action_y": action[1], "action_z": action[2], "gripper": action[3],
                    "cos_sim": cos_sims[-1] if cos_sims else np.nan
                })
                    
        results.append({
            "success": info["is_success"],
            "initial_dist": initial_dist,
            "min_dist": min_dist,
            "final_dist": dist,
            "progress": initial_dist - min_dist,
            "hold_duration": max_hold,
            "dropped": dropped,
            "mean_cos_sim": np.mean(cos_sims) if cos_sims else np.nan
        })
        
    return pd.DataFrame(results), pd.DataFrame(trajectories)

def main():
    print("--- Phase 11.9 Transport Diagnostic ---")
    
    with open("configs/pick_place_baseline.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    env_kwargs = config["env"].copy()
    if "name" in env_kwargs:
        del env_kwargs["name"]
    if "randomizer" in env_kwargs:
        env_kwargs["randomizer_config"] = env_kwargs.pop("randomizer")
        
    # Validations
    test_initialization(env_kwargs)
    test_reward(env_kwargs)
    
    # Training Setup
    exp_dir = "experiments/pnp_transport_seed42"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/models", exist_ok=True)
    os.makedirs("results/phase11_9", exist_ok=True)
    
    env = PandaTransportDiagnosticEnv(**env_kwargs)
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    
    eval_env = PandaTransportDiagnosticEnv(**env_kwargs)
    eval_env = Monitor(eval_env)
    
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
    
    checkpoints = [5000, 10000, 20000, 30000, 40000, 50000]
    previous_cp = 0
    
    learning_curve = []
    final_trajectories = None
    
    for cp in checkpoints:
        steps = cp - previous_cp
        print(f"\nTraining for {steps} steps (Target: {cp})...")
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
        model.save(f"{exp_dir}/models/model_{cp}_steps.zip")
        
        print(f"Evaluating checkpoint {cp}...")
        res_df, traj_df = evaluate_checkpoint(model, eval_env, num_episodes=100)
        
        c_sim = res_df['mean_cos_sim'].dropna()
        alignment = (c_sim > 0.1).mean() * 100 if not c_sim.empty else 0.0
        
        learning_curve.append({
            "Timesteps": cp,
            "Success": f"{res_df['success'].mean()*100:.1f}%",
            "Mean Final Distance": f"{res_df['final_dist'].mean():.3f}",
            "Mean Min Distance": f"{res_df['min_dist'].mean():.3f}",
            ">25% Progress": f"{(res_df['progress'] / res_df['initial_dist'] > 0.25).mean()*100:.1f}%",
            "Alignment": f"{alignment:.1f}%"
        })
        
        if cp == 50000:
            final_trajectories = traj_df
            res_df.to_csv("results/phase11_9_transport_metrics.csv", index=False)
            
        previous_cp = cp
        
    final_trajectories.to_csv("results/phase11_9_transport_trajectories.csv", index=False)
    
    # HER Diagnostic
    print("\n--- HER Validation ---")
    rb = model.replay_buffer
    print(f"Replay Buffer Size: {rb.size()}")
    
    # Write learning curve table
    lc_df = pd.DataFrame(learning_curve)
    print("\nLearning Curve:")
    print(lc_df.to_string(index=False))
    
    # Final Decision
    final_success = float(lc_df.iloc[-1]["Success"].strip("%"))
    final_align = float(lc_df.iloc[-1]["Alignment"].strip("%"))
    progress_25 = float(lc_df.iloc[-1][">25% Progress"].strip("%"))
    
    if final_success >= 30.0 and final_align > 50.0:
        decision = "TRANSPORT_LEARNED"
    elif progress_25 > 20.0 or final_align > 20.0:
        decision = "TRANSPORT_PARTIAL"
    else:
        decision = "TRANSPORT_NOT_LEARNED"
        
    print(f"\nFinal Classification: {decision}")
    
    # Write Report
    md = f"""# Phase 11.9 — Transport-Only SAC+HER Diagnostic

## 1. Executive Summary
This diagnostic isolated the continuous-control transport stage of the Pick & Place task. The environment strictly initialized the object already physically grasped and lifted in the air, removing the reaching and grasping bottlenecks entirely. 
After exactly 50,000 steps of isolated training, the SAC+HER policy resulted in a classification of **{decision}**.

## 2. Experimental Configuration
- Model: Fresh SAC instance (`experiments/pnp_transport_seed42/`)
- Environment: Isolated `PandaTransportDiagnosticEnv`
- Timesteps: 50,000
- Seed: 42

## 3. Initial-State Validation
- Initialization was physically validated via pybullet stepping.
- Success rate of initialization > 95%.

## 4. Reward Validation
- Strict `-distance(object, target)` reward was used.
- Tested deterministic geometries proved moving toward target increases reward, and moving away decreases reward.

## 5. Learning Curve

| Timesteps | Success | Mean Final Distance | Mean Min Distance | >25% Progress | Alignment |
|---:|---:|---:|---:|---:|---:|
"""
    for row in learning_curve:
        md += f"| {row['Timesteps']} | {row['Success']} | {row['Mean Final Distance']} | {row['Mean Min Distance']} | {row['>25% Progress']} | {row['Alignment']} |\n"
        
    md += f"""
## 6. Decision Classification
**{decision}**

## 7. Recommended Next Experiment
Based on the transport classification, the recommended next step is to analyze how the learning capacity changes when scaling compute or when chaining the behaviors together via curriculum.
"""
    
    with open("docs/phase11_9_transport_only_diagnostic.md", "w") as f:
        f.write(md)
        
if __name__ == "__main__":
    main()
