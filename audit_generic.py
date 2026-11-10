import os
import sys
import json
import glob
import yaml
import pandas as pd
from stable_baselines3 import SAC
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from src.rl.evaluate import make_env

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_eval_raw(model, env, num_episodes, seed_start, model_id, training_seed, condition):
    rows = []
    for ep in range(num_episodes):
        eval_seed = seed_start + ep
        obs, info = env.reset(seed=eval_seed)
        done = False
        ep_rew = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, term, trunc, info = env.step(action)
            ep_rew += rew
            done = term or trunc
            
        success = info.get("is_success", False)
        grasp = info.get("grasp_success", False)
        dist = info.get("distance", 0.0)
        length = info.get("episode_length", 100)
        colls = info.get("collisions", 0)
        
        failure_cat = "unknown"
        if success:
            failure_cat = "none"
        elif not grasp:
            failure_cat = "failed_grasp"
        elif dist > 0.05:
            failure_cat = "failed_place"
        
        rows.append({
            "model_id": model_id,
            "training_seed": training_seed,
            "evaluation_condition": condition,
            "evaluation_seed": eval_seed,
            "success": success,
            "final_distance": dist,
            "grasp_success": grasp,
            "episode_length": length,
            "reward": ep_rew,
            "collision_count": colls,
            "timeout": length >= 100 and not success,
            "failure_category": failure_cat
        })
    return rows

def main():
    if len(sys.argv) < 2:
        print("Usage: python audit_generic.py <dr_prefix>")
        sys.exit(1)
        
    exp_prefix = sys.argv[1] # e.g., pnp_dr2
    dr_num = exp_prefix.split('_')[1] # e.g., dr2
    seeds = [42, 43, 44, 45, 46]
    
    nominal_cfg = load_config("configs/pick_place_baseline.yaml")["env"]
    heldout_cfg = load_config("configs/pick_place_heldout.yaml")["env"]
    
    raw_episodes = []
    
    for seed in seeds:
        exp_id = f"{exp_prefix}_seed{seed}"
        exp_dir = f"experiments/{exp_id}"
        
        # 1. VERIFY
        metadata_path = f"{exp_dir}/metadata.json"
        config_path = f"{exp_dir}/config.yaml"
        model_path = f"{exp_dir}/models/final_model.zip"
        
        actual_timesteps = 0
        status = "UNKNOWN"
        if os.path.exists(metadata_path) and os.path.exists(model_path):
            with open(metadata_path) as f:
                meta = json.load(f)
            if meta.get("training_timesteps") == 100000:
                actual_timesteps = 100000
                status = "COMPLETE"
            else:
                status = "INCOMPLETE"
        elif not os.path.exists(model_path):
            status = "FAILED"
            
        if status != "COMPLETE":
            print(f"Error: {exp_id} did not complete successfully (Status: {status})")
            sys.exit(1)
            
        print(f"Auditing {exp_id}...")
        
        # 2. CHECKPOINT INTEGRITY & SAC CONFIG
        env_nom = make_env(nominal_cfg, render_mode=None)
        try:
            model = SAC.load(model_path, env=env_nom)
        except Exception as e:
            print(f"Error loading model for {exp_id}: {e}")
            sys.exit(1)
            
        if seed == 42:
            print("Running Reproducibility Test on Seed 42...")
            res1 = run_eval_raw(model, env_nom, 10, 1001, exp_id, seed, "nominal_rep1")
            res2 = run_eval_raw(model, env_nom, 10, 1001, exp_id, seed, "nominal_rep2")
            for r1, r2 in zip(res1, res2):
                assert r1["reward"] == r2["reward"], "Reproducibility failed!"
                assert r1["final_distance"] == r2["final_distance"], "Reproducibility failed!"
            print("Reproducibility Test PASS")

        # EVALUATE NOMINAL
        print("Evaluating Nominal...")
        raw_nom = run_eval_raw(model, env_nom, 100, 1001, exp_id, seed, "Nominal")
        raw_episodes.extend(raw_nom)
        env_nom.close()
        
        # EVALUATE HELDOUT
        print("Evaluating Heldout...")
        env_held = make_env(heldout_cfg, render_mode=None)
        raw_held = run_eval_raw(model, env_held, 100, 1001, exp_id, seed, "Held-out")
        raw_episodes.extend(raw_held)
        env_held.close()

    if raw_episodes:
        df = pd.DataFrame(raw_episodes)
        os.makedirs("results", exist_ok=True)
        df.to_csv(f"results/phase11_{dr_num}_episode_metrics.csv", index=False)
        
        summary = df.groupby(["model_id", "training_seed", "evaluation_condition"]).agg(
            episodes=("success", "count"),
            success_rate=("success", "mean"),
            mean_reward=("reward", "mean"),
            median_reward=("reward", "median"),
            mean_final_distance=("final_distance", "mean"),
            median_final_distance=("final_distance", "median"),
            grasp_success_rate=("grasp_success", "mean"),
            mean_episode_length=("episode_length", "mean"),
            collision_rate=("collision_count", "mean"),
            timeout_rate=("timeout", "mean")
        ).reset_index()
        
        summary.to_csv(f"results/phase11_{dr_num}_summary.csv", index=False)
        
        seed_summary = summary.groupby(["training_seed", "evaluation_condition"]).mean(numeric_only=True).reset_index()
        seed_summary.to_csv(f"results/phase11_{dr_num}_seed_summary.csv", index=False)
        
        print(f"Audit Complete. Saved to results/phase11_{dr_num}_*")
        
        with open(f"docs/phase11_batch_{dr_num}.md", "w") as f:
            f.write(f"# Phase 11 — {dr_num.upper()} Audit\n\nAudit successful for all 5 seeds.\nNominal Mean Success: {seed_summary[seed_summary['evaluation_condition'] == 'Nominal']['success_rate'].mean()}\nHeld-out Mean Success: {seed_summary[seed_summary['evaluation_condition'] == 'Held-out']['success_rate'].mean()}\n\n")

if __name__ == "__main__":
    main()
