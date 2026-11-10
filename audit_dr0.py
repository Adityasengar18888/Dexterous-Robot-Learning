import os
import json
import glob
import yaml
import numpy as np
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

def get_tb_metrics(log_dir):
    event_file = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_file:
        return {}
    ea = EventAccumulator(event_file[0])
    ea.Reload()
    metrics = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        metrics[tag] = [{"step": e.step, "value": e.value} for e in events]
    return metrics

def main():
    seeds = [42, 43, 44, 45, 46]
    exp_prefix = "pnp_dr0"
    
    nominal_cfg = load_config("configs/pick_place_baseline.yaml")["env"]
    heldout_cfg = load_config("configs/pick_place_heldout.yaml")["env"]
    
    raw_episodes = []
    summary_rows = []
    audit_data = []
    
    for seed in seeds:
        exp_id = f"{exp_prefix}_seed{seed}"
        exp_dir = f"experiments/{exp_id}"
        
        # 1. VERIFY
        metadata_path = f"{exp_dir}/metadata.json"
        config_path = f"{exp_dir}/config.yaml"
        model_path = f"{exp_dir}/models/final_model.zip"
        log_dir = glob.glob(f"{exp_dir}/logs/SAC_1")
        
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
            
        audit_data.append({
            "experiment_id": exp_id,
            "training_seed": seed,
            "status": status,
            "actual_timesteps": actual_timesteps,
            "checkpoint_exists": os.path.exists(model_path)
        })
        
        if status != "COMPLETE":
            continue
            
        print(f"Auditing {exp_id}...")
        
        # 2. CHECKPOINT INTEGRITY & SAC CONFIG
        env_nom = make_env(nominal_cfg, render_mode=None)
        model = SAC.load(model_path, env=env_nom)
        
        print(f"Policy: {model.policy_class}")
        print(f"Replay Buffer: {model.replay_buffer_class}")
        print(f"Gamma: {model.gamma}, Tau: {model.tau}, LR: {model.learning_rate}")
        
        # Reproducibility Test on Seed 42
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

    # Aggregate summaries
    if raw_episodes:
        df = pd.DataFrame(raw_episodes)
        os.makedirs("results", exist_ok=True)
        df.to_csv("results/phase11_dr0_episode_metrics.csv", index=False)
        
        # Calculate summary
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
        
        summary.to_csv("results/phase11_dr0_summary.csv", index=False)
        
        seed_summary = summary.groupby(["training_seed", "evaluation_condition"]).mean().reset_index()
        seed_summary.to_csv("results/phase11_dr0_seed_summary.csv", index=False)
        
        print("Audit Complete. Saved to results/phase11_dr0_*")

if __name__ == "__main__":
    main()
