import os
import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from src.rl.evaluate import make_env

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_eval(model, env, num_episodes=100, seed_start=1001):
    successes = []
    distances = []
    lengths = []
    grasps = []
    collisions = []
    rewards = []
    
    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed_start + ep)
        done = False
        ep_rew = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, term, trunc, info = env.step(action)
            ep_rew += rew
            done = term or trunc
            
        successes.append(info.get("is_success", False))
        distances.append(info.get("distance", 0.0))
        lengths.append(info.get("episode_length", 100))
        grasps.append(info.get("grasp_success", False))
        collisions.append(info.get("collisions", 0))
        rewards.append(ep_rew)
        
    return {
        "success_rate": np.mean(successes),
        "mean_reward": np.mean(rewards),
        "mean_final_distance": np.mean(distances),
        "grasp_success": np.mean(grasps),
        "collision_rate": np.mean(collisions),
        "mean_episode_length": np.mean(lengths)
    }

def main():
    experiments = [
        ("pnp_dr0", "configs/pick_place_baseline.yaml"),
        ("pnp_dr1", "configs/pick_place_dr_position.yaml"),
        ("pnp_dr2", "configs/pick_place_dr_position_target.yaml"),
        ("pnp_dr3", "configs/pick_place_dr_pose.yaml"),
        ("pnp_dr4", "configs/pick_place_dr_physics.yaml"),
    ]
    
    seeds = [42, 43, 44, 45, 46]
    
    heldout_cfg = load_config("configs/pick_place_heldout.yaml")["env"]
    nominal_cfg = load_config("configs/pick_place_baseline.yaml")["env"]
    
    results = []
    
    for exp_prefix, id_config_path in experiments:
        id_cfg = load_config(id_config_path)["env"]
        
        for seed in seeds:
            exp_id = f"{exp_prefix}_seed{seed}"
            model_path = f"experiments/{exp_id}/models/final_model.zip"
            
            if not os.path.exists(model_path):
                print(f"Skipping {exp_id}, model not found.")
                continue
                
            print(f"Evaluating {exp_id}...")
            
            # 1. Nominal
            env_nom = make_env(nominal_cfg, render_mode=None)
            model = SAC.load(model_path, env=env_nom)
            res_nom = run_eval(model, env_nom)
            env_nom.close()
            
            # 2. In-Distribution
            env_id = make_env(id_cfg, render_mode=None)
            model = SAC.load(model_path, env=env_id)
            res_id = run_eval(model, env_id)
            env_id.close()
            
            # 3. Heldout
            env_held = make_env(heldout_cfg, render_mode=None)
            model = SAC.load(model_path, env=env_held)
            res_held = run_eval(model, env_held)
            env_held.close()
            
            results.append({
                "experiment_id": exp_id,
                "randomization_level": exp_prefix,
                "seed": seed,
                "training_timesteps": 100000,
                "nominal_success": res_nom["success_rate"],
                "randomized_success": res_id["success_rate"],
                "heldout_success": res_held["success_rate"],
                "mean_reward": res_held["mean_reward"], # taking heldout as primary stats
                "mean_final_distance": res_held["mean_final_distance"],
                "grasp_success": res_held["grasp_success"],
                "collision_rate": res_held["collision_rate"],
                "mean_episode_length": res_held["mean_episode_length"],
            })
            
    if results:
        df = pd.DataFrame(results)
        os.makedirs("results", exist_ok=True)
        df.to_csv("results/phase11_summary.csv", index=False)
        print("Evaluation complete. Results saved to results/phase11_summary.csv")
        print(df)
    else:
        print("No models evaluated.")

if __name__ == "__main__":
    main()
