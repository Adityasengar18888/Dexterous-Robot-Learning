import os
import yaml
import pandas as pd
from stable_baselines3 import SAC
from src.rl.evaluate import make_env
from audit_generic import run_eval_raw

def main():
    with open("configs/pick_place_baseline.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    env_kwargs = config["env"]
    seed = 42
    exp_id = "pnp_dr0_dense_seed42"
    exp_dir = f"experiments/{exp_id}"
    
    # Evaluation using ORIGINAL SUCCESS DEFINITION (original env)
    eval_env = make_env(env_kwargs, render_mode=None)
    
    eval_model = SAC.load(f"{exp_dir}/models/final_model.zip", env=eval_env)
    print("Evaluating learned policy on nominal configuration...")
    
    raw_nom = run_eval_raw(eval_model, eval_env, 100, 1001, exp_id, seed, "Nominal")
    df = pd.DataFrame(raw_nom)
    df.to_csv(f"results/phase11_6_dense_reward_metrics.csv", index=False)
    
    summary = df.groupby(["model_id", "evaluation_condition"]).agg(
        episodes=("success", "count"),
        success_rate=("success", "mean"),
        grasp_success_rate=("grasp_success", "mean"),
        mean_final_distance=("final_distance", "mean"),
        mean_episode_length=("episode_length", "mean"),
        timeout_rate=("timeout", "mean")
    ).reset_index()
    
    print("\n--- Evaluation Summary ---")
    print(summary.to_string(index=False))
    
    eval_env.close()

if __name__ == "__main__":
    main()
