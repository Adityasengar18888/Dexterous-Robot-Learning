import os
import yaml
import json
import pandas as pd
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environments.dense_pick_place_env import PandaPickPlaceDenseRewardEnv
from audit_generic import run_eval_raw

def main():
    # 1. Setup config (DR-0)
    with open("configs/pick_place_baseline.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    env_kwargs = config["env"].copy()
    if "name" in env_kwargs:
        del env_kwargs["name"]
    if "randomizer" in env_kwargs:
        env_kwargs["randomizer_config"] = env_kwargs.pop("randomizer")
    seed = 42
    total_timesteps = 100000
    
    exp_id = "pnp_dr0_dense_seed42"
    exp_dir = f"experiments/{exp_id}"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/models", exist_ok=True)
    
    # 2. Instantiate env
    env = PandaPickPlaceDenseRewardEnv(**env_kwargs)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    policy_kwargs = config["rl"].get("policy_kwargs", dict(net_arch=[256, 256, 256]))
    
    model = SAC(
        "MultiInputPolicy",
        env,
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
        seed=seed
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=f"{exp_dir}/models",
        name_prefix="model",
    )
    
    print(f"Starting Dense Reward Diagnostic Training ({total_timesteps} steps)...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    model.save(f"{exp_dir}/models/final_model.zip")
    print("Training finished.")
    
    # 4. Evaluation using ORIGINAL SUCCESS DEFINITION
    # The evaluation env should ideally be the original PandaPickPlaceEnv to ensure 
    # we aren't evaluating using dense logic, but dense logic doesn't change `info["is_success"]`.
    # However, to be absolutely safe, we will evaluate in the original env!
    from src.rl.evaluate import make_env
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
