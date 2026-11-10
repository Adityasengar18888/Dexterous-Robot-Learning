import os
import pandas as pd
from stable_baselines3 import SAC
from src.environments.dense_pick_place_env import PandaPickPlaceDenseRewardEnv

def verify_and_get_stats(seed):
    print(f"\n--- Seed {seed} ---")
    exp_dir = f"experiments/phase13_dense_dr/dr1/seed{seed}"
    cp_path = os.path.join(exp_dir, "models", "checkpoint_500000.zip")
    
    if not os.path.exists(cp_path):
        print("ERROR: Checkpoint missing:", cp_path)
        return
    print("Checkpoint exists:", cp_path)
    
    try:
        env = PandaPickPlaceDenseRewardEnv(
            distance_threshold=0.05, max_episode_steps=100, action_scale=0.05,
            randomizer_config={"enabled": True, "object_position": {"enabled": True}}
        )
        model = SAC.load(cp_path, env=env, device="cpu")
        print("Checkpoint successfully loadable.")
        del model
        env.close()
    except Exception as e:
        print("ERROR loading checkpoint:", e)
        return

    metrics_path = os.path.join(exp_dir, "metrics", "checkpoint_metrics.csv")
    if not os.path.exists(metrics_path):
        print("ERROR: Metrics missing.")
        return
    
    df = pd.read_csv(metrics_path)
    df_500k = df[df["checkpoint"] == 500000]
    
    for regime in ["Nominal_DR0", "Matched_DR1", "Heldout_Stress"]:
        row = df_500k[df_500k["regime"] == regime]
        if not row.empty:
            r = row.iloc[0]
            # Handle possible missing columns or JSON failure taxonomy
            taxonomy = r.get('failure_taxonomy', 'N/A')
            print(f"[{regime}] Success: {r['success_rate']*100:.1f}% | "
                  f"Reach: {r['reach_rate']*100:.1f}% | "
                  f"Grasp: {r['grasp_rate']*100:.1f}% | "
                  f"Lift: {r['lift_rate']*100:.1f}% | "
                  f"FinalDist: {r['mean_final_dist']:.3f}m | "
                  f"Failures: {taxonomy}")
        else:
            print(f"[{regime}] Missing evaluation data!")

verify_and_get_stats(44)
verify_and_get_stats(45)
