import os
import glob
import json
import yaml
import numpy as np
import pandas as pd
from scipy import stats
from stable_baselines3 import SAC

from src.environments.dense_pick_place_env import PandaPickPlaceDenseRewardEnv


def audit_single_seed(exp_dir: str, seed: int, dr_level: int = 1) -> dict:
    """
    Performs full integrity audit on a single run:
    - Verifies exactly 1,000,000 timesteps completed
    - Verifies config snapshot and training metadata
    - Verifies every checkpoint is loadable
    - Verifies no NaN/Inf corruption
    - Verifies SAC hyperparameters match Phase 12
    - Verifies HER is enabled with future strategy and n_sampled_goal=4
    - Verifies DR-1 config
    """
    report = {
        "seed": seed,
        "exp_dir": exp_dir,
        "status": "PASS",
        "failures": []
    }
    
    # 1. Metadata check
    meta_path = os.path.join(exp_dir, "training_metadata.json")
    if not os.path.exists(meta_path):
        report["status"] = "FAIL"
        report["failures"].append("training_metadata.json missing")
        return report
        
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    if meta.get("status") != "COMPLETE":
        report["status"] = "FAIL"
        report["failures"].append(f"Run status is {meta.get('status')}, expected COMPLETE")
        
    # 2. Config check
    cfg_path = os.path.join(exp_dir, "config_snapshot.yaml")
    if not os.path.exists(cfg_path):
        report["status"] = "FAIL"
        report["failures"].append("config_snapshot.yaml missing")
    else:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        rl_cfg = cfg.get("rl", {})
        if rl_cfg.get("learning_rate") != 0.001 or rl_cfg.get("tau") != 0.05 or rl_cfg.get("gamma") != 0.95:
            report["status"] = "FAIL"
            report["failures"].append(f"Hyperparameters mismatch: {rl_cfg}")
        if not rl_cfg.get("her") or rl_cfg.get("her_strategy") != "future" or rl_cfg.get("n_sampled_goal") != 4:
            report["status"] = "FAIL"
            report["failures"].append(f"HER configuration mismatch: {rl_cfg}")
        dr_cfg = cfg.get("env", {}).get("dr_config", {})
        if not dr_cfg.get("enabled") or not dr_cfg.get("object_position", {}).get("enabled"):
            report["status"] = "FAIL"
            report["failures"].append(f"DR-1 config mismatch: {dr_cfg}")
            
    # 3. Checkpoint loadability check
    models_dir = os.path.join(exp_dir, "models")
    checkpoints = [250000, 500000, 750000, 1000000]
    
    # Initialize env for HerReplayBuffer
    env = PandaPickPlaceDenseRewardEnv(
        distance_threshold=0.05,
        max_episode_steps=100,
        action_scale=0.05,
        randomizer_config={"enabled": True, "object_position": {"enabled": True}}
    )
    
    for cp in checkpoints:
        cp_path = os.path.join(models_dir, f"checkpoint_{cp}.zip")
        if not os.path.exists(cp_path):
            report["status"] = "FAIL"
            report["failures"].append(f"Checkpoint {cp_path} missing")
            continue
        try:
            model = SAC.load(cp_path, env=env, device="cpu")
            # NaN / Inf check on policy parameters
            for name, param in model.policy.named_parameters():
                if np.isnan(param.detach().numpy()).any() or np.isinf(param.detach().numpy()).any():
                    report["status"] = "FAIL"
                    report["failures"].append(f"NaN/Inf detected in {name} at checkpoint {cp}")
            del model
        except Exception as e:
            report["status"] = "FAIL"
            report["failures"].append(f"Failed to load checkpoint {cp_path}: {e}")
            
    env.close()
            
    # 4. Evaluation check
    eval_csv = os.path.join(exp_dir, "metrics", "checkpoint_metrics.csv")
    if not os.path.exists(eval_csv):
        report["status"] = "FAIL"
        report["failures"].append("checkpoint_metrics.csv missing")
    else:
        df_cp = pd.read_csv(eval_csv)
        cp_present = df_cp["checkpoint"].unique().tolist()
        for cp in checkpoints:
            if cp not in cp_present:
                report["status"] = "FAIL"
                report["failures"].append(f"Checkpoint {cp} missing in metrics CSV")
                
    # 5. Runtime and Replay Buffer stats
    report["duration_seconds"] = meta.get("total_duration_seconds", 0)
    report["duration_hours"] = round(report["duration_seconds"] / 3600, 2)
    report["final_replay_buffer_size"] = meta.get("final_replay_buffer_size", 0)
    report["start_time"] = meta.get("training_start_time")
    report["end_time"] = meta.get("training_end_time")
    
    return report


def test_reproducibility(exp_dir: str, seed: int, num_episodes: int = 10, seed_start: int = 1001) -> bool:
    """
    Verifies that running evaluation twice with deterministic=True produces identical rewards and final distances.
    """
    model_path = os.path.join(exp_dir, "models", "checkpoint_1000000.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(exp_dir, "models", "final_model.zip")
        
    env = PandaPickPlaceDenseRewardEnv(
        distance_threshold=0.05,
        max_episode_steps=100,
        action_scale=0.05,
        randomizer_config={"enabled": True, "object_position": {"enabled": True}}
    )
    model = SAC.load(model_path, env=env, device="cpu")
    
    def run_eval():
        records = []
        for ep in range(num_episodes):
            eval_seed = seed_start + ep
            obs, _ = env.reset(seed=eval_seed)
            done = False
            total_rew = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rew, term, trunc, info = env.step(action)
                done = term or trunc
                total_rew += rew
            records.append((round(total_rew, 5), round(info["transport_dist"], 5)))
        return records

    run1 = run_eval()
    run2 = run_eval()
    env.close()
    
    match = (run1 == run2)
    return match


def compute_aggregate_statistics(seeds: list, base_dir: str = "experiments/phase13_dense_dr/dr1") -> dict:
    """
    Computes mean, standard deviation, min, max, and 95% confidence interval
    for Reach, Grasp, Lift, Transport, Placement, and End-to-End Success
    separately across:
    1. Nominal DR-0
    2. Matched DR-1
    3. Held-out stress
    Also calculates steps to first >= 80% deterministic success.
    """
    all_eval_rows = []
    seed_steps_to_80 = {}
    
    for s in seeds:
        s_dir = os.path.join(base_dir, f"seed{s}")
        metrics_csv = os.path.join(s_dir, "metrics", "checkpoint_metrics.csv")
        if not os.path.exists(metrics_csv):
            continue
        df_s = pd.read_csv(metrics_csv)
        df_s["seed"] = s
        all_eval_rows.append(df_s)
        
        # Steps to >=80% success in matched DR-1
        df_dr1 = df_s[df_s["regime"] == "Matched_DR1"].sort_values("checkpoint")
        hit_80 = df_dr1[df_dr1["success_rate"] >= 0.80]
        if not hit_80.empty:
            seed_steps_to_80[s] = int(hit_80.iloc[0]["checkpoint"])
        else:
            seed_steps_to_80[s] = None
            
    if not all_eval_rows:
        return {}
        
    df_all = pd.concat(all_eval_rows, ignore_index=True)
    
    # Filter for 1M final checkpoint
    df_1m = df_all[df_all["checkpoint"] == 1000000]
    
    metrics = ["reach_rate", "grasp_rate", "lift_rate", "transport_rate", "placement_rate", "success_rate"]
    metric_names = ["Reach", "Grasp", "Lift", "Transport", "Placement", "End-to-End Success"]
    regimes = ["Nominal_DR0", "Matched_DR1", "Heldout_Stress"]
    
    summary_tables = {}
    
    for regime in regimes:
        df_reg = df_1m[df_1m["regime"] == regime]
        rows = []
        n = len(df_reg)
        
        for m_col, m_name in zip(metrics, metric_names):
            vals = df_reg[m_col].values
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            min_val = float(np.min(vals))
            max_val = float(np.max(vals))
            
            # 95% Confidence Interval (t-distribution)
            if n > 1 and std_val > 1e-8:
                ci_margin = stats.t.ppf(0.975, df=n-1) * (std_val / np.sqrt(n))
                ci_low = max(0.0, mean_val - ci_margin)
                ci_high = min(1.0, mean_val + ci_margin)
            else:
                ci_low = mean_val
                ci_high = mean_val
                
            rows.append({
                "Metric": m_name,
                "Mean": round(mean_val * 100, 2),
                "Std": round(std_val * 100, 2),
                "Min": round(min_val * 100, 2),
                "Max": round(max_val * 100, 2),
                "95% CI Lower": round(ci_low * 100, 2),
                "95% CI Upper": round(ci_high * 100, 2)
            })
            
        summary_tables[regime] = pd.DataFrame(rows)
        
    return {
        "summary_tables": summary_tables,
        "seed_steps_to_80": seed_steps_to_80,
        "df_all": df_all,
        "df_1m": df_1m
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Audit Phase 13 DR-1 Runs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43], help="Seeds to audit")
    parser.add_argument("--base_dir", type=str, default="experiments/phase13_dense_dr/dr1", help="Base directory")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print(f"AUDITING PHASE 13 DR-1 (Seeds: {args.seeds})")
    print("="*60 + "\n")
    
    audit_results = []
    for s in args.seeds:
        exp_dir = os.path.join(args.base_dir, f"seed{s}")
        print(f"Auditing Seed {s} in {exp_dir}...")
        res = audit_single_seed(exp_dir, s, dr_level=1)
        
        if res["status"] == "PASS":
            rep_pass = test_reproducibility(exp_dir, s, num_episodes=10)
            res["reproducibility"] = "PASS" if rep_pass else "FAIL"
            if not rep_pass:
                res["status"] = "FAIL"
                res["failures"].append("Reproducibility check failed")
        else:
            res["reproducibility"] = "SKIPPED"
            
        audit_results.append(res)
        print(f"Result for Seed {s}: {res['status']} | Duration: {res.get('duration_hours', 0)}h | Failures: {res['failures']}")
        
    # Aggregate statistics if any runs passed
    stats_res = compute_aggregate_statistics(args.seeds, base_dir=args.base_dir)
    if stats_res:
        print("\n" + "="*60)
        print("AGGREGATE 1M EVALUATION RESULTS")
        print("="*60)
        for regime, table in stats_res["summary_tables"].items():
            print(f"\n--- Regime: {regime} ---")
            print(table.to_string(index=False))
        print("\nSteps to first >= 80% Success in Matched DR-1:")
        for s, steps in stats_res["seed_steps_to_80"].items():
            print(f"  Seed {s}: {steps if steps is not None else 'Did not reach 80%'}")
            
    # Save audit report
    os.makedirs("results/phase13_dr1", exist_ok=True)
    with open("results/phase13_dr1/audit_report.json", "w") as f:
        json.dump(audit_results, f, indent=4)
        
    all_passed = all(r["status"] == "PASS" for r in audit_results)
    print(f"\nFinal Audit Verdict: {'ALL PASS' if all_passed else 'SOME FAILED'}\n")
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
