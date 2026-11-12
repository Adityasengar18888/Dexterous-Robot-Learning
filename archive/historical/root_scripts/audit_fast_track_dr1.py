import os
import json
import numpy as np
import pandas as pd

def analyze_seed_eval(seed, n_eps=30):
    eval_csv = f"experiments/phase13_dense_dr/dr1/seed{seed}/evaluations/eval_checkpoint_500000.csv"
    if not os.path.exists(eval_csv):
        print(f"File not found: {eval_csv}")
        return {}
    df = pd.read_csv(eval_csv)
    
    results = {}
    regimes = ["Nominal_DR0", "Matched_DR1", "Heldout_Stress"]
    for reg in regimes:
        sub = df[df["regime"] == reg].head(n_eps)
        tax = sub["failure_type"].value_counts().to_dict()
        res = {
            "n_episodes": len(sub),
            "reach_rate": float(sub["reach"].mean()),
            "grasp_rate": float(sub["grasped"].mean()),
            "lift_rate": float(sub["lifted"].mean()),
            "transport_rate": float(sub["transport"].mean()),
            "placement_rate": float(sub["placed"].mean()),
            "success_rate": float(sub["success"].mean()),
            "mean_final_dist": float(sub["final_dist"].mean()),
            "mean_hold_duration": float(sub["hold_duration"].mean()),
            "drop_rate": float(sub["dropped"].mean()),
            "timeout_rate": float(sub["timeout"].mean()),
            "mean_reward": float(sub["reward"].mean()),
            "failure_taxonomy": tax
        }
        results[reg] = res
    return results

def main():
    print("=================================================================")
    print("PHASE 13 DR-1 AUDIT: SEED 44 & 45 EVALUATION (30-EPISODE SCREENING)")
    print("=================================================================")
    
    screening_results = {}
    for seed in [44, 45]:
        print(f"\n--- Seed {seed} (30 episodes screening) ---")
        res = analyze_seed_eval(seed, n_eps=30)
        screening_results[f"seed_{seed}"] = res
        for reg, m in res.items():
            print(f"  [{reg}]")
            print(f"    Success: {m['success_rate']*100:.1f}% | Reach: {m['reach_rate']*100:.1f}% | Grasp: {m['grasp_rate']*100:.1f}% | Lift: {m['lift_rate']*100:.1f}%")
            print(f"    Transport: {m['transport_rate']*100:.1f}% | Placement: {m['placement_rate']*100:.1f}%")
            print(f"    Mean Final Dist: {m['mean_final_dist']:.4f}m | Mean Hold: {m['mean_hold_duration']:.1f} steps")
            print(f"    Drop Rate: {m['drop_rate']*100:.1f}% | Timeout Rate: {m['timeout_rate']*100:.1f}% | Mean Reward: {m['mean_reward']:.1f}")
            print(f"    Failure Taxonomy: {m['failure_taxonomy']}")

    print("\n=================================================================")
    print("PHASE 13 DR-1 AUDIT: SEED 44 & 45 EVALUATION (100-EPISODE FULL)")
    print("=================================================================")
    full_results = {}
    for seed in [44, 45]:
        print(f"\n--- Seed {seed} (100 episodes full) ---")
        res = analyze_seed_eval(seed, n_eps=100)
        full_results[f"seed_{seed}"] = res
        for reg, m in res.items():
            print(f"  [{reg}]")
            print(f"    Success: {m['success_rate']*100:.1f}% | Reach: {m['reach_rate']*100:.1f}% | Grasp: {m['grasp_rate']*100:.1f}% | Lift: {m['lift_rate']*100:.1f}%")
            print(f"    Transport: {m['transport_rate']*100:.1f}% | Placement: {m['placement_rate']*100:.1f}%")
            print(f"    Mean Final Dist: {m['mean_final_dist']:.4f}m | Mean Hold: {m['mean_hold_duration']:.1f} steps")
            print(f"    Drop Rate: {m['drop_rate']*100:.1f}% | Timeout Rate: {m['timeout_rate']*100:.1f}% | Mean Reward: {m['mean_reward']:.1f}")
            print(f"    Failure Taxonomy: {m['failure_taxonomy']}")

    # Save to json
    os.makedirs("results/phase13_fast_track", exist_ok=True)
    with open("results/phase13_fast_track/dr1_seed44_45_screening.json", "w") as f:
        json.dump({"screening_30ep": screening_results, "full_100ep": full_results}, f, indent=4)
        
    # Check decision rule
    s44_matched = screening_results["seed_44"]["Matched_DR1"]["success_rate"]
    s45_matched = screening_results["seed_45"]["Matched_DR1"]["success_rate"]
    print("\n" + "="*60)
    print("DECISION RULE EVALUATION (GATE: >= 90% Matched DR1 Success)")
    print(f"  Seed 44 Matched DR1 Success: {s44_matched*100:.1f}% -> {'PASS' if s44_matched >= 0.90 else 'FAIL'}")
    print(f"  Seed 45 Matched DR1 Success: {s45_matched*100:.1f}% -> {'PASS' if s45_matched >= 0.90 else 'FAIL'}")
    if s44_matched >= 0.90 and s45_matched >= 0.90:
        print(">> GATE STATUS: PASSED. Accept 500k results for fast-track benchmark.")
        print(">> Extension to 750k / 1M: NOT REQUIRED.")
    print("="*60)

if __name__ == "__main__":
    main()

