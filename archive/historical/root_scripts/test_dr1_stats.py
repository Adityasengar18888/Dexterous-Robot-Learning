import os
import json
import numpy as np
import pandas as pd
from scipy import stats

def compute_dr1_4seed_stats():
    # Seeds 42 and 43 evaluated at 1M (and 500k)
    # Seeds 44 and 45 evaluated at 500k (accepted by gate)
    
    # Let's inspect both at 500k and at final accepted step
    seeds = [42, 43, 44, 45]
    records_500k = []
    records_final = []
    
    for s in seeds:
        csv_path = f"experiments/phase13_dense_dr/dr1/seed{s}/metrics/checkpoint_metrics.csv"
        df = pd.read_csv(csv_path)
        
        # 500k
        df_500 = df[df["checkpoint"] == 500000].copy()
        df_500["seed"] = s
        records_500k.append(df_500)
        
        # Final accepted (1M for 42,43; 500k for 44,45)
        max_cp = df["checkpoint"].max()
        df_fin = df[df["checkpoint"] == max_cp].copy()
        df_fin["seed"] = s
        df_fin["final_budget"] = max_cp
        records_final.append(df_fin)
        
    df_500_all = pd.concat(records_500k, ignore_index=True)
    df_fin_all = pd.concat(records_final, ignore_index=True)
    
    print("=== DR1 4-SEED BENCHMARK (AT 500K CHECKPOINT) ===")
    for reg in ["Nominal_DR0", "Matched_DR1", "Heldout_Stress"]:
        sub = df_500_all[df_500_all["regime"] == reg]
        succs = sub["success_rate"].values * 100
        mean = np.mean(succs)
        std = np.std(succs, ddof=1)
        print(f"[{reg}] Mean: {mean:.2f}% | Std: {std:.2f}% | Seeds: {succs}")
        
    print("\n=== DR1 4-SEED BENCHMARK (FAST-TRACK ACCEPTED BUDGET: 42/43 @ 1M, 44/45 @ 500k) ===")
    for reg in ["Nominal_DR0", "Matched_DR1", "Heldout_Stress"]:
        sub = df_fin_all[df_fin_all["regime"] == reg]
        succs = sub["success_rate"].values * 100
        mean = np.mean(succs)
        std = np.std(succs, ddof=1)
        ci = stats.t.ppf(0.975, df=len(succs)-1) * (std / np.sqrt(len(succs))) if std > 1e-6 else 0.0
        print(f"[{reg}] Mean: {mean:.2f}% ± {ci:.2f}% (Std: {std:.2f}%) | Seeds: {succs}")
        
if __name__ == "__main__":
    compute_dr1_4seed_stats()

