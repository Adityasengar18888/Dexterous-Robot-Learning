"""Phase 3 Runner: Complete Frozen-Policy Generalization & Robustness Suite."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

import numpy as np

from scripts.eval_phase15_phase3_robustness import (
    eval_pickplace_dr_frozen,
    eval_single_pregrasp_condition,
)


def main() -> None:
    os.makedirs("results/phase15_phase3", exist_ok=True)
    t0 = time.time()

    print("=== 1. PickPlace DR4 Frozen Policies (E_ADAPTIVE_TRANSPORT: Seed 42 & Seed 43) ===")
    pp_res = eval_pickplace_dr_frozen(n_per_seed=20)

    print("=== 2. SinglePreGrasp Frozen Checkpoints (5B2_17.5k SAC+HER vs 5B7_BC_Warmstart) ===")
    seed_batches = {
        "batch_A_seeds_4001_4020": list(range(4001, 4021)),
        "batch_B_seeds_5001_5020": list(range(5001, 5021)),
    }
    sp_models = [
        ("5B2_17500_SAC_HER", "experiments/multi_sort_phase15_5B2/seed42/models/checkpoint_17500_steps.zip"),
        ("5B7_BC_WARMSTART", "experiments/multi_sort_phase15_5B7/seed42/models/bc_warmstart_model.zip"),
    ]
    sp_conditions = [
        ("Nominal", dict(randomize_pos=True, randomize_ori=False, randomize_mass_fric=False, obs_noise_m=0.0, obs_delay_steps=0, target_z_span=None)),
        ("DR1_Pos_Ori", dict(randomize_pos=True, randomize_ori=True, randomize_mass_fric=False, obs_noise_m=0.0, obs_delay_steps=0, target_z_span=None)),
        ("DR2_TargetZ_Ori", dict(randomize_pos=True, randomize_ori=True, randomize_mass_fric=False, obs_noise_m=0.0, obs_delay_steps=0, target_z_span=(0.11, 0.17))),
        ("DR4_Phys_Noise_Delay", dict(randomize_pos=True, randomize_ori=True, randomize_mass_fric=True, obs_noise_m=0.003, obs_delay_steps=1, target_z_span=(0.11, 0.17))),
    ]

    sp_rows: List[Dict[str, Any]] = []
    for m_label, m_path in sp_models:
        for c_name, c_kw in sp_conditions:
            batch_metrics = []
            for b_name, b_seeds in seed_batches.items():
                r = eval_single_pregrasp_condition(m_path, f"{c_name}_{b_name}", b_seeds, **c_kw)
                batch_metrics.append(r)
            row = {
                "model": m_label,
                "condition": c_name,
                "n_episodes_total": sum(x["n_episodes"] for x in batch_metrics),
                "mean_grasp_rate": round(float(np.mean([x["grasp_rate"] for x in batch_metrics])), 4),
                "std_grasp_rate": round(float(np.std([x["grasp_rate"] for x in batch_metrics])), 4),
                "mean_lift_rate": round(float(np.mean([x["lift_rate"] for x in batch_metrics])), 4),
                "std_lift_rate": round(float(np.std([x["lift_rate"] for x in batch_metrics])), 4),
                "mean_hold_rate": round(float(np.mean([x["hold_success_rate"] for x in batch_metrics])), 4),
                "mean_table_coll_ep_rate": round(float(np.mean([x["table_coll_ep_rate"] for x in batch_metrics])), 4),
                "batches": batch_metrics,
            }
            sp_rows.append(row)
            print(
                f"[SinglePreGrasp {m_label:>18s} | {c_name:>20s}]"
                f" Grasp={row['mean_grasp_rate']*100:5.1f}±{row['std_grasp_rate']*100:4.1f}%"
                f" | Lift={row['mean_lift_rate']*100:5.1f}±{row['std_lift_rate']*100:4.1f}%"
                f" | Hold={row['mean_hold_rate']*100:5.1f}%"
            )

    payload = {
        "total_elapsed_s": round(time.time() - t0, 2),
        "pickplace_dr_results": pp_res["pickplace_dr_results"],
        "single_pregrasp_robustness": sp_rows,
    }
    with open("results/phase15_phase3/phase15_phase3_robustness_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved Phase 3 results in {payload['total_elapsed_s']}s.")


if __name__ == "__main__":
    main()

