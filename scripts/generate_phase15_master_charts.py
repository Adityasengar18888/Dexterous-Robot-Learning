"""Generate verified Phase 15 master charts from experiment JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 180,
    }
)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_single_grasp_chart(out_dir: Path) -> Path:
    b2 = _load_json("results/phase15_2_5B2/phase15_2_5B2_training_results.json")["checkpoint_evaluations"]
    steps_b2 = [0 if x["checkpoint"] == "0_initial" else int(x["checkpoint"]) for x in b2]
    reach_b2 = [100 * x["reach_rate"] for x in b2]
    align_b2 = [100 * x["alignment_rate"] for x in b2]
    grasp_b2 = [100 * x["bilateral_grasp_rate"] for x in b2]
    lift_b2 = [100 * x["lift_rate"] for x in b2]

    runs = [
        ("5B3", "results/phase15_2_5B3/phase15_2_5B3_5k_diag.json", 0),
        ("5B4", "results/phase15_2_5B4/phase15_2_5B4_training_results.json", 1500),
        ("5B5", "results/phase15_2_5B5/phase15_2_5B5_training_results.json", 2600),
        ("5B6", "results/phase15_2_5B6/phase15_2_5B6_training_results.json", 2600),
        ("5B7", "results/phase15_2_5B7/phase15_2_5B7_training_results.json", 2600),
        ("5B8", "results/phase15_2_5B8/phase15_2_5B8_training_results.json", 4500),
    ]
    labels, bc_grasp, bc_lift, sac_peak_lift = [], [], [], []
    for name, p, act_start in runs:
        evals = _load_json(p)["checkpoint_evaluations"]
        bc_row = [r for r in evals if r["checkpoint"] == "bc_warmstart_only"][0]
        # Exclude checkpoints before actor updates begin so BC warm-start is never mislabeled as online SAC
        online_rows = [r for r in evals if str(r["checkpoint"]).isdigit() and int(r["checkpoint"]) >= act_start and int(r["checkpoint"]) > 0]
        labels.append(name)
        bc_grasp.append(100 * bc_row["bilateral_grasp_rate"])
        bc_lift.append(100 * bc_row["lift_rate"])
        sac_peak_lift.append(100 * max((r["lift_rate"] for r in online_rows), default=0.0))

    b3_100 = _load_json("results/phase15_2_5B3/phase15_2_5B3_100ep_results.json")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    # Panel A: 5B2 learning trajectory
    ax = axes[0]
    ax.plot(steps_b2, reach_b2, "o-", lw=2, label="Reach %", color="#2563eb")
    ax.plot(steps_b2, align_b2, "s--", lw=1.6, label="Align %", color="#0891b2")
    ax.plot(steps_b2, grasp_b2, "D-", lw=2.2, label="Bilateral Grasp %", color="#16a34a")
    ax.plot(steps_b2, lift_b2, "^-", lw=2.0, label="Lift %", color="#dc2626")
    ax.annotate("Peak Grasp\n75% @ 17.5k", xy=(17500, 75), xytext=(14200, 42),
                arrowprops=dict(arrowstyle="->", color="#16a34a"), fontsize=8, fontweight="bold", color="#16a34a")
    ax.set_title("A. 5B2 SAC+HER Learning Curve (0-20k)")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(-4, 108)
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", frameon=True)

    # Panel B: BC Warmstart vs Online SAC+HER Retention (5B3-5B8)
    ax = axes[1]
    x = np.arange(len(labels))
    w = 0.25
    b1 = ax.bar(x - w, bc_grasp, width=w, label="BC Warmstart Grasp%", color="#60a5fa")
    b2_bar = ax.bar(x, bc_lift, width=w, label="BC Warmstart Lift%", color="#1d4ed8")
    b3 = ax.bar(x + w, sac_peak_lift, width=w, label="Online SAC Peak Lift%", color="#f97316")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 108)
    ax.set_title("B. BC Warm-Start vs Online SAC+HER Retention")
    ax.set_xlabel("Phase 1 Ablation Run (See Report)")
    ax.set_ylabel("Rate (%)")
    ax.grid(axis="y", alpha=0.25)
    for bars in (b1, b2_bar, b3):
        for rect in bars:
            h = rect.get_height()
            if h > 0:
                ax.text(rect.get_x() + rect.get_width() / 2, h + 1.5, f"{h:.0f}", ha="center", va="bottom", fontsize=7.5)
    ax.legend(loc="upper right", frameon=True)

    # Panel C: 100-Episode Unseen Generalization (B.3)
    ax = axes[2]
    m_names = ["5B7 BC Warmstart\n(BEHAVIOR_CLONING)", "5B2 17.5k\n(SAC+HER)"]
    metrics = ["reach_rate", "bilateral_grasp_rate", "lift_rate", "hold_success_rate", "table_coll_ep_rate"]
    m_labels = ["Reach", "Bilateral Grasp", "Lift", "Hold Success", "Table Collision"]
    colors = ["#3b82f6", "#16a34a", "#f59e0b", "#8b5cf6", "#ef4444"]
    x_c = np.arange(len(m_labels))
    w_c = 0.36
    vals_bc = [100 * b3_100["bc_warmstart_5b7"][m] for m in metrics]
    vals_sac = [100 * b3_100["sac_her_5b2_17500"][m] for m in metrics]
    r1 = ax.bar(x_c - w_c / 2, vals_bc, width=w_c, label="5B7 BC (BEHAVIOR_CLONING)", color="#2563eb")
    r2 = ax.bar(x_c + w_c / 2, vals_sac, width=w_c, label="5B2 17.5k (LEARNED_SAC_HER)", color="#ea580c")
    for rect in list(r1) + list(r2):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 1.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x_c)
    ax.set_xticklabels(m_labels, rotation=18, ha="right")
    ax.set_ylim(0, 112)
    ax.set_title("C. B.3 100-Ep Unseen Seed Generalization")
    ax.set_ylabel("Episode Rate (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=True)

    fig.suptitle("Phase 1 Single-Object Grasp Acquisition, BC->SAC Stability, & 100-Ep Generalization", fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = out_dir / "phase15_single_grasp_learning_and_ablation.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_robustness_chart(out_dir: Path) -> Path:
    p3 = _load_json("results/phase15_phase3/phase15_phase3_robustness_results.json")
    pp = p3["pickplace_dr_results"]
    sp = p3["single_pregrasp_robustness"]

    conds_pp = ["DR1_POS", "DR2_POSE_TARGET", "DR4_FULL_PHYSICS", "DR4_OBS_NOISE_5MM", "DR4_OBS_DELAY_1"]
    pp_labels = ["DR1 Pos", "DR2 Pose+Tgt", "DR4 Physics", "DR4 Noise 5mm", "DR4 Delay 1-step"]
    s42_succ = [100 * [r for r in pp if r["model"] == "DR4_Seed42" and r["condition"] == c][0]["success_rate"] for c in conds_pp]
    s43_succ = [100 * [r for r in pp if r["model"] == "DR4_Seed43" and r["condition"] == c][0]["success_rate"] for c in conds_pp]
    s42_grasp = [100 * [r for r in pp if r["model"] == "DR4_Seed42" and r["condition"] == c][0]["grasp_rate"] for c in conds_pp]
    s43_grasp = [100 * [r for r in pp if r["model"] == "DR4_Seed43" and r["condition"] == c][0]["grasp_rate"] for c in conds_pp]

    conds_sp = ["Nominal", "DR1_Pos_Ori", "DR2_TargetZ_Ori", "DR4_Phys_Noise_Delay"]
    sp_labels = ["Nominal", "DR1 Pos+Ori", "DR2 TgtZ+Ori", "DR4 Phys+Noise+Delay"]

    def get_sp(model: str, metric: str):
        vals = []
        for c in conds_sp:
            row = [r for r in sp if r["model"] == model and r["condition"] == c][0]
            vals.append(100 * row[metric])
        return vals

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.9))

    # Left: PickPlace Frozen E_ADAPTIVE_TRANSPORT
    ax = axes[0]
    x = np.arange(len(conds_pp))
    w = 0.27
    b1 = ax.bar(x - w, s42_succ, width=w, label="DR4 Seed 42 Success%", color="#16a34a")
    b2 = ax.bar(x, s43_succ, width=w, label="DR4 Seed 43 Success%", color="#0284c7")
    b3 = ax.bar(x + w, [0.5 * (a + b) for a, b in zip(s42_grasp, s43_grasp)], width=w, label="Mean Grasp%", color="#64748b", alpha=0.8)
    for rect in list(b1) + list(b2):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 1.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(pp_labels, rotation=15, ha="right")
    ax.set_ylim(0, 114)
    ax.set_title("A. PickPlace Frozen E_ADAPTIVE_TRANSPORT (N=20/seed/cond)")
    ax.set_ylabel("Rate (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left", frameon=True)

    # Right: SinglePreGrasp Frozen Multi-Seed (40 eps/condition)
    ax = axes[1]
    x2 = np.arange(len(conds_sp))
    w2 = 0.20
    sac_g = get_sp("5B2_17500_SAC_HER", "mean_grasp_rate")
    sac_g_err = get_sp("5B2_17500_SAC_HER", "std_grasp_rate")
    sac_l = get_sp("5B2_17500_SAC_HER", "mean_lift_rate")
    bc_g = get_sp("5B7_BC_WARMSTART", "mean_grasp_rate")
    bc_g_err = get_sp("5B7_BC_WARMSTART", "std_grasp_rate")
    bc_l = get_sp("5B7_BC_WARMSTART", "mean_lift_rate")
    bc_l_err = get_sp("5B7_BC_WARMSTART", "std_lift_rate")

    r1 = ax.bar(x2 - 1.5 * w2, bc_g, width=w2, yerr=bc_g_err, capsize=3, label="5B7 BC Grasp%", color="#2563eb")
    r2 = ax.bar(x2 - 0.5 * w2, bc_l, width=w2, yerr=bc_l_err, capsize=3, label="5B7 BC Lift%", color="#1d4ed8")
    r3 = ax.bar(x2 + 0.5 * w2, sac_g, width=w2, yerr=sac_g_err, capsize=3, label="5B2 SAC Grasp%", color="#f97316")
    r4 = ax.bar(x2 + 1.5 * w2, sac_l, width=w2, label="5B2 SAC Lift%", color="#dc2626")

    for bars, errs in ((r1, bc_g_err), (r2, bc_l_err), (r3, sac_g_err), (r4, [0.0] * len(sac_l))):
        for rect, err in zip(bars, errs):
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + err + 2.0, f"{h:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x2)
    ax.set_xticklabels(sp_labels, rotation=15, ha="right")
    ax.set_ylim(0, 118)
    ax.set_title("B. SinglePreGrasp Multi-Seed DR Suite (2x20=40 eps/cond)")
    ax.set_ylabel("Mean Rate (%) +/- std")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", ncol=2, frameon=True)

    fig.suptitle("Phase 3 Robustness Under Pose, Physics, Observation Noise (+/-5 mm) & 1-Step Delay", fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = out_dir / "phase15_robustness_generalization.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_assembly_chart(out_dir: Path) -> Path:
    scr = _load_json("results/phase15_3_assembly/phase15_3_assembly_scripted_results.json")
    lrn = _load_json("results/phase15_3_assembly/phase15_3_assembly_learned_results.json")

    policies = [
        ("Scripted\n(SCRIPTED)", scr),
        ("5k BC Warmstart\n(BEHAVIOR_CLONING)", lrn["bc_warmstart_only"]),
        ("SAC+HER 2.5k\n(LEARNED_SAC_HER)", lrn["sac_her_2500"]),
        ("SAC+HER 5.0k\n(LEARNED_SAC_HER)", lrn["sac_her_5000"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 4.8))

    # Left: Stage Funnel Rates
    ax = axes[0]
    p_labels = [p[0] for p in policies]
    x = np.arange(len(policies))
    w = 0.2
    grasp = [100 * p[1]["grasp_rate"] for p in policies]
    lift = [100 * p[1]["lift_rate"] for p in policies]
    align = [100 * p[1]["pre_insert_align_rate"] for p in policies]
    assem = [100 * p[1]["assembly_success_rate"] for p in policies]

    b1 = ax.bar(x - 1.5 * w, grasp, width=w, label="Grasp %", color="#3b82f6")
    b2 = ax.bar(x - 0.5 * w, lift, width=w, label="Lift %", color="#0891b2")
    b3 = ax.bar(x + 0.5 * w, align, width=w, label="Pre-Insert Align %", color="#f59e0b")
    b4 = ax.bar(x + 1.5 * w, assem, width=w, label="Assembly Success %", color="#16a34a")
    for rect in list(b1) + list(b2) + list(b3) + list(b4):
        h = rect.get_height()
        if h > 0:
            ax.text(rect.get_x() + rect.get_width() / 2, h + 1.5, f"{h:.1f}" if h % 1 else f"{h:.0f}", ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(x)
    ax.set_xticklabels(p_labels)
    ax.set_ylim(0, 114)
    ax.set_title("A. Assembly Stage Completion Rates (N=30 episodes each)")
    ax.set_ylabel("Episode Rate (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=True)

    # Right: Orientation Error & Episode Return
    ax = axes[1]
    ang_err = [p[1]["mean_final_angle_deg"] for p in policies]
    rets = [p[1]["mean_episode_reward"] for p in policies]

    bars = ax.bar(x - 0.18, ang_err, width=0.36, color=["#16a34a", "#60a5fa", "#f97316", "#dc2626"], label="Final Orientation Error (deg)")
    ax.set_ylabel("Mean Final Orientation Error (deg)")
    ax.set_ylim(0, 102)
    ax.set_xticks(x)
    ax.set_xticklabels(p_labels)
    ax.set_title("B. Orientation Error (deg) & Mean Return Comparison")
    ax.grid(axis="y", alpha=0.25)
    for rect in bars:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 1.5, f"{h:.2f}°", ha="center", va="bottom", fontsize=8)

    ax2 = ax.twinx()
    bars2 = ax2.bar(x + 0.18, rets, width=0.36, color="#64748b", alpha=0.65, label="Mean Episode Return")
    ax2.set_ylabel("Mean Episode Return")
    ax2.set_ylim(-11, 10)
    ax2.axhline(0, color="black", lw=0.8, ls=":")
    for rect in bars2:
        h = rect.get_height()
        va = "bottom" if h >= 0 else "top"
        off = 0.25 if h >= 0 else -0.3
        ax2.text(rect.get_x() + rect.get_width() / 2, h + off, f"{h:+.2f}", ha="center", va=va, fontsize=8, color="#334155")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)

    fig.suptitle("Phase 4 Orientation-Aware Peg-in-Hole Assembly (PandaAssemblyDenseEnv)", fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = out_dir / "phase15_orientation_assembly_benchmark.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_capability_matrix_chart(out_dir: Path) -> Path:
    # Summarize verified cross-task Phase 15 capabilities strictly separated by policy type
    b3_100 = _load_json("results/phase15_2_5B3/phase15_2_5B3_100ep_results.json")
    scr_asm = _load_json("results/phase15_3_assembly/phase15_3_assembly_scripted_results.json")
    lrn_asm = _load_json("results/phase15_3_assembly/phase15_3_assembly_learned_results.json")
    p3 = _load_json("results/phase15_phase3/phase15_phase3_robustness_results.json")["pickplace_dr_results"]
    pp_mean_dr4 = 100 * np.mean([r["success_rate"] for r in p3 if r["condition"] in ("DR1_POS", "DR2_POSE_TARGET", "DR4_FULL_PHYSICS")])
    pp_mean_grasp = 100 * np.mean([r["grasp_rate"] for r in p3 if r["condition"] in ("DR1_POS", "DR2_POSE_TARGET", "DR4_FULL_PHYSICS")])

    categories = [
        "PickPlace\nE_ADAPTIVE_TRANSPORT\n(LEARNED SAC+HER)",
        "SinglePreGrasp\n5B7 Warm-Start\n(BEHAVIOR_CLONING)",
        "SinglePreGrasp\n5B2 17.5k Unseen\n(LEARNED SAC+HER)",
        "Two-Object Sort\nPhase 15.2.5A\n(SCRIPTED)",
        "Orientation Assembly\nPhase 4 Controller\n(SCRIPTED)",
        "Orientation Assembly\n5k Warm-Start\n(BEHAVIOR_CLONING)",
    ]
    grasp_vals = [
        float(pp_mean_grasp),
        100 * b3_100["bc_warmstart_5b7"]["bilateral_grasp_rate"],
        100 * b3_100["sac_her_5b2_17500"]["bilateral_grasp_rate"],
        100.0,  # Phase 15.2.5A verified 50/50
        100 * scr_asm["grasp_rate"],
        100 * lrn_asm["bc_warmstart_only"]["grasp_rate"],
    ]
    task_vals = [
        float(pp_mean_dr4),
        100 * b3_100["bc_warmstart_5b7"]["lift_rate"],
        100 * b3_100["sac_her_5b2_17500"]["lift_rate"],
        100.0,  # Phase 15.2.5A verified 50/50
        100 * scr_asm["assembly_success_rate"],
        100 * lrn_asm["bc_warmstart_only"]["assembly_success_rate"],
    ]

    fig, ax = plt.subplots(figsize=(12.8, 4.8))
    x = np.arange(len(categories))
    w = 0.34
    b1 = ax.bar(x - w / 2, grasp_vals, width=w, label="Grasp Acquisition Rate (%)", color="#3b82f6")
    b2 = ax.bar(x + w / 2, task_vals, width=w, label="Task Completion Rate (Lift / Place / Assemble %)", color="#10b981")

    for rect in list(b1) + list(b2):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8.5)
    ax.set_ylim(0, 124)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Phase 15 Master Capability Overview (Strictly Stratified by Policy Source)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", ncol=2, frameon=True)

    fig.tight_layout()
    out_path = out_dir / "phase15_milestone_summary_funnel.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    out_dir = Path("results/phase15_master")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        build_single_grasp_chart(out_dir),
        build_robustness_chart(out_dir),
        build_assembly_chart(out_dir),
        build_capability_matrix_chart(out_dir),
    ]
    for p in paths:
        print("GENERATED:", p)


if __name__ == "__main__":
    main()
