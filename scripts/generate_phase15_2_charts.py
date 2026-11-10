"""Generate publication-quality Matplotlib figures from verified Phase 15.2 result JSONs."""

from __future__ import annotations

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.grid": True,
        "grid.alpha": 0.35,
    }
)


def make_phase15_2_4_chart() -> str:
    d4 = json.load(open("results/phase15_2_4/phase15_2_4_training_results.json"))
    ckpts = d4["checkpoint_evaluations"]
    steps = [c["num_timesteps"] / 1000.0 for c in ckpts]
    d0 = [c["mean_min_ee_d0_m"] for c in ckpts]
    d1 = [c["mean_min_ee_d1_m"] for c in ckpts]
    o1_reach = [c["obj1_reach_rate"] * 100.0 for c in ckpts]
    ret = [c["mean_episode_reward"] for c in ckpts]
    colls = [c["mean_collisions"] for c in ckpts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.4), dpi=180)
    fig.suptitle("Phase 15.2.4 — Two-Object Continuous Proximity Reach Learning (0k -> 50k Steps)", fontsize=12, fontweight="bold", y=1.02)

    ax1.plot(steps, d0, "o-", color="#1f77b4", lw=2.2, label="Min EE-Obj1 Dist (m)")
    ax1.plot(steps, d1, "s--", color="#ff7f0e", lw=2.0, label="Min EE-Obj2 Dist (m)")
    ax1.set_xlabel("Training Steps (k)")
    ax1.set_ylabel("Minimum EE-Object Distance (m)")
    ax1.set_title("Spatial Approach Distance & Obj1 Reach Rate")
    ax1.set_ylim(0.0, 0.58)
    ax1b = ax1.twinx()
    ax1b.bar(steps, o1_reach, width=2.6, alpha=0.22, color="#2ca02c", label="Obj1 Reach (%)")
    ax1b.set_ylabel("Obj1 Reach Rate (%)", color="#2ca02c")
    ax1b.set_ylim(0, 40)
    ax1b.grid(False)
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax1b.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper right")

    ax2.plot(steps, ret, "o-", color="#2ca02c", lw=2.2, label="Mean Episode Return")
    ax2.set_xlabel("Training Steps (k)")
    ax2.set_ylabel("Mean Episode Return", color="#2ca02c")
    ax2.set_title("Episode Return Gain vs Premature-Closure Collisions")
    ax2b = ax2.twinx()
    ax2b.plot(steps, colls, "^--", color="#d62728", lw=2.0, label="Mean Collisions / Ep")
    ax2b.set_ylabel("Collisions / Ep", color="#d62728")
    ax2b.grid(False)
    l3, lb3 = ax2.get_legend_handles_labels()
    l4, lb4 = ax2b.get_legend_handles_labels()
    ax2.legend(l3 + l4, lb3 + lb4, loc="lower right")

    out_path = "results/phase15_2_4/phase15_2_4_reach_learning.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_phase15_2_5B_vs_B1_chart() -> str:
    b0 = json.load(open("results/phase15_2_5B/phase15_2_5B_training_results.json"))
    b1 = json.load(open("results/phase15_2_5B1/phase15_2_5B1_training_results.json"))

    b0_c = b0["checkpoint_evaluations"]
    b1_c = b1["checkpoint_evaluations"]

    s_b0 = [c["num_timesteps"] / 1000.0 for c in b0_c]
    d3_b0 = [c["mean_min_ee_dist_m"] for c in b0_c]
    dxy_b0 = [c["mean_min_xy_dist_m"] for c in b0_c]
    reach_b0 = [c["reach_rate"] * 100.0 for c in b0_c]
    grasp_b0 = [c["bilateral_grasp_rate"] * 100.0 for c in b0_c]
    rew_b0 = [c["mean_episode_reward"] for c in b0_c]

    s_b1 = [c["num_timesteps"] / 1000.0 for c in b1_c]
    minz_b1 = [c["mean_min_ee_z_m"] for c in b1_c]
    zerr_b1 = [c["mean_vert_err_m"] for c in b1_c]
    dxy_b1 = [c["mean_min_xy_dist_m"] for c in b1_c]
    reach_b1 = [c["reach_rate"] * 100.0 for c in b1_c]
    safed_b1 = [c["safe_descent_rate"] * 100.0 for c in b1_c]
    auth_b1 = [c["grasp_auth_rate"] * 100.0 for c in b1_c]
    grasp_b1 = [c["bilateral_grasp_rate"] * 100.0 for c in b1_c]
    prem_b1 = [c["premature_closure_ep_rate"] * 100.0 for c in b1_c]
    rew_b1 = [c["mean_episode_reward"] for c in b1_c]

    tb0_ent = b0["tensorboard_scalars"].get("train/ent_coef", [])
    tb1_ent = b1["tensorboard_scalars"].get("train/ent_coef", [])

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6), dpi=180)
    fig.suptitle(
        "Phase 15.2.5B vs Phase 15.2.5B.1 — Solving Vertical Descent & Exposing Gripper-Bias Bottleneck",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )

    # Panel A: Vertical descent
    ax = axes[0, 0]
    ax.plot(s_b0, d3_b0, "o--", color="#7f7f7f", lw=2.0, label="Phase B Min 3D Dist (m) [0.50m Stall]")
    ax.plot(s_b1, minz_b1, "s-", color="#1f77b4", lw=2.3, label="Phase B.1 Min EE Z (m)")
    ax.plot(s_b1, zerr_b1, "^-", color="#17becf", lw=2.0, label="Phase B.1 Mean Vertical Error (m)")
    ax.axhline(0.025, color="#2ca02c", ls=":", lw=1.4, label="Target Grasp Z (0.025m)")
    ax.set_title("A. Vertical Approach Resolution (Phi_z + Demo Buffer)")
    ax.set_xlabel("Training Steps (k)")
    ax.set_ylabel("Meters (m)")
    ax.set_ylim(0.0, 0.58)
    ax.legend(loc="center right")

    # Panel B: XY alignment & Reach
    ax = axes[0, 1]
    ax.plot(s_b0, reach_b0, "o--", color="#7f7f7f", lw=1.8, label="Phase B Reach (%)")
    ax.plot(s_b1, reach_b1, "s-", color="#2ca02c", lw=2.3, label="Phase B.1 Reach (%)")
    ax.plot(s_b1, safed_b1, "^-", color="#1f77b4", lw=2.0, label="Phase B.1 Safe Descent (%)")
    ax.set_title("B. Reach & Safe Open-Gripper Descent Rates")
    ax.set_xlabel("Training Steps (k)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(-3, 78)
    ax_xy = ax.twinx()
    ax_xy.plot(s_b0, dxy_b0, "x:", color="#ff7f0e", lw=1.6, label="Phase B Min XY (m)")
    ax_xy.plot(s_b1, dxy_b1, "d:", color="#d62728", lw=1.6, label="Phase B.1 Min XY (m)")
    ax_xy.set_ylabel("Min XY Error (m)", color="#d62728")
    ax_xy.set_ylim(0.0, 0.16)
    ax_xy.grid(False)
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax_xy.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, loc="upper right", ncol=1)

    # Panel C: Grasp / Auth / Premature Closure
    ax = axes[1, 0]
    ax.plot(s_b0, grasp_b0, "o--", color="#7f7f7f", lw=1.8, label="Phase B Bilateral Grasp (%)")
    ax.plot(s_b1, grasp_b1, "s-", color="#9467bd", lw=2.4, label="Phase B.1 Bilateral Grasp (%)")
    ax.plot(s_b1, auth_b1, "^-", color="#17becf", lw=1.9, label="Phase B.1 Grasp Auth (%)")
    ax.plot(s_b1, prem_b1, "v:", color="#d62728", lw=2.0, label="Phase B.1 Premature Closure Ep (%)")
    ax.annotate("Peak: 40% Grasp,\n20% Auth @ 7.5k", xy=(7.5, 40), xytext=(0.4, 58),
                arrowprops=dict(arrowstyle="->", color="#9467bd", lw=1.2), fontsize=8.5, fontweight="bold", color="#4b2e83")
    ax.set_title("C. Grasp Emergence vs Closed-Gripper Bias Regression")
    ax.set_xlabel("Training Steps (k)")
    ax.set_ylabel("Episode Rate (%)")
    ax.set_ylim(-5, 114)
    ax.legend(loc="center right", bbox_to_anchor=(0.99, 0.62))

    # Panel D: Return & Entropy
    ax = axes[1, 1]
    ax.plot(s_b0, rew_b0, "o--", color="#7f7f7f", lw=2.0, label="Phase B Eval Return")
    ax.plot(s_b1, rew_b1, "s-", color="#ff7f0e", lw=2.2, label="Phase B.1 Eval Return")
    ax.set_title("D. Deterministic Return & SAC Entropy Coefficient")
    ax.set_xlabel("Training Steps (k)")
    ax.set_ylabel("Mean Episode Return")
    ax.set_ylim(-10.8, -0.5)
    ax_e = ax.twinx()
    if tb0_ent:
        ax_e.plot([p[0] / 1000.0 for p in tb0_ent], [p[1] for p in tb0_ent], ":", color="#8c564b", lw=1.8, label="Phase B ent_coef")
    if tb1_ent:
        ax_e.plot([p[0] / 1000.0 for p in tb1_ent], [p[1] for p in tb1_ent], "-", color="#e377c2", lw=1.8, alpha=0.85, label="Phase B.1 ent_coef")
    ax_e.set_ylabel("SAC ent_coef", color="#b03060")
    ax_e.grid(False)
    l3, lb3 = ax.get_legend_handles_labels()
    l4, lb4 = ax_e.get_legend_handles_labels()
    ax.legend(l3 + l4, lb3 + lb4, loc="center right")

    out_path = "results/phase15_2_5B1/phase15_2_5B_vs_B1_progress.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_phase15_2_5A_funnel_chart() -> str:
    d4 = json.load(open("results/phase15_2_4/phase15_2_4_training_results.json"))
    d5a = json.load(open("results/phase15_2_5A/phase15_2_5A_diagnostics.json"))["sequential_50ep_summary"]
    d5b = json.load(open("results/phase15_2_5B/phase15_2_5B_training_results.json"))["checkpoint_evaluations"]
    d5b1 = json.load(open("results/phase15_2_5B1/phase15_2_5B1_training_results.json"))["checkpoint_evaluations"]

    # Pick exact verified rows
    r4_50k = [c for c in d4["checkpoint_evaluations"] if c["num_timesteps"] == 50000][0]
    r5b_20k = [c for c in d5b if c["num_timesteps"] == 20000][0]
    r5b1_75 = [c for c in d5b1 if c["num_timesteps"] == 7500][0]
    r5b1_20k = [c for c in d5b1 if c["num_timesteps"] == 20000][0]

    stages = ["Reach\n(<=5cm)", "Open XY\nAlign", "Safe Descent /\nAuth", "Bilateral\nGrasp", "Lift\nSuccess", "Full Hold /\nSort"]
    series = {
        "15.2.5A Scripted PreGrasp (N=50, 2-Obj)": [
            d5a["object_1"]["approach_rate"] * 100.0,
            d5a["object_1"]["alignment_rate"] * 100.0,
            d5a["object_1"]["safe_descent_rate"] * 100.0,
            d5a["object_1"]["bilateral_grasp_rate"] * 100.0,
            d5a["object_1"]["lift_verified_rate"] * 100.0,
            d5a["full_sorting_success_rate"] * 100.0,
        ],
        "15.2.4 SAC+HER @50k (2-Obj Prox)": [
            max(r4_50k["obj1_reach_rate"], r4_50k["obj2_reach_rate"]) * 100.0,
            0.0,
            0.0,
            r4_50k["obj1_grasp_rate"] * 100.0,
            r4_50k["obj1_lift_rate"] * 100.0,
            r4_50k["both_success_rate"] * 100.0,
        ],
        "15.2.5B SAC+HER @20k (1-Obj)": [
            r5b_20k["reach_rate"] * 100.0,
            r5b_20k["alignment_rate"] * 100.0,
            r5b_20k["safe_descent_rate"] * 100.0,
            r5b_20k["bilateral_grasp_rate"] * 100.0,
            r5b_20k["lift_rate"] * 100.0,
            r5b_20k["hold_success_rate"] * 100.0,
        ],
        "15.2.5B.1 SAC+HER @7.5k Peak (1-Obj)": [
            r5b1_75["reach_rate"] * 100.0,
            r5b1_75["alignment_rate"] * 100.0,
            r5b1_75["safe_descent_rate"] * 100.0,
            r5b1_75["bilateral_grasp_rate"] * 100.0,
            r5b1_75["lift_rate"] * 100.0,
            r5b1_75["hold_success_rate"] * 100.0,
        ],
        "15.2.5B.1 SAC+HER @20k Final (1-Obj)": [
            r5b1_20k["reach_rate"] * 100.0,
            r5b1_20k["alignment_rate"] * 100.0,
            r5b1_20k["safe_descent_rate"] * 100.0,
            r5b1_20k["bilateral_grasp_rate"] * 100.0,
            r5b1_20k["lift_rate"] * 100.0,
            r5b1_20k["hold_success_rate"] * 100.0,
        ],
    }
    colors = ["#2ca02c", "#8c564b", "#7f7f7f", "#1f77b4", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(12.2, 4.8), dpi=180)
    x = np.arange(len(stages))
    w = 0.16
    for i, (label, vals) in enumerate(series.items()):
        offset = (i - (len(series) - 1) / 2.0) * w
        bars = ax.bar(x + offset, vals, width=w, color=colors[i], label=label, edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            if v > 0.0:
                ax.text(b.get_x() + b.get_width() / 2.0, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Stage Completion Rate (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Phase 15.2 Manipulation Funnel — Scripted Feasibility vs Learned SAC+HER Checkpoints", fontweight="bold", pad=38)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=True)

    out_path = "results/phase15_2_5A/phase15_2_5A_scripted_vs_rl_funnel.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    p1 = make_phase15_2_4_chart()
    p2 = make_phase15_2_5B_vs_B1_chart()
    p3 = make_phase15_2_5A_funnel_chart()
    print("GENERATED:", p1, p2, p3)
