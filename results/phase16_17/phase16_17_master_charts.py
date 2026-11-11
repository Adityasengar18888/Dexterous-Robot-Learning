"""Generate Phase 16 & 17 Master Benchmark Charts cleanly from verified JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8.2,
        "figure.dpi": 180,
    }
)


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_phase16_hrl_vision_chart(out_dir: Path) -> Path:
    hrl = _load("results/phase16_17/phase16_1_hrl_results.json")
    vis = _load("results/phase16_17/phase16_2_vision_results.json")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.7))

    # Panel A: HRL SMDP Learning Curve (0 -> 12k)
    ax = axes[0]
    lc = hrl["hrl_learning_curve"]
    xs = [p["env_steps"] for p in lc]
    succs = [100 * p["success_rate"] for p in lc]
    grasps = [100 * p["grasp_rate"] for p in lc]
    ents = [p["entropy"] for p in lc]
    ax.plot(xs, grasps, "o--", lw=1.8, color="#2563eb", label="HRL Grasp %")
    ax.plot(xs, succs, "D-", lw=2.2, color="#16a34a", label="HRL PickPlace %")
    ax2 = ax.twinx()
    ax2.plot(xs, ents, "^:", lw=1.6, color="#ea580c", label="Policy Entropy (nats)")
    ax.set_title("A. Learned HRL Progression (0 -> 12k steps)")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Success / Grasp Rate (%)")
    ax2.set_ylabel("Categorical Policy Entropy")
    ax.set_ylim(-5, 65)
    ax2.set_ylim(0.0, 2.0)
    ax.grid(alpha=0.25)
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, loc="upper left")

    # Panel B: 30-Episode Comparison (HRL Untrained vs HRL 12k vs Flat SAC 12k vs Phase 14 500k)
    ax = axes[1]
    cmp_keys = [
        ("HRL_Untrained_Step0", "HRL Step 0\n(Untrained)"),
        ("Flat_SAC_HER_12k", "Flat SAC+HER\n(12k Scratch)"),
        ("HRL_Learned_DiscreteSAC_HER_12k", "Learned HRL\n(12k Scratch)"),
        ("Phase14_Adaptive_E_Frozen_500k_Base", "Phase 14\n(500k+Ctrl E)"),
    ]
    x = np.arange(len(cmp_keys))
    w = 0.35
    g_vals = [100 * hrl["comparisons_30ep"][k]["grasp_rate"] for k, _ in cmp_keys]
    s_vals = [100 * hrl["comparisons_30ep"][k]["success_rate"] for k, _ in cmp_keys]
    b1 = ax.bar(x - w / 2, g_vals, width=w, color="#3b82f6", label="Grasp %")
    b2 = ax.bar(x + w / 2, s_vals, width=w, color="#10b981", label="PickPlace Success %")
    for r in list(b1) + list(b2):
        h = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2, h + 1.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in cmp_keys], fontsize=8)
    ax.set_ylim(0, 114)
    ax.set_title("B. Phase 16.1 Control Comparison (30 eps)")
    ax.set_ylabel("Rate (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")

    # Panel C: Phase 16.2 Privileged GT vs RGB-D Vision
    ax = axes[2]
    v_keys = [
        ("SAC_Privileged_GT_20ep", "SAC\n(GT State)"),
        ("SAC_RGBD_Vision_20ep", "SAC\n(RGB-D Vis)"),
        ("HRL_Privileged_GT_20ep", "HRL\n(GT State)"),
        ("HRL_RGBD_Vision_20ep", "HRL\n(RGB-D Vis)"),
    ]
    xv = np.arange(len(v_keys))
    vg = [100 * vis[k]["grasp_rate"] for k, _ in v_keys]
    vs = [100 * vis[k]["success_rate"] for k, _ in v_keys]
    r1 = ax.bar(xv - w / 2, vg, width=w, color="#6366f1", label="Grasp %")
    r2 = ax.bar(xv + w / 2, vs, width=w, color="#f59e0b", label="PickPlace Success %")
    for r in list(r1) + list(r2):
        h = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2, h + 1.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(xv)
    ax.set_xticklabels([lbl for _, lbl in v_keys], fontsize=8.5)
    ax.set_ylim(0, 114)
    ax.set_title("C. Phase 16.2 Vision vs GT State (20 eps)")
    ax.set_ylabel("Rate (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    fig.suptitle("Phase 16.1 Hierarchical RL & Phase 16.2 Oblique RGB-D Vision Benchmarks", fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = out_dir / "phase16_hrl_and_vision_benchmarks.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def build_phase16_dynamic_contact_chart(out_dir: Path) -> Path:
    dyn = _load("results/phase16_17/phase16_3_dynamic_results.json")
    asm = _load("results/phase16_17/phase16_4_contact_assembly_results.json")
    auto = _load("results/phase16_17/phase16_6_autonomous_results.json")

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.7))

    # Left: Dynamic Moving-Object & Target Benchmarks
    ax = axes[0]
    dk = [
        ("linear_no_prediction", "Linear\n(No Pred)"),
        ("linear_predictive", "Linear\n(Predictive)"),
        ("sinusoidal_predictive", "Sinusoidal\n(Predictive)"),
        ("disturbance_step_predictive", "Disturbance\n(Predictive)"),
        ("moving_target_and_cube", "Moving Tgt\n+ Cube"),
    ]
    xd = np.arange(len(dk))
    w = 0.35
    dg = [100 * dyn[k]["grasp_rate"] for k, _ in dk]
    ds = [100 * dyn[k]["success_rate"] for k, _ in dk]
    b1 = ax.bar(xd - w / 2, dg, width=w, color="#0ea5e9", label="Grasp %")
    b2 = ax.bar(xd + w / 2, ds, width=w, color="#16a34a", label="PickPlace Success %")
    for r in list(b1) + list(b2):
        h = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2, h + 1.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(xd)
    ax.set_xticklabels([l for _, l in dk], fontsize=8.5)
    ax.set_ylim(0, 114)
    ax.set_title("A. Phase 16.3 Dynamic Tracking & Predictive Grasp (20 eps)")
    ax.set_ylabel("Rate (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")

    # Right: Contact Assembly & Autonomous Execution Comparison
    ax = axes[1]
    ak = [
        ("Phase 15 Scripted\nAssembly", asm["phase15_scripted_baseline_30ep"]["grasp_rate"] * 100, asm["phase15_scripted_baseline_30ep"]["assembly_success_rate"] * 100),
        ("Phase 16 Compliant\nAssembly", asm["phase16_compliant_impedance_30ep"]["grasp_rate"] * 100, asm["phase16_compliant_impedance_30ep"]["assembly_success_rate"] * 100),
        ("Phase 15 5k SAC\nAssembly (RL)", asm["phase15_learned_reference"]["sac_her_5000_LEARNED_SAC_HER"]["grasp_rate"] * 100, asm["phase15_learned_reference"]["sac_her_5000_LEARNED_SAC_HER"]["assembly_success_rate"] * 100),
        ("Phase 16.6 Auton.\n2-Obj (Obj1)", auto["obj1_success_rate"] * 100, auto["full_2obj_success_rate"] * 100),
    ]
    xa = np.arange(len(ak))
    ag = [v[1] for v in ak]
    asucc = [v[2] for v in ak]
    r1 = ax.bar(xa - w / 2, ag, width=w, color="#8b5cf6", label="Grasp / Obj1 Rate %")
    r2 = ax.bar(xa + w / 2, asucc, width=w, color="#f97316", label="Full Task Completion %")
    for r in list(r1) + list(r2):
        h = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(xa)
    ax.set_xticklabels([v[0] for v in ak], fontsize=8.5)
    ax.set_ylim(0, 114)
    ax.set_title("B. Phase 16.4 Contact Assembly & 16.6 Autonomous Task")
    ax.set_ylabel("Rate (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    fig.suptitle("Phase 16 Dynamic Manipulation, Compliant Contact Assembly & Autonomous Execution", fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = out_dir / "phase16_dynamic_contact_autonomous.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def build_phase17_cross_sim_safety_chart(out_dir: Path) -> Path:
    cs = _load("results/phase16_17/phase17_cross_simulator_results.json")
    sf = _load("results/phase16_17/phase17_safety_assessment.json")

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 4.8))

    # Left: PyBullet -> MuJoCo Cross-Sim Transfer
    ax = axes[0]
    mkeys = [
        ("ZeroShot SAC Seed42", cs["zero_shot_eval"]["sac_dr4_seed42"]),
        ("ZeroShot SAC Seed43", cs["zero_shot_eval"]["sac_dr4_seed43"]),
        ("ZeroShot HRL", cs["zero_shot_eval"]["hrl_meta"]),
        ("Calibrated SAC Seed42", cs["calibrated_physics_eval"]["sac_dr4_seed42"]),
        ("Calibrated HRL", cs["calibrated_physics_eval"]["hrl_meta"]),
    ]
    xm = np.arange(len(mkeys))
    w = 0.25
    b1 = ax.bar(xm - w, [100 * m[1]["bilateral_contact_rate"] for m in mkeys], width=w, color="#3b82f6", label="Bilateral Contact %")
    b2 = ax.bar(xm, [100 * m[1]["grasp_rate"] for m in mkeys], width=w, color="#f59e0b", label="Grasp / Lift %")
    b3 = ax.bar(xm + w, [100 * m[1]["success_rate"] for m in mkeys], width=w, color="#16a34a", label="Full PickPlace %")
    for r in list(b1) + list(b2) + list(b3):
        h = r.get_height()
        if h > 0:
            ax.text(r.get_x() + r.get_width() / 2, h + 1.2, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(xm)
    ax.set_xticklabels([m[0] for m in mkeys], rotation=15, ha="right", fontsize=8.5)
    ax.set_ylim(0, 68)
    ax.set_title("A. Phase 17.1 PyBullet Policies Evaluated in MuJoCo 4.0.1")
    ax.set_ylabel("Rate (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    # Right: Phase 17.2 Sim-to-Real Stress Suite
    ax = axes[1]
    s_conds = [
        ("A0_Clean_Nominal", "Clean\nNominal"),
        ("S1_Sensor_Noise_5mm", "Sensor\n±5mm"),
        ("S2_Camera_Calib_Err_8mm", "Cam Calib\n±8mm"),
        ("S3_Actuator_Latency_1step", "Latency\n1-step"),
        ("S4_Actuator_Latency_2step", "Latency\n2-step"),
        ("S5_Control_Freq_Jitter", "Freq\nJitter"),
        ("S6_Low_Friction_Heavy_Mass", "LowFric\nHeavy"),
    ]
    xs = np.arange(len(s_conds))
    w2 = 0.36
    sg = [100 * sf[k]["grasp_rate"] for k, _ in s_conds]
    ss = [100 * sf[k]["success_rate"] for k, _ in s_conds]
    r1 = ax.bar(xs - w2 / 2, sg, width=w2, color="#0284c7", label="Grasp %")
    r2 = ax.bar(xs + w2 / 2, ss, width=w2, color="#16a34a", label="Success % (with SafetyMonitor)")
    for r in list(r1) + list(r2):
        h = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2, h + 1.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([l for _, l in s_conds], fontsize=8)
    ax.set_ylim(0, 114)
    ax.set_title("B. Phase 17.2 Simulation-Side Sim-to-Real Stress Suite (N=20)")
    ax.set_ylabel("Rate (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left")

    fig.suptitle("Phase 17 MuJoCo Cross-Simulator Validation & Simulation Sim-to-Real Stress Assessment", fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = out_dir / "phase17_cross_sim_and_sim2real_safety.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def consolidate_benchmark_json(out_dir: Path) -> None:
    payload = {
        "phase16_1_hrl": _load("results/phase16_17/phase16_1_hrl_results.json"),
        "phase16_2_vision": _load("results/phase16_17/phase16_2_vision_results.json"),
        "phase16_3_dynamic": _load("results/phase16_17/phase16_3_dynamic_results.json"),
        "phase16_4_contact_assembly": _load("results/phase16_17/phase16_4_contact_assembly_results.json"),
        "phase16_6_autonomous": _load("results/phase16_17/phase16_6_autonomous_results.json"),
    }
    with open(out_dir / "phase16_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    out_dir = Path("results/phase16_17")
    out_dir.mkdir(parents=True, exist_ok=True)
    consolidate_benchmark_json(out_dir)
    for p in (
        build_phase16_hrl_vision_chart(out_dir),
        build_phase16_dynamic_contact_chart(out_dir),
        build_phase17_cross_sim_safety_chart(out_dir),
    ):
        print("GENERATED:", p)


if __name__ == "__main__":
    main()

