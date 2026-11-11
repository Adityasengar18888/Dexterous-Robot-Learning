"""Builds the complete Phase 0-18 Monograph content tree, loads full empirical
tables directly from verified repository JSON/CSV/YAML files, and exports:
  - publication/research_monograph_source.md
  - publication/Dexterous_Robot_Learning_Monograph.tex
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parent.parent
PUB = ROOT / "publication"


def load_registries() -> Dict[str, Any]:
    return {
        "hist": json.loads((PUB / "historical_phase_registry.json").read_text(encoding="utf-8")),
        "ev": json.loads((PUB / "source_evidence_registry.json").read_text(encoding="utf-8")),
        "comp": json.loads((PUB / "phase_completion_matrix.json").read_text(encoding="utf-8")),
        "files": json.loads((PUB / "project_file_inventory.json").read_text(encoding="utf-8")),
        "tests": list(csv.DictReader((PUB / "complete_test_inventory.csv").open(encoding="utf-8"))),
        "bugs": list(csv.DictReader((PUB / "debugging_history.csv").open(encoding="utf-8"))),
        "exps": list(csv.DictReader((PUB / "experiment_inventory.csv").open(encoding="utf-8"))),
        "tb": json.loads((PUB / "benchmark_data" / "tb_scalars_all.json").read_text(encoding="utf-8")),
        "p11": list(csv.DictReader((ROOT / "results" / "phase11_final_summary.csv").open(encoding="utf-8"))),
        "p12": list(csv.DictReader((ROOT / "results" / "phase12_1m" / "checkpoint_metrics.csv").open(encoding="utf-8"))),
        "p13_mass": list(csv.DictReader((ROOT / "results" / "phase13_5" / "d2_mass_sensitivity" / "mass_summary.csv").open(encoding="utf-8"))),
        "p13_fric": list(csv.DictReader((ROOT / "results" / "phase13_5" / "d3_friction_sensitivity" / "friction_summary.csv").open(encoding="utf-8"))),
        "p13_geo": list(csv.DictReader((ROOT / "results" / "phase13_5" / "d4_contact_geometry" / "contact_geometry_summary.csv").open(encoding="utf-8"))),
        "p14_conf": json.loads((ROOT / "results" / "phase14_4" / "phase14_4_confirmation_results.json").read_text(encoding="utf-8")),
        "p15_1": json.loads((ROOT / "results" / "phase15_1" / "phase15_1_results.json").read_text(encoding="utf-8")),
        "p16_1": json.loads((ROOT / "results" / "phase16_17" / "phase16_1_hrl_results.json").read_text(encoding="utf-8")),
        "p16_2": json.loads((ROOT / "results" / "phase16_17" / "phase16_2_vision_results.json").read_text(encoding="utf-8")),
        "p16_3": json.loads((ROOT / "results" / "phase16_17" / "phase16_3_dynamic_results.json").read_text(encoding="utf-8")),
        "p16_4": json.loads((ROOT / "results" / "phase16_17" / "phase16_4_contact_assembly_results.json").read_text(encoding="utf-8")),
        "p16_6": json.loads((ROOT / "results" / "phase16_17" / "phase16_6_autonomous_results.json").read_text(encoding="utf-8")),
        "p17_c": json.loads((ROOT / "results" / "phase16_17" / "phase17_cross_simulator_results.json").read_text(encoding="utf-8")),
        "p17_s": json.loads((ROOT / "results" / "phase16_17" / "phase17_safety_assessment.json").read_text(encoding="utf-8")),
    }


def load_yaml_matrix() -> List[List[str]]:
    cfg_paths = [
        "configs/reach.yaml",
        "configs/pick.yaml",
        "configs/pick_place_her.yaml",
        "configs/pick_place_dr_physics.yaml",
        "configs/sort.yaml",
        "configs/multi_sort.yaml",
        "configs/assembly.yaml",
        "configs/multi_sort_phase15_5B2.yaml",
        "configs/multi_sort_phase15_5B8.yaml",
        "configs/assembly_phase15_3.yaml",
    ]
    rows = []
    for p in cfg_paths:
        y = yaml.safe_load((ROOT / p).read_text(encoding="utf-8"))
        rl = y.get("rl", {})
        env = y.get("env", {})
        rows.append([
            p.replace("configs/", ""),
            str(env.get("max_episode_steps", "-")),
            str(rl.get("learning_rate", "-")),
            str(rl.get("gamma", "-")),
            str(rl.get("tau", "-")),
            str(rl.get("buffer_size", "-")),
            str(rl.get("total_timesteps", "-")),
            str(rl.get("learning_starts", "default")),
        ])
    return rows


def build_sections(reg: Dict[str, Any]) -> List[Dict[str, Any]]:
    secs: List[Dict[str, Any]] = []
    yaml_rows = load_yaml_matrix()

    # Phase 11 all 50-row summary table (compact formatted)
    p11_rows = [
        [
            r["dr_condition"].upper(),
            r["training_seed"],
            r["evaluation_condition"],
            f"{float(r['success_rate'])*100:.1f}%",
            f"{float(r['grasp_success_rate'])*100:.1f}%",
            f"{float(r['mean_final_distance']):.4f} m",
            f"{float(r['collision_rate']):.2f}%",
        ]
        for r in reg["p11"]
    ]

    p12_rows = [
        [
            f"{int(float(r['Total Steps'])):,}",
            f"{float(r['Reach'])*100:.0f}%",
            f"{float(r['Grasp'])*100:.0f}%",
            f"{float(r['Lift'])*100:.0f}%",
            f"{float(r['End-to-End Success'])*100:.0f}%",
            f"{float(r['Mean Final Dist']):.4f} m",
            f"{float(r['Hold Duration']):.2f}",
        ]
        for r in reg["p12"]
    ]

    p13_mass_rows = [
        [r["mass"] + " kg", f"{float(r['success_rate'])*100:.1f}%", f"{float(r['grasp_rate'])*100:.1f}%",
         f"{float(r['mean_slip_events']):.2f}", f"{float(r['mean_normal_force']):.1f} N", f"{float(r['mean_lift_height']):.4f} m"]
        for r in reg["p13_mass"]
    ]
    p13_fric_rows = [
        [r["friction"], f"{float(r['success_rate'])*100:.1f}%", f"{float(r['grasp_rate'])*100:.1f}%",
         f"{float(r['mean_hold_duration']):.2f}", f"{float(r['mean_slip_events']):.2f}", f"{float(r['drop_rate'])*100:.1f}%"]
        for r in reg["p13_fric"]
    ]

    p14_rows = [
        [
            s["controller"][:20],
            s["regime"],
            str(s["training_seed"]),
            f"{s['success_rate']*100:.1f}%",
            f"{s['grasp_rate']*100:.1f}%",
            f"{s['bilateral_rate']*100:.1f}%",
            f"{s['mean_final_dist']:.4f} m",
        ]
        for s in reg["p14_conf"]["summaries"]
    ]

    p15_1_rows = [
        [
            s["condition"][:26],
            s["group"][:14],
            f"{s['success_rate']*100:.1f}%",
            f"{s['delta_pp_vs_baseline']:+.1f} pp",
            f"{s['grasp_rate']*100:.1f}%",
            s["readiness_class"],
        ]
        for s in reg["p15_1"]["summaries"]
    ]

    p16_2_rows = [
        [k.replace("_20ep", ""), v["observation_source"], f"{v['grasp_rate']*100:.1f}%",
         f"{v['success_rate']*100:.1f}%", f"{v['mean_collisions']:.2f}",
         f"{v['mean_3d_loc_err_mm']:.2f} mm" if v["mean_3d_loc_err_mm"] is not None else "GT Oracle",
         f"{v['mean_seg_iou']:.4f}" if v["mean_seg_iou"] is not None else "N/A"]
        for k, v in reg["p16_2"].items()
    ]

    p16_3_rows = [
        [k, f"{v['cube_speed_mps']:.3f} m/s", str(v["predictive_tracking"]),
         f"{v['grasp_rate']*100:.1f}%", f"{v['success_rate']*100:.1f}%",
         f"{v['mean_pred_err_mm']:.2f} mm", str(v["replan_triggers"])]
        for k, v in reg["p16_3"].items()
    ]

    p17_s_rows = [
        [k.replace("_", " "), str(v["n_episodes"]), f"{v['grasp_rate']*100:.1f}%",
         f"{v['success_rate']*100:.1f}%", f"{v['mean_collisions']:.2f}", f"{v['mean_safety_ws_clamps']:.2f}"]
        for k, v in reg["p17_s"].items()
        if isinstance(v, dict) and "success_rate" in v
    ]

    # ------------------------------------------------------------------
    # CH 1
    # ------------------------------------------------------------------
    secs.append({
        "id": "ch1",
        "title": "Chapter 1: Introduction, Historical Context, and Architectural Foundations",
        "blocks": [
            ("h2", "1.1 Problem Statement and Scientific Positioning"),
            ("p",
             "Autonomous robotic manipulation requires continuous synthesis across kinematics, frictional contact mechanics, goal-conditioned "
             "credit assignment, sensorimotor robustness, and safety engineering. Throughout Phases 0 through 17 of Dexterous-Robot-Learning, "
             "research focused on controlling a 7-degree-of-freedom (7-DOF) Franka Emika Panda manipulator using Soft Actor-Critic (SAC), "
             "Hindsight Experience Replay (HER), Behavioral Cloning (BC), Hierarchical Reinforcement Learning (HRL), and physics-informed "
             "inference controllers strictly under CPU-only PyTorch and single-process simulation constraints."),
            ("p",
             "Contrary to simplified continuous-control tasks with dense monotonic gradients, tabletop object manipulation exhibits brittle "
             "non-linear barriers: millisecond bilateral finger-pad slip during lift acceleration, sparse hindsight relabeling stagnation on "
             "unmoved objects, severe overhead camera self-occlusion, and non-negligible contact stiffness differences across independent "
             "physics solvers (PyBullet versus MuJoCo 4.0.1). Every architectural pivot documented in this monograph emerged directly from "
             "empirical anomaly localization."),
            ("h2", "1.2 Complete Stack and Milestone Summary (Phases 0-18)"),
            ("fig", "publication/architecture_diagrams/arch_01_system_stack_ph1_to_17.png",
             "Figure 1.1: Four-layer software, kinematics, control, perception, and verification stack unifying Phases 1-17."),
            ("fig", "publication/figures_2d/fig10_project_timeline_tests_throughput.png",
             "Figure 1.2: Regression suite scaling (37 -> 122 passing tests), CPU step throughput comparison, and Phase 1-18 completion scorecard."),
            ("h2", "1.3 Hyperparameter and Environment Configuration Evolution"),
            ("p",
             "Table 1.1 documents the exact evolution of YAML hyperparameters preserved across configs/*.yaml from early 10k sparse "
             "benchmarks (lr=1e-3, gamma=0.95, tau=0.05) to tuned long-horizon BC-SAC and assembly stabilization regimes (lr=3e-4 -> 1e-4, "
             "gamma=0.98, tau=0.005, extended learning_starts up to 2,500 steps)."),
            ("table", "Table 1.1: Verbatim YAML Hyperparameter Matrix Across Key Phase Configurations",
             ["Config File", "Max Steps", "LR", "Gamma", "Tau", "Buffer", "Budget", "Learn Starts"],
             yaml_rows),
        ],
    })

    # ------------------------------------------------------------------
    # CH 2
    # ------------------------------------------------------------------
    secs.append({
        "id": "ch2",
        "title": "Chapter 2: Foundation Kinematics, PyBullet Physics, Reach, and Pick (Phases 1-4)",
        "blocks": [
            ("h2", "2.1 Phase 0-1: Franka Emika Panda Kinematics and Cartesian Controller"),
            ("p",
             "Earliest repository commits (3a345f3, a5cc06b, and 7ad9848 on 2026-10-03) built PyBulletSim and FrankaPanda atop "
             "franka_panda/panda.urdf. Arm joints [0..6] operate under motor torque saturation [87, 87, 87, 87, 12, 12, 12] N*m with 0.04 "
             "linear and angular damping, while finger joints [9, 10] actuate up to 25 N across a symmetric [0.00, 0.08] m parallel-jaw span. "
             "Control commands dx in R^3 (bounded to +-0.05 m per 20 Hz cycle inside x=[0.25, 0.75], y=[-0.35, 0.35], z=[0.02, 0.50] m) "
             "resolve joint increments through damped least-squares inverse kinematics (lambda=0.05, max_iter=100, tol=1e-4, Equation (1)). "
             "Five deterministic validation waypoints confirmed end-effector error between 0.7 mm and 4.5 mm alongside 37/37 unit tests in 4.71 s."),
            ("eq", "publication/equations/eq_01.png", "Equation (1): Damped least-squares IK with null-space neutral posture regularization."),
            ("fig", "publication/figures_3d/fig3d_01_panda_fk_and_workspace.png",
             "Figure 2.1: Live PyBullet 3D forward-kinematic link chains for FrankaPanda inside workspace bounds."),
            ("h2", "2.2 Phases 2-4: Reach, Pick, and Reconstructed Early TensorBoard Diagnostics"),
            ("p",
             "Phase 2 (PandaReachEnv, 4D obs + 6D goals, 50 steps) and Phase 3 (SB3 SAC+HER callback harness) produced three reach event streams: "
             "logs/reach_sac/SAC_1 and SAC_2 contain 0 scalar events (aborted pre-warmup launches), whereas SAC_3 logged 10,000 steps. "
             "Actor loss climbed monotonically (-9.127 -> +19.518) as entropy collapsed (0.907 -> 0.002), while rollout success hit 1.0% "
             "against 0% deterministic evaluation (-50.0 reward). Phase 4 (PandaPickEnv, target_altitude=0.20 m, 100 steps) mirrored this "
             "trajectory (actor_loss -14.331 -> +20.005, ent_coef 0.742 -> 0.000, eval -100.0). Scripted controllers solved Reach at Step 6 "
             "and Pick at Step 67. Standalone Phase 5 reports do not exist in the repository and remain marked NOT_RECOVERABLE."),
            ("fig", "publication/figures_2d/fig01_early_tb_diagnostics.png",
             "Figure 2.2: Exact TensorBoard training metrics across Phases 2-8 (actor loss, critic loss, entropy decay, and CPU FPS)."),
            ("table", "Table 2.1: Reconstructed Early Phase TensorBoard & Scripted Baseline Comparison (Exact Measurements)",
             ["Run ID", "Env / Steps", "Actor Loss Trajectory", "Ent Coef Decay", "FPS", "Outcome / Baseline"],
             [
                 ["logs/reach_sac/SAC_3", "Reach / 50", "-9.127 -> +19.518", "0.907 -> 0.002", "106 -> 49", "Rollout 1%; Eval -50; Script Step 6"],
                 ["logs/pick_sac/SAC_1", "Pick / 100", "-14.331 -> +20.005", "0.742 -> 0.000", "72 -> 41", "Eval -100; Script Step 67"],
                 ["E1_no_her/SAC_1", "PickPlace / 100", "-14.135 -> +17.767", "0.875 -> 0.000", "74 -> 21", "7.7 colls/ep; Eval -100"],
                 ["E2_her/SAC_1", "PickPlace / 100", "-22.183 -> +2.594", "0.892 -> 0.001", "81 -> 35", "0.0 colls; Rollout 2%; Script Step 74"],
                 ["logs/sort/SAC_2", "Sort / 150", "-24.686 -> +3.710", "0.870 -> 0.001", "80 -> 30", "Eval -150; Script Step 79"],
                 ["logs/multi_sort/SAC_1", "MultiSort / 300", "-18.002 -> +7.383", "0.869 -> 0.001", "54 -> 18", "Eval -600; Script Step 209"],
                 ["logs/assembly/SAC_1", "Assembly / 200", "-32.398 -> +3.264", "0.885 -> 0.001", "57 -> 21", "Eval -200; Script Step 143"],
             ]),
        ],
    })

    # ------------------------------------------------------------------
    # CH 3
    # ------------------------------------------------------------------
    secs.append({
        "id": "ch3",
        "title": "Chapter 3: Pick-and-Place, Hindsight Experience Replay, Sorting, and Early Assembly (Phases 5-8)",
        "blocks": [
            ("h2", "3.1 Phase 6: HER Ablation and Finger Friction Physics Fix (ERR-001)"),
            ("p",
             "Phase 6 implemented PandaPickPlaceEnv and revealed ERR-001: low default tangential friction in Bullet allowed gripped 0.10 kg "
             "cubes to slide downward during liftoff. Configuring lateralFriction=2.0 and spinningFriction=0.1 on finger links 9 and 10 inside "
             "FrankaPanda.__init__ stabilized scripted pick-and-place at Step 74. Comparing E1 (No-HER SAC) vs E2 (SAC+HER, k=4 future, "
             "Equations (2)-(7)) across 10k smoke runs showed HER curbed Q-value blowup (actor_loss +2.594/+3.061 vs +17.767) and reduced "
             "table collisions from 7.7/ep to 0.0/ep."),
            ("eq", "publication/equations/eq_02.png", "Equation (2): Double Q-critic MSE loss with clipped target values."),
            ("eq", "publication/equations/eq_03.png", "Equation (3): Soft Bellman target including entropy bonus."),
            ("eq", "publication/equations/eq_04.png", "Equation (4): Reparameterized SAC policy gradient objective."),
            ("eq", "publication/equations/eq_05.png", "Equation (5): Lagrange dual update for automatic entropy coefficient alpha."),
            ("eq", "publication/equations/eq_06.png", "Equation (6): Binary sparse Euclidean tolerance reward (-1 outside 5 cm)."),
            ("eq", "publication/equations/eq_07.png", "Equation (7): Future-strategy HER virtual goal relabeling rule (k=4)."),
            ("fig", "publication/figures_2d/fig02_her_vs_noher_sparse_vs_dense.png",
             "Figure 3.1: (a) Phase 6 HER regularization effect; (b-c) Phase 11 25-run sparse failure distance and collision breakdown."),
            ("h2", "3.2 Phases 7-8: Color Sorting, Dual-Cube Multi-Sorting, and Assembly Prototyping"),
            ("p",
             "Phase 7 added color-matched SortingBin placement (PandaSortEnv, 150 steps, scripted Step 79). Phase 8 scaled to two sequential "
             "objects in PandaMultiSortEnv (31D observation, 6D desired_goal, 300 steps, scripted Step 209) and 5D orientation-aware peg-in-slot "
             "control [dx, dy, dz, dyaw, gripper] in PandaAssemblyEnv (22D observation, 7D goal, 200 steps, scripted Step 143). Brief 10k sparse "
             "RL evaluations remained uniformly unsolved (-150, -600, -200 mean return)."),
        ],
    })

    # ------------------------------------------------------------------
    # CH 4
    # ------------------------------------------------------------------
    secs.append({
        "id": "ch4",
        "title": "Chapter 4: System Audit, Domain Randomization, and Sparse HER Failure Forensics (Phases 9-11)",
        "blocks": [
            ("h2", "4.1 Phases 9-10: Lazy Client Repair and Five-Tier Domain Randomization"),
            ("p",
             "Phase 9 resolved ERR-002 (deferred client_id injection in PandaAssemblyEnv.reset()) raising verified passing unit tests to 52/52 "
             "(6.47 s). Phase 10 formalized physics/geometry randomization R0..R4, passing 56/56 tests and achieving scripted DR baseline rates "
             "of 90% (R0), 85% (R1), 100% (R2), 85% (R3), and 75% (R4) across 20 episodes per condition."),
            ("h2", "4.2 Phase 11: Complete 25-Run (2.5M-Step) Sparse Evaluation Audit"),
            ("p",
             "Phase 11 trained 25 independent sparse SAC+HER agents (5 DR tiers x seeds 42..46, 100k steps each) and evaluated all 25 models on "
             "100 Nominal and 100 Held-Out DR4 episodes (5,000 total episodes, Table 4.1). All 25 runs scored 0.0% task completion (mean held-out "
             "distance 0.5698 m, nominal ~0.243 m). Phase 11.5 isolated ERR-003 (the HER Static-Goal Trap): because sparse exploration rarely "
             "touched the cube, future goals repeated initial resting coordinates, reinforcing passive hovering. Phase 11.9 proved transport "
             "itself was straightforward (100% in 40-50k on pre-grasped TransportDiagnosticEnv), while 11.10b reached 28% Reach / 23% Grasp @ 225k."),
            ("table", "Table 4.1: Complete 50-Row Phase 11 Sparse Evaluation Matrix (results/phase11_final_summary.csv)",
             ["DR Tier", "Seed", "Eval Split", "Success", "Grasp", "Mean Dist", "Coll Rate"],
             p11_rows),
        ],
    })

    # ------------------------------------------------------------------
    # CH 5
    # ------------------------------------------------------------------
    secs.append({
        "id": "ch5",
        "title": "Chapter 5: 1M Dense Breakthrough, OOD Generalization Cliff, and Force-Slip Forensics (Phases 12-13.5)",
        "blocks": [
            ("h2", "5.1 Phase 12: One-Million-Step Dense Pick-and-Place Breakthrough"),
            ("p",
             "Using staged dense shaping Equation (8) over canonical 10D observation + 6D goals (16D Dict total, Equation (9)), Phase 12 trained DR0 Seed 42 to 1,000,000 "
             "steps (Table 5.1). Between 300k and 500k steps, sub-skills cascaded rapidly: Reach rose 33%->100%, Grasp 20%->100%, Lift 1%->98%, "
             "and end-to-end Placement leaped from 0% to 99%, locking at 100% from 600k through 1M (0.0142 m final error, 89.84 hold steps)."),
            ("eq", "publication/equations/eq_08.png", "Equation (8): Multi-stage dense manipulation shaping function."),
            ("eq", "publication/equations/eq_09.png", "Equation (9): Unified 10D proprioceptive + object state and 6D goal representation (16D Dict total)."),
            ("table", "Table 5.1: Phase 12 Checkpoint Sub-Skill Cascade (results/phase12_1m/checkpoint_metrics.csv)",
             ["Total Steps", "Reach", "Grasp", "Lift", "Success", "Final Dist", "Hold Steps"],
             p12_rows),
            ("fig", "publication/figures_2d/fig03_phase12_1m_learning_curve.png",
             "Figure 5.1: Phase 12 1M dense S-curve and positioning precision stabilization."),
            ("h2", "5.2 Phases 13 & 13.5: OOD DR4 Cliff and 1,520-Episode Microscopic Slip Investigation"),
            ("p",
             "Matched DR4 policies achieved 99.0% in-distribution accuracy but collapsed on Held-Out DR4 to 57.0% final (Seed 42: 81.0%, "
             "Seed 43: 33.0%). Across 1,520 diagnostic runs (D1-D5, Tables 5.2-5.3), mass up to 0.25 kg alone yielded 100% and low friction "
             "mu=0.3 alone yielded 90%, yet coupled heavy-mass + low-friction triggered 30.5% GRASP_BUT_NO_LIFT via a 0.14 m/s^2 liftoff jerk "
             "that snapped normal force from 68.4 N to 0.0 N (Figures 5.2-5.4, Equation (10))."),
            ("eq", "publication/equations/eq_10.png", "Equation (10): Bilateral finger normal-force and slip detection lift-gating invariant."),
            ("table", "Table 5.2: Phase 13.5 D2 Mass Sensitivity Sweep (d2_mass_sensitivity/mass_summary.csv)",
             ["Mass", "Success", "Grasp", "Mean Slips", "Normal Force", "Lift Height"],
             p13_mass_rows),
            ("table", "Table 5.3: Phase 13.5 D3 Friction Sensitivity Sweep (d3_friction_sensitivity/friction_summary.csv)",
             ["Friction mu", "Success", "Grasp", "Hold Steps", "Mean Slips", "Drop Rate"],
             p13_fric_rows),
            ("fig", "publication/figures_2d/fig04_phase13_dr_generalization_heatmap.png",
             "Figure 5.2: Phase 13 matched vs held-out DR4 asymmetry and Phase 13.5 mass/friction sweep responses."),
            ("fig", "publication/figures_2d/fig05_phase13_5_force_collapse_telemetry.png",
             "Figure 5.3: Step-level normal force, aperture, and cube Z comparing ep5001 success against ep5002 force collapse."),
            ("fig", "publication/figures_3d/fig3d_02_3d_manipulation_trajectories.png",
             "Figure 5.4: Real 3D end-effector and cube trajectory divergence during GRASP_BUT_NO_LIFT slippage."),
        ],
    })

    # ------------------------------------------------------------------
    # CH 6
    # ------------------------------------------------------------------
    secs.append({
        "id": "ch6",
        "title": "Chapter 6: Adaptive Zero-Retraining Manipulation Controllers (Phase 14)",
        "blocks": [
            ("h2", "6.1 Safety Shield Architecture (Controllers A through E)"),
            ("p",
             "Phase 14 introduced inference-time wrappers (src/rl/adaptive_controllers.py) intercepting actions from frozen Phase 13 models. "
             "Controller D (Hold+Settle) introduced a 3-step zero-XY squeeze period; Controller E (Adaptive Transport Shield, Figure 6.1) added "
             "contact-force-modulated vertical liftoff throttling, eliminating impulsive Stage-3 slip without a single RL weight update."),
            ("fig", "publication/architecture_diagrams/arch_02_hrl_option_fsm_and_phase14_shield.png",
             "Figure 6.1: Phase 14 Adaptive Manipulation Shield integrated beneath the Phase 16.1/16.6 five-skill FSM."),
            ("h2", "6.2 Complete Phase 14.4 Confirmation Results (N=100 per Cell)"),
            ("p",
             "Table 6.1 and Figure 6.2 summarize every Phase 14.4 confirmation evaluation across Seeds 42 and 43 in Nominal DR0, Matched DR4, "
             "and Held-Out DR4 regimes. Controller E elevated combined Held-Out DR4 accuracy from 57.0% to 98.0% (+41.0 pp), lifting Seed 43 "
             "from 33.0% to 99.0% while holding 100.0% DR0 and 98.5% Matched DR4 accuracy."),
            ("table", "Table 6.1: Full Phase 14.4 Confirmation Results Across Controllers, Seeds, and Regimes",
             ["Controller", "Regime", "Seed", "Success", "Grasp", "Bilateral", "Mean Final Dist"],
             p14_rows),
            ("fig", "publication/figures_2d/fig06_phase14_adaptive_controller_comparison.png",
             "Figure 6.2: Phase 14.4 Held-Out DR4 generalization breakthrough (+41.0 pp) and suite-wide consistency."),
        ],
    })

    # ------------------------------------------------------------------
    # CH 7
    # ------------------------------------------------------------------
    secs.append({
        "id": "ch7",
        "title": "Chapter 7: Stress Testing, Multi-Object Pre-Grasp Sorting, Masked BC-SAC, and Assembly (Phase 15)",
        "blocks": [
            ("h2", "7.1 Phase 15.1 Perturbation Sensitivity Matrix (320 Episodes)"),
            ("p",
             "Stress evaluation under observation noise, actuator delay, and unprivileged observations (Table 7.1) proved high resilience to "
             "Gaussian spatial noise (+-3 mm: 97.5%, +-5 mm: 95.0%), moderate tolerance to 1-step delay (72.5%), total breakdown at >=2-step "
             "latency (0.0%), and a drop to 42.5% when ground-truth velocity/orientation cues were masked."),
            ("table", "Table 7.1: Complete Phase 15.1 Readiness Stress Test Suite (results/phase15_1/phase15_1_results.json)",
             ["Scenario Condition", "Group", "Success", "Delta vs Base", "Grasp", "Class"],
             p15_1_rows),
            ("h2", "7.2 Phase 15.2 & 15.3: Multi-Object Sorting Gates, Vertical Approach Conditioning, and Assembly"),
            ("p",
             "Phase 15.2.4 exposed ERR-005 (un-gated dense sorting suffered 163.2 collisions/ep and 0% grasp). Phase 15.2.5A resolved this via "
             "5-check bilateral grasp gating in PreGraspController, yielding 50/50 (100%) scripted 2-object sorting in 76.4 mean steps. On "
             "single-object RL (PandaSinglePreGraspEnv), resolving ERR-006 (adding continuous vertical potential Equation (11)) and ERR-007 "
             "(removing closed-gripper HOLD_AT_GOAL bias via masked BC Equations (12)-(13)) propelled pure BC (5B7) to 75% grasp / 45% lift and "
             "Hybrid BC-SAC (5B8) to 66% grasp / 26% lift, compared against 7% lift for pure SAC (5B2). Full 28D->43D multi-object RL transfer "
             "(Phase 15.2.5C) remained BLOCKED. In Phase 15.3 peg-in-slot assembly, scripted baseline achieved 56.67% insertion (1.41 deg align), "
             "while BC achieved 23.33% pre-align / 0% insertion. Phase 15 closed with 111/111 tests passing in 29.47 s."),
            ("eq", "publication/equations/eq_11.png", "Equation (11): Continuous vertical potential Phi_z eliminating z=0.50m hover deadlocks."),
            ("eq", "publication/equations/eq_12.png", "Equation (12): Progress-masked Behavioral Cloning L2 regression loss."),
            ("eq", "publication/equations/eq_13.png", "Equation (13): Exponentially decaying Hybrid BC-SAC objective."),
            ("fig", "publication/figures_2d/fig07_phase15_grasp_bc_vs_sac_ablations.png",
             "Figure 7.1: Phase 15.1 stress resilience, 15.2.5B-B8 grasp/lift ablations, and Phase 15.3 assembly benchmark."),
        ],
    })

    # ------------------------------------------------------------------
    # CH 8
    # ------------------------------------------------------------------
    secs.append({
        "id": "ch8",
        "title": "Chapter 8: Hierarchical RL, Oblique RGB-D Vision, Dynamic Tracking, and Compliant Assembly (Phase 16)",
        "blocks": [
            ("h2", "8.1 Hierarchical Option-Critic, Oblique 128x128 RGB-D Perception, and Dynamic Kalman Control"),
            ("p",
             "Phase 16 expanded manipulation along five fronts: (16.1) Five-skill SMDP option decomposition (Equation (14)) solving 30.0% "
             "PickPlace at 12k steps after fixing ERR-008/009/010 vs 0.0% Flat SAC; (16.2) Repositioning SimulatedRGBDCamera obliquely at "
             "(0.92, 0.28, 0.45) to overcome ERR-011 overhead self-occlusion, raising detection from 18.65% to 99.9-100.0% and clipping "
             "unprojection error to 14.81-15.55 mm (Equation (15), Figure 8.1, Table 8.1); (16.3) Constant-velocity Kalman target tracking "
             "(Equation (16)) reaching 95-100% dynamic interception across moving cubes, sinusoids, and disturbances (Table 8.2); (16.4) Wall-force "
             "impedance compliance (Equation (17)) raising peg-in-slot assembly from 50.0% to 53.33%; and (16.6) Autonomous HRL 2-object evaluation "
             "yielding 25.0% Object-1 / 0.0% 2-object completion across 229 replans and 1,049 safety clamps."),
            ("eq", "publication/equations/eq_14.png", "Equation (14): Option-Critic five-skill hierarchical action policy."),
            ("eq", "publication/equations/eq_15.png", "Equation (15): Inverse projection from clip-space depth buffer to world XYZ coordinates."),
            ("eq", "publication/equations/eq_16.png", "Equation (16): Discrete Kalman predictor and innovation gain update."),
            ("eq", "publication/equations/eq_17.png", "Equation (17): Cartesian compliant wall-force insertion law (Phase 16.4)."),
            ("fig", "publication/figures_3d/fig3d_03_oblique_rgbd_frustum_and_mujoco_workspace.png",
             "Figure 8.1: Live PyBullet 3D oblique RGB-D frustum (55 deg FOV) and unprojected 91-pixel segmented cube cloud."),
            ("table", "Table 8.1: Phase 16.2 Vision vs Privileged State Evaluation (phase16_2_vision_results.json)",
             ["Agent Variant", "Obs Source", "Grasp %", "Place %", "Colls", "3D Loc Err", "Seg IoU"],
             p16_2_rows),
            ("table", "Table 8.2: Phase 16.3 Dynamic Moving Target & Disturbance Results (phase16_3_dynamic_results.json)",
             ["Motion Mode", "Speed", "Predictive", "Grasp %", "Success %", "Pred Err", "Replans"],
             p16_3_rows),
            ("fig", "publication/figures_2d/fig08_phase16_hrl_vision_dynamic_contact.png",
             "Figure 8.2: Phase 16 comparative benchmarks across HRL, RGB-D vision, dynamic Kalman tracking, and compliant assembly."),
        ],
    })

    # ------------------------------------------------------------------
    # CH 9
    # ------------------------------------------------------------------
    secs.append({
        "id": "ch9",
        "title": "Chapter 9: Cross-Simulator Validation (MuJoCo 4.0.1), Safety Envelope, and Hardware Boundary (Phase 17)",
        "blocks": [
            ("h2", "9.1 Cross-Engine Verification and Stress Suite Results"),
            ("p",
             "After repairing massless dummy links in MJCF (ERR-012), Phase 17.1 measured zero-shot PyBullet->MuJoCo 4.0.1 transfer of Seed 42: "
             "40.0% bilateral contact, 5.0% grasp, 0.0% completion, rising to 50.0% contact / 15.0% grasp / 5.0% completion after calibrating "
             "PD gains and tabletop Z offset (9.4 mm physics discrepancy). Phase 17.2 validated SafetySupervisor bounds (Equation (18)) across "
             "scenarios S1-S6 (Table 9.1) and confirmed 32.41 us E-stop cutoff across 250 faulty commands. Phase 17.3 physical robot execution "
             "was NOT_ATTEMPTED; repository suite verification stands at 122/122 passed tests in 42.47 s."),
            ("eq", "publication/equations/eq_18.png", "Equation (18): Workspace safety clipping and emergency velocity/force cutoff gate."),
            ("fig", "publication/architecture_diagrams/arch_03_sim2sim_and_rgbd_pipeline.png",
             "Figure 9.1: RGB-D pipeline, MuJoCo 4.0.1 cross-simulator bridge, and explicit unexecuted Phase 17.3/18 hardware boundary."),
            ("table", "Table 9.1: Phase 17.2 Deployment Safety & Perturbation Scenarios (phase17_safety_assessment.json)",
             ["Scenario", "Eps", "Grasp %", "Success %", "Collisions", "Safety Clamps/Ep"],
             p17_s_rows),
            ("fig", "publication/figures_2d/fig09_phase17_sim2sim_and_safety_stress.png",
             "Figure 9.2: Phase 17.1 MuJoCo cross-engine transfer gap and Phase 17.2 deployment safety robustness."),
        ],
    })

    # ------------------------------------------------------------------
    # CH 10
    # ------------------------------------------------------------------
    secs.append({
        "id": "ch10",
        "title": "Chapter 10: Consolidated Mathematical Formulations and Environment Registry",
        "blocks": [
            ("h2", "10.1 Complete Gymnasium Environment Specification Table (Phases 2-17)"),
            ("p",
             "Every task family adheres to consistent observation/goal/action tensor contracts, ensuring seamless transfer across SB3 SAC, "
             "HER, BC, Hybrid BC-SAC, HRL wrappers, and the MuJoCo cross-simulator parity adapter."),
            ("table", "Table 10.1: Environment Observation, Goal, Action, Horizon, and Reward Taxonomy (Verified Shapes)",
             ["Environment Class", "Obs Dim", "Goal Dim (AG+DG)", "Action Dim", "Horizon", "Reward Design"],
             [
                 ["PandaReachEnv", "4D", "3D + 3D (10D tot)", "4D", "50", "Sparse / L2 Distance"],
                 ["PandaPickEnv", "10D", "3D + 3D (16D tot)", "4D", "100", "Sparse / Altitude 0.2m"],
                 ["PandaPickPlaceEnv", "10D", "3D + 3D (16D tot)", "4D", "100", "Sparse -1 (tol 0.05m)"],
                 ["PandaPickPlaceDenseRewardEnv", "10D", "3D + 3D (16D tot)", "4D", "100", "Staged Dense +5 Place"],
                 ["TransportDiagnosticEnv", "10D", "3D + 3D (16D tot)", "4D", "100", "Pre-Grasped Transport"],
                 ["PandaSortEnv", "22D", "3D + 3D (28D tot)", "4D", "150", "Color Bin Placement"],
                 ["PandaMultiSortEnv", "31D", "6D + 6D (43D tot)", "4D", "300", "Dual Sequential Sort"],
                 ["PandaMultiSortPreGraspEnv", "31D", "6D + 6D (43D tot)", "4D", "300", "5-Check Grasp Gate"],
                 ["PandaSinglePreGraspEnv", "22D", "3D + 3D (28D tot)", "4D", "80", "Continuous Phi_z"],
                 ["PandaAssemblyEnv / Dense", "22D", "7D + 7D (36D tot)", "5D (+dyaw)", "200 / 120", "SE(2)+Z Insertion"],
                 ["PandaContactAssemblyEnv", "22D", "7D + 7D (36D tot)", "5D (+dyaw)", "120", "Compliant Wall-Force"],
                 ["PandaDynamicPickPlaceEnv", "10D", "3D + 3D (16D tot)", "4D", "100", "Kalman Intercept"],
                 ["MuJoCoPandaPickPlaceEnv", "10D", "3D + 3D (16D tot)", "4D", "100", "Cross-Sim Parity"],
             ]),
        ],
    })

    # ------------------------------------------------------------------
    # CH 11
    # ------------------------------------------------------------------
    bug_rows = [[b["error_id"], b["phase"], b["affected_file"], b["symptom_or_message"][:54], b["fix"][:54]] for b in reg["bugs"]]
    test_rows = [[t["test_id"], t["phase"], t["file_path"].replace("tests/", ""), t["test_name"][:40], t["parametrized_cases"], t["status"]] for t in reg["tests"]]
    secs.append({
        "id": "ch11",
        "title": "Chapter 11: Complete Debugging History and Full Regression Test Inventory",
        "blocks": [
            ("h2", "11.1 Root-Cause Forensics of All Twelve Verified System Defects (ERR-001 to ERR-012)"),
            ("table", "Table 11.1: Complete Registry of Twelve Verified Bugs, Root Causes, and Verified Resolutions",
             ["Error ID", "Phase", "File", "Symptom", "Verified Fix"],
             bug_rows),
            ("h2", "11.2 Complete Repository Automated Verification Suite (96 AST Functions / 122 Passed Items)"),
            ("p",
             "All 96 test definitions in tests/ expand via pytest parameterization into exactly 122 executed test cases passing in 42.47 s."),
            ("table", "Table 11.2: Full 96-Function Unit & Integration Test Inventory (complete_test_inventory.csv)",
             ["Test ID", "Phase", "Module", "Function Name", "Cases", "Status"],
             test_rows),
        ],
    })

    # ------------------------------------------------------------------
    # CH 12
    # ------------------------------------------------------------------
    exp_rows = [[e["exp_id"], e["phase"], e["environment"][:20], e["method"][:20], e["budget_or_eval"], e["verified_outcome"][:50]] for e in reg["exps"]]
    comp_rows = [[c["phase"], c["title"][:44], c["status"], c["evidence_quality"]] for c in reg["comp"]["completed_or_partial_phases"]]
    secs.append({
        "id": "ch12",
        "title": "Chapter 12: Cross-Phase Meta-Analysis, Completion Matrix, and Experiment Registry",
        "blocks": [
            ("h2", "12.1 Phase 0-18 Completion & Verification Provenance Matrix"),
            ("table", "Table 12.1: Complete Phase 0-18 Completion Status & Evidence Classification Matrix",
             ["Phase", "Capability / Milestone Title", "Status", "Evidence Class"],
             comp_rows),
            ("h2", "12.2 Master Registry of All 27 Executed Experiments"),
            ("table", "Table 12.2: Complete Experimental Ledger (publication/experiment_inventory.csv)",
             ["Exp ID", "Phase", "Environment", "Method", "Budget / Eval", "Verified Outcome"],
             exp_rows),
        ],
    })

    # ------------------------------------------------------------------
    # CH 13
    # ------------------------------------------------------------------
    secs.append({
        "id": "ch13",
        "title": "Chapter 13: Technical Limitations, Failure Mode Taxonomy, and Scientific Conclusions",
        "blocks": [
            ("h2", "13.1 Remaining Open Bottlenecks"),
            ("p",
             "Five verified limitations bound current project readiness: (1) Multi-object autonomous RL (Phases 15.2.5C & 16.6) has achieved 25% "
             "Object-1 sorting but 0% autonomous end-to-end 2-object RL transfer vs 100% scripted; (2) PyBullet-to-MuJoCo zero-shot accuracy "
             "drops from 98% to 0% (5% calibrated); (3) Unprivileged RGB-D visual execution without visual fine-tuning achieves 15-20% vs 100% GT; "
             "(4) Learned peg-in-slot insertion remains unsolved (23.33% BC align, 0% full insert vs 53.33% compliant scripted); and (5) No physical "
             "Franka hardware execution was attempted."),
            ("h2", "13.2 Key Scientific Conclusions"),
            ("p",
             "Across 17 phases, breakthroughs consistently came from coupling reward geometry with physical mechanics: fingertip friction parity (Ph 6), "
             "dense credit shaping to break static HER equilibria (Ph 11-12), inference force/slip shields (+41 pp zero-retraining boost in Ph 14), "
             "unbiased action-progress masking in Behavioral Cloning (Ph 15.2.5B7), non-occluded camera geometry (Ph 16.2), and cross-engine falsification (Ph 17.1)."),
        ],
    })

    # ------------------------------------------------------------------
    # CH 14
    # ------------------------------------------------------------------
    secs.append({
        "id": "ch14",
        "title": "Chapter 14: Future Scope — Phase 18 Architecture Roadmap [UNIMPLEMENTED FUTURE WORK]",
        "blocks": [
            ("h2", "14.1 Strict Scope Boundary Statement"),
            ("p",
             "Notice: Phase 18 is NOT implemented, executed, or empirically tested anywhere in this repository. All proposals below define future "
             "research directions building upon the verified Phase 1-17 artifacts."),
            ("h2", "14.2 Proposed Phase 18 Engineering Workstreams"),
            ("p",
             "18A Dual-Engine MuJoCo/PyBullet Co-Training to close the 98%->5% solver compliance gap; "
             "18B Closed-Loop Visuomotor Fine-Tuning over noisy 15 mm unprojected RGB-D inputs; "
             "18C Long-Horizon Multi-Object BC -> Option-Critic Curriculum; "
             "18D Physical System Identification & Dry-Run libfranka FCI Calibration; and "
             "18E 6-Axis Wrist Wrench Compliant Peg-in-Hole Insertion."),
        ],
    })

    # ------------------------------------------------------------------
    # CH 15
    # ------------------------------------------------------------------
    bib_items = [
        "[1] Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL. ICML.",
        "[2] Haarnoja, T. et al. (2018). Soft Actor-Critic Algorithms and Applications. arXiv:1812.05905.",
        "[3] Andrychowicz, M. et al. (2017). Hindsight Experience Replay. NeurIPS.",
        "[4] Plappert, M. et al. (2018). Multi-Goal Reinforcement Learning. arXiv:1802.09464.",
        "[5] Raffin, A. et al. (2021). Stable-Baselines3. JMLR, 22(268), 1-8.",
        "[6] Coumans, E., & Bai, Y. (2016-2021). PyBullet Physics Simulation.",
        "[7] Todorov, E., Erez, T., & Tassa, Y. (2012). MuJoCo: A physics engine for model-based control. IROS.",
        "[8] Towers, M. et al. (2023). Gymnasium. Farama Foundation.",
        "[9] Tobin, J. et al. (2017). Domain Randomization. IROS.",
        "[10] Sutton, R. S., Precup, D., & Singh, S. (1999). Between MDPs and semi-MDPs (Options). AIJ.",
        "[11] Ross, S., Gordon, G., & Bagnell, D. (2011). DAgger. AISTATS.",
        "[12] Nair, A. et al. (2018). Overcoming Exploration in RL with Demonstrations. ICRA.",
        "[13] Hogan, N. (1985). Impedance Control. ASME JDSMC.",
        "[14] Kalman, R. E. (1960). Linear Filtering and Prediction. ASME JBE.",
        "[15] Haddadin, S., Parusel, S., & Albu-Schaffer, A. (2017). Franka Panda Interface. Franka Emika.",
    ]
    secs.append({
        "id": "ch15",
        "title": "Chapter 15: Annotated Bibliography and Peer-Reviewed References",
        "blocks": [("p", b) for b in bib_items],
    })

    # ------------------------------------------------------------------
    # APP A: Reproducibility & Repository File Index
    # ------------------------------------------------------------------
    total_f = len(reg["files"]["files"])
    cat_counts = Counter(f["category"] for f in reg["files"]["files"])
    cat_rows = [[k, str(v), f"{100.0 * v / total_f:.1f}%"] for k, v in sorted(cat_counts.items())]
    all_key_files = [
        [f["path"], f["category"], str(f["lines"]), f"{f['size_bytes']} B"]
        for f in reg["files"]["files"]
        if f["path"].startswith(("src/", "scripts/", "tests/", "configs/", "docs/"))
    ]
    secs.append({
        "id": "appA",
        "title": "Appendix A: Reproducibility Checklists and Repository Artifact Inventory",
        "blocks": [
            ("h2", "A.1 Deterministic Execution Environment"),
            ("p",
             f"Environment: .conda-env/python.exe with KMP_DUPLICATE_LIB_OK=TRUE. Total tracked files: {total_f}. "
             "Latest verification: 122 passed tests in 42.47 s."),
            ("table", "Table A.1: Repository File Distribution by Category",
             ["Category", "Count", "Share (%)"],
             cat_rows),
            ("table", "Table A.2: Complete Core Repository Index (src/, scripts/, tests/, configs/, docs/)",
             ["Path", "Category", "Lines", "Size"],
             all_key_files),
        ],
    })
    return secs


def tex_esc(s: str) -> str:
    return (
        str(s)
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def write_markdown_and_tex(reg: Dict[str, Any], secs: List[Dict[str, Any]]) -> None:
    md_lines = [
        "# DEXTEROUS ROBOT LEARNING: PHASE 1-17 RESEARCH MONOGRAPH",
        "",
        "**Subtitle:** Comprehensive Technical Documentation of Franka Panda RL, Control, Perception, and Simulation-Based Validation",
        "",
        "> **Verification Boundary:** All empirical claims reflect verified PyBullet & MuJoCo 4.0.1 simulations. Physical robot hardware execution (Phase 17.3) was NOT_ATTEMPTED; Phase 18 is unexecuted future scope.",
        "",
    ]
    tex_lines = [
        r"\documentclass[11pt,a4paper,oneside]{book}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern,amsmath,amssymb,graphicx,booktabs,longtable,hyperref,geometry}",
        r"\geometry{a4paper,margin=2.0cm}",
        r"\title{\textbf{Dexterous Robot Learning: Phase 1--17 Research Monograph}}",
        r"\author{Dexterous-Robot-Learning Engineering Archive}",
        r"\date{Compiled from Verified Repository State (122/122 Tests Passing)}",
        r"\begin{document}",
        r"\frontmatter\maketitle\tableofcontents\listoffigures\listoftables\mainmatter",
    ]
    for sec in secs:
        md_lines.append(f"## {sec['title']}\n")
        tex_lines.append("\\chapter{" + tex_esc(sec["title"]) + "}")
        for blk in sec["blocks"]:
            kind = blk[0]
            if kind == "h2":
                md_lines.append(f"### {blk[1]}\n")
                tex_lines.append("\\section{" + tex_esc(blk[1]) + "}")
            elif kind == "p":
                md_lines.append(f"{blk[1]}\n")
                tex_lines.append(tex_esc(blk[1]) + "\n")
            elif kind == "algo":
                plain = blk[1].replace("<b>", "**").replace("</b>", "**").replace("<br/>", "\n> ")
                md_lines.append(f"> {plain}\n")
                t_algo = tex_esc(blk[1].replace("<b>", "").replace("</b>", "").replace("<br/>", r" \\ "))
                tex_lines.append("\\begin{quote}\\small\\textbf{Algorithm Summary}\\\\" + t_algo + "\\end{quote}\n")
            elif kind in ("eq", "fig"):
                md_lines.append(f"![{blk[2]}]({blk[1]})\n\n*{blk[2]}*\n")
                tex_lines.append(
                    "\\begin{figure}[htbp]\\centering\\includegraphics[width=0.92\\linewidth]{"
                    + blk[1] + "}\\caption{" + tex_esc(blk[2]) + "}\\end{figure}"
                )
            elif kind == "table":
                _, cap, hdrs, rows = blk
                md_lines.append(f"**{cap}**\n")
                md_lines.append("| " + " | ".join(hdrs) + " |")
                md_lines.append("| " + " | ".join(["---"] * len(hdrs)) + " |")
                for r in rows:
                    md_lines.append("| " + " | ".join(str(c).replace("|", "/") for c in r) + " |")
                md_lines.append("")
                colspec = " ".join(["l"] * len(hdrs))
                tex_lines.append("\\begin{longtable}{" + colspec + "}\\caption{" + tex_esc(cap) + "}\\\\\\toprule")
                tex_lines.append(" & ".join(tex_esc(h) for h in hdrs) + r" \\ \midrule \endhead")
                for r in rows:
                    tex_lines.append(" & ".join(tex_esc(c) for c in r) + r" \\")
                tex_lines.append(r"\bottomrule \end{longtable}")
    tex_lines.append(r"\end{document}")
    (PUB / "research_monograph_source.md").write_text("\n".join(md_lines), encoding="utf-8")
    (PUB / "Dexterous_Robot_Learning_Monograph.tex").write_text("\n".join(tex_lines), encoding="utf-8")
    print("Exported research_monograph_source.md and Dexterous_Robot_Learning_Monograph.tex")
