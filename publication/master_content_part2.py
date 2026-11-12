"""Parts III through VIII content for the redesigned publication-quality robotics monograph:
- Part III: Project Architecture & Software Verification
- Part IV: Phase-Wise Research & Implementation (Phases 1-17)
- Part V: Cross-Phase Research Analysis & Failure Taxonomy
- Part VI: Web-Based Simulation Control Center (FastAPI + WebSockets + Three.js)
- Part VII: Phase 18 Future Research Scope (Multi-Agent Cooperative Robotics)
- Part VIII: Conclusions, Annotated IEEE Bibliography, Further Reading & Appendices.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
PUB = ROOT / "publication"


def build_parts_3_to_8() -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    hist = json.loads((PUB / "historical_phase_registry.json").read_text(encoding="utf-8"))
    bugs = list(csv.DictReader((PUB / "debugging_history.csv").open(encoding="utf-8")))
    exps = list(csv.DictReader((PUB / "experiment_inventory.csv").open(encoding="utf-8")))
    p14 = json.loads((ROOT / "results" / "phase14_4" / "phase14_4_confirmation_results.json").read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # PART III: PROJECT ARCHITECTURE AND DEVELOPMENT
    # ------------------------------------------------------------------
    sections.append({
        "id": "part3_architecture",
        "part": "PART III — PROJECT ARCHITECTURE AND DEVELOPMENT",
        "title": "Chapter 4: Modular System Stack, Simulation Design, and Engineering Verification",
        "blocks": [
            ("h2", "4.1 Four-Layer Robotics Architecture and Authoritative PyBullet Core"),
            ("p",
             "Across 17 research phases and subsequent full-stack refactoring, the platform converged onto four decoupled architectural planes "
             "(Figure 4.1): (1) Physics Simulation Core (PyBullet 240 Hz engine paired with MuJoCo 4.0.1 MJCF parity validator); (2) Robotics "
             "Kinematics, Perception & Controllers (`robotics/robots/franka_panda/`, `src/controllers/`, `src/perception/`, `src/rl/adaptive_controllers.py`); "
             "(3) Goal-Conditioned Gymnasium/HER Task Suites (`PandaReachEnv` through `PandaContactAssemblyEnv`); and (4) Training/Workstation "
             "Execution (`SB3`, `BCDecayingSAC`, `FastAPI SimulationManager`, `React + Three.js` digital twin)."),
            ("fig", "publication/assets/diagrams/diag_01_system_architecture.png",
             "Figure 4.1: Complete four-layer Dexterous-Robot-Learning architecture unifying physics, control, RL/perception, and web workstation."),
            ("fig", "publication/assets/diagrams/diag_02_pybullet_sim_loop.png",
             "Figure 4.2: Authoritative PyBullet 240 Hz stepping cycle, adaptive lift guard, DLS IK projection, and 20 Hz observation broadcast."),
            ("h2", "4.2 Environment Dimensionality & Verification Discipline"),
            ("p",
             "A key engineering strength is zero silent assumption about tensor dimensions. Live environment introspection verifies exact observation, "
             "goal, and action spaces: `PandaReachEnv` exposes `(4,)` observation + `(3,)` achieved/desired goals (`10D` total Dict); `PandaPickEnv`, "
             "`PandaPickPlaceEnv`, `PandaDynamicPickPlaceEnv`, and `MuJoCoPandaPickPlaceEnv` share `(10,)` + `(3,)+(3,)` (`16D` Dict); `PandaSortEnv` "
             "and `PandaSinglePreGraspEnv` expose `(22,)` + `6D` (`28D` Dict); `PandaMultiSortEnv` / `PreGrasp` expose `(31,)` + `(6,)+(6,)` (`43D` "
             "Dict); and assembly environments expose `(22,)` + `(7,)+(7,)` `SE(3)` pose goals with 5D `[dx, dy, dz, dyaw, gripper]` actions. "
             "The automated regression suite expanded monotonically from 37 tests (Phase 1) -> 52 (Phase 9) -> 56 (Phase 10) -> 111 (Phase 15) -> 126 "
             "combined Python + backend integration tests today with zero skipped or relaxed assertions."),
            ("table", "Table 4.1: Verified Gymnasium & Cross-Simulator Environment Contracts",
             ["Environment Class", "Obs", "Goals (AG+DG)", "Action", "Horizon", "Provenance / Phase"],
             [
                 ["PandaReachEnv", "4D", "3D + 3D", "4D", "50", "Phase 2 / Sparse Goal"],
                 ["PandaPickPlaceEnv / Dense", "10D", "3D + 3D", "4D", "100", "Phase 6 / 12–14 Core"],
                 ["PandaSinglePreGraspEnv", "22D", "3D + 3D", "4D", "80", "Phase 15.2.5B PreGrasp"],
                 ["PandaMultiSortPreGraspEnv", "31D", "6D + 6D", "4D", "300", "Phase 15.2.5A 2-Obj"],
                 ["PandaContactAssemblyEnv", "22D", "7D + 7D", "5D", "120", "Phase 15.3 / 16.4 ASM"],
                 ["PandaDynamicPickPlaceEnv", "10D", "3D + 3D", "4D", "100", "Phase 16.3 Kalman"],
                 ["MuJoCoPandaPickPlaceEnv", "10D", "3D + 3D", "4D", "100", "Phase 17.1 Cross-Sim"],
             ]),
        ],
    })

    # ------------------------------------------------------------------
    # PART IV: PHASE-WISE RESEARCH AND IMPLEMENTATION (PHASES 1-17)
    # ------------------------------------------------------------------
    sections.append({
        "id": "part4_phases_1_11",
        "part": "PART IV — PHASE-WISE RESEARCH AND IMPLEMENTATION",
        "title": "Chapter 5: Early Foundations Through Sparse Deadlocks (Phases 1–11)",
        "blocks": [
            ("h2", "5.1 Reconstructed History of Phases 1–8 (Git & Telemetry Evidence)"),
            ("p",
             "Phases 0 and 1 (commits 3a345f3, a5cc06b, 7ad9848 on 2026-10-03) built URDF modeling, PyBullet 240 Hz stepping, and damped "
             "least-squares IK, verifying 0.7–4.5 mm accuracy across 5 waypoints (37/37 tests in 4.71 s). Phases 2–4 implemented `PandaReachEnv`, "
             "SB3 SAC+HER callbacks, and `PandaPickEnv`. Note on historical provenance: standalone Phase 5 markdown logs and isolated pre-squash "
             "Phase 2 vs Phase 3 commits are NOT RECOVERABLE FROM AVAILABLE EVIDENCE and are never extrapolated beyond verified TensorBoard runs "
             "(`reach_sac/SAC_3`, `pick_sac/SAC_1`) and scripted benchmarks (Reach Step 6, Pick Step 67)."),
            ("p",
             "Phase 6 (`PandaPickPlaceEnv`) resolved `ERR-001` (finger pad `lateralFriction=2.0`, `spinningFriction=0.1`) and contrasted E1 (No-HER: "
             "7.7 table collisions/ep, actor loss +17.767) against E2 (`HER k=4`: 0.0 collisions, actor loss +2.594) across 10k smoke runs (Figure 5.1). "
             "Phases 7–8 scaled to colored `SortingBin` tasks (`Sort` Step 79, `MultiSort` Step 209, `Assembly` Step 143 scripted). Phases 9–10 repaired "
             "lazy simulation connection `client_id=-1` in `PandaAssemblyEnv.reset()` (`ERR-002`, 52/52 tests) and verified scripted DR R0–R4 (75–100%)."),
            ("fig", "publication/assets/plots/plot_01_early_tb_curves.png",
             "Figure 5.1: Reconstructed Phases 2–8 TensorBoard actor loss and entropy coefficient decay trajectories (CPU PyTorch, 10k runs)."),
            ("h2", "5.2 Phase 11: Systematic Diagnosis of the Sparse HER Equilibrium (2.5M Steps)"),
            ("p",
             "Evaluating 25 sparse SAC+HER models (5 DR tiers x 5 seeds, 100k steps each; 5,000 total evaluation episodes) produced uniformly 0.0% "
             "completion across both Nominal and Held-Out DR4 splits (Figure 5.2). Isolating transport via `TransportDiagnosticEnv` (Phase 11.9) "
             "proved 100.0% transport convergence in 40k–50k steps (0.009 m error), demonstrating that sparse reward failure stemmed strictly from "
             "initial grasp acquisition starvation rather than spatial goal transport capacity."),
            ("fig", "publication/assets/plots/plot_02_phase11_sparse_vs_transport.png",
             "Figure 5.2: Phase 11 25-model sparse stagnation across 5,000 evaluation episodes (Mean +/- SD across 5 seeds per DR tier)."),
        ],
    })

    sections.append({
        "id": "part4_phases_12_15",
        "part": "PART IV — PHASE-WISE RESEARCH AND IMPLEMENTATION",
        "title": "Chapter 6: 1M Breakthrough, Adaptive Control, and Multi-Task Scaling (Phases 12–15)",
        "blocks": [
            ("h2", "6.1 Phases 12–13.5: Dense 1M Convergence and OOD Slip Discovery"),
            ("p",
             "Scaling dense potential shaping (Equation 10) to 1,000,000 steps in Phase 12 triggered an abrupt capability cascade between 300k and "
             "500k steps: Reach (33%->100%), Grasp (20%->100%), Lift (1%->98%), and Placement (0%->99%), locking at 100% through 1M steps with 0.0142 m "
             "final error and 89.84 hold steps (Figure 6.1). Extending training to DR1/DR2/DR4 in Phase 13 achieved 99–100% Matched evaluation accuracy "
             "but revealed severe Held-Out DR4 collapse (57.0% combined; Seed 43 only 33.0%, Figure 6.2). Phase 13.5 forensics (1,520 episodes) "
             "captured the exact 68.4 N -> 0.0 N slip event under heavy mass (0.25 kg) and low friction (mu=0.3) (Figure 6.3)."),
            ("fig", "publication/assets/plots/plot_03_phase12_1m_breakthrough.png",
             "Figure 6.1: Phase 12 1M-step dense Pick-and-Place S-curve (N=100 evaluation episodes per checkpoint, DR0 Seed 42)."),
            ("fig", "publication/assets/plots/plot_04_phase13_dr_with_errorbars.png",
             "Figure 6.2: Phase 13 Matched vs. Held-Out DR4 generalization gap (multi-seed Mean +/- SD) and Phase 13.5 isolated friction sweep."),
            ("fig", "publication/assets/plots/plot_05_phase13_5_slip_telemetry_2d_3d.png",
             "Figure 6.3: Real Phase 13.5 D5 telemetry contrasting successful lift (ep5001) against Stage-3 slip collapse (ep5002) in 2D & 3D."),
            ("h2", "6.2 Phase 14: Zero-Retraining Adaptive Manipulation Shields (+41.0 pp Gain)"),
            ("p",
             "To eliminate liftoff slip without retraining frozen neural weights, Phase 14 wrapped the policy output inside `AdaptiveTransportController` "
             "(Figure 6.4). Adding a 3-step zero-XY grasp settle floor (F_n >= 5 N) and dynamic vertical velocity throttling propelled Held-Out DR4 "
             "confirmation (N=200 combined) from 57.0% (Controller A) -> 89.5% (Controller D) -> 98.0% (Controller E), rescuing worst-case Seed 43 "
             "from 33.0% to 99.0% while retaining 100.0% DR0 Nominal accuracy (Figure 6.5)."),
            ("fig", "publication/assets/diagrams/diag_09_adaptive_shield_and_pregrasp.png",
             "Figure 6.4: Phase 14 Adaptive Manipulation Shield integrated with the Phase 15.2.5A five-check bilateral PreGrasp FSM."),
            ("fig", "publication/assets/plots/plot_06_phase14_controllers_comparison.png",
             "Figure 6.5: Phase 14.4 Held-Out DR4 confirmation across Controllers A, D, and E (Seeds 42, 43, and N=200/600 combined suites)."),
            ("h2", "6.3 Phase 15: Perturbation Sensitivity, Pre-Grasp Sorting & Masked BC-SAC"),
            ("p",
             "Phase 15.1 evaluated Controller E across 320 perturbed episodes: ±3 mm (97.5%) and ±5 mm (95.0%) spatial noise remained robust, 1-step "
             "delay dropped to 72.5%, >=2-step delay collapsed to 0.0%, and stripping oracle forces fell to 42.5%. Phase 15.2.5A achieved 50/50 (100%) "
             "scripted two-object sorting (76.4 steps, 0 collisions, Figure 6.6a), whereas single-object RL ablations 5B2–5B8 proved pure masked BC "
             "(5B7: 75% grasp / 45% lift, 0% coll) outperformed pure 17.5k SAC (5B2: 62%/7%) and Hybrid BC-SAC (5B8: 66%/26%) on 100 unseen episodes. "
             "Multi-object RL transfer (Phase 15.2.5C) remained BLOCKED; Phase 15.3 scripted assembly scored 56.67% vs 23.33% BC pre-align (Figure 6.7)."),
            ("fig", "publication/assets/diagrams/diag_10_task_lifecycles_multisort_assembly.png",
             "Figure 6.6: Sequential dual-cube MultiSort workflow (50/50 scripted) and SE(2)+Z compliant peg-in-slot insertion geometry."),
            ("fig", "publication/assets/plots/plot_07_phase15_ablations_and_stress.png",
             "Figure 6.7: Phase 15.1 stress resilience, 15.2.5B–B8 grasp/lift ablations, and Phase 15.3 assembly results."),
        ],
    })

    sections.append({
        "id": "part4_phases_16_17",
        "part": "PART IV — PHASE-WISE RESEARCH AND IMPLEMENTATION",
        "title": "Chapter 7: HRL, Oblique RGB-D Vision, Dynamic Tracking & MuJoCo Parity (Phases 16–17)",
        "blocks": [
            ("h2", "7.1 Phase 16 (Completed Subsystems vs. Partial Multi-Object Benchmarks)"),
            ("p",
             "Phase 16 delivered four verified breakthroughs and two partial multi-step benchmarks (Figure 7.1, Figure 7.2): "
             "(1) Phase 16.1 (`COMPLETE`): Discrete SAC meta-controller over 5 neural option skills reached 30.0% PickPlace (33.33% grasp, entropy "
             "1.7905 -> 0.5940) in 12k steps vs 0.0% flat SAC; (2) Phase 16.2 (`COMPLETE`): Repositioning RGB-D obliquely to [0.92, 0.28, 0.45] solved "
             "overhead self-occlusion (`ERR-011`: 18.65% -> 99.9–100% detection, 14.81–15.55 mm 3D unprojection error, Figure 7.3), enabling zero-shot "
             "vision SAC 60% grasp / 15% place; (3) Phase 16.3 (`COMPLETE`): Kalman target prediction hit 95–100% across moving cubes, sinusoids, and "
             "step disturbances; (4) Phase 16.4 (`PARTIALLY_COMPLETE`): Compliant wall-force insertion improved scripted assembly 50.0% -> 53.33% "
             "(learned RL assembly remains blocked); (5) Phase 16.6 (`PARTIALLY_COMPLETE`): Autonomous RGB-D HRL 2-object sorting hit 25% Object-1 "
             "but 0% full 2-object sorting across 229 replans and 1,049 safety clamps."),
            ("fig", "publication/assets/diagrams/diag_11_hrl_rgbd_and_sim2sim.png",
             "Figure 7.1: Integration across oblique RGB-D perception, 5-skill HRL selection, Kalman dynamic prediction, and MuJoCo validation."),
            ("fig", "publication/assets/plots/plot_08_phase16_hrl_vision_dynamic_contact.png",
             "Figure 7.2: Empirical Phase 16 suite across HRL (12k), RGB-D Vision (20 ep), Dynamic Kalman (20 ep/cond), and Compliant Assembly (30 ep)."),
            ("fig", "publication/assets/plots/plot_11_oblique_rgbd_3d_frustum.png",
             "Figure 7.3: Live PyBullet 3D oblique camera frustum (FOV 55 deg) and unprojected 91-pixel cube point cloud."),
            ("h2", "7.2 Phase 17: Cross-Simulator Falsification in MuJoCo 4.0.1 & Deployment Safety"),
            ("p",
             "Porting PickPlace identically into MuJoCo 4.0.1 (`MuJoCoPandaPickPlaceEnv`, Phase 17.1 `PARTIALLY_COMPLETE` as cross-sim diagnostic) "
             "exposed genuine physics-solver differences (`ERR-012` massless MJCF links repaired + 9.4 mm tabletop rest Z offset): frozen PyBullet Seed 42 "
             "dropped from 98% Held-Out DR4 to 40% bilateral contact / 5% grasp / 0% completion zero-shot, improving to 50% contact / 15% grasp / 5% "
             "completion after actuator friction/force calibration (Figure 7.4a). Phase 17.2 (`COMPLETE`) validated `SafetySupervisor` runtime E-stop "
             "latency at 32.41 us across 250 faulty injections and 100% stress completion under ±5 mm noise and frequency jitter (Figure 7.4b). "
             "Phase 17.3 physical robot execution was explicitly NOT_ATTEMPTED."),
            ("fig", "publication/assets/plots/plot_09_phase17_sim2sim_and_safety.png",
             "Figure 7.4: Phase 17.1 PyBullet-to-MuJoCo 4.0.1 transfer gap and Phase 17.2 simulated stress & E-stop assessment."),
        ],
    })

    # ------------------------------------------------------------------
    # PART V: RESEARCH ANALYSIS
    # ------------------------------------------------------------------
    sections.append({
        "id": "part5_analysis",
        "part": "PART V — RESEARCH ANALYSIS",
        "title": "Chapter 8: Cross-Phase Synthesis, Negative Results, and Provenance Analysis",
        "blocks": [
            ("h2", "8.1 Strict Provenance Comparison (`SCRIPTED` vs `BEHAVIOR_CLONING` vs `LEARNED`)"),
            ("p",
             "Conflating scripted kinematic heuristics with end-to-end neural reinforcement learning leads to misleading engineering claims. Across all "
             "27 registered experiments, deterministic scripted controllers (`Phase 15.2.5A`: 100% 2-obj sorting; `Phase 16.4`: 53.33% compliant insertion) "
             "excel when exact geometric state and contact thresholds are known, yet brittle failure ensues outside calibrated bounds. Pure SAC+HER "
             "achieves 100% DR0 at 600k–1M steps (`Phase 12`), suffers OOD slip collapse without runtime shielding (`Phase 13.5`: 57% -> `Phase 14`: 98%), "
             "and outperforms sparse HER under balanced Behavioral Cloning warm-starts at low sample budgets (`5B7`: 45% lift vs `5B2`: 7% lift)."),
            ("h2", "8.2 Value of Negative Results and Statistical Failure Taxonomy"),
            ("p",
             "Four negative experimental outcomes fundamentally shaped system architecture: (1) Phase 11's 25x100k sparse 0% plateau proved static-goal "
             "HER equilibria; (2) Phase 15.2.5B's 0% hover at z=0.50 m identified strict XY tolerance cylinder deadzones; (3) Phase 15.2.5B.1's 78.4% "
             "closed-gripper bias exposed how unmasked demo tails poison critic/actor rollouts; and (4) Phase 17.1's 98% -> 5% PyBullet-to-MuJoCo "
             "drop falsified any premature assumption that PyBullet DR4 success implies physical FCI hardware readiness."),
            ("table", "Table 8.1: Cross-Phase Defect Root-Cause Registry (ERR-001 through ERR-012)",
             ["ID", "Phase", "Module", "Observed Failure", "Verified Resolution"],
             [[b["error_id"], b["phase"], b["affected_file"], b["symptom_or_message"], b["fix"]] for b in bugs]),
        ],
    })

    # ------------------------------------------------------------------
    # PART VI: WEB-BASED SIMULATION CONTROL CENTER
    # ------------------------------------------------------------------
    sections.append({
        "id": "part6_web_workstation",
        "part": "PART VI — WEB-BASED SIMULATION CONTROL CENTER",
        "title": "Chapter 9: Refactored Robotics Workstation, FastAPI Core, and Three.js Digital Twin",
        "blocks": [
            ("h2", "9.1 Authoritative PyBullet-to-Browser Architecture"),
            ("p",
             "Following Phase 0 repository auditing (`docs/repository_audit.md`, pinning `backup/pre-restructure-ph17`, archiving 41 loose root scripts "
             "into `archive/historical/root_scripts/`), the project added a restrained dark industrial web workstation under `backend/` and `frontend/`. "
             "Crucially, PyBullet remains the single authoritative source of truth (Figure 9.1): a thread-safe `SimulationManager` steps PyBullet at 20 Hz "
             "and streams 11 real `panda.urdf` link frame transforms, cube/peg poses, sorting bins, and trajectory coordinates via WebSocket `/ws/simulation`. "
             "React Three Fiber renders official `.obj`/`.mtl` Panda geometry via `OBJLoader` + `MTLLoader` strictly from streamed physics frames — never "
             "running client-side forward kinematics animations that could diverge from simulation truth."),
            ("fig", "publication/assets/diagrams/diag_12_website_websocket_architecture.png",
             "Figure 9.1: Authoritative FastAPI SimulationManager, REST validation gates, 20 Hz WebSocket stream, and Three.js renderer."),
            ("fig", "publication/assets/plots/plot_10_panda_3d_fk_and_rgbd_cloud.png",
             "Figure 9.2: Three-dimensional workspace bounds and multi-target Panda kinematic chains mirrored between PyBullet and workstation."),
            ("h2", "9.2 Implemented Web Controls, Task Verification & Known Limits"),
            ("p",
             "Verified functionality (`126` combined Python + backend integration tests, `tsc -b && vite build`, and `2/2` Vitest UI checks): "
             "(i) 7-DOF joint sliders enforcing `panda.urdf` radian limits (`HTTP 422` on violations); (ii) Cartesian `(X, Y, Z, Roll, Pitch, Yaw)` "
             "command resolution via `FrankaPanda.calculate_ik` displaying requested vs. achieved pose and millimeter tracking error; (iii) Parallel-jaw "
             "gripper open/close/width actuation; (iv) Operator E-Stop halting active tasks and locking motors (`HTTP 409` until reset); (v) Live task "
             "runner (`reach_target`, `pick_lift`, `pick_place`, `multi_sort`, `assembly_insert`, `dynamic_manip`, returning `HTTP 501` for "
             "unimplemented physical FCI); and (vi) Experiment & Evaluation browser (`experiment_inventory.csv` + CSV export + Recharts). "
             "Known limitation: interactive live Chromium browser WebGL rendering validation remains manual startup verification rather than headless Playwright CI."),
        ],
    })

    # ------------------------------------------------------------------
    # PART VII: PHASE 18 FUTURE RESEARCH SCOPE
    # ------------------------------------------------------------------
    sections.append({
        "id": "part7_phase18_future",
        "part": "PART VII — PHASE 18 FUTURE RESEARCH [UNIMPLEMENTED]",
        "title": "Chapter 10: Future Scope — Multi-Agent Cooperative Manipulation Architecture",
        "blocks": [
            ("h2", "10.1 Explicit Future-Scope Disclaimer"),
            ("p",
             "CRITICAL PROVENANCE NOTICE: Phase 18 has NOT been implemented, simulated, or verified in this repository. Every architecture and "
             "methodology detailed in Chapter 10 constitutes a forward research roadmap building upon the verified Phase 1–17 single-arm baseline."),
            ("h2", "10.2 Multi-Robot Cooperative Manipulation, MARL, Task Allocation & Human-Robot Teaming"),
            ("p",
             "A. Multi-Robot Simulation & Cooperative Manipulation: Extending single-arm cells into dual-Panda tabletop bays requires synchronized "
             "contact constraints inside PyBullet/MuJoCo so two manipulators can jointly lift oversized rigid payloads or perform mid-air bimanual "
             "re-orientation without inter-arm self-collision.\n"
             "B. Multi-Agent Reinforcement Learning (MARL) & CTDE: Centralized Training with Decentralized Execution (Figure 10.1) resolves non-stationarity "
             "by feeding global joint states and both action vectors into centralized mixing critics Q_tot(s_all, a_A, a_B) while executing decentralized "
             "local policies pi_A(a_A|o_A) and pi_B(a_B|o_B) at runtime.\n"
             "C. Distributed Task Allocation, Cooperative Assembly & Inter-Robot Communication: Combinatorial auctions or Graph Neural Network allocators "
             "partition multi-object sorting sequences dynamically between arms, exchanging latent intent embeddings m_t to prevent workspace deadlocks "
             "during shared peg-in-hole handovers.\n"
             "D. Human-Robot Collaboration, Swarm Intelligence & Multi-Agent Robustness: Safe velocity separation scaling (ISO/TS 15066 principles), "
             "intent-aware trajectory replanning around moving human operators, and robustness under packet-loss delays across multi-agent links remain "
             "open theoretical and experimental challenges. Proposed evaluation metrics include joint bimanual grasp success rate, collision-free cycle "
             "time, communication bandwidth overhead, and degraded performance when one agent experiences injected sensor latency."),
            ("fig", "publication/assets/diagrams/diag_13_phase18_multiagent_future.png",
             "Figure 10.1: Proposed Phase 18 Centralized Training / Decentralized Execution (CTDE) Multi-Agent Cooperative Manipulation architecture [UNIMPLEMENTED]."),
            ("photo", "publication/assets/photographs/photo_02_industrial_robot_arm.jpg",
             "Figure 10.2: Multi-axis industrial serial manipulator illustrating industrial automation hardware scale beyond single tabletop simulation. "
             "Source: Wikimedia Commons (KUKA industrial robotic arm performing The Beach - 03.jpg), Author: Oleg Yunakov, License: CC BY-SA 4.0."),
        ],
    })

    # ------------------------------------------------------------------
    # PART VIII: CONCLUSION, BIBLIOGRAPHY, FURTHER READING & APPENDICES
    # ------------------------------------------------------------------
    bib_entries = [
        "[1] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, 'Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,' ICML, 2018. https://arxiv.org/abs/1801.01290",
        "[2] T. Haarnoja et al., 'Soft Actor-Critic Algorithms and Applications,' arXiv:1812.05905, 2018.",
        "[3] M. Andrychowicz et al., 'Hindsight Experience Replay,' Advances in Neural Information Processing Systems (NeurIPS), 2017. https://arxiv.org/abs/1707.01495",
        "[4] A. Raffin et al., 'Stable-Baselines3: Reliable Reinforcement Learning Implementations,' Journal of Machine Learning Research (JMLR), vol. 22, no. 268, pp. 1-8, 2021.",
        "[5] E. Coumans and Y. Bai, 'PyBullet, a Python Module for Physics Simulation for Games, Robotics and Machine Learning,' PyBullet Official Documentation, 2016-2021. https://pybullet.org",
        "[6] E. Todorov, T. Erez, and Y. Tassa, 'MuJoCo: A physics engine for model-based control,' IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2012. DOI: 10.1109/IROS.2012.6386109",
        "[7] J. Tobin et al., 'Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World,' IEEE/RSJ IROS, 2017. DOI: 10.1109/IROS.2017.8202133",
        "[8] S. Haddadin, S. Parusel, and A. Albu-Schaffer, 'Franka Emika Panda Technical Documentation & FCI Architecture,' Franka Robotics GmbH, 2017.",
        "[9] R. S. Sutton, D. Precup, and S. Singh, 'Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning,' Artificial Intelligence, vol. 112, pp. 181-211, 1999.",
        "[10] A. Nair et al., 'Overcoming Exploration in Reinforcement Learning with Demonstrations,' IEEE ICRA, 2018. DOI: 10.1109/ICRA.2018.8463162",
        "[11] N. Hogan, 'Impedance Control: An Approach to Manipulation,' ASME Journal of Dynamic Systems, Measurement, and Control, 1985.",
        "[12] R. E. Kalman, 'A New Approach to Linear Filtering and Prediction Problems,' ASME Journal of Basic Engineering, 1960.",
        "[13] M. Towers et al., 'Gymnasium: A Standard Interface for Reinforcement Learning Environments,' Farama Foundation, 2023. https://gymnasium.farama.org",
        "[14] S. Ramirez, 'FastAPI Framework Documentation,' Tiangolo / FastAPI Official Docs, 2024. https://fastapi.tiangolo.com",
        "[15] R. Cabello et al., 'Three.js 3D Library Documentation,' Three.js Authors, 2024. https://threejs.org/docs/",
    ]
    further_reading = [
        ["Robotics Fundamentals & Kinematics", "Lynch & Park, Modern Robotics (Cambridge Univ. Press, 2017) — screw theory, Jacobians, manipulability ellipsoids, and dynamics."],
        ["Reinforcement Learning", "Sutton & Barto, Reinforcement Learning: An Introduction (MIT Press, 2nd Ed.) — MDPs, Bellman optimality, actor-critic derivations."],
        ["Robot Learning & Sim-to-Real", "OpenAI et al., Solving Rubik's Cube with a Robot Hand (arXiv:1910.07113) — Automatic Domain Randomization at scale."],
        ["Simulation & Digital Twins", "PyBullet Quickstart Guide & MuJoCo Documentation — solver integrators, elliptic vs. pyramidal friction cones, MJCF modeling."],
        ["Multi-Agent Robotics (Phase 18)", "Rashid et al., QMIX / Lowe et al., MADDPG — centralized critics for decentralized multi-robot policy coordination."],
        ["Web & Control Stack", "FastAPI WebSockets + React Three Fiber — deterministic state-sync architectures separating physics stepping from UI frame rates."],
    ]
    sections.append({
        "id": "part8_conclusion_bib",
        "part": "PART VIII — CONCLUSIONS, BIBLIOGRAPHY & APPENDICES",
        "title": "Chapter 11: Verified Conclusions, Bibliography, Further Reading, and Registries",
        "blocks": [
            ("h2", "11.1 Verified Contributions & Explicit Technical Limitations"),
            ("p",
             "Major Contributions: (1) Resolved sparse HER static-goal stagnation (Phase 11->12: 0% -> 100% DR0 at 600k-1M); (2) Diagnosed micro-scale "
             "OOD slip collapse and delivered zero-retraining adaptive shielding (+41.0 pp Held-Out DR4 to 98.0% in Phase 14); (3) Built collision-free "
             "two-object PreGrasp sorting (50/50 scripted in Phase 15.2.5A) and unbiased progress-masked BC (75% grasp / 45% lift); (4) Delivered "
             "hierarchical options (30% vs 0%), oblique RGB-D unprojection (14.8 mm), predictive Kalman tracking (95-100%), MuJoCo 4.0.1 parity verification, "
             "32.41 us E-stop bounds, and a live Three.js/FastAPI Digital Twin workstation backed by 126 passing automated tests.\n"
             "Principal Limitations: Full 2-object autonomous RL sorting (0% in 16.6), learned compliant peg insertion (0% full insert vs 53.33% scripted), "
             "and cross-simulator MuJoCo transfer without retraining (5% calibrated) remain open problems. No physical Panda hardware was executed."),
            ("h2", "11.2 Annotated IEEE Bibliography"),
            *[("p", b) for b in bib_entries],
            ("h2", "11.3 Further Reading and Learning Resources"),
            ("table", "Table 11.1: Curated Robotics & Reinforcement Learning Study Guide",
             ["Research Domain", "Recommended Resource & Technical Takeaways"],
             further_reading),
            ("h2", "11.4 Reproducibility, Experiment Summary & Historical Provenance Registry"),
            ("table", "Table 11.2: Representative Executed Experiments Ledger (Strict Provenance Separation)",
             ["Exp ID", "Phase", "Env", "Method", "Budget / Eval", "Verified Outcome"],
             [[e["exp_id"], e["phase"], e["environment"], e["method"], e["budget_or_eval"], e["verified_outcome"]] for e in exps]),
            ("table", "Table 11.3: Phase 0–17 Verification Status Summary",
             ["Phase", "Title", "Period", "Status", "Evidence Quality"],
             [[h["phase"], h["title"], h["approx_period"], h["status"], h["evidence_quality"]] for h in hist]),
        ],
    })
    return sections

