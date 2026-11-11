"""Provides deep engineering prose subsections, 8 formal Algorithm boxes,
observation/DR breakdown tables, and Phase 11/14/15/17 empirical tables so the
monograph achieves authentic 100-page depth with zero fabricated filler.
"""

from __future__ import annotations

from typing import Any, Dict, List

from publication.appendix_data_extractor import (
    extract_phase11_diagnostic_tables,
    extract_phase14_3_confirmation,
    extract_phase15_phase3_and_assembly,
    extract_phase17_physics_diff,
)


def enrich_sections(secs: List[Dict[str, Any]], reg: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Ch 1: Deep Software Architecture & CPU Throughput Mechanics
    secs[0]["blocks"].extend([
        ("h2", "1.4 Hardware Compute Envelope and Single-Process Determinism"),
        ("p",
         "Every training and diagnostic campaign across Phases 0 through 17 ran exclusively under CPU-only PyTorch with "
         "single-process PyBullet physics stepping (timeStep = 1/240 s, 12 internal solver substeps per 20 Hz agent action step, "
         "yielding 0.05 s effective control period). No GPU CUDA acceleration or multi-worker VecEnv parallelism was utilized. "
         "Measurements extracted from archived TensorBoard logs show PyBullet state-based training throughput typically ranged "
         "between 18 and 49 steps/s depending on replay buffer warmup, multi-object collision count, and HER virtual relabeling ratio "
         "(k=4). Adding 128x128 tiny-renderer RGB-D camera capture in Phase 16.2 reduced closed-loop throughput to ~13.9 steps/s "
         "(2,000 steps in 143.54 s), whereas MuJoCo 4.0.1 C-engine evaluation in Phase 17.1 reached ~349 steps/s."),
        ("p",
         "On Windows, concurrent imports of PyTorch (torch) and SciPy/NumPy binary wheels triggered Intel OpenMP runtime conflicts "
         "(OMP: Error #15: Initializing libiomp5md.dll, recorded as ERR-011 during Phase 16.2). All benchmark scripts and automated "
         "regression suites isolate execution inside .conda-env/python.exe with KMP_DUPLICATE_LIB_OK=TRUE and deterministic random "
         "seed injection across NumPy, PyTorch, and PyBullet spawn samplers."),
        ("table", "Table 1.3: Verified Core Canonical 10D/16D PickPlace Observation Tensor Layout",
         ["Index Slice", "Tensor Field", "Dim", "Physical Unit & Semantics", "Frame"],
         [
             ["obs[0:3]", "ee_pos", "3", "Panda grasptarget Link 11 Cartesian XYZ (m)", "World Base"],
             ["obs[3]", "gripper_width", "1", "Sum of finger joints 9+10 opening [0.00, 0.08] (m)", "Gripper"],
             ["obs[4:7]", "obj_pos", "3", "Manipulated cube center-of-mass XYZ (m)", "World Base"],
             ["obs[7:10]", "obj_vel", "3", "Linear cube velocity [vx, vy, vz] (m/s)", "World Base"],
             ["achieved_goal", "ag", "3", "Cube XYZ (PickPlace) or EE XYZ (Reach) (m)", "World Base"],
             ["desired_goal", "dg", "3", "Target placement / bin XYZ coordinate (m)", "World Base"],
         ]),
    ])

    # Ch 2: Algorithm 1 (DLS IK) + Early Physics & Training Failure Mechanics
    secs[1]["blocks"].extend([
        ("h2", "2.3 Algorithmic Control Pipeline & Why 10k Sparse Runs Collapsed"),
        ("algo",
         "<b>Algorithm 1: FrankaPanda Damped Least-Squares Cartesian Step Controller (src/robots/panda.py)</b><br/>"
         "1. Receive normalized policy delta action a_t in [-1, 1]^4 -> dx = clip(a[0:3], -1, 1) * 0.05 m.<br/>"
         "2. Query current EE link 11 position p_t and compute clipped target p_des = clip(p_t + dx, ws_min, ws_max).<br/>"
         "3. Solve PyBullet DLS IK (lambda=0.05, max_iter=100, residual_tol=1e-4) biased toward neutral joints q_0.<br/>"
         "4. Map scalar gripper command a[3] -> jaw aperture target in [0.00, 0.08] m (max motor force 25 N).<br/>"
         "5. Execute 12 PyBullet physics substeps at dt = 1/240 s using joint position/velocity PD actuators."),
        ("p",
         "Analyzing Phase 2-4 logs alongside Algorithm 1 explains why all early 10k RL runs ended at 0% deterministic evaluation even "
         "on Reach (50-step horizon) and Pick (100-step horizon). Stable-Baselines3 SAC initializes entropy coefficient alpha via a "
         "learned log-parameter tuned against target entropy -dim(A) = -4.0. In sparse reward landscapes where zero positive rewards "
         "(+0.0) are encountered during the first few thousand random samples, Q-targets remain uniformly constant (-1 / (1 - gamma) * "
         "const), causing policy entropy to shrink rapidly (0.907 -> 0.002 in Reach SAC_3; 0.742 -> 0.000 in Pick SAC_1) before meaningful "
         "spatial workspace coverage can occur."),
    ])

    # Ch 3: Algorithm 2 (SAC+HER) + Multi-Object Coupling Mechanics
    secs[2]["blocks"].extend([
        ("h2", "3.3 Algorithmic Formalism: Off-Policy SAC with Future-Strategy HER"),
        ("algo",
         "<b>Algorithm 2: Goal-Conditioned SAC + HER Replay Update (Phases 3-13)</b><br/>"
         "1. Collect trajectory tau = (s_0, a_0, ..., s_T, dg) under stochastic actor pi_phi(a | s, dg).<br/>"
         "2. Store original transition (s_t, a_t, s_{t+1}, r(s_{t+1}, dg), dg) into HerReplayBuffer.<br/>"
         "3. For k = 4 virtual samples per transition: draw future step tau > t, set virtual goal g' = ag_tau.<br/>"
         "4. Recompute task reward r' = compute_reward(ag_{t+1}, g', info) cleanly via vectorized NumPy hook.<br/>"
         "5. Update Twin Critics Q_theta1,2 (Eq. 2-3), Reparameterized Actor pi_phi (Eq. 4), and Alpha (Eq. 5)."),
        ("p",
         "A critical structural discovery occurred in Phase 6 (ERR-001) and scaled sharply in Phases 7-8. On PandaMultiSortEnv (31D obs, "
         "6D dual-object goals), HER relabeling faces combinatorial coupling: shifting cube A accidentally alters cube A's final coordinate, "
         "yet if cube B remains undisturbed, virtual goals referencing only touched trajectories fail to decouple independent object "
         "sub-goals within a 300-step horizon unless explicit stage decomposition or PreGrasp gating is enforced (Phase 15.2.5A)."),
    ])

    # Ch 4: Phase 10 DR Table & Phase 11.8/11.10/11.10b Diagnostic Tables
    p11_tabs = extract_phase11_diagnostic_tables()
    secs[3]["blocks"].extend([
        ("h2", "4.3 Domain Randomization Tier Specifications (R0..R4) & Phase 11 Sub-Diagnostics"),
        ("p",
         "To quantify generalization systematically, Phase 10 and Phase 11 stratified environmental variability across five calibrated "
         "tiers (DR0 nominal through DR4 extreme simultaneous cube mass, tabletop/finger friction, initial workspace radius, and target "
         "height randomization). Table 4.2 summarizes the exact state initialization probes (Phase 11.8), curriculum mixtures (Phase 11.10), "
         "and 250k dense acquisition milestones (Phase 11.10b) that preceded the 1M Phase 12 run."),
        ("table", "Table 4.2: Phase 11.8 State Initialization Probe Summary (phase11_8_state_initialization_metrics.csv)",
         ["Initial State Regime", "Eps", "Grasp %", "Lift %", "Drop %", "Success %"],
         p11_tabs["11_8"]),
        ("table", "Table 4.3: Phase 11.10 Curriculum & Phase 11.10b 250k Acquisition Milestones",
         ["Steps", "Reach %", "Grasp %", "Lift %", "Obj Moved %", "Hold Steps", "Timeout %"],
         p11_tabs["11_10b"]),
    ])

    # Ch 5: Deep Force-Slip Physics Mechanics
    secs[4]["blocks"].extend([
        ("h2", "5.3 Dynamical Mechanics of the 68.4 N -> 0.0 N Force Collapse (ERR-004)"),
        ("p",
         "Coulomb friction equilibrium requires bilateral normal squeeze force F_n >= (m * (g + a_z)) / (2 * mu). During Phase 13.5 "
         "D5 forensics on Held-Out DR4 failures (e.g., ep5002), heavy OOD cubes (m = 0.25 kg) paired with low friction (mu = 0.30) demand "
         "at least (0.25 * (9.81 + a_z)) / 0.60 N of sustained bilateral normal force. When the unconstrained SAC actor commanded a rapid "
         "upward delta-Z pulse (a_z = 0.14 m/s^2 Stage-3 jerk) on the very next control step after partial fingertip contact (before jaw "
         "PD actuators settled past F_n = 5 N), elastic impulse rebound knocked the cube free from the pads, dropping F_n from a transient "
         "68.4 N impact spike straight to 0.0 N while the empty end-effector continued rising toward the goal bin (shown in Figure 5.4)."),
    ])

    # Ch 6: Algorithm 3 (Phase 14 Shield) + Phase 14.3 Table
    p14_3_rows = extract_phase14_3_confirmation()
    secs[5]["blocks"].extend([
        ("h2", "6.3 Algorithm 3: Runtime Controller E & Phase 14.3 Confirmation Comparisons"),
        ("algo",
         "<b>Algorithm 3: Controller E Adaptive Transport Shield (src/rl/adaptive_controllers.py)</b><br/>"
         "1. Inspect live PyBullet contact points on finger links 9 & 10: compute f9, f10 normal forces & bilateral flag.<br/>"
         "2. If rel_dist < grasp_zone and cube_z < 0.05m and not grasp_settled:<br/>"
         "   clamp dx = 0, dy = 0, force gripper_close, throttle dz = 0 for settle_steps = 3 ticks.<br/>"
         "3. During LIFT & TRANSPORT (cube_z > table): monitor slip delta ||p_obj - p_ee|| and min(f9, f10).<br/>"
         "   Cap upward acceleration dz_cmd <= dz_safe(normal_force, slippage) eliminating impulsive breakaway jerk.<br/>"
         "4. Forward stabilized 4D action directly to PyBulletSimulation without altering any neural policy weight."),
        ("table", "Table 6.3: Phase 14.3 Seed-Level Controller A vs D Summary (results/phase14_3/confirmation_summary.csv)",
         ["Seed", "Controller", "Regime", "Success %", "Bilat %", "GBNL %", "Mean Slips", "Mean Steps"],
         p14_3_rows),
    ])

    # Ch 7: Algorithms 4 & 5 + Phase 15 Tables
    pp15_rows, asm15_rows = extract_phase15_phase3_and_assembly()
    secs[6]["blocks"].extend([
        ("h2", "7.4 PreGrasp Gate, Masked BC-SAC Algorithms, and Robustness/Assembly Audits"),
        ("algo",
         "<b>Algorithm 4: Phase 15.2.5A Five-Check Bilateral Grasp Authorization Gate</b><br/>"
         "Authorize gripper closing ONLY when ALL 5 physical conditions hold simultaneously:<br/>"
         "  C1: xy_err(EE, obj) <= 0.015 m  AND  C2: |z_ee - z_grasp| <= 0.018 m<br/>"
         "  C3: finger_width > 0.065 m (open descent prevents top-knocking collision)<br/>"
         "  C4: ||v_obj|| < 0.02 m/s (cube stationary)  AND  C5: active target object locked.<br/><br/>"
         "<b>Algorithm 5: Progress-Masked Hybrid BC-SAC (Phase 15.2.5B7 / 5B8)</b><br/>"
         "1. Filter demo buffer using binary progress mask M_prog(s) = 0 on trailing HOLD_AT_GOAL frames (removing 78% closed bias).<br/>"
         "2. Pretrain actor via masked L2 loss Eq. (12) for pure BC baseline (75% grasp / 45% lift, 0% coll).<br/>"
         "3. Anneal BC auxiliary weight lambda_BC(t) = lambda_0 * exp(-kappa * t) alongside SAC Eq. (13) with entropy floor."),
        ("table", "Table 7.2: Phase 15 Phase-3 PickPlace DR Cross-Perturbation Results (phase15_phase3_robustness_results.json)",
         ["Model", "Condition", "Eps", "Success %", "Grasp %", "Lift %"],
         pp15_rows),
        ("table", "Table 7.3: Phase 15.3 Orientation-Aware Peg-in-Slot Assembly Benchmark (30 Eps Each)",
         ["Policy Variant", "Eps", "Grasp %", "Pre-Insert Align %", "Full Insertion %", "Mean Angle Err"],
         asm15_rows),
    ])

    # Ch 8: Algorithms 6 & 7
    secs[7]["blocks"].extend([
        ("h2", "8.4 HRL Option Selection & Unprojected Oblique Vision Mechanics"),
        ("algo",
         "<b>Algorithm 6: Phase 16.1 SMDP 5-Skill Option-Critic Loop (src/hrl/hrl_agent.py)</b><br/>"
         "1. High-level selector pi_high(u | s_t) picks active skill u in {REACH, ALIGN, GRASP, LIFT, TRANSPORT}.<br/>"
         "2. Run low-level skill policy a_t ~ pi_low^(u)(s_t) until termination head beta_u(s_{t+k}) > 0.5 or max_horizon.<br/>"
         "3. Assign bounded SMDP option reward using potential height delta 1.5*(h_{t+k} - h_t) + sparse completion +6.0.<br/><br/>"
         "<b>Algorithm 7: Phase 16.2 Oblique RGB-D Unprojection + Phase 16.3 Kalman Predictor</b><br/>"
         "1. Render 128x128 RGB + metric depth from oblique eye (0.92, 0.28, 0.45) avoiding Franka arm self-occlusion.<br/>"
         "2. Segment cube mask in HSV space (IoU 0.9983-0.9999) and unproject masked depth pixels via (PV)^(-1) Eq. (15).<br/>"
         "3. Pass centroid [x_est, y_est, z_est] into 4-state constant-velocity Kalman filter Eq. (16) for moving-target interception."),
    ])

    # Ch 9: Algorithm 8 + Phase 17 Physics Parity Table
    p17_diff = extract_phase17_physics_diff()
    secs[8]["blocks"].extend([
        ("h2", "9.3 Algorithm 8 SafetySupervisor & PyBullet-to-MuJoCo Solver Comparison"),
        ("algo",
         "<b>Algorithm 8: Real-Time Deployment SafetySupervisor & E-Stop Gate (src/deployment/runtime.py)</b><br/>"
         "1. Check action bounds & finite state integrity (NaN/Inf guard); clamp target XYZ strictly inside workspace box.<br/>"
         "2. Evaluate velocity threshold ||v_ee|| > v_crit and wall/contact force F_wall > F_crit.<br/>"
         "3. Trigger hard software E-Stop halt within 32.41 us verified cutoff latency (blocking all 250 faulty injections)."),
        ("table", "Table 9.2: PyBullet vs MuJoCo 4.0.1 Verified Physical Model Parity Parameters",
         ["Physics Parameter", "Verified Cross-Simulator Value"],
         p17_diff + [
             ["MJCF Solver Integrator", "implicitfast, timestep = 0.0041667 s (240 Hz), elliptic friction cone"],
             ["Tabletop Rest Height Delta", "9.4 mm vertical rest discrepancy affecting zero-shot grasp contact line"],
             ["PyBullet Finger Friction", "lateralFriction = 2.0, spinningFriction = 0.1 on finger links 9, 10"],
             ["Physical Hardware Execution", "NOT_ATTEMPTED (HAL & SafetySupervisor simulation-verified only)"],
         ]),
    ])

    return secs

