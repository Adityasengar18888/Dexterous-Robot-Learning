"""Part I (Foundations) & Part II (Mathematical and Algorithmic Foundations) content
for the redesigned publication-quality robotics monograph.
Every concept follows a first-principles academic exposition with real math definitions,
exact repository file mappings, concrete simulated Panda examples, real-world engineering
context, limitations/alternatives, and verified experimental findings.
"""

from __future__ import annotations

from typing import Any, Dict, List


def build_front_and_parts_1_2() -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # PART I: FOUNDATIONS
    # ------------------------------------------------------------------
    sections.append({
        "id": "part1_intro",
        "part": "PART I — FOUNDATIONS",
        "title": "Chapter 1: Robotic Manipulation, Project Objectives, and Historical Scope",
        "blocks": [
            ("h2", "1.1 The Fundamental Challenge of Contact-Rich Manipulation"),
            ("p",
             "Autonomous robotic manipulation differs profoundly from unconstrained motion planning or classical game benchmarks. "
             "When a 7-degree-of-freedom (7-DOF) serial manipulator approaches, grasps, lifts, transports, and inserts rigid objects in a "
             "tabletop workspace, control policies must negotiate discontinuous contact transitions, frictional shear limits across millimeter "
             "jaw apertures, non-convex kinematic joint bounds, partial observation noise, and severe long-horizon credit-assignment delays. "
             "A millimeter lateral misalignment or an excessive 0.14 m/s^2 vertical acceleration pulse at grasp closure can instantly break "
             "bilateral friction equilibrium and drop a payload."),
            ("p",
             "The Dexterous-Robot-Learning project investigates how off-policy reinforcement learning (Soft Actor-Critic paired with Hindsight "
             "Experience Replay), dense potential shaping, progress-masked Behavioral Cloning, Hierarchical Reinforcement Learning (HRL), "
             "oblique RGB-D computer vision, and physics-informed inference shields can be systematically composed to achieve reliable "
             "simulated manipulation on the Franka Emika Panda platform under strict CPU-only single-process compute constraints."),
            ("h2", "1.2 Verified Research Scope (Phases 1–17 Active; Phase 18 Proposed)"),
            ("p",
             "Project development spans Phases 0 through 17 in verifiable simulation artifacts (PyBullet primary physics engine and MuJoCo 4.0.1 "
             "cross-simulator verification), alongside a refactored web-based simulation workstation built over FastAPI and Three.js. Phases 1–8 "
             "are reconstructed strictly from Git history (earliest commits 3a345f3, a5cc06b, 7ad9848 dated 2026-10-03), YAML snapshots, and "
             "TensorBoard scalars; where standalone historical artifacts (such as standalone Phase 5 notes) do not exist in the repository, they "
             "are explicitly documented as NOT RECOVERABLE FROM AVAILABLE EVIDENCE. Phase 17.3 hardware execution on physical Franka hardware "
             "was recorded as NOT_ATTEMPTED, and Phase 18 (Multi-Agent Cooperative Manipulation) represents proposed future research scope only."),
            ("photo", "publication/assets/photographs/photo_03_research_lab_arm.jpg",
             "Figure 1.1: Real-world laboratory compliant 7-DOF manipulator grasping a machined component above an industrial test rail. "
             "Source: Wikimedia Commons (File:Baxter_1.jpg), Author: Salammbo31, License: CC BY-SA 4.0. Note: Illustrates general laboratory manipulation context; the current project evaluates Franka Panda strictly in simulation."),
            ("h2", "1.3 Franka Emika Panda Architecture and PyBullet Simulation Fundamentals"),
            ("p",
             "In real industrial and university laboratories, torque-controlled 7-DOF collaborative manipulators such as the Franka Emika Panda "
             "provide kinematic redundancy: seven revolute axes allow reaching an arbitrary 6-DOF end-effector pose (position [x, y, z] and "
             "orientation [qx, qy, qz, qw]) while exploiting the extra null-space degree of freedom to avoid joint stops and singular elbow locking. "
             "Within simulation (`src/robots/panda.py`, wrapped cleanly in `robotics/robots/franka_panda/robot_model.py`), the official URDF "
             "`panda.urdf` models arm joints [0..6] with maximum effort limits [87, 87, 87, 87, 12, 12, 12] N*m, finger joints [9, 10] with 25 N "
             "maximum squeeze force over [0.00, 0.08] m aperture, and grasp target frame link 11 (`panda_grasptarget`)."),
            ("fig", "publication/assets/diagrams/diag_03_panda_kinematics_frames.png",
             "Figure 1.2: Franka Emika Panda 7-DOF kinematic chain, effort ratings, joint limits, and neutral reference posture q_0."),
            ("h2", "1.4 Real-World Applications and Engineering Relevance"),
            ("p",
             "Manipulator architectures of this class underpin modern flexible automation across: (1) Industrial bin picking and machine tending "
             "(extracting randomized parts from shallow trays); (2) Warehouse order fulfillment and parcel sorting (rapid sequential color/SKU "
             "binning analogous to Phase 15.2 MultiSort); (3) Precision manufacturing assembly (connector mating and peg-in-slot insertion under "
             "lateral wall forces analogous to Phases 15.3 and 16.4); and (4) Healthcare and assistive robotics (adaptive object transfer near "
             "fragile surfaces). Deploying simulated policies onto hardware in any of these sectors would require empirical system identification, "
             "sensor calibration, real-time FCI driver certification (1 kHz control loop), and hardware safety interlocks beyond current simulation scope."),
        ],
    })

    # ------------------------------------------------------------------
    # PART II: MATHEMATICAL & ALGORITHMIC FOUNDATIONS
    # ------------------------------------------------------------------
    sections.append({
        "id": "part2_kinematics_contact",
        "part": "PART II — MATHEMATICAL & ALGORITHMIC FOUNDATIONS",
        "title": "Chapter 2: Kinematics, Jacobian Control, Contact Mechanics, and Coulomb Friction",
        "blocks": [
            ("h2", "2.1 Forward Kinematics, Inverse Kinematics, and Jacobian Control"),
            ("p",
             "A. Definition & Purpose: Forward Kinematics (FK) maps joint coordinates q in R^7 into end-effector Cartesian pose x = FK(q) in SE(3) "
             "through homogeneous link transforms T_0^1(q_1)...T_6^7(q_7). Conversely, Inverse Kinematics (IK) determines joint increments dq achieving "
             "a desired Cartesian displacement dx, solving the problem of commanding intuitive 3D task goals [dx, dy, dz] with a 7-DOF arm."),
            ("p",
             "B. Why Selected & Mathematical Formulation: Rather than asking RL policies to directly regress 7 coupled joint torques (which requires "
             "orders of magnitude more exploration samples), this repository exposes compact 4D/5D task-space actions mapped via PyBullet's "
             "Damped Least-Squares (DLS) Jacobian pseudoinverse (Equation 1–2). The Jacobian J(q) in R^(6x7) relates joint velocities dq/dt to spatial "
             "twist dx/dt = J(q) dq/dt. Pure pseudoinverses J^T (J J^T)^(-1) explode near kinematic singularities where det(J J^T) -> 0; adding damping "
             "lambda = 0.05 stabilizes inversion while projecting neutral posture regularization (I - J# J) k_0 (q_0 - q_t) into the 1-DOF null space."),
            ("eq", "publication/assets/equations/eq_01.png", "Equation (1): Damped Least-Squares Inverse Kinematics with null-space posture regularization."),
            ("eq", "publication/assets/equations/eq_02.png", "Equation (2): Instantaneous Jacobian velocity mapping for 7-DOF spatial twist."),
            ("fig", "publication/assets/diagrams/diag_04_jacobian_and_control.png",
             "Figure 2.1: Jacobian inversion, damping lambda=0.05 conditioning, and null-space posture regularization in FrankaPanda.calculate_ik."),
            ("code", "src/robots/panda.py :: FrankaPanda.calculate_ik & clip_to_workspace",
             "clipped_pos = self.clip_to_workspace(target_position)\n"
             "all_ik_angles = p.calculateInverseKinematics(\n"
             "    bodyUniqueId=self.body_id, endEffectorLinkIndex=self.ee_link_idx,\n"
             "    targetPosition=list(clipped_pos), targetOrientation=orn,\n"
             "    lowerLimits=self.joint_lower_limits, upperLimits=self.joint_upper_limits,\n"
             "    jointRanges=self.joint_ranges, restPoses=self.neutral_joint_positions,\n"
             "    maxNumIterations=100, residualThreshold=1e-4, physicsClientId=self.cid,\n"
             ")\n"
             "# Clips XYZ within ws_x=[0.25,0.75], ws_y=[-0.35,0.35], ws_z=[0.02,0.50]; solves DLS IK in <1ms."),
            ("p",
             "C. Practical Example, Alternatives & Real-World Context: In Phase 1, 5 validation waypoints confirmed 0.7 mm to 4.5 mm tracking accuracy. "
             "Alternatives include QP whole-body velocity controllers (OSQP) and operational-space torque control, which offer superior dynamic impact "
             "compliance at the cost of requiring accurate inertia matrices. In real manufacturing and surgical arms, damped Jacobian IK forms the "
             "universal inner kinematics layer beneath high-level planners."),
            ("h2", "2.2 Contact Mechanics, Parallel-Jaw Grasping, and Coulomb Friction Cones"),
            ("p",
             "A. Definition & Internal Physics: Grasp stability during lifting is governed by unilateral normal squeeze forces F_{n,9}, F_{n,10} applied "
             "by Panda finger links 9 and 10 onto opposite cube faces, creating shear load capacity limited by Coulomb's friction cone |f_s| <= mu * F_n "
             "(Equation 9). To elevate an object of mass m under upward liftoff acceleration a_z without slip, total bilateral squeeze force must obey "
             "F_{n,req} >= m(g + a_z) / (2 * mu)."),
            ("eq", "publication/assets/equations/eq_09.png", "Equation (9): Coulomb friction cone limit and minimum normal squeeze force invariant."),
            ("fig", "publication/assets/diagrams/diag_05_contact_and_friction_cones.png",
             "Figure 2.2: (a) Bilateral gripper free-body force diagram; (b) Minimum required normal force across friction mu showing OOD slip cliff."),
            ("photo", "publication/assets/photographs/photo_01_parallel_gripper.jpg",
             "Figure 2.3: Real-world parallel-jaw robotic gripper and wrist vision camera in a university research workstation. "
             "Source: Wikimedia Commons (File:Baxter_robot's_gripper.JPG), Author: Gustavo A. Casan, License: CC BY-SA 4.0."),
            ("p",
             "B. Evidence & Project Lessons Learned: Two major failures traced directly back to Equation (9). First, in Phase 6 (ERR-001), default "
             "PyBullet finger friction could not hold a 0.10 kg cube; configuring lateralFriction=2.0 and spinningFriction=0.1 fixed scripted pick-and-place. "
             "Second, in Phase 13.5, heavy OOD cubes (0.25 kg) under low friction (mu=0.3) combined with an impulsive a_z=0.14 m/s^2 liftoff jerk "
             "spiked normal force to 68.4 N before snapping contact to 0.0 N (30.5% GRASP_BUT_NO_LIFT failure rate), directly inspiring Phase 14 "
             "Controller E's 3-tick settle + lift-rate safety shield."),
        ],
    })

    sections.append({
        "id": "part2_rl_algorithms",
        "part": "PART II — MATHEMATICAL & ALGORITHMIC FOUNDATIONS",
        "title": "Chapter 3: Reinforcement Learning, SAC, HER, Imitation, Randomization, and HRL",
        "blocks": [
            ("h2", "3.1 Goal-Conditioned Markov Decision Processes and Soft Actor-Critic (SAC)"),
            ("p",
             "A. Definition & Role: Task execution is modeled as a Goal-Conditioned MDP (S, A, G, P, r, gamma), where observations follow the Gymnasium "
             "GoalEnv Dict schema {observation, achieved_goal, desired_goal}. Soft Actor-Critic (SAC) is an off-policy maximum-entropy actor-critic "
             "algorithm that simultaneously maximizes expected discounted return and expected policy entropy H(pi(.|s,g))."),
            ("p",
             "B. Internal Mechanics & Equations: SAC trains twin Q-critics Q_{theta_1}, Q_{theta_2} against Polyak-averaged targets (tau=0.05 early; "
             "0.005 late) using clipped double-Q minimization to curb overestimation bias (Equations 3–4). The stochastic actor samples continuous 4D "
             "actions via the reparameterization trick a = tanh(mu_phi(s,g) + sigma_phi(s,g)*eps) and optimizes Equation (5), while automatic "
             "Lagrange dual updates adapt temperature alpha toward target entropy H_tgt = -dim(A) = -4 (Equation 6)."),
            ("eq", "publication/assets/equations/eq_03.png", "Equation (3): Clipped double Q-learning critic regression objective."),
            ("eq", "publication/assets/equations/eq_04.png", "Equation (4): Soft Bellman backup target with entropy bonus."),
            ("eq", "publication/assets/equations/eq_05.png", "Equation (5): Reparameterized SAC actor objective."),
            ("eq", "publication/assets/equations/eq_06.png", "Equation (6): Automatic entropy temperature dual minimization."),
            ("p",
             "C. Alternatives, Limitations & Evidence: On-policy methods like PPO discard off-policy replay transitions and lack native HER compatibility; "
             "deterministic off-policy methods like DDPG/TD3 require external exploration noise tuning. Across 10k sparse runs in Phases 2–8, SAC "
             "suffered premature alpha collapse (0.907 -> 0.002), demonstrating that without informative positive gradients, entropy scheduling alone "
             "cannot bridge long manipulation horizons."),
            ("h2", "3.2 Hindsight Experience Replay (HER) and Sparse vs. Dense Reward Shaping"),
            ("p",
             "A. Mechanism & Static-Goal Exploration Trap (ERR-003): Under binary sparse rewards r = -I(||p_obj - p_goal|| > 0.05 m) (Equation 7), "
             "random policies almost never hit desired goals. Future-strategy HER (k=4, Equation 8) retrospectively substitutes achieved states g' = ag_tau "
             "(tau > t) into replay mini-batches, turning failed trajectories into valid goal-reaching examples. However, Phase 11 (25 models x 100k "
             "steps = 0.0% across 5,000 evaluation episodes) uncovered a critical failure mode: when an agent fails even to move the object off the table, "
             "every future achieved goal g' equals the static initial resting coordinate, reinforcing stationary hovering."),
            ("eq", "publication/assets/equations/eq_07.png", "Equation (7): Binary sparse Euclidean tolerance reward."),
            ("eq", "publication/assets/equations/eq_08.png", "Equation (8): Future-strategy HER virtual goal relabeling (k=4)."),
            ("fig", "publication/assets/diagrams/diag_06_sac_and_her_architecture.png",
             "Figure 3.1: Off-policy SAC actor-critic dataflow coupled with HER k=4 virtual relabeling and the Phase 15.2.5B1 lift-gradient guard."),
            ("p",
             "B. Dense Staged Shaping & Continuous Vertical Potentials: Breaking the static trap in Phase 12 required staged dense shaping (Equation 10) "
             "uniting approach attraction, bilateral contact bonuses, lift altitude increments, goal transport potential, and collision penalties. "
             "Similarly, Phase 15.2.5B1 overcame an XY-gated vertical hover deadlock at z=0.50 m by replacing cylinder-gated Z rewards with an un-gated "
             "continuous vertical potential Phi_z = -2.0 * clip(|z_ee - z_grasp|, 0, 1) (Equation 11)."),
            ("eq", "publication/assets/equations/eq_10.png", "Equation (10): Multi-stage dense manipulation shaping function (Phase 12)."),
            ("eq", "publication/assets/equations/eq_11.png", "Equation (11): Continuous un-gated vertical approach potential Phi_z (Phase 15.2.5B1)."),
            ("fig", "publication/assets/diagrams/diag_14_reward_shaping_potential.png",
             "Figure 3.2: (a) Sparse plateau vs. dense gradient; (b) Solving the z=0.50 m hover stall via un-gated vertical potential Phi_z."),
            ("h2", "3.3 Progress-Masked Behavioral Cloning, Hybrid BC-SAC, Domain Randomization & HRL"),
            ("p",
             "Behavioral Cloning (BC) initializes actor policies from expert demonstrations via supervised regression. Phase 15.2.5B.1 proved that naive "
             "BC over complete scripted episodes overfits trailing stationary HOLD_AT_GOAL frames (78.4% closed-jaw bias), causing premature air-pinching. "
             "Introducing balanced open-approach demos (70.1% open descent) and binary progress masking M_prog(s) (Equations 12–13) propelled pure masked BC "
             "(5B7) to 75% grasp and 45% lift across 100 episodes with 0% table collisions, while Decaying Hybrid BC-SAC (5B8) reached 66% / 26%."),
            ("eq", "publication/assets/equations/eq_12.png", "Equation (12): Progress-masked Behavioral Cloning L2 objective."),
            ("eq", "publication/assets/equations/eq_13.png", "Equation (13): Exponentially decaying Hybrid BC-SAC objective with entropy floor."),
            ("fig", "publication/assets/diagrams/diag_07_bc_and_hybrid_bcsac.png",
             "Figure 3.3: Transition from unbalanced closed-jaw BC bias (5B1) to balanced masked BC (5B7) and decaying Hybrid BC-SAC (5B8)."),
            ("p",
             "Domain Randomization (DR0–DR4, Figure 3.4) varies cube mass (0.04–0.25 kg), Coulomb friction (0.3–1.5), and spawn/goal geometry so policies "
             "do not overfit fixed nominal constants. Hierarchical RL (Phase 16.1, Equation 14) further mitigates horizon complexity by training a "
             "high-level discrete meta-policy pi_high(u|s) over five modular closed-loop skill policies {REACH, ALIGN, GRASP, LIFT, TRANSPORT}."),
            ("fig", "publication/assets/diagrams/diag_08_domain_randomization_dr0_4.png",
             "Figure 3.4: Calibrated physical mass and friction parameter spans across Domain Randomization tiers DR0 through DR4."),
            ("eq", "publication/assets/equations/eq_14.png", "Equation (14): Five-skill hierarchical option policy marginalization (Phase 16.1)."),
        ],
    })
    return sections

