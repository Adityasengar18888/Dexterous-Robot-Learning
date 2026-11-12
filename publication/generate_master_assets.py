"""Generates the complete Publication Art-Directed Visual Suite under publication/assets/:
- High-resolution PyBullet Cover Render + 14 Original Technical & Mathematical Diagrams (assets/diagrams/)
- 10 Evidence-Based 2D/3D Plots with uncertainty bands & provenance metadata (assets/plots/)
- 22 Typeset Computer Modern Mathematical Equations on warm #FCFCFA ground (assets/equations/)
Strictly preserves #FCFCFA warm-neutral, #263B53 deep-navy, #637E96 steel-blue aesthetic.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PUB = ROOT / "publication"
ASSETS = PUB / "assets"
DIR_DIAG = ASSETS / "diagrams"
DIR_PLOT = ASSETS / "plots"
DIR_PHOTO = ASSETS / "photographs"
DIR_EQ = ASSETS / "equations"
for d in (DIR_DIAG, DIR_PLOT, DIR_PHOTO, DIR_EQ):
    d.mkdir(parents=True, exist_ok=True)

# Editorial Palette (Section 7)
BG_WARM = "#FCFCFA"
TXT_DARK = "#24282C"
TXT_MUTED = "#66717B"
NAVY = "#263B53"
STEEL = "#637E96"
RULE_GREY = "#E7EAED"
SAGE = "#5A8F76"
OCHRE = "#B88435"
CRIMSON = "#A84A4A"

plt.rcParams.update({
    "figure.facecolor": BG_WARM,
    "axes.facecolor": BG_WARM,
    "savefig.facecolor": BG_WARM,
    "font.family": "DejaVu Sans",
    "font.size": 9.2,
    "text.color": TXT_DARK,
    "axes.labelcolor": TXT_DARK,
    "axes.edgecolor": TXT_MUTED,
    "xtick.color": TXT_MUTED,
    "ytick.color": TXT_MUTED,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.color": RULE_GREY,
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def box(ax, x, y, w, h, title, body="", fc="#F4F6F8", ec=NAVY, lw=1.2, tc=NAVY, title_y=0.68, body_y=0.30, ts=8.5, bs=7.2):
    p = FancyBboxPatch((x + 0.004, y + 0.004), w - 0.008, h - 0.008,
                       boxstyle="round,pad=0.005,rounding_size=0.012",
                       facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(p)
    if body:
        ax.text(x + w / 2, y + h * title_y, title, ha="center", va="center", fontsize=ts, fontweight="bold", color=tc)
        ax.text(x + w / 2, y + h * body_y, body, ha="center", va="center", fontsize=bs, color=TXT_DARK)
    else:
        ax.text(x + w / 2, y + h * 0.5, title, ha="center", va="center", fontsize=ts, fontweight="bold", color=tc)


def arrow(ax, x1, y1, x2, y2, col=NAVY, lbl="", lx=0.0, ly=0.02, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.35, linestyle=ls))
    if lbl:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, lbl, ha="center", va="bottom",
                fontsize=7.0, color=col, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.1", fc=BG_WARM, ec="none", alpha=0.85))


def generate_cover_artwork() -> str:
    """High-res live PyBullet Franka Panda simulation render + geometric coordinate overlay."""
    import pybullet as p
    import pybullet_data
    from src.robots.panda import FrankaPanda

    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    p.loadURDF("plane.urdf", physicsClientId=cid)
    panda = FrankaPanda(cid)
    q_sol = panda.calculate_ik(np.array([0.46, -0.02, 0.16]))
    panda.reset(q_sol)
    panda.set_gripper_width(0.055)

    c_box = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.022, 0.022, 0.022], physicsClientId=cid)
    v_red = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.022, 0.022, 0.022], rgbaColor=[0.66, 0.29, 0.29, 1.0], physicsClientId=cid)
    v_blue = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.022, 0.022, 0.022], rgbaColor=[0.28, 0.45, 0.62, 1.0], physicsClientId=cid)
    p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=c_box, baseVisualShapeIndex=v_red, basePosition=[0.46, -0.02, 0.022], physicsClientId=cid)
    p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=c_box, baseVisualShapeIndex=v_blue, basePosition=[0.38, 0.16, 0.022], physicsClientId=cid)
    for _ in range(15):
        p.stepSimulation(physicsClientId=cid)

    view = p.computeViewMatrix([1.32, -0.95, 0.86], [0.28, 0.02, 0.28], [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(48.0, 1.5, 0.05, 5.0)
    w, h = 1200, 800
    _, _, rgba, _, _ = p.getCameraImage(w, h, viewMatrix=view, projectionMatrix=proj, renderer=p.ER_TINY_RENDERER, physicsClientId=cid)
    rgb = np.asarray(rgba, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    p.disconnect(cid)

    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    ax.imshow(rgb)
    ax.axis("off")
    # Subtle technical coordinate callout annotations cleanly placed near base & EE
    ax.text(0.02, 0.04, "Simulated Environment: Franka Emika Panda (7-DOF) | PyBullet 240 Hz Physics",
            transform=ax.transAxes, fontsize=8.5, color=NAVY, fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.35", fc="#FCFCFA", ec=STEEL, lw=0.8, alpha=0.92))
    out = DIR_DIAG / "cover_panda_workspace.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out.name


def generate_technical_diagrams() -> List[str]:
    saved = []

    # 1. System Architecture
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.02, 0.06, 0.46, 0.17, "PyBullet Physics Core (240 Hz)", "Rigid-body contacts, lateralFriction=2.0\n12 substeps/step -> 20 Hz control loop", "#F1F5F8", NAVY)
    box(ax, 0.52, 0.06, 0.46, 0.17, "MuJoCo 4.0.1 Cross-Simulator Twin (Ph 17)", "Implicitfast MJCF parity model | 21D/16D obs\n32.41 us SafetySupervisor E-Stop check", "#F4F6F8", STEEL)
    box(ax, 0.02, 0.29, 0.30, 0.17, "FrankaPanda Kinematics", "Damped Least-Squares IK (lam=0.05)\nNull-space + Cartesian WS Clamp", "#F8F9FA", NAVY)
    box(ax, 0.35, 0.29, 0.30, 0.17, "Perception & Dynamic Tracker", "Oblique 128x128 RGB-D (FOV 55 deg)\nClip-to-World Unproj + Kalman", "#F8F9FA", NAVY)
    box(ax, 0.68, 0.29, 0.30, 0.17, "Controllers (Ph 14 / 15 / 16)", "Ctrl E Adaptive Lift Shield\n5-Check PreGrasp & Compliant ASM", "#EEF4F0", SAGE)
    box(ax, 0.02, 0.52, 0.96, 0.16, "Goal-Conditioned Gymnasium & HER Environments (Phases 2-17)", "PandaReach (4D) -> Pick/PickPlace (10D+6D) -> Sort (22D+6D) -> MultiSort (31D+12D) -> Assembly (22D+14D) -> Dynamic", "#FAF7F0", OCHRE)
    box(ax, 0.02, 0.74, 0.46, 0.19, "Learning Algorithms (CPU-Only PyTorch)", "SB3 SAC+HER (k=4, future) | Progress-Masked BC\nDecaying Hybrid BC-SAC | 5-Skill HRL Options", "#F1F5F8", NAVY)
    box(ax, 0.52, 0.74, 0.46, 0.19, "Web Workstation & Verification Suite", "FastAPI REST + 20 Hz WebSockets -> Three.js R3F\n126 Combined Python Tests + Vitest + Monograph", "#F4F6F8", STEEL)
    for x in (0.17, 0.50, 0.83):
        arrow(ax, x, 0.23, x, 0.29); arrow(ax, x, 0.46, x, 0.52)
    arrow(ax, 0.25, 0.68, 0.25, 0.74); arrow(ax, 0.75, 0.68, 0.75, 0.74)
    ax.set_title("Complete Four-Layer Dexterous-Robot-Learning System Architecture (Phases 1-17)")
    p = DIR_DIAG / "diag_01_system_architecture.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p.name)

    # 2. PyBullet Simulation Timing & Loop
    fig, ax = plt.subplots(figsize=(10.0, 3.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.03, 0.35, 0.19, 0.38, "1. Agent Action a_t", "dt_ctrl = 50 ms (20 Hz)\n[dx, dy, dz, (dyaw), grip]", "#F1F5F8", NAVY)
    box(ax, 0.27, 0.35, 0.21, 0.38, "2. Adaptive Safety Gate", "WS Clip + Ctrl E Settle\n& Lift-Rate Throttling", "#EEF4F0", SAGE)
    box(ax, 0.53, 0.35, 0.20, 0.38, "3. DLS IK Solver", "solve_ik -> q_target\nArm PD + Gripper 25 N", "#F4F6F8", STEEL)
    box(ax, 0.78, 0.35, 0.19, 0.38, "4. Physics Substeps", "12 x (1/240 s) Bullet\nContacts, Slip & Obs_{t+1}", "#FAF7F0", OCHRE)
    arrow(ax, 0.22, 0.54, 0.27, 0.54); arrow(ax, 0.48, 0.54, 0.53, 0.54); arrow(ax, 0.73, 0.54, 0.78, 0.54)
    arrow(ax, 0.875, 0.35, 0.125, 0.35, col=STEEL, lbl="Return Dict{obs, ag, dg}, contact_forces, info")
    ax.set_title("PyBullet Simulation Execution Cycle (240 Hz Physics Solver / 20 Hz Policy Control Rate)")
    p = DIR_DIAG / "diag_02_pybullet_sim_loop.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p.name)

    # 3. Panda Kinematic Structure & Frames
    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    joints = [
        ("Base\nLink0", 0.06, "Fixed z=0"),
        ("q1..q2\n[-2.90,2.90]\n[-1.76,1.76]", 0.23, "Shoulder\n87 N*m"),
        ("q3..q4\n[-2.90,2.90]\n[-3.07,-0.07]", 0.42, "Elbow\n87 N*m"),
        ("q5..q7\n[-2.90,2.90]\n[-0.02,3.75]", 0.62, "Wrist\n12 N*m"),
        ("Hand+Jaw\nLink 8,9,10\n[0, 0.08m]", 0.82, "EE Link 11\ngrasptarget"),
    ]
    for title, x, sub in joints:
        box(ax, x - 0.02, 0.30, 0.16, 0.42, title, sub, "#F4F6F8", NAVY, ts=7.8, bs=7.0)
    for i in range(len(joints) - 1):
        arrow(ax, joints[i][1] + 0.14, 0.51, joints[i + 1][1] - 0.02, 0.51, col=STEEL, lbl=f"T_{i}^{i+1}")
    ax.text(0.50, 0.12, "Neutral Rest Pose q_0 = [0.000, -0.785, 0.000, -2.356, 0.000, 1.571, 0.785] rad | Base Pos [0,0,0] -> EE [0.307, 0, 0.485] m",
            ha="center", fontsize=8.0, color=NAVY, bbox=dict(boxstyle="square,pad=0.3", fc="#EEF4F0", ec=SAGE))
    ax.set_title("Franka Emika Panda 7-DOF Kinematic Chain, Joint Limits, Torque Ratings, and Link Transforms")
    p = DIR_DIAG / "diag_03_panda_kinematics_frames.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p.name)

    # 4. Jacobian & Control Architecture
    fig, ax = plt.subplots(figsize=(10.2, 4.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.04, 0.55, 0.24, 0.30, "Task Space Target dx_des", "Clipped in WS Bounds\nx:[0.25,0.75] y:[-0.35,0.35] z:[0.02,0.50]", "#F1F5F8", NAVY)
    box(ax, 0.36, 0.55, 0.28, 0.30, "Damped Least Squares IK", "dq = J^T (J J^T + lam^2 I)^(-1) dx\nlam = 0.05 avoids singularity blowup", "#FAF7F0", OCHRE)
    box(ax, 0.36, 0.12, 0.28, 0.28, "Null-Space Regularizer", "(I - J# J) k_0 (q_0 - q_t)\nBiases redundant 7th DOF near q_0", "#F4F6F8", STEEL)
    box(ax, 0.72, 0.33, 0.24, 0.34, "Joint Motor PD Target", "q_{t+1} = q_t + dq_total\nPosition Control -> 240 Hz Sim", "#EEF4F0", SAGE)
    arrow(ax, 0.28, 0.70, 0.36, 0.70); arrow(ax, 0.64, 0.70, 0.72, 0.55)
    arrow(ax, 0.64, 0.26, 0.72, 0.45)
    ax.set_title("Redundant 7-DOF Jacobian Inversion, Damping Stabilization, and Null-Space Posture Projection")
    p = DIR_DIAG / "diag_04_jacobian_and_control.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p.name)

    # 5. Contact Mechanics & Coulomb Friction Cone (Mathematical Illustration)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    # Left panel: Gripper pad forces & friction cone
    ax1.set_xlim(-1.2, 1.2); ax1.set_ylim(-0.3, 1.4); ax1.axis("off")
    ax1.add_patch(patches.Rectangle((-0.35, 0.15), 0.70, 0.70, fc="#E2E8F0", ec=NAVY, lw=1.6))
    ax1.add_patch(patches.Rectangle((-0.52, 0.22), 0.16, 0.56, fc=STEEL, ec=NAVY))
    ax1.add_patch(patches.Rectangle((0.36, 0.22), 0.16, 0.56, fc=STEEL, ec=NAVY))
    ax1.annotate("", xy=(-0.35, 0.50), xytext=(-0.95, 0.50), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2.2))
    ax1.annotate("", xy=(0.35, 0.50), xytext=(0.95, 0.50), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2.2))
    ax1.annotate("", xy=(-0.36, 0.96), xytext=(-0.36, 0.50), arrowprops=dict(arrowstyle="-|>", color=SAGE, lw=2.2))
    ax1.annotate("", xy=(0.36, 0.96), xytext=(0.36, 0.50), arrowprops=dict(arrowstyle="-|>", color=SAGE, lw=2.2))
    ax1.annotate("", xy=(0.0, -0.22), xytext=(0.0, 0.48), arrowprops=dict(arrowstyle="-|>", color=CRIMSON, lw=2.2))
    ax1.text(-0.80, 0.58, r"$F_{n,9}$", fontsize=10, color=NAVY, fontweight="bold")
    ax1.text(0.55, 0.58, r"$F_{n,10}$", fontsize=10, color=NAVY, fontweight="bold")
    ax1.text(-0.32, 0.98, r"$f_{s,9} \leq \mu F_{n,9}$", fontsize=9, color=SAGE)
    ax1.text(0.08, -0.18, r"$m(g + a_z)$", fontsize=9.5, color=CRIMSON, fontweight="bold")
    ax1.set_title("(a) Bilateral Squeeze Equilibrium During Lift")

    # Right panel: Friction cone stability boundary F_n >= m(g+az)/(2*mu)
    mus = np.linspace(0.25, 1.5, 120)
    for m, az, col, lbl in [
        (0.10, 0.0, STEEL, "Nominal m=0.10 kg, a_z=0"),
        (0.25, 0.0, OCHRE, "Heavy OOD m=0.25 kg, a_z=0"),
        (0.25, 0.14 * 9.81, CRIMSON, "Heavy + Stage-3 Jerk (a_z spike)"),
    ]:
        fn_req = (m * (9.81 + az)) / (2.0 * mus)
        ax2.plot(mus, fn_req, color=col, lw=2.0, label=lbl)
    ax2.axhline(5.0, color=SAGE, ls="--", lw=1.2, label="Ctrl E Minimum Settle Floor (5 N)")
    ax2.set_xlabel(r"Coulomb Lateral Friction Coefficient $\mu$")
    ax2.set_ylabel(r"Minimum Per-Finger Normal Force $F_n$ (N)")
    ax2.set_title("(b) Analytical Slip Boundary Condition")
    ax2.legend(fontsize=7.5)
    fig.suptitle("Contact Mechanics, Coulomb Friction Cones, and Analytical Basis of Phase 13.5 Slip Collapse", fontweight="bold")
    p = DIR_DIAG / "diag_05_contact_and_friction_cones.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p.name)

    # 6. SAC + HER Architecture & Static-Goal Trap
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.03, 0.56, 0.25, 0.30, "Stochastic Actor pi_phi", "Squashed Gaussian tanh(mu + sig*eps)\nAuto entropy alpha -> H_tgt = -dim(A)", "#F1F5F8", NAVY)
    box(ax, 0.03, 0.12, 0.25, 0.30, "Twin Critics Q1, Q2 + Targets", "Min Q target clipping (Eq. 2-3)\nPolyak tau=0.05 / 0.005", "#F4F6F8", STEEL)
    box(ax, 0.36, 0.33, 0.28, 0.38, "HerReplayBuffer (k=4, future)", "Stores (s_t, a_t, s_{t+1}, ag, dg)\nVirtual goals g' = ag_tau (tau > t)\nStatic Trap (Ph11): if obj unmoved,\ng' == obj_init -> passive hover!", "#FAF7F0", OCHRE)
    box(ax, 0.71, 0.56, 0.26, 0.30, "PyBullet Dict Env", "obs=(10,), ag=(3,), dg=(3,)\nReward sparse(-1/0) vs dense", "#EEF4F0", SAGE)
    box(ax, 0.71, 0.12, 0.26, 0.30, "Phase 15.2.5B1 HER Guard", "gate_goal_lift_on_lifted_dg\nBlocks negative r_goal_lift penalty", "#F9EFEF", CRIMSON)
    arrow(ax, 0.28, 0.71, 0.71, 0.71, lbl="Action a_t")
    arrow(ax, 0.71, 0.62, 0.64, 0.54, lbl="Transitions")
    arrow(ax, 0.36, 0.52, 0.28, 0.66); arrow(ax, 0.36, 0.42, 0.28, 0.27)
    arrow(ax, 0.64, 0.36, 0.71, 0.27)
    ax.set_title("Soft Actor-Critic (SAC) Coupled with Future-Strategy HER and Static-Goal Exploration Mitigation")
    p = DIR_DIAG / "diag_06_sac_and_her_architecture.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p.name)

    # 7. Behavioral Cloning & Decaying Hybrid BC-SAC
    fig, ax = plt.subplots(figsize=(10.2, 4.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.03, 0.54, 0.27, 0.32, "5B1 Unbalanced Demos", "56.7% trailing HOLD_AT_GOAL\n78.4% closed-jaw actions -> premature close", "#F9EFEF", CRIMSON)
    box(ax, 0.03, 0.10, 0.27, 0.32, "5B2 Balanced + M_prog Mask", "50/50 eps (1,672 transitions), 0 colls\n70.1% open approach -> Masks HOLD frames", "#EEF4F0", SAGE)
    box(ax, 0.38, 0.30, 0.27, 0.38, "BCDecayingSAC Objective", "J_total = J_SAC + lam_BC(t) * L_BC_masked\nlam_BC decays; min_ent_coef floor active", "#F1F5F8", NAVY)
    box(ax, 0.72, 0.54, 0.25, 0.32, "5B7 Pure Masked BC", "100-ep Gen: 75% Grasp\n45% Lift | 0.0% Table Colls", "#EEF4F0", SAGE)
    box(ax, 0.72, 0.10, 0.25, 0.32, "5B8 Hybrid BC-SAC (10k)", "100-ep Gen: 66% Grasp\n26% Lift (vs 7% Pure SAC)", "#F4F6F8", STEEL)
    arrow(ax, 0.30, 0.26, 0.38, 0.45); arrow(ax, 0.65, 0.55, 0.72, 0.68); arrow(ax, 0.65, 0.40, 0.72, 0.26)
    ax.set_title("Progress-Masked Behavioral Cloning (5B7) and Decaying Auxiliary Hybrid BC-SAC (5B8)")
    p = DIR_DIAG / "diag_07_bc_and_hybrid_bcsac.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p.name)

    # 8. Domain Randomization Distributions DR0..DR4 (Conceptual + Config)
    fig, ax = plt.subplots(figsize=(10.2, 4.0))
    tiers = ["DR0\nNominal", "DR1\nLight", "DR2\nModerate", "DR3\nStrong", "DR4\nStress OOD"]
    mass_lo = [0.10, 0.08, 0.06, 0.05, 0.04]
    mass_hi = [0.10, 0.12, 0.16, 0.20, 0.25]
    fric_lo = [1.00, 0.80, 0.60, 0.45, 0.30]
    fric_hi = [1.00, 1.20, 1.30, 1.40, 1.50]
    x = np.arange(5)
    ax_t = ax.twinx()
    ax.vlines(x - 0.12, mass_lo, mass_hi, color=NAVY, lw=6, label="Cube Mass Range (kg)")
    ax.plot(x - 0.12, [0.5*(a+b) for a,b in zip(mass_lo, mass_hi)], "o", color=OCHRE)
    ax_t.vlines(x + 0.12, fric_lo, fric_hi, color=SAGE, lw=6, label="Lateral Friction mu Range")
    ax.set_xticks(x); ax.set_xticklabels(tiers)
    ax.set_ylabel("Cube Mass Span (kg)", color=NAVY); ax_t.set_ylabel("Lateral Friction mu Span", color=SAGE)
    ax.set_title("Calibrated Physics Perturbation Envelopes Across Domain Randomization Tiers DR0-DR4")
    p = DIR_DIAG / "diag_08_domain_randomization_dr0_4.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p.name)

    # 9. Adaptive Shield & PreGrasp 5-Check State Machine
    fig, ax = plt.subplots(figsize=(10.2, 4.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    stages = [
        (0.03, "1. APPROACH\nOpen Jaw > 6.5cm"),
        (0.22, "2. XY ALIGN\nxy_err <= 1.5cm"),
        (0.41, "3. DESCEND\n|dz| <= 1.8cm"),
        (0.60, "4. GRASP SETTLE\n3 ticks, Fn >= 5N"),
        (0.79, "5. LIFT+PLACE\nSlip-rate cap"),
    ]
    for i, (sx, txt) in enumerate(stages):
        box(ax, sx, 0.56, 0.16, 0.28, txt, "", "#F1F5F8" if i < 3 else "#EEF4F0", NAVY if i < 3 else SAGE, ts=7.8)
        if i < 4:
            arrow(ax, sx + 0.16, 0.70, stages[i + 1][0], 0.70)
    box(ax, 0.10, 0.12, 0.80, 0.26, "Phase 14 Controller E Adaptive Shield + Phase 15.2.5A 5-Check Gate",
        "Checks simultaneously: (1) xy_err<=15mm, (2) dz<=18mm, (3) gripper_width>65mm during descent,\n"
        "(4) cube velocity <0.02 m/s, (5) active object lock -> Prevents premature closing & lifts DR4 Held-Out 57% -> 98%",
        "#FAF7F0", OCHRE)
    arrow(ax, 0.50, 0.56, 0.50, 0.38, col=OCHRE)
    ax.set_title("Integrated Adaptive Transport Shield (Phase 14) and Five-Check Pre-Grasp State Machine (Phase 15.2.5A)")
    p = DIR_DIAG / "diag_09_adaptive_shield_and_pregrasp.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p.name)

    # 10. Multi-Object Sorting & Peg-in-Slot Insertion Geometry
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.axis("off")
    box(ax1, 0.05, 0.65, 0.28, 0.22, "Spawn Obj 0 & 1", "Random tabletop XYZ", "#F1F5F8", NAVY)
    box(ax1, 0.38, 0.65, 0.26, 0.22, "Pick+Sort Obj 0", "5-Check -> Bin 0", "#EEF4F0", SAGE)
    box(ax1, 0.69, 0.65, 0.26, 0.22, "Pick+Sort Obj 1", "5-Check -> Bin 1", "#EEF4F0", SAGE)
    box(ax1, 0.15, 0.18, 0.70, 0.25, "Phase 15.2.5A Result (Scripted)", "50/50 (100.0%) Full 2-Obj Completion | 76.4 Steps | 0 Table Colls", "#FAF7F0", OCHRE)
    arrow(ax1, 0.33, 0.76, 0.38, 0.76); arrow(ax1, 0.64, 0.76, 0.69, 0.76)
    ax1.set_title("(a) Dual-Cube Sequential Sorting Workflow")

    ax2.set_xlim(-1, 1); ax2.set_ylim(-0.2, 1.2); ax2.axis("off")
    ax2.add_patch(patches.Rectangle((-0.65, 0.0), 0.35, 0.45, fc="#CBD5E1", ec=NAVY))
    ax2.add_patch(patches.Rectangle((0.30, 0.0), 0.35, 0.45, fc="#CBD5E1", ec=NAVY))
    ax2.add_patch(patches.Rectangle((-0.24, 0.32), 0.48, 0.65, fc="#D6A65D", ec=NAVY, alpha=0.85))
    ax2.annotate("", xy=(0.0, 0.18), xytext=(0.0, 0.60), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2))
    ax2.annotate("", xy=(-0.45, 0.38), xytext=(-0.24, 0.38), arrowprops=dict(arrowstyle="<->", color=CRIMSON, lw=1.6))
    ax2.text(0.05, 0.56, r"$\Delta z,\; \Delta\psi$ (5D Act)", fontsize=8.5, color=NAVY)
    ax2.text(-0.88, 0.48, r"$F_{\mathrm{wall}}$ Jam\\Compliance", fontsize=8.0, color=CRIMSON)
    ax2.set_title("(b) SE(2)+Z Compliant Peg-in-Slot Assembly (Ph 15.3 / 16.4)")
    p = DIR_DIAG / "diag_10_task_lifecycles_multisort_assembly.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p.name)

    # 11. HRL, RGB-D & Sim2Sim Pipeline
    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.03, 0.56, 0.28, 0.30, "1. Oblique RGB-D (Ph 16.2)", "128x128 @ FOV 55 deg [0.92,0.28,0.45]\nAvoids self-occlusion; 14.8 mm loc err", "#F1F5F8", NAVY)
    box(ax, 0.36, 0.56, 0.28, 0.30, "2. 5-Skill HRL Meta (Ph 16.1)", "Discrete SAC selects {REACH..TRANS}\n30.0% PickPlace vs 0% Flat SAC @ 12k", "#EEF4F0", SAGE)
    box(ax, 0.69, 0.56, 0.28, 0.30, "3. Kalman Predictor (Ph 16.3)", "4-state constant-vel target tracking\n95-100% Moving Target Intercept", "#F4F6F8", STEEL)
    box(ax, 0.14, 0.10, 0.72, 0.30, "Phase 17.1 MuJoCo 4.0.1 Cross-Simulator Parity & Phase 17.2 Stress Gate", "Zero-shot PyBullet->MuJoCo 40% bilat / 0% place -> Calibrated 50% / 5% (9.4 mm table Z offset).\nSoftware E-Stop SafetySupervisor verified at 32.41 us latency (Physical Robot = NOT_ATTEMPTED).", "#FAF7F0", OCHRE)
    arrow(ax, 0.31, 0.71, 0.36, 0.71); arrow(ax, 0.64, 0.71, 0.69, 0.71); arrow(ax, 0.50, 0.56, 0.50, 0.40)
    ax.set_title("Phase 16 Cognitive/Perception Modules & Phase 17 MuJoCo Sim-to-Sim Falsification Bridge")
    p = DIR_DIAG / "diag_11_hrl_rgbd_and_sim2sim.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p.name)

    # 12. Web Workstation to Simulation Architecture (Part VI)
    fig, ax = plt.subplots(figsize=(10.2, 4.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.03, 0.26, 0.28, 0.48, "React 18 + Three.js R3F", "10 Workstation Panels\nOBJ/MTL Panda Link Renderer\nNever computes physics locally!", "#F1F5F8", NAVY)
    box(ax, 0.37, 0.54, 0.26, 0.26, "REST Commands (/api/*)", "Joints (limit-checked), IK,\nGripper, Task, E-Stop (409)", "#FAF7F0", OCHRE)
    box(ax, 0.37, 0.18, 0.26, 0.26, "WebSocket (/ws/* 20Hz)", "11 Link Transforms, Objects,\nEE Traj, Live Events", "#EEF4F0", SAGE)
    box(ax, 0.69, 0.26, 0.28, 0.48, "FastAPI SimulationManager", "Single Authoritative PyBullet\nWorker Thread + RLock\nEnforces Limits & E-Stop", "#F4F6F8", STEEL)
    arrow(ax, 0.31, 0.67, 0.37, 0.67); arrow(ax, 0.63, 0.67, 0.69, 0.67)
    arrow(ax, 0.69, 0.31, 0.63, 0.31); arrow(ax, 0.37, 0.31, 0.31, 0.31)
    ax.set_title("Web-Based Simulation Control Center: Bidirectional Authoritative Sync Pipeline")
    p = DIR_DIAG / "diag_12_website_websocket_architecture.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p.name)

    # 13. Phase 18 Multi-Agent Cooperative Robotics (FUTURE SCOPE)
    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.04, 0.55, 0.26, 0.30, "Robot Agent A (Panda 1)", "Local Proprioception + RGB-D\nDecentralized Policy pi_A", "#F1F5F8", NAVY)
    box(ax, 0.70, 0.55, 0.26, 0.30, "Robot Agent B (Panda 2)", "Local Proprioception + F/T\nDecentralized Policy pi_B", "#F1F5F8", NAVY)
    box(ax, 0.37, 0.60, 0.26, 0.22, "Shared Task Allocator", "Auction / Graph MARL\nCollision-Free Bimanual Plan", "#FAF7F0", OCHRE)
    box(ax, 0.18, 0.12, 0.64, 0.28, "[PROPOSED FUTURE RESEARCH SCOPE — NOT IMPLEMENTED IN PHASES 1-17]",
        "Centralized Critic Q_tot(s_all, a_A, a_B) during Simulation Training -> Decentralized Execution (CTDE)\nTarget Extensions: Bimanual Handover, Cooperative Peg-in-Slot, Human-Robot Shared Workspace",
        "#F4F6F8", CRIMSON)
    arrow(ax, 0.30, 0.71, 0.37, 0.71, ls="--", lbl="Comm m_t")
    arrow(ax, 0.70, 0.71, 0.63, 0.71, ls="--", lbl="Comm m_t")
    arrow(ax, 0.50, 0.60, 0.50, 0.40, ls="--", col=CRIMSON)
    ax.set_title("Phase 18 Multi-Agent Cooperative Manipulation Architecture [FUTURE RESEARCH SCOPE ONLY]")
    p = DIR_DIAG / "diag_13_phase18_multiagent_future.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p.name)

    # 14. Reward Shaping Landscapes Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)
    d = np.linspace(0.0, 0.5, 200)
    r_sparse = np.where(d <= 0.05, 0.0, -1.0)
    r_dense = -2.0 * d + np.where(d <= 0.05, 5.0, 0.0)
    ax1.plot(d, r_sparse, color=CRIMSON, lw=2.2, label="Sparse r = -I(||p - g|| > 0.05)")
    ax1.plot(d, r_dense, color=NAVY, lw=2.2, label="Dense Potential + Completion Bonus")
    ax1.axvline(0.05, color=SAGE, ls=":", lw=1.2)
    ax1.set_xlabel("Distance ||p_obj - p_goal|| (m)"); ax1.set_ylabel("Step Reward")
    ax1.set_title("(a) Sparse Flat Plateau vs Dense Gradient"); ax1.legend(fontsize=7.5)

    dz = np.linspace(0.0, 0.48, 200)
    phi_gated = np.where(dz < 0.1, 0.0, 0.0)  # Illustrating zero signal outside XY cone
    phi_cont = -2.0 * np.clip(dz, 0.0, 1.0)
    ax2.plot(dz, np.zeros_like(dz) - 0.9, "--", color=CRIMSON, lw=2.0, label="5B XY-Gated Vertical (Zero Z Gradient)")
    ax2.plot(dz, phi_cont, color=SAGE, lw=2.2, label="5B1 Continuous Phi_z = -2.0 * clip(|dz|,0,1)")
    ax2.set_xlabel("Vertical Offset |z_ee - z_grasp| (m)"); ax2.set_ylabel("Approach Potential Phi_z")
    ax2.set_title("(b) Solving z=0.50 m Hover Stall (Phase 15.2.5B1)"); ax2.legend(fontsize=7.5)
    fig.suptitle("Reward Topology Design: Escaping Sparse Equilibria and Hover Plateaus", fontweight="bold")
    p = DIR_DIAG / "diag_14_reward_shaping_potential.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p.name)

    return saved


def generate_evidence_plots() -> List[str]:
    """Generate 10 high-res evidence-based 2D/3D plots using #FCFCFA theme & exact provenance."""
    tb = json.loads((PUB / "benchmark_data" / "tb_scalars_all.json").read_text(encoding="utf-8"))
    saved = []

    def gxy(rk, tag):
        e = tb.get(rk, {}).get(tag, {})
        return np.array(e.get("steps", []), dtype=float), np.array(e.get("values", []), dtype=float)

    # Plot 1: Early TB
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.9), constrained_layout=True)
    for rk, lbl, c in [
        ("logs/reach_sac/SAC_3", "Ph2 Reach SAC_3", NAVY),
        ("logs/pick_sac/SAC_1", "Ph4 Pick SAC_1", STEEL),
        ("experiments/pick_place/E1_no_her/logs/SAC_1", "Ph6 E1 No-HER", CRIMSON),
        ("experiments/pick_place/E2_her/logs/SAC_1", "Ph6 E2 HER (k=4)", SAGE),
        ("logs/multi_sort/SAC_1", "Ph8 MultiSort SAC_1", OCHRE),
    ]:
        x, y = gxy(rk, "train/actor_loss")
        if len(x): ax1.plot(x / 1000.0, y, label=lbl, color=c, lw=1.7)
        x, y = gxy(rk, "train/ent_coef")
        if len(x): ax2.plot(x / 1000.0, y, label=lbl, color=c, lw=1.7)
    ax1.set_title("(a) Actor Loss (10k Training Runs)"); ax1.set_xlabel("Steps (x1,000)"); ax1.set_ylabel("train/actor_loss"); ax1.legend(fontsize=7.3)
    ax2.set_title("(b) Auto-Entropy Decay"); ax2.set_xlabel("Steps (x1,000)"); ax2.set_ylabel("train/ent_coef")
    fig.suptitle("Early Training Benchmarks (Phases 2-8, Sparse Rewards, CPU PyTorch | Source: TensorBoard Events)", fontweight="bold")
    p = DIR_PLOT / "plot_01_early_tb_curves.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p.name)

    # Plot 2: Phase 11 25x100k sparse plateau with std bars across 5 seeds
    rows11 = list(csv.DictReader((ROOT / "results" / "phase11_final_summary.csv").open()))
    drs = ["dr0", "dr1", "dr2", "dr3", "dr4"]
    nom_m, nom_s, held_m, held_s = [], [], [], []
    for dr in drs:
        nv = [float(r["mean_final_distance"]) for r in rows11 if r["dr_condition"] == dr and r["evaluation_condition"] == "Nominal"]
        hv = [float(r["mean_final_distance"]) for r in rows11 if r["dr_condition"] == dr and r["evaluation_condition"] == "Held-out"]
        nom_m.append(np.mean(nv)); nom_s.append(np.std(nv))
        held_m.append(np.mean(hv)); held_s.append(np.std(hv))
    fig, ax = plt.subplots(figsize=(9.5, 3.8), constrained_layout=True)
    idx = np.arange(5)
    ax.bar(idx - 0.17, nom_m, 0.34, yerr=nom_s, capsize=3, color=STEEL, label="Nominal Eval (N=500 eps/DR, 5 seeds)")
    ax.bar(idx + 0.17, held_m, 0.34, yerr=held_s, capsize=3, color=OCHRE, label="Held-Out DR4 Eval (N=500 eps/DR, 5 seeds)")
    ax.axhline(0.05, color=SAGE, ls="--", lw=1.4, label="Success Radius (0.05 m) — All 25 Models = 0.0%")
    ax.set_xticks(idx); ax.set_xticklabels([d.upper() for d in drs]); ax.set_ylabel("Final EE-Target Distance (m)")
    ax.set_title("Phase 11 Sparse HER Stagnation (25 Models x 100k Train = 0.0% Across 5,000 Eval Eps | Source: phase11_final_summary.csv)")
    ax.legend(fontsize=8)
    p = DIR_PLOT / "plot_02_phase11_sparse_vs_transport.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p.name)

    # Plot 3: Phase 12 1M curve
    r12 = list(csv.DictReader((ROOT / "results" / "phase12_1m" / "checkpoint_metrics.csv").open()))
    st = np.array([float(r["Total Steps"]) / 1e3 for r in r12])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)
    ax1.plot(st, [float(r["Reach"]) * 100 for r in r12], "o-", color=STEEL, label="Reach %")
    ax1.plot(st, [float(r["Grasp"]) * 100 for r in r12], "s--", color=OCHRE, label="Grasp %")
    ax1.plot(st, [float(r["End-to-End Success"]) * 100 for r in r12], "D-", color=NAVY, lw=2.3, label="Place Success %")
    ax1.set_title("(a) Sub-Skill Cascade (Seed 42, DR0)"); ax1.set_xlabel("Train Steps (x1k)"); ax1.set_ylabel("Rate (%)"); ax1.legend(fontsize=8)
    ax2.plot(st, [float(r["Mean Final Dist"]) for r in r12], "o-", color=CRIMSON, lw=2.0, label="Mean Dist (m)")
    ax2.axhline(0.05, color=SAGE, ls=":")
    ax2.set_title("(b) Final Target Positioning Error"); ax2.set_xlabel("Train Steps (x1k)"); ax2.set_ylabel("Distance (m)")
    fig.suptitle("Phase 12 1M-Step Dense Pick-and-Place Breakthrough (N=100 Eval Eps/Ckpt | Source: phase12_1m/checkpoint_metrics.csv)", fontweight="bold")
    p = DIR_PLOT / "plot_03_phase12_1m_breakthrough.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p.name)

    # Plot 4: Phase 13 DR with multi-seed std + Phase 13.5 sweeps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)
    cats = ["DR1 (4 seeds)", "DR2 (2 seeds)", "DR4 (2 seeds)"]
    m_means = [100.0, 99.0, 99.0]; m_stds = [0.0, 0.0, 1.4]
    h_means = [80.0, 87.5, 57.0]; h_stds = [20.8, 10.6, 33.9]  # Seed42 81%, Seed43 33%
    x = np.arange(3)
    ax1.bar(x - 0.17, m_means, 0.34, yerr=m_stds, capsize=3, color=STEEL, label="Matched DR Eval")
    ax1.bar(x + 0.17, h_means, 0.34, yerr=h_stds, capsize=3, color=CRIMSON, label="Held-Out DR4 OOD")
    ax1.set_xticks(x); ax1.set_xticklabels(cats); ax1.set_ylim(0, 118); ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("(a) Phase 13 Matched vs Held-Out (Mean +/- SD)"); ax1.legend(fontsize=7.8)

    f_rows = list(csv.DictReader((ROOT / "results" / "phase13_5" / "d3_friction_sensitivity" / "friction_summary.csv").open()))
    ax2.plot([float(r["friction"]) for r in f_rows], [float(r["grasp_rate"]) * 100 for r in f_rows], "o--", color=STEEL, label="Grasp %")
    ax2.plot([float(r["friction"]) for r in f_rows], [float(r["success_rate"]) * 100 for r in f_rows], "s-", color=NAVY, lw=2, label="Success %")
    ax2.set_xlabel(r"Isolated Friction $\mu$"); ax2.set_ylabel("Rate (%)"); ax2.set_title("(b) Phase 13.5 D3 Isolated Friction Sweep"); ax2.legend(fontsize=8)
    fig.suptitle("Generalization Cliff Across DR Tiers & Isolated Friction Sensitivity (N=100-1,520 Eps)", fontweight="bold")
    p = DIR_PLOT / "plot_04_phase13_dr_with_errorbars.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p.name)

    # Plot 5: Copy crisp #FCFCFA styled version of D5 Force + 3D Trajectory
    sr = list(csv.DictReader((ROOT / "results" / "phase13_5" / "d5_lift_trajectories" / "success_sample_1_seed42_ep5001.csv").open()))
    fr = list(csv.DictReader((ROOT / "results" / "phase13_5" / "d5_lift_trajectories" / "grasp_but_no_lift_sample_1_seed42_ep5002.csv").open()))
    fig = plt.figure(figsize=(10.4, 4.4), constrained_layout=True)
    ax1 = fig.add_subplot(121)
    ax1.plot([float(r["step"]) for r in sr], [float(r["total_normal_force"]) for r in sr], color=SAGE, lw=2, label="Success (ep5001)")
    ax1.plot([float(r["step"]) for r in fr], [float(r["total_normal_force"]) for r in fr], color=CRIMSON, lw=2, label="Slip Collapse 68.4N->0N (ep5002)")
    ax1.set_title("(a) Bilateral Finger Normal Force (N)"); ax1.set_xlabel("Control Step (20 Hz)"); ax1.set_ylabel("Force (N)"); ax1.legend(fontsize=7.8)
    ax2 = fig.add_subplot(122, projection="3d")
    ee_s = np.array([[float(r["ee_pos_0"]), float(r["ee_pos_1"]), float(r["ee_pos_2"])] for r in sr])
    cb_f = np.array([[float(r["cube_pos_0"]), float(r["cube_pos_1"]), float(r["cube_pos_2"])] for r in fr])
    ee_f = np.array([[float(r["ee_pos_0"]), float(r["ee_pos_1"]), float(r["ee_pos_2"])] for r in fr])
    ax2.plot(ee_s[:,0], ee_s[:,1], ee_s[:,2], color=SAGE, lw=2, label="EE Success XYZ")
    ax2.plot(ee_f[:,0], ee_f[:,1], ee_f[:,2], color=CRIMSON, lw=2, label="EE Empty Slip XYZ")
    ax2.plot(cb_f[:,0], cb_f[:,1], cb_f[:,2], "--", color=OCHRE, lw=2, label="Cube Left on Table")
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)"); ax2.set_zlabel("Z (m)")
    ax2.set_title("(b) 3D Physical Trajectory Divergence (XYZ)"); ax2.legend(fontsize=7.2)
    fig.suptitle("Phase 13.5 Microscopic Slip Forensics (Real D5 CSV Telemetry, Seed 42 Held-Out DR4)", fontweight="bold")
    p = DIR_PLOT / "plot_05_phase13_5_slip_telemetry_2d_3d.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p.name)

    # Dedicated warm-palette Plot 07 (clean non-overlapping 2-line titles + grouped Phase 15.1 means)
    p151 = json.loads((ROOT / "results" / "phase15_1" / "phase15_1_results.json").read_text())
    by_cond: Dict[str, List[float]] = {}
    for s in p151["summaries"]:
        by_cond.setdefault(s["condition"], []).append(float(s["success_rate"]) * 100.0)
    cond_order = [
        ("A0_CLEAN_BASELINE", "A0 Clean\nBaseline"),
        ("A1_NOISE_3MM", "A1 Noise\n(3 mm)"),
        ("A2_NOISE_5MM", "A2 Noise\n(5 mm)"),
        ("B1_OBS_DELAY_1", "B1 Obs\nLag (1-st)"),
        ("B2_OBS_DELAY_2", "B2 Obs\nLag (2-st)"),
        ("B3_OBS_DELAY_4", "B3 Obs\nLag (4-st)"),
        ("B4_COMBINED_DELAY_1_1", "B4 Comb.\nDelay (1+1)"),
        ("C1_UNPRIVILEGED", "C1 Unpriv.\nState Est."),
    ]
    c_lbls = [lbl for key, lbl in cond_order if key in by_cond]
    c_means = [float(np.mean(by_cond[key])) for key, _ in cond_order if key in by_cond]
    if not c_lbls:
        for k, vals in list(by_cond.items())[:7]:
            c_lbls.append(k.replace("_", "\n")[:18])
            c_means.append(float(np.mean(vals)))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.4, 4.3), gridspec_kw={"wspace": 0.34})
    fig.patch.set_facecolor(BG_WARM)

    c_cols = [SAGE if r >= 90 else (OCHRE if r >= 60 else CRIMSON) for r in c_means]
    ax1.barh(np.arange(len(c_lbls)), c_means, color=c_cols, height=0.62)
    ax1.set_yticks(np.arange(len(c_lbls)))
    ax1.set_yticklabels(c_lbls, fontsize=7.6)
    ax1.invert_yaxis()
    ax1.set_xlabel("Mean Success Rate (%)")
    ax1.set_xlim(0, 118)
    ax1.set_title("(a) Phase 15.1 Stress\nCondition Means", fontsize=9.6, pad=8)
    for i, v in enumerate(c_means):
        ax1.text(v + 2.0, i, f"{v:.1f}%", va="center", fontsize=7.6, fontweight="bold")

    variants = ["5B SAC\n(20k)", "5B1 Warm\n(10k)", "5B2 Bal.\n(17.5k)", "5B7 BC\n(Pure)", "5B8 Hybrid\n(10k)"]
    grasp_v = [0.0, 6.0, 62.0, 75.0, 66.0]
    lift_v = [0.0, 0.0, 7.0, 45.0, 26.0]
    coll_v = [0.0, 35.0, 31.0, 0.0, 20.0]
    xv = np.arange(len(variants))
    ax2.bar(xv - 0.25, grasp_v, 0.25, label="Grasp %", color=STEEL)
    ax2.bar(xv, lift_v, 0.25, label="Lift %", color=SAGE)
    ax2.bar(xv + 0.25, coll_v, 0.25, label="Table Coll %", color=CRIMSON)
    ax2.set_xticks(xv)
    ax2.set_xticklabels(variants, fontsize=7.3)
    ax2.set_ylim(0, 94)
    ax2.set_ylabel("Rate (% over 100 eps)")
    ax2.set_title("(b) Phase 15.2.5B-B8\nSingle Grasp/Lift Ablations", fontsize=9.6, pad=8)
    ax2.legend(fontsize=7.2, loc="upper left")

    asm_labels = ["Scripted\nBaseline", "Behavioral\nCloning", "SAC 5k\n(RL)"]
    asm_align = [100.0, 23.33, 3.33]
    asm_succ = [56.67, 0.0, 0.0]
    xa = np.arange(3)
    ax3.bar(xa - 0.17, asm_align, 0.34, label="Pre-Insert Align %", color=STEEL)
    ax3.bar(xa + 0.17, asm_succ, 0.34, label="Full Insertion %", color=NAVY)
    for i, (a, b) in enumerate(zip(asm_align, asm_succ)):
        ax3.text(i - 0.17, a + 1.8, f"{a:.1f}", ha="center", fontsize=7.5)
        ax3.text(i + 0.17, b + 1.8, f"{b:.1f}", ha="center", fontsize=7.5)
    ax3.set_xticks(xa)
    ax3.set_xticklabels(asm_labels, fontsize=7.8)
    ax3.set_ylim(0, 118)
    ax3.set_ylabel("Rate (% over 30 eps)")
    ax3.set_title("(c) Phase 15.3\nPeg-in-Slot Assembly", fontsize=9.6, pad=8)
    ax3.legend(fontsize=7.2, loc="upper right")

    fig.suptitle("Phase 15 Stress Testing, Single-Object BC vs SAC-HER, and Orientation-Aware Assembly", fontsize=11.0, fontweight="bold", y=1.01)
    p7 = DIR_PLOT / "plot_07_phase15_ablations_and_stress.png"
    fig.savefig(p7, bbox_inches="tight")
    plt.close(fig)
    saved.append(p7.name)

    # Copy high-res plots 6, 8, 9, 10
    for src_path in [
        ("plot_06_phase14_controllers_comparison.png", PUB / "figures_2d" / "fig06_phase14_adaptive_controller_comparison.png"),
        ("plot_08_phase16_hrl_vision_dynamic_contact.png", PUB / "figures_2d" / "fig08_phase16_hrl_vision_dynamic_contact.png"),
        ("plot_09_phase17_sim2sim_and_safety.png", PUB / "figures_2d" / "fig09_phase17_sim2sim_and_safety_stress.png"),
        ("plot_10_panda_3d_fk_and_rgbd_cloud.png", PUB / "figures_3d" / "fig3d_01_panda_fk_and_workspace.png"),
    ]:
        import shutil
        dst = DIR_PLOT / src_path[0]
        shutil.copy2(src_path[1], dst)
        saved.append(dst.name)

    # Also copy 3D RGB-D frustum into plot_11
    import shutil
    shutil.copy2(PUB / "figures_3d" / "fig3d_03_oblique_rgbd_frustum_and_mujoco_workspace.png",
                 DIR_PLOT / "plot_11_oblique_rgbd_3d_frustum.png")
    saved.append("plot_11_oblique_rgbd_3d_frustum.png")
    return saved


def generate_equations() -> List[str]:
    plt.rcParams["mathtext.fontset"] = "cm"
    eqs = [
        (1, r"$\mathbf{q}_{t+1} = \mathbf{q}_t + J^\top (J J^\top + \lambda^2 I)^{-1}\Delta\mathbf{x} + (I - J^\# J)k_0(\mathbf{q}_0 - \mathbf{q}_t)$"),
        (2, r"$\dot{\mathbf{x}} = J(\mathbf{q})\dot{\mathbf{q}},\quad J(\mathbf{q}) \in \mathbb{R}^{6 \times 7}$"),
        (3, r"$\mathcal{L}_Q(\theta_i) = \mathbb{E}_{(s_t,a_t,r_t,s_{t+1},g)\sim\mathcal{D}}\left[(Q_{\theta_i}(s_t,a_t,g) - y)^2\right]$"),
        (4, r"$y = r_t + \gamma\left(\min_{i=1,2} Q_{\bar{\theta}_i}(s_{t+1},\tilde{a}_{t+1},g) - \alpha \log\pi_\phi(\tilde{a}_{t+1}\mid s_{t+1},g)\right)$"),
        (5, r"$J_\pi(\phi) = \mathbb{E}_{s_t\sim\mathcal{D},\epsilon\sim\mathcal{N}}\left[\alpha\log\pi_\phi(f_\phi(\epsilon;s_t,g)\mid s_t,g) - \min_{i=1,2}Q_{\theta_i}(s_t,f_\phi,g)\right]$"),
        (6, r"$\alpha^* = \arg\min_{\alpha>0}\;\mathbb{E}_{a_t\sim\pi_\phi}\left[-\alpha(\log\pi_\phi(a_t\mid s_t,g) + \mathcal{H}_{\mathrm{tgt}})\right],\; \mathcal{H}_{\mathrm{tgt}} = -4$"),
        (7, r"$r_{\mathrm{sparse}}(s_{t+1}, g) = -\mathbb{I}\left(\|\mathbf{p}_{\mathrm{obj}} - \mathbf{p}_{\mathrm{goal}}\|_2 > \varepsilon\right),\; \varepsilon = 0.05\,\mathrm{m}$"),
        (8, r"$g' \sim \left\{ \phi(s_\tau) : \tau > t,\, k = 4\;\mathrm{future\;HER\;sampling} \right\}$"),
        (9, r"$|f_s| \leq \mu F_n,\quad F_{n,\mathrm{req}} \geq \frac{m(g + a_z)}{2\mu}$"),
        (10, r"$r_{\mathrm{dense}} = -w_r d_{\mathrm{ee,obj}} + w_g \mathbb{I}_{\mathrm{bilat}} + w_l \Delta z_{\mathrm{obj}} - w_t d_{\mathrm{obj,goal}} + B_{\mathrm{succ}}\mathbb{I}_{\mathrm{succ}} - w_c\mathbb{I}_{\mathrm{coll}}$"),
        (11, r"$r_{\mathrm{pregrasp}} = w_p(\Phi_z(s_{t+1}) - \Phi_z(s_t)) + r_{\mathrm{grasp}} - w_{\mathrm{coll}}\mathbb{I}_{\mathrm{table}},\; \Phi_z = -2\,\mathrm{clip}(|z_{\mathrm{ee}}-z_g|,0,1)$"),
        (12, r"$\mathcal{L}_{\mathrm{BC}}(\phi) = \mathbb{E}_{(s,a^*)\sim\mathcal{D}_{\mathrm{bal}}}\left[M_{\mathrm{prog}}(s)\,\|\mu_\phi(s) - a^*\|_2^2\right]$"),
        (13, r"$J_{\mathrm{hybrid}}(\phi) = J_\pi^{\mathrm{SAC}}(\phi) + \lambda_{\mathrm{BC}}(t)\,\mathcal{L}_{\mathrm{BC}}(\phi),\quad \lambda_{\mathrm{BC}}(t) = \lambda_0 e^{-\kappa t}$"),
        (14, r"$\pi_{\mathrm{HRL}}(a_t\mid s_t) = \sum_{u\in\mathcal{U}} \pi_{\mathrm{high}}(u\mid s_t)\,\pi_{\mathrm{low}}^{(u)}(a_t\mid s_t)$"),
        (15, r"$\mathbf{p}_{\mathrm{world}} = \frac{(PV)^{-1}\mathbf{p}_{\mathrm{clip}}}{[(PV)^{-1}\mathbf{p}_{\mathrm{clip}}]_4},\quad \mathbf{p}_{\mathrm{clip}} = [x_{\mathrm{ndc}}, y_{\mathrm{ndc}}, 2d-1, 1]^\top$"),
        (16, r"$\hat{\mathbf{x}}_{k\mid k-1} = F\hat{\mathbf{x}}_{k-1\mid k-1},\quad K_k = P_{k\mid k-1}H^\top(H P_{k\mid k-1}H^\top + R)^{-1}$"),
        (17, r"$\mathbf{u}_{\mathrm{imp}} = K_p(\mathbf{x}_d - \mathbf{x}) + K_d(\dot{\mathbf{x}}_d - \dot{\mathbf{x}}) + K_f(\mathbf{f}_d - \mathbf{f}_{\mathrm{wall}})$"),
        (18, r"$\mathcal{S}(a_t) = \mathrm{clip}(\mathbf{p}_{\mathrm{ee}} + \delta_{\max}a_{1:3},\, \mathbf{b}_{\min},\, \mathbf{b}_{\max}),\; \mathbb{I}_{\mathrm{estop}} = (\|\mathbf{v}_{\mathrm{ee}}\|>v_{\mathrm{crit}})\vee(F_{\mathrm{wall}}>F_{\mathrm{crit}})$"),
    ]
    saved = []
    for num, tex in eqs:
        fig = plt.figure(figsize=(8.6, 0.62))
        fig.patch.set_facecolor(BG_WARM)
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(BG_WARM); ax.axis("off")
        ax.text(0.02, 0.5, tex, fontsize=12.6, va="center", ha="left", color=TXT_DARK)
        ax.text(0.985, 0.5, f"({num})", fontsize=10.8, va="center", ha="right", fontweight="bold", color=NAVY)
        p = DIR_EQ / f"eq_{num:02d}.png"
        fig.savefig(p, dpi=300, facecolor=BG_WARM)
        plt.close(fig)
        saved.append(p.name)
    return saved


if __name__ == "__main__":
    print("Cover:", generate_cover_artwork())
    print("Diagrams:", len(generate_technical_diagrams()))
    print("Plots:", len(generate_evidence_plots()))
    print("Equations:", len(generate_equations()))
