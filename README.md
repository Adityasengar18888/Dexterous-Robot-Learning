# Goal-Conditioned Robotic Manipulation Platform (SAC + HER)

A modular, reproducible, and production-quality reinforcement learning research platform for goal-conditioned robotic manipulation using **Soft Actor-Critic (SAC)** paired with **Hindsight Experience Replay (HER)** in continuous control.

The platform simulates a 7-DOF **Franka Emika Panda** robotic arm in **PyBullet**, adhering strictly to Gymnasium `GoalEnv` conventions and sparse-reward learning formulations.

---

## 1. Project Objective

The goal of this platform is to formulate and solve complex robotic manipulation benchmarks under sparse binary reward regimes. By treating manipulation as a goal-conditioned reinforcement learning problem and applying Hindsight Experience Replay, the Franka Panda agent learns to achieve precise target configurations from unshaped reward signals without human reward engineering.

The primary research formulation uses:
- **Observation Space**: Dictionary containing `observation`, `achieved_goal`, and `desired_goal`.
- **Reward Function**: Sparse binary reward $r \in \{-1, 0\}$, where $r = 0$ upon achieving the goal within a distance threshold $\epsilon = 5\text{ cm}$, and $-1$ otherwise.
- **Control Interface**: Cartesian delta end-effector control $[dx, dy, dz, \text{gripper}]$ resolved to joint positions via Inverse Kinematics (IK).

---

## 2. Planned Task Progression

The project progresses through seven distinct manipulation capabilities:

1. **Phase 2 & 3: Reach**
   - The end-effector must reach arbitrary 3D target coordinates in the free workspace.
   - Goal space $\mathcal{G} \subset \mathbb{R}^3$.
2. **Phase 4: Pick**
   - Grasp a cube/cylinder from a tabletop and hold it stably at a specified target altitude.
3. **Phase 5: Pick and Place**
   - Pick an object and place it accurately onto a target position on a surface or tray.
4. **Phase 6: Sorting**
   - Identify, pick, and sort multi-colored or shaped objects into their designated target bins.
5. **Phase 7: Multi-Object Manipulation**
   - Sequential manipulation, obstacle avoidance, and clearing clutter to manipulate occluded targets.
6. **Phase 8: Multi-Part Assembly**
   - High-precision insertion (peg-in-hole) and tight-tolerance multi-component mechanical assembly.
7. **Phase 9: Robust Manipulation under Domain Randomization**
   - Visual, mass, friction, damping, and latency randomization for robust policy transfer.

---

## 3. Architecture

The codebase enforces strict separation of concerns across simulation physics, kinematic control, environment logic, RL algorithms, and experiment telemetry:

```
robotic-manipulation-her/
├── README.md                # Project documentation and guide
├── LICENSE                  # MIT License
├── .gitignore               # Ignored build, cache, and artifact paths
├── requirements.txt         # Pinned pip dependency specifications
├── pyproject.toml           # Project metadata, build specs, and test configs
├── configs/                 # Declarative YAML configurations
│   ├── default.yaml         # Simulation timestep, frequency, seeds, RL hyperparams
│   └── robot/               # Robot geometry, joint limits, kinematic bounds
├── src/                     # Core library
│   ├── environments/        # Gymnasium GoalEnv environments (Reach, Pick, etc.)
│   ├── robots/              # Robot models, base interfaces, and IK controllers
│   ├── objects/             # Objects, tables, bins, and collision geometries
│   ├── rl/                  # Algorithm wrappers, HER replay buffers, policy evaluation
│   ├── metrics/             # Telemetry, success rate calculators, TensorBoard hooks
│   └── utils/               # Config loaders, PyBullet helpers, math utilities
├── tests/                   # Test suite (unit, kinematic, and smoke tests)
├── experiments/             # Experiment runners, training scripts, sweep manifests
├── checkpoints/             # Saved policy weights and replay buffer dumps
├── videos/                  # Recorded evaluation rollouts
├── logs/                    # Training metrics and TensorBoard event logs
├── notebooks/               # Research analysis, trajectory inspection, and plotting
└── docs/                    # Formal technical specifications and results logs
    ├── architecture.md      # Detailed layer separation and component diagram
    ├── methodology.md       # Mathematical RL formulation, HER, and kinematics
    ├── experiments.md       # Registry of formal benchmark runs
    └── results.md           # Empirical verification log and test evidence
```

---

## 4. Setup

### Prerequisites
- Python 3.11+
- Micromamba or Conda environment manager (recommended on Windows for PyBullet binary builds)

### Environment Installation
```powershell
# 1. Create isolated environment with Python 3.11 and precompiled PyBullet
micromamba create -p .\.conda-env -c conda-forge python=3.11 pybullet -y

# 2. Install RL and scientific computing dependencies
uv pip install --python .\.conda-env\python.exe -r requirements.txt
```

---

## 5. Testing

The repository maintains automated testing for configuration integrity, kinematics, and dependency verification.

### Running Test Suite
```powershell
# Run all tests verbosely
.\.conda-env\python.exe -m pytest -v

# Run smoke tests only
.\.conda-env\python.exe -m pytest tests/test_imports.py -v
```

---

## 6. Development Phases

| Phase | Milestone | Scope | Status |
|---|---|---|---|
| **Phase 0** | **Project Initialization** | Repository scaffold, dependencies, configs, docs, import smoke tests | **COMPLETED** |
| **Phase 1** | **Robot Kinematics & Control** | Franka Panda model, IK solver, Cartesian delta control, limits | **COMPLETED** |
| **Phase 2** | **GoalEnv & Reach Task** | Gymnasium GoalEnv base, sparse reward engine, Reach task | *Pending Approval* |
| **Phase 3** | **SAC + HER Training Pipeline** | SB3 SAC + HerReplayBuffer, evaluation protocol, Reach benchmark | *Pending Approval* |
| **Phase 4** | **Pick & Grasping Mechanics** | Object physics, contact points, grasping stabilization | Planned |
| **Phase 5** | **Pick and Place Task** | Dual-phase goal conditioning, target placement | Planned |
| **Phase 6** | **Sorting Task** | Multi-class sorting into target bins | Planned |
| **Phase 7** | **Multi-Object Manipulation** | Clutter, sequential picking, obstacle negotiation | Planned |
| **Phase 8** | **Multi-Part Assembly** | High-precision peg-in-hole and multi-part mating | Planned |
| **Phase 9** | **Domain Randomization & Robustness** | Physics and sensory domain randomization | Planned |
| **Phase 10** | **Production Platform** | FastAPI monitoring, React + TypeScript dashboard, Docker | Planned |
