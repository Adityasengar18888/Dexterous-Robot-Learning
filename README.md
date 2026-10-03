# Goal-Conditioned Robotic Manipulation Platform

A production-quality, reproducible robotic manipulation research platform using **Soft Actor-Critic (SAC)** with **Hindsight Experience Replay (HER)** in continuous control.

The system trains a simulated **7-DOF Franka Emika Panda** manipulator in PyBullet across a progressive curriculum of manipulation skills:
1. **Reach** (Phase 2 & 3)
2. **Pick** (Phase 4)
3. **Pick and Place** (Phase 5)
4. **Sorting** (Phase 6)
5. **Multi-Object Manipulation** (Phase 7)
6. **Multi-Part Assembly** (Phase 8)
7. **Robust Manipulation under Domain Randomization** (Phase 9)

---

## Architecture Overview

```
src/
├── robots/          # Robot kinematic models and controllers (Franka Panda, IK solver)
├── environments/    # Gymnasium GoalEnv compliant environments (Reach, Pick, etc.)
├── objects/         # Object models, tables, bins, fixtures, collision geometry
├── rl/              # SAC + HER training wrappers, policy evaluation, replay buffers
├── metrics/         # Telemetry, success rate calculators, TensorBoard loggers
└── utils/          # Configuration management, PyBullet client helpers, math utilities

configs/             # Parameterized YAML configurations (robot, tasks, algorithms)
docs/                # Architecture, methodology, experiments, and results logs
tests/               # Unit, kinematic, environment, and integration test suites
experiments/         # Experiment run scripts and launch manifests
checkpoints/         # Saved policy weights and replay buffer states
```

---

## Quickstart

### Prerequisites
- Python 3.11+
- Micromamba or Conda environment manager

### Environment Setup
```powershell
# Create environment with precompiled PyBullet & core dependencies
micromamba create -p .\.conda-env -c conda-forge python=3.11 pybullet -y
uv pip install --python .\.conda-env\python.exe gymnasium stable-baselines3 torch torchvision scipy pyyaml pandas tensorboard pytest
```

### Running Tests
```powershell
.\.conda-env\python.exe -m pytest -v
```

---

## Documentation
- [Architecture Details](file:///docs/architecture.md)
- [Methodology & RL Formulation](file:///docs/methodology.md)
- [Experiment Registry](file:///docs/experiments.md)
- [Empirical Results](file:///docs/results.md)
