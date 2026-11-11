# Section 1 — Phase 16 & 17 Baseline Audit and Preservation

**Audit Timestamp:** `2026-11-11T22:39:05+05:30`  
**Compute Platform:** CPU-only PyTorch `2.14.1+cpu` (`16` threads, `22` OS CPUs, `3.14 GB / 16.66 GB` RAM free)  
**Primary Simulator:** PyBullet (`franka_panda/panda.urdf` verified in `.conda-env`)  
**Secondary Simulator:** `mujoco 4.0.1` (installed & verified in `.conda-env`)  
**Regression Baseline:** `111 passed` (`.\.conda-env\python.exe -m pytest -q`)  

## 1. Preserved Phase 15 Checkpoints & Results

All Phase 15 models, configs, replay buffers, and reports under `experiments/` and `results/phase15_*` remain untouched:
- `experiments/pnp_dr4_seed42/models/best_model.zip` (`PickPlace` DR1–DR4 `95–100%`)
- `experiments/pnp_dr4_seed43/models/best_model.zip` (`PickPlace` DR1–DR4 `90–100%`, Delay1 `65%`)
- `experiments/multi_sort_phase15_5B7/seed42/models/bc_warmstart_model.zip` (`SinglePreGrasp` BC: `75%` Grasp / `45%` Lift, `100` eps)
- `experiments/multi_sort_phase15_5B2/seed42/models/checkpoint_17500_steps.zip` (`SinglePreGrasp` SAC: `62%` Grasp / `7%` Lift, `100` eps)
- `results/phase15_master/PHASE15_FINAL_TECHNICAL_REPORT.md` & `results/phase15_2_5A/phase15_2_5A_diagnostics.json` (`50/50` scripted sorting)

## 2. Measured Baseline CPU Throughput

- `PickPlace`: `426.1 env_steps/sec`
- `SinglePreGrasp`: `370.4 env_steps/sec`
- `AssemblyDense`: `310.5 env_steps/sec`
- `MultiSortPreGrasp`: `282.1 env_steps/sec`
- Isolated output workspaces initialized: `experiments/phase16_17/`, `results/phase16_17/`, `configs/phase16_17/`.

