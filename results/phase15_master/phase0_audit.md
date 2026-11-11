# Phase 0 Project Audit & 12-Hour Execution Dependency Graph

**Audit Timestamp:** `2026-11-11T20:44:37+05:30` (`12h` Hard Budget Deadline: `2026-11-12T08:44:37+05:30`)  
**Repository Baseline Check:** Full `pytest -q` suite: **`87 passed` (`0` failed, `0` skipped)**

---

## 1. Existing Milestones & Preserved Artifacts

| Historical Phase | Key Implementation / Artifacts | Verified Baseline | Preservation Action |
| :--- | :--- | :--- | :--- |
| **Phases 0–13.5** | `src/robots/panda.py`, `pick_place_env.py`, `experiments/phase13_dense_dr/dr4/seed42,43/` | Matched DR4 `99%`, Held-out DR4 `57%` (`33%` S43, `81%` S42) | Preserved untouched (`SHA-256` verified) |
| **Phase 14 (14-Lite/3/4)** | `src/rl/adaptive_controllers.py`, `configs/phase14_4_config.yaml` | `E_ADAPTIVE_TRANSPORT`: **`98.0%`** Held-out DR4 (`N=200`), **`100%`** DR0, **`98.5%`** Matched DR4 | Preserved untouched; used only as frozen baseline |
| **Phase 15.1** | `src/rl/sim_to_real_*_wrapper.py`, `unprivileged_contact_wrapper.py`, `run_phase15_1.py` | Clean `97.5%`, `±5 mm` noise `95.0%` (Class A), `>=2-step` delay `0%` (Class D), unpriv force `42.5%` | Preserved untouched as frozen sim-to-real baseline |
| **Phase 15.2.1–15.2.4** | `multi_sort_env.py`, `dense_multi_sort_env.py`, `experiments/multi_sort_phase15_4/` | `50k`: min EE-O1 `0.489 m -> 0.175 m` (`20%` Reach, `0%` Grasp) | Preserved untouched |
| **Phase 15.2.5A** | `src/controllers/pregrasp_controller.py`, `src/environments/pregrasp_multi_sort_env.py` | Scripted two-object sequential sorting: **`50/50 (100%)`**, `0` table collisions | Preserved untouched; reused as demo generator |
| **Phase 15.2.5B / B.1** | `src/environments/single_pregrasp_env.py`, `src/rl/transfer_utils.py`, `experiments/multi_sort_phase15_5B1/` | `B`: XY align `0.011 m`, stuck at `z=0.50 m`. `B.1`: `z_err 0.479->0.0424 m`, peak `60% Reach / 40% Grasp`, `0% Lift` (`78.4%` closed-gripper demo tail, `ent_coef=0.00804`) | Preserved untouched; new B.2 run uses `experiments/multi_sort_phase15_5B2/` |
| **Phase 9/15.3 Base** | `src/environments/assembly_env.py`, `src/objects/parts.py`, `tests/test_assembly_env.py` | `PandaAssemblyEnv` (`22D` obs, `7D` pos+quat goal, `5D` action `[dx,dy,dz,dyaw,gripper]`, Peg `0.06m` into Receptacle `0.08m` hole) | Separate assembly extension (`PandaAssemblyV2Env` / controller) preserves base env |

---

## 2. Compact Execution Dependency Graph

```text
[PHASE 0: Audit & Graph (87/87 pytest)]
        |
        v
[PHASE 1: Single-Object Grasp & Lift (Phase 15.2.5B.2 + 15.2.5B.3)]
  - Rebalanced scripted demos (65-75% open descent, 10-15% auth/close, 10-15% lift, 5-10% hold)
  - Auxiliary BC objective (lambda_BC: 0.305 -> 0.02) + Entropy floor (min_ent_coef >= 0.02)
  - 5k diagnostic -> 10k-20k -> 100-ep B.3 generalization benchmark
        |
        +--- Gate PASS (Grasp & Lift learned) ---> [PHASE 2: Two-Object Sequential Sorting (15.2.5C)]
        |                                          Transfer 28D->43D (transfer_single_to_multi_sac)
        |                                          Train O1 -> O2 sequential -> 100-ep benchmark
        |
        v
[PHASE 3: Frozen-Policy Generalization & Robustness (DR1/DR2/DR4, Pose/Ori/Noise/Delay, Multi-Seed)]
  - Evaluates Phase 1/2 learned checkpoints (if passed) + Phase 14 DR4 Seed 42/43 frozen baselines
        |
        v
[PHASE 4: Orientation-Aware Assembly (Peg-in-Hole Manipulation, Contact/Jam Detection, Alignment)]
  - Quaternion-aware orientation control + scripted feasibility + learned/BC-SAC assembly evaluation
        |
        v
[PHASE 5: Full Regression Suite, Benchmark Consolidation & Final Phase 15 Master Report]
```

