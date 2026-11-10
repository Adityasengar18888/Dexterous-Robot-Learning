# Phase 15.2.5A Final Report — Pre-Grasp Acquisition & Collision-Aware Reward Engineering

**Project:** Robotic Manipulation using SAC + HER  
**Parent Phase:** 15.2 Multi-Object Sorting  
**Execution Scope:** Pre-grasp state machine, collision-aware reward & HER audit, `50`-episode diagnostic benchmark, and unit/HER verification (`NO SAC training executed`)  
**Preservation Status:** Phase 14 controllers, original sparse & Phase 15.2.2/15.2.4 dense environments, and all checkpoints/replay buffers (`experiments/multi_sort*`, `multi_sort_phase15_dense`, `multi_sort_phase15_4`) remain completely untouched.

---

## 1. Pre-Grasp Controller Architecture & State Machine

Implemented in [`src/controllers/pregrasp_controller.py`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/controllers/pregrasp_controller.py) ([`PreGraspController`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/controllers/pregrasp_controller.py#L140-L333), [`PreGraspConfig`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/controllers/pregrasp_controller.py#L31-L61)):

```text
APPROACH (z_obj + 0.115m, gw open +1.0)
   └─> ALIGNMENT (z_obj + 0.055m, xy_err <= 0.011m, orn_dot >= 0.985, gw open)
         └─> CONTROLLED_DESCENT (dz <= 0.016m/step, gw open, monitor table/cube disp; abort if unsafe)
               └─> GRASP_AUTHORIZATION (verify 5 gates: xy<=0.015m, z<=0.012m, orn>=0.985, table=0, disp<=20mm, gw>0.065m)
                     └─> GRASP (close cmd=-1.0; verify PyBullet f9 & f10 bilateral >= 10N)
                           └─> LIFT_VERIFICATION (Phase 14-Lite accel/jerk smooth lift >= 0.075m, verify held)
                                 └─> ACQUIRED (hand off to Phase 14-style transport/place)
```

Every transition is recorded in `self.transition_log` (`step`, `active_obj_idx`, `from`, `to`, `reason`, `ee_z`, `gw`).

---

## 2. Collision-Aware Reward Design & Measured Component Magnitudes

Implemented in [`src/environments/pregrasp_multi_sort_env.py`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/environments/pregrasp_multi_sort_env.py) ([`PandaMultiSortPreGraspEnv`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/environments/pregrasp_multi_sort_env.py#L46-L386), [`PreGraspRewardConfig`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/environments/pregrasp_multi_sort_env.py#L20-L43)):

| Component | Equation / Gate (`k = active_obj_idx`) | Scripted `10`-Ep Mean Sum (`Trig Freq`) | Idle (`300` Steps) | Premature Close (`300` Steps) |
| :--- | :--- | :---: | :---: | :---: |
| **A. Pre-Grasp Align** | `4.0*clip(Δxy,-0.05,0.05) - 0.25*clip(xy,0,1)` (ungrasped) | `+1.2028` (`45.95%`) | `-13.9448` | `+0.6634` |
| **B. Safe Vertical** | `2.0*clip(Δz_err,-0.05,0.05)` gated on `xy<=0.03, gw>0.065, tcoll=0, orn` | `+0.9098` (`29.50%`) | `0.0000` | `0.0000` |
| **C. Open Compliance** | `-0.06 * I[ungrasped & ~auth & gw<=0.065]` | **`0.0000`** (`0.0%`) | `-18.0000` | `-0.7200` |
| **D. Collision** | `-0.10*table_coll - 0.08*other_cube_contact` | `-0.0160` (`0.26%`) | `0.0000` | `0.0000` |
| **E. Cube Knock Disp** | `-6.0*clip(disp_xy-0.003,0,0.05)` (ungrasped, unplaced) | **`0.0000`** (`0.0%`) | `0.0000` | `0.0000` |
| **F. Grasp Auth** | `+0.50*(valid_curr - valid_prev)` (retained on bilateral grasp) | **`+1.0000`** (`2.61%`) | `0.0000` | `0.0000` |
| **G. Bilateral Grasp** | `+1.5*(bilat_curr - bilat_prev)` (`f9 & f10 >= 5.0N`) | `0.0000` net (`5.22%`) | `0.0000` | `+1.5000` |
| **H. Lift Potential** | `1.5*Δclip((z-0.025)/0.095,0,1)*outside_bin*held` | **`+3.0000`** (`7.96%`) | `0.0000` | `0.0000` |
| **Transport / Place / Done / Step** | `6.0*Δg_d` / `+4.0,-4.5` / `+8.0` / `-0.02` | `+4.21` / `+8.0` / `+8.0` / `-1.51` | `0` / `0` / `0` / `-6.0` | `+0.28` / `0` / `0` / `-6.0` |
| **Total Episode Return** | Sum of all bounded terms | **`+24.7964`** | **`-37.9491`** | **`-4.2783`** |

---

## 3. HER Compatibility Audit Findings

1. **Zero Stale Simulator State**: Controller internal state (`transition_log`, `stage_steps`, `anchor_xy`) is strictly isolated in `PreGraspController` and NEVER referenced inside `PandaMultiSortPreGraspEnv.compute_reward()`.
2. **Stateless Transition Contract (`copy_info_dict=True`)**: `step()` stores only goal-agnostic `prev`/`curr` physical quantities (`prev_ee_pos`, `ee_pos`, `prev_ag`, `prev_gw`, `gripper_width`, `prev_bilateral_contact`, `bilateral_contact`, `prev_orn_ok`, `orn_ok`, `prev_table_collision`, `table_collision`, `any_cube_contact`, `bin_positions`, `active_color_indices`).
3. **Relabeled Goal Reconstruction**: Verified in [`test_pregrasp_her_compatibility.py`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/tests/test_pregrasp_her_compatibility.py) that when HER relabels `desired_goal[0:3]` onto Cube 0's position, `active_obj_idx` dynamically switches from `0 -> 1` cleanly, mutating live simulator `cubes[0]` afterwards produces identical batched rewards (`atol = 1e-5`), and `HerReplayBuffer(future, n=4, copy_info_dict=True).sample(32)` yields finite rewards (`NaN/Inf = 0`).

---

## 4. Scripted Diagnostic Benchmark & Acceptance Criteria Verification

Full per-episode data in [`experiments/multi_sort_phase15_5A/phase15_2_5A_diagnostics.json`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/experiments/multi_sort_phase15_5A/phase15_2_5A_diagnostics.json) and [`results/phase15_2_5A/phase15_2_5A_diagnostics.json`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/results/phase15_2_5A/phase15_2_5A_diagnostics.json):

| Metric | Target | Object 1 Standalone (`N=20`) | Object 1 Sequential (`N=50`) | Object 2 After O1 (`N=50`) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Approach Success** | `>= 95%` | `100.0%` (`10.0` steps) | **`100.0%`** (`10.0` steps) | **`100.0%`** (`8.14` steps) | **PASS** |
| **Alignment Success** | `>= 90%` | `100.0%` (`3.0` steps) | **`100.0%`** (`3.06` steps) | **`100.0%`** (`3.02` steps) | **PASS** |
| **Safe Descent Success** | `>= 95%` | `100.0%` (`3.1` steps) | **`100.0%`** (`3.08` steps) | **`100.0%`** (`3.04` steps) | **PASS** |
| **Grasp Authorization** | — | `100.0%` (`1.0` step) | `100.0%` (`1.0` step) | `100.0%` (`1.0` step) | **PASS** |
| **Premature Closure** | `0%` | `0.0%` | **`0.0%`** | **`0.0%`** | **PASS** |
| **Bilateral Grasp** | `>= 90%` | `100.0%` (`1.0` step) | **`100.0%`** (`1.0` step) | **`100.0%`** (`1.0` step) | **PASS** |
| **Lift \| Bilateral Grasp** | `>= 90%` | `100.0%` (`4.05` steps) | **`100.0%`** (`4.08` steps) | **`100.0%`** (`4.02` steps) | **PASS** |
| **Table Collisions (Episodes)** | `<= 5%` | `0.0%` | **`0.0%`** (`0` steps) | **`0.0%`** (`0` steps) | **PASS** |
| **Disp Failures (`> 20 mm`)** | `<= 5%` | `0.0%` (`0.0 mm`) | **`0.0%`** (`0.0 mm`) | **`0.0%`** (`0.0 mm`) | **PASS** |

- **Sequential Sorting Comparison vs Phase 15.2.2 Baseline**:
  - Phase 15.2.2 scripted (`N = 20`): `100.0%` (`20/20`), `mean_steps = 76.0`, `0` drops/collisions/wrong bins.
  - Phase 15.2.5A `PreGraspController` sequential (`N = 50`): **`100.0%` (`50/50`)**, `mean_steps = 76.4` (`+0.4` step overhead for full 5-gate pre-grasp authorization & verification), `0` drops, `0` wrong bins, `0` table collisions, `0` Object 1 disturbances. *(Note: scripted benchmark validates environment/controller physics and reward calibration; it is not learned SAC performance.)*

---

## 5. Unit & Regression Tests, Runtime & Limitations

- **Pytest Suite** (`pytest tests\test_pregrasp_controller.py tests\test_pregrasp_reward.py tests\test_pregrasp_her_compatibility.py tests\test_dense_multi_sort_env.py tests\test_multi_sort_env.py tests\test_phase15_wrappers.py -v`):
  **`24 passed in 27.08s`** (`0` failed, `0` skipped — `6` new Phase 15.2.5A tests + `11` V1/V2 dense tests + `3` sparse MultiSort + `4` Phase 15.1 wrappers).
- **Hardware & Runtime**: CPU-only (`0` SAC training), `task-1254` diagnostic benchmark completed in `~155 s` (`2.58 min`), Python RSS `149.9 MB`, `1.73 GB` free system RAM.
- **Remaining Limitations & Recommendation for Phase 15.2.5B**:
  - `PandaMultiSortPreGraspEnv` (`MultiSortPreGrasp`) is configured in [`configs/multi_sort_phase15_5A.yaml`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/configs/multi_sort_phase15_5A.yaml) and tested via `Monitor` + `HerReplayBuffer`, but is not yet registered in [`src/rl/train.py`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/rl/train.py) / [`src/rl/evaluate.py`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/rl/evaluate.py).
  - **Recommendation**: In **Phase 15.2.5B**, register `MultiSortPreGrasp` in `train.py`/`evaluate.py` and launch a `20k–50k` smoke/validation SAC+HER run (optionally pairing with a small replay-buffer warm-start from `PreGraspController` trajectories if needed) to verify whether open-gripper alignment & collision penalties unlock `grasp` emergence cleanly.

**FINAL STATUS: COMPLETE**  
**STOPPED — Awaiting explicit user approval before Phase 15.2.5B.**

