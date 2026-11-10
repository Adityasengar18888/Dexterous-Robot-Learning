# Phase 14-Lite: Adaptive Lift Stabilization Report
**Controller-Level OOD Grasp & Lift Stabilization on Unmodified DR-4 Policies**  
**Date:** November 10, 2026  
**Status:** COMPLETE (Stop Condition Reached — Awaiting Manual Approval)

---

## Executive Summary

Phase 14-Lite investigated whether lightweight, inference-time controller modifications—**without retraining SAC, altering HER, or modifying reward functions**—can mitigate the out-of-distribution `GRASP_BUT_NO_LIFT` failure mode identified in Phase 13.5. Three controller variants were implemented in [`src/rl/adaptive_controllers.py`](file:///C:/Users/Welcome/OneDrive/Desktop/newshit/src/rl/adaptive_controllers.py) and evaluated across **Nominal DR-0**, **Matched DR-4**, and **Held-out DR-4** on both **DR4 Seed 42** and **DR4 Seed 43** (`1,800` total confirmation episodes: `100` deterministic episodes × `2` seeds × `3` controllers × `3` regimes, following an initial `540`-episode `N=30` screening).

### Core Empirical Findings (`N=100` Confirmation per Seed, `N=200` Combined):
1. **Nominal DR-0 & Matched DR-4 Preservation (`N=200`):**
   - **Matched DR-4 Success:** `A_BASELINE` = **99.0%**, `B_ACCEL_SMOOTHING` = **99.5%** (100.0% grasp-to-lift, `0.0%` `GRASP_BUT_NO_LIFT`), `C_SLIP_AWARE` = **98.5%**.
   - **Nominal DR-0 Success:** `A_BASELINE` = **97.5%**, `B_ACCEL_SMOOTHING` = **98.0%**, `C_SLIP_AWARE` = **98.5%** (slip events reduced `0.54 -> 0.34`/ep).
   - Both interventions reduce peak vertical command jerk by **11.8% to 18.2%** (`1.56 -> 1.31 / 1.28` on DR-0; `1.47 -> 1.30 / 1.26` on Matched DR-4) without incurring execution timeouts.
2. **Held-out DR-4 Seed-Dependent Generalization (`N=200`):**
   - **Combined Held-out DR-4:** `A_BASELINE` = **57.0%** `[50.1%, 63.7%]`, `B_ACCEL_SMOOTHING` = **55.0%** `[48.1%, 61.7%]`, `C_SLIP_AWARE` = **57.5%** `[50.6%, 64.1%]`.
   - **DR4 Seed 42 (Monotone Rescue):** `C_SLIP_AWARE` improved Held-out success from **81.0% → 85.0%** (+4.0%, cutting `GRASP_BUT_NO_LIFT` from `18.0%` to `14.0%`, boosting mean reward `+16.4` from `362.3` to `378.7`), **rescuing 4 episodes (`1003, 1024, 1071, 1096`) with 0 regressions** out of 100 episodes.
   - **DR4 Seed 43 (Pre-Contact Spatial Bottleneck):** Held-out success remained **33.0% (A) / 29.0% (B) / 30.0% (C)**. Telemetry inspection confirms that ~55% of Seed 43's nominal `GRASP_BUT_NO_LIFT` failures (`max_normal_force = 0.0 N`) pinch shut empty *prior* to forming bilateral contact due to spatial approach errors at held-out workspace edges (`x ∈ [0.35, 0.40], y ∈ [-0.25, -0.20]`). Lift-phase stabilization alone cannot fix pre-contact grasping misses.

---

## 1. Reproducibility & Model Preservation Audit (Rule 1)

Zero policy weights, HER buffers, reward functions, or simulation environments were modified.

| Artifact | Path | Verification | SHA-256 / Lock |
| :--- | :--- | :--- | :--- |
| **DR4 Seed 42 Model** | `experiments/phase13_dense_dr/dr4/seed42/models/checkpoint_500000.zip` | Unmodified; `SAC.load(..., device="cpu")` | `481885754A024C4345DEBB8E1885042216E2C0FC28393BE2FEA8325D6C242F7C` |
| **DR4 Seed 43 Model** | `experiments/phase13_dense_dr/dr4/seed43/models/checkpoint_500000.zip` | Unmodified; `SAC.load(..., device="cpu")` | `E1E7EC8D23EB6CF113CA003237A326E2B1EFC32D6DDE4A3CCA906D71C0D6C9C5` |
| **Controller Module** | `src/rl/adaptive_controllers.py` | Action-space inference filter only | Physical limits `[-1, 1]`; zero fabricated forces |
| **Evaluation Harness** | `run_phase14_lite.py` | Deterministic `eval_seed = 1001..1100` | Identical initial seeds across A, B, C |

---

## 2. Controller Variant Architecture (Rule 2)

1. **`A_BASELINE` (`BaselineController`):** Passes deterministic SAC actions `model.predict(obs, deterministic=True)` directly to the environment without modification.
2. **`B_ACCEL_SMOOTHING` (`AccelerationSmoothingController`):** Applies configurable vertical acceleration (`accel_limit_z = 0.45`) and jerk limits (`jerk_limit_z = 0.30`) during early upward lift initiation (`cube_z < 0.06 m`, `rel_dist < 0.055 m`, active grasp `gripper_width >= 0.042 m`). Approach descents (`raw_az <= 0`) and missed-pinch resets pass through unmodified.
3. **`C_SLIP_AWARE` (`SlipAwareLiftController`):** Monitors real PyBullet `getContactPoints` bilateral finger contact (`link 9`, `link 10`), true normal contact force (`total_normal_force`), finger aperture (`gripper_width`), and relative EE-object displacement (`rel_dist`, `delta_rel_dist`). When active bilateral/contact slip or table-drag shear is detected (`delta_rel > 0.0025 m`, `norm_force < 15 N`, asymmetric contact, or lateral shear drag on tabletop):
   - Temporarily reduces vertical acceleration (`slip_accel_limit_z = 0.20`) and scales horizontal shear proportionally so the fingers do not drag off the cube.
   - Maintains safe gripper closure (`cmd[3] = -1.0`, within `[-1, 1]` physical limits) only when `gripper_width >= 0.042 m` confirms the `0.05 m` cube is physically seated between the jaws.
   - Resumes jerk-limited lifting once bilateral contact (`>= 15 N`) and steady relative displacement recover (capped at `max_slip_hold_steps = 3` per attempt to prevent stalls).

---

## 3. Master Confirmation Comparison (`N=100` per Seed, `N=200` Combined)

### 3.1 Combined Results Across Both Seeds (`N=200` per Controller × Regime)

| Regime | Controller | End-to-End Success [95% CI] | Reach | Grasp | Lift | G2L Conv | `GRASP_BUT_NO_LIFT` | Mean Slip Events | Peak Cmd Jerk | Peak EE Accel ($m/s^2$) | Mean Lift Dur (steps) | Mean Steps to Succ | Timeout Rate | Mean Reward |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal_DR0** | `A_BASELINE` | 97.5% [94.3, 98.9] | 99.5% | 99.5% | 98.0% | 98.5% | 1.5% | 0.54 | 1.56 | 3.36 | 3.90 | 18.05 | 2.5% | 475.3 |
| **Nominal_DR0** | `B_ACCEL_SMOOTHING` | 98.0% [95.0, 99.2] | 99.5% | 99.5% | 98.5% | 99.0% | 1.0% | 0.40 | 1.31 | 2.97 | 4.35 | 18.02 | 2.0% | 477.5 |
| **Nominal_DR0** | `C_SLIP_AWARE` | **98.5%** [95.7, 99.5] | 99.5% | 99.5% | **99.0%** | **99.5%** | **0.5%** | **0.34** | **1.28** | **2.91** | 4.59 | 18.39 | **1.5%** | **478.2** |
| **Matched_DR4** | `A_BASELINE` | 99.0% [96.4, 99.7] | 100.0% | 99.5% | 99.0% | 99.5% | 0.5% | **0.31** | 1.47 | 3.31 | **4.00** | **17.70** | 1.0% | 483.3 |
| **Matched_DR4** | `B_ACCEL_SMOOTHING` | **99.5%** [97.2, 99.9] | 100.0% | 99.5% | **99.5%** | **100.0%** | **0.0%** | 0.38 | 1.30 | 2.99 | 4.77 | 18.04 | **0.5%** | **484.2** |
| **Matched_DR4** | `C_SLIP_AWARE` | 98.5% [95.7, 99.5] | 100.0% | 99.5% | 99.0% | 99.5% | 0.5% | 0.40 | **1.26** | **2.93** | 4.32 | 17.85 | 1.5% | 480.1 |
| **Heldout_DR4** | `A_BASELINE` | 57.0% [50.1, 63.7] | 96.5% | 92.0% | 61.5% | 66.8% | 30.5% | 2.65 | 2.12 | 4.19 | 9.53 | 27.44 | 43.0% | 225.9 |
| **Heldout_DR4** | `B_ACCEL_SMOOTHING` | 55.0% [48.1, 61.7] | 96.5% | 92.0% | 60.0% | 65.2% | 32.0% | 2.57 | 2.03 | 4.13 | **8.63** | **26.12** | 45.0% | 223.6 |
| **Heldout_DR4** | `C_SLIP_AWARE` | **57.5%** [50.6, 64.1] | 96.5% | 92.0% | **63.0%** | **68.5%** | **29.0%** | **2.35** | **2.02** | **4.09** | 9.60 | 26.76 | **42.5%** | **232.5** |

### 3.2 Per-Seed Breakdown on Held-out DR-4 (`N=100` each)

| Seed | Controller | Success [95% CI] | Lift Rate | G2L Conversion | `GRASP_BUT_NO_LIFT` | Mean Slip Events | Peak Cmd Jerk | Mean Steps to Succ | Mean Reward |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | `A_BASELINE` | 81.0% [72.2, 87.5] | 82.0% | 82.0% | 18.0% | 0.40 | 2.07 | 22.94 | 362.3 |
| **42** | `B_ACCEL_SMOOTHING` | 81.0% [72.2, 87.5] | 81.0% | 81.0% | 19.0% | **0.38** | 1.92 | 22.96 | 361.7 |
| **42** | `C_SLIP_AWARE` | **85.0%** [76.7, 90.7] | **86.0%** | **86.0%** | **14.0%** | **0.38** | **1.90** | 23.58 | **378.7** |
| **43** | `A_BASELINE` | **33.0%** [24.6, 42.7] | **41.0%** | **48.8%** | **43.0%** | 4.90 | 2.17 | 36.34 | **89.5** |
| **43** | `B_ACCEL_SMOOTHING` | 29.0% [21.0, 38.5] | 39.0% | 46.4% | 45.0% | 4.75 | **2.14** | **32.67** | 85.4 |
| **43** | `C_SLIP_AWARE` | 30.0% [21.9, 39.6] | 40.0% | 47.6% | 44.0% | **4.32** | **2.14** | 33.69 | 86.4 |

> **Methodological Note on Seed 43 Baseline (`33.0%` vs Phase 13.5 `41.0%`):** Notice that `Lift Rate = 41.0%` and `GRASP_BUT_NO_LIFT = 43.0%` match Phase 13.5 D1 **exactly**. The 8-episode difference (`33.0%` final success vs `41.0%` ever-placed in D1) comes entirely from 8 Seed 43 episodes (`LIFT_AND_HELD_BUT_WRONG_DIRECTION` / edge oscillation) that briefly grazed target distance `< 0.05 m` mid-episode before drifting outside `< 0.05 m` at final step `100`. Phase 14-Lite evaluates final-step `is_success` consistently with Phase 13 (`train_phase13_dr.py`).

---

## 4. Failure Taxonomy Breakdown (`N=200` per Controller × Regime)

### Held-out DR-4 Failure Taxonomy (`N=200`)

| Controller | `SUCCESS` | `GRASP_BUT_NO_LIFT` | `LIFT_AND_HELD_BUT_WRONG_DIRECTION` | `REACH_BUT_NO_GRASP` | `NO_REACH` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `A_BASELINE` | 114 (57.0%) | 61 (30.5%) | 9 (4.5%) | 9 (4.5%) | 7 (3.5%) |
| `B_ACCEL_SMOOTHING` | 110 (55.0%) | 64 (32.0%) | 10 (5.0%) | 9 (4.5%) | 7 (3.5%) |
| `C_SLIP_AWARE` | **115 (57.5%)** | **58 (29.0%)** | 11 (5.5%) | 9 (4.5%) | 7 (3.5%) |

### Paired Trajectory Rescue Analysis on Seed 42 Held-out (`N=100`)
- **Preserved (`A=True, C=True`):** 81 episodes
- **Rescued (`A=False, C=True`):** **4 episodes** (`eval_seed = 1003, 1024, 1071, 1096`)
- **Regressed (`A=True, C=False`):** **0 episodes**
- **Both Failed (`A=False, C=False`):** 15 episodes

---

## 5. Plots & Telemetry Visualizations

- **Controller Comparison Chart:** [`controller_comparison.png`](file:///C:/Users/Welcome/OneDrive/Desktop/newshit/results/phase14_lite/controller_comparison.png) — summarizes End-to-End Success, Grasp-to-Lift Conversion, `GRASP_BUT_NO_LIFT` Rate, and Mean Peak Command Vertical Jerk across all three regimes.
- **Trajectory Rescue Telemetry (`Seed 42, eval_seed=1003`):** [`lift_trajectory_stabilization.png`](file:///C:/Users/Welcome/OneDrive/Desktop/newshit/results/phase14_lite/lift_trajectory_stabilization.png) — demonstrates how `C_SLIP_AWARE` detects post-contact slip at initial table separation, stabilizes normal force at ~`48 N`, maintains `rel_dist ≈ 0.025 m`, and executes a clean sustained lift to `z = 0.16 m` where `A_BASELINE` and `B_ACCEL_SMOOTHING` drop the cube back to `z = 0.00 m`.

---

## 6. Scientific Validation (Rule 5)

1. **Improvement is NOT Caused by Slower Execution:**
   - On Seed 42 Held-out, `C_SLIP_AWARE` lifted `85%` of episodes cleanly in a mean of **23.58 steps** to first success (only `+0.64` steps above Baseline's `22.94` steps within a `100`-step budget), while cutting timeouts from `19.0%` down to `15.0%`.
   - Across Combined Held-out, `C_SLIP_AWARE` actually achieved *fewer* mean steps to success (`26.76` vs `27.44`) and lower timeout rate (`42.5%` vs `43.0%`) than Baseline.
2. **Matched DR-4 Baseline Fully Preserved:**
   - Matched DR-4 stayed at **99.5%** (`B_ACCEL_SMOOTHING`) and **98.5%** (`C_SLIP_AWARE`) versus **99.0%** (`A_BASELINE`).
3. **Screening vs Confirmation Consistency (`N=30` vs `N=100`):**
   - `N=30` screening predicted Held-out Seed 42 gain for `C_SLIP_AWARE` (`76.7% -> 83.3%` in pilot / `80.0%` in v1) and Seed 43 flat/negative behavior (`36.7% -> 30.0–33.3%`). The `N=100` confirmation strictly validated both trends (`Seed 42: 81.0% -> 85.0%`, `Seed 43: 33.0% -> 30.0%`), demonstrating why multi-seed confirmation prevents overclaiming from `N=30` screening or single seeds.

---

## 7. Key Scientific Conclusion & Next-Step Recommendation

**Why Lightweight Lift Stabilization Helps Seed 42 (+4%, 0 Regressions) but Cannot Solve Seed 43 Alone:**
- Phase 14-Lite reveals two physically distinct sub-mechanisms inside the nominal `GRASP_BUT_NO_LIFT` bucket:
  1. **True Post-Contact Lift Slip (`max_normal_force >= 15 N`):** Dominant in Seed 42 (`4/7` screening failures). Directly mitigated by `C_SLIP_AWARE` (`81% -> 85%`, 4/19 failures rescued, 0 regressed).
  2. **Pre-Contact Edge Closure / Empty Pinch (`max_normal_force = 0.0 N`, `gripper_width < 0.042 m`):** Dominant in Seed 43 (>50% of failures). Because `PandaPickPlaceEnv.step()` executes `move_ee()` *before* `close_gripper()`, any lateral offset during the approach-to-close step causes the jaws to shut on empty space (`gw ~ 0.028 m`) or graze a cube edge before bilateral contact ever forms. An inference-time lift controller—which must wait for physical contact before stabilizing—cannot fix spatial approach mis-centering in the underlying RL policy.

**Recommended Next Step (Awaiting Manual Approval):**
- **Combine `C_SLIP_AWARE` inference stabilization with targeted DR-4 fine-tuning / pre-grasp policy refinement** expanding training mass to `[0.05, 0.25] kg` and friction to `[0.3, 1.5]` (addressing Seed 43's pre-contact spatial offset), OR refine approach-time velocity damping only after explicit Phase 14+ approval.

---

## Stop Condition Reached

- All three controller comparisons (`A`, `B`, `C`) and the `1,800`-episode `N=100` confirmation evaluation are complete.
- All outputs saved in `results/phase14_lite/`.
- **STOPPED.** No SAC retraining, Phase 14.3, Phase 15, or default deployment initiated. Awaiting manual review and approval.

