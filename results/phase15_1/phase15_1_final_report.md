# Phase 15.1 Final Report — Sim-to-Real Transfer Readiness Benchmark

**Project:** Robotic Manipulation using SAC + HER  
**Phase:** 15.1 (Observation Noise, Control Latency & Unprivileged-Contact Robustness)  
**Target Controller:** Frozen Phase 14 `E_ADAPTIVE_TRANSPORT` ([`AdaptiveTransportController`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/rl/adaptive_controllers.py))  
**Authorization Scope:** Phase 15.1 benchmark + Phase 15.2/15.3 planning only (Phase 15.2/15.3 implementation NOT executed)

---

## 1. Executive Summary & Readiness Verdict

Evaluating the frozen Phase 14 `E_ADAPTIVE_TRANSPORT` controller with unmodified DR4 Seed 42 and Seed 43 SAC checkpoints on Held-Out DR4 (`N = 20` paired episodes per seed per condition, `320` total episodes, `0` NaN detections) reveals a sharp dichotomy in sim-to-real transfer readiness:

1. **Pose Observation Noise (`A1: ±3 mm`, `A2: ±5 mm`) — Class `A - ROBUST`**:
   - Combined held-out success holds steady at **`97.5%`** (`0.0 pp`) at `±3 mm` (`σ = 1.5 mm`) and **`95.0%`** (`-2.5 pp`) at `±5 mm` (`σ = 2.5 mm`), relative to the **`97.5%`** clean baseline (`A0`).
   - Seed 42 improves from `95.0%` (`0.0243 m`) to `100.0%` (`0.0061 m` at `±3 mm`, `0.0068 m` at `±5 mm`), while Seed 43 degrades gracefully from `100.0%` to `95.0%` (`±3 mm`) and `90.0%` (`±5 mm`).
2. **Closed-Loop Control/Observation Latency (`B1..B4`) — Class `C - FRAGILE` at 1 step, Class `D - FAILED` at >= 2 steps**:
   - At `1-step` observation delay (`B1`, ~`83.3–104.2 ms`), combined success drops **`-25.0 pp`** to **`72.5%`** (`55.0%` Seed 42 / `90.0%` Seed 43), driven by `GRASP_BUT_NO_LIFT` (`10/40`).
   - At `2-step` observation delay (`B2`), `4-step` observation delay (`B3`), and `1-step` obs + `1-step` action delay (`B4`, total 2-step loop RTT), combined success collapses to **`0.0%`** (**`-97.5 pp`**). Notice `B4 (1+1)` (`0.0%`, `Reach=80%`, `Grasp=35%`) closely mirrors `B2 (2+0)` (`0.0%`, `Reach=75%`, `Grasp=30%`), confirming closed-loop breakdown scales with total round-trip transport lag.
3. **Unprivileged-Contact Kinematic Fallback (`C1_UNPRIVILEGED`) — Class `D - FAILED`**:
   - Stripping PyBullet contact points/forces cleanly inside [`strip_privileged_telemetry`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/rl/unprivileged_contact_wrapper.py#L43-L51) causes combined success to drop **`-55.0 pp`** to **`42.5%`** (`45.0%` Seed 42 / `40.0%` Seed 43).
   - Reach (`100.0%`) and Grasp (`100.0%`) remain intact, and every single failure (`23/23` across both seeds) is `GRASP_BUT_NO_LIFT`, proving that the current Phase 14 lift gate relies critically on simulator-only bilateral contact force telemetry (`f9`/`f10`) that a simple jaw-width dwell heuristic cannot safely replace on heavier, slipperier held-out objects.

| Perturbation Axis | Condition | Seed 42 (`N=20`) | Seed 43 (`N=20`) | Combined (`N=40`) | Delta vs Clean | Readiness Class |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Clean Reference (`B0`/`C0`) | `A0_CLEAN_BASELINE` | `95.0%` | `100.0%` | **`97.5%`** | `0.0 pp` | **`A - ROBUST`** |
| Observation Noise (`±3 mm`) | `A1_NOISE_3MM` | `100.0%` | `95.0%` | **`97.5%`** | `0.0 pp` | **`A - ROBUST`** |
| Observation Noise (`±5 mm`) | `A2_NOISE_5MM` | `100.0%` | `90.0%` | **`95.0%`** | `-2.5 pp` | **`A - ROBUST`** |
| Latency (`1-step` obs) | `B1_OBS_DELAY_1` | `55.0%` | `90.0%` | **`72.5%`** | `-25.0 pp` | **`C - FRAGILE`** |
| Latency (`2-step` obs) | `B2_OBS_DELAY_2` | `0.0%` | `0.0%` | **`0.0%`** | `-97.5 pp` | **`D - FAILED`** |
| Latency (`4-step` obs) | `B3_OBS_DELAY_4` | `0.0%` | `0.0%` | **`0.0%`** | `-97.5 pp` | **`D - FAILED`** |
| Latency (`1 obs + 1 act`) | `B4_COMBINED_DELAY_1_1` | `0.0%` | `0.0%` | **`0.0%`** | `-97.5 pp` | **`D - FAILED`** |
| Contact Mode (No PyBullet Force) | `C1_UNPRIVILEGED` | `45.0%` | `40.0%` | **`42.5%`** | `-55.0 pp` | **`D - FAILED`** |

---

## 2. Audit, Integrity & Experimental Protocol

### 2.1 Frozen Checkpoints & SHA-256 Verification
Verified before evaluation in [run_phase15_1.py](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/run_phase15_1.py) against [phase15_1_config.yaml](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/configs/phase15_1_config.yaml):
- **DR4 Seed 42**: `experiments/phase13_dense_dr/dr4/seed42/models/checkpoint_500000.zip`  
  - SHA-256: `481885754A024C4345DEBB8E1885042216E2C0FC28393BE2FEA8325D6C242F7C` (Verified)
- **DR4 Seed 43**: `experiments/phase13_dense_dr/dr4/seed43/models/checkpoint_500000.zip`  
  - SHA-256: `E1E7EC8D23EB6CF113CA003237A326E2B1EFC32D6DDE4A3CCA906D71C0D6C9C5` (Verified)
- **Immutability Compliance**: Zero modifications to `src/rl/adaptive_controllers.py`, SAC weights, HER buffers, or reward functions.

### 2.2 Benchmark Configuration & Unit Tests
- **Environment**: [`PandaPickPlaceDenseRewardEnv`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/environments/dense_pick_place_env.py) configured via [pick_place_heldout.yaml](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/configs/pick_place_heldout.yaml) (`distance_threshold=0.05 m`, `max_episode_steps=100`, `action_scale=0.05`, edge spawn `x ∈ [0.35, 0.40]`, `y ∈ [-0.25, -0.20]`, high target `x ∈ [0.65, 0.70]`, `y ∈ [0.20, 0.25]`, `z ∈ [0.15, 0.20]`, randomized orientation, mass `0.20–0.25 kg`, friction `0.3–0.5`).
- **Effective Control Period**: Each RL step executes `10–15` IK substeps + `10` gripper substeps at `240 Hz` = `20–25` PyBullet substeps (`83.3–104.2 ms/step`, `~9.6–12.0 Hz`).
- **Paired Initial Conditions**: Fixed `eval_seed` range `1001..1020` (`20` episodes per seed per condition).
- **Wrapper Verification**: `pytest tests/test_phase15_wrappers.py -v` -> **`4 passed in 1.06s`** (verifying noise bounds/reset repeatability, FIFO obs/action queue timing, strict removal/assertion of all `9` privileged contact keys, and kinematic lift behavior).

---

## 3. Perturbation Wrappers (Implemented Outside Frozen Controller)

1. **Observation Noise** — [`ObservationNoiseWrapper`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/rl/sim_to_real_observation_wrapper.py#L14-L90): Adds reproducible (`episode_seed`-seeded) zero-mean Gaussian noise (`σ = bound / 2.0`, clipped to `±bound`) consistently to both RL Dict observation and controller telemetry for `ee_pos`, `cube_pos`, and `desired_goal`, leaving simulator physics, `obj_vel`, and `gripper_width` intact.
2. **Latency** — [`LatencyWrapper`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/rl/sim_to_real_latency_wrapper.py#L15-L64): Bounded FIFO `deque` queues pre-filled at `t=0` for `obs_delay_steps` (`0, 1, 2, 4`) and `action_delay_steps` (`0, 1` with neutral zero-motion open command prefill).
3. **Unprivileged Contact** — [`strip_privileged_telemetry`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/rl/unprivileged_contact_wrapper.py#L43-L51) & [`UnprivilegedAdaptiveController`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/rl/unprivileged_contact_wrapper.py#L54-L268): Removes and asserts absence of `contact_count`, `f9_contact_count`, `f10_contact_count`, `f9_normal_force`, `f10_normal_force`, `total_normal_force`, `total_lateral_force`, `contact_symmetry`, and `has_contact`. Replaces simulator contact detection with a `2`-step kinematic jaw-stall dwell (`gripper_width ∈ [0.042, 0.066] m`, `|Δgw| < 0.010 m`, `rel_dist < 0.048 m`) and kinematic slip monitoring (`delta_rel_dist`).

---

## 4. Detailed Results — Seed-Wise & Combined

### 4.1 Seed 42 Results (`N = 20` per condition)

| Condition | Success | Δ pp | Class | Reach | Grasp | Bilateral | Lift | Transport | Placement | Timeout | Mean Steps to Succ | Mean Final Dist (m) | Mean Reward |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `A0_CLEAN_BASELINE` | `95.0%` | `0.0` | `A - ROBUST` | `100.0%` | `100.0%` | `100.0%` | `95.0%` | `95.0%` | `95.0%` | `5.0%` | `35.05` | `0.02434` | `358.85` |
| `A1_NOISE_3MM` | `100.0%` | `+5.0` | `A - ROBUST` | `100.0%` | `100.0%` | `100.0%` | `100.0%` | `100.0%` | `100.0%` | `0.0%` | `34.10` | `0.00614` | `385.87` |
| `A2_NOISE_5MM` | `100.0%` | `+5.0` | `A - ROBUST` | `100.0%` | `100.0%` | `100.0%` | `100.0%` | `100.0%` | `100.0%` | `0.0%` | `35.85` | `0.00675` | `374.98` |
| `B1_OBS_DELAY_1` | `55.0%` | `-40.0` | `C - FRAGILE` | `100.0%` | `95.0%` | `85.0%` | `55.0%` | `55.0%` | `55.0%` | `45.0%` | `40.73` | `0.24110` | `161.18` |
| `B2_OBS_DELAY_2` | `0.0%` | `-95.0` | `D - FAILED` | `70.0%` | `25.0%` | `20.0%` | `5.0%` | `0.0%` | `0.0%` | `100.0%` | `36.00` | `0.67678` | `-72.73` |
| `B3_OBS_DELAY_4` | `0.0%` | `-95.0` | `D - FAILED` | `30.0%` | `0.0%` | `0.0%` | `0.0%` | `0.0%` | `0.0%` | `100.0%` | `-1.00` | `0.64909` | `-76.16` |
| `B4_COMBINED_DELAY_1_1` | `0.0%` | `-95.0` | `D - FAILED` | `80.0%` | `40.0%` | `35.0%` | `0.0%` | `0.0%` | `0.0%` | `100.0%` | `-1.00` | `0.63764` | `-72.79` |
| `C1_UNPRIVILEGED` | `45.0%` | `-50.0` | `D - FAILED` | `100.0%` | `100.0%` | `N/A (Unpriv)` | `45.0%` | `45.0%` | `45.0%` | `55.0%` | `27.22` | `0.31680` | `160.18` |

### 4.2 Seed 43 Results (`N = 20` per condition)

| Condition | Success | Δ pp | Class | Reach | Grasp | Bilateral | Lift | Transport | Placement | Timeout | Mean Steps to Succ | Mean Final Dist (m) | Mean Reward |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `A0_CLEAN_BASELINE` | `100.0%` | `0.0` | `A - ROBUST` | `100.0%` | `100.0%` | `100.0%` | `100.0%` | `100.0%` | `100.0%` | `0.0%` | `32.65` | `0.00553` | `394.99` |
| `A1_NOISE_3MM` | `95.0%` | `-5.0` | `A - ROBUST` | `100.0%` | `100.0%` | `100.0%` | `95.0%` | `95.0%` | `95.0%` | `5.0%` | `30.89` | `0.03756` | `382.61` |
| `A2_NOISE_5MM` | `90.0%` | `-10.0` | `A - ROBUST` | `100.0%` | `100.0%` | `100.0%` | `90.0%` | `90.0%` | `90.0%` | `10.0%` | `32.33` | `0.05407` | `351.44` |
| `B1_OBS_DELAY_1` | `90.0%` | `-10.0` | `A - ROBUST` | `100.0%` | `100.0%` | `100.0%` | `90.0%` | `90.0%` | `90.0%` | `10.0%` | `42.06` | `0.07502` | `294.36` |
| `B2_OBS_DELAY_2` | `0.0%` | `-100.0` | `D - FAILED` | `80.0%` | `35.0%` | `25.0%` | `5.0%` | `0.0%` | `0.0%` | `100.0%` | `-1.00` | `0.62396` | `-74.00` |
| `B3_OBS_DELAY_4` | `0.0%` | `-100.0` | `D - FAILED` | `60.0%` | `10.0%` | `5.0%` | `0.0%` | `0.0%` | `0.0%` | `100.0%` | `-1.00` | `0.51756` | `-67.39` |
| `B4_COMBINED_DELAY_1_1` | `0.0%` | `-100.0` | `D - FAILED` | `80.0%` | `30.0%` | `30.0%` | `5.0%` | `0.0%` | `0.0%` | `100.0%` | `83.00` | `0.59456` | `-74.99` |
| `C1_UNPRIVILEGED` | `40.0%` | `-60.0` | `D - FAILED` | `100.0%` | `100.0%` | `N/A (Unpriv)` | `40.0%` | `40.0%` | `40.0%` | `60.0%` | `30.75` | `0.34321` | `126.42` |

### 4.3 Combined Results (Seed 42 + Seed 43, `N = 40` per condition)

| Condition | Success | Δ pp | Class | Reach | Grasp | Bilateral | Lift | Drop | WrongDir | Overshoot/Ep | Timeout | Mean Final Dist (m) | Mean Reward |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `A0_CLEAN_BASELINE` | `97.5%` | `0.0` | `A - ROBUST` | `100.0%` | `100.0%` | `100.0%` | `97.5%` | `0.0%` | `0.0%` | `0.00` | `2.5%` | `0.01493` | `376.92` |
| `A1_NOISE_3MM` | `97.5%` | `0.0` | `A - ROBUST` | `100.0%` | `100.0%` | `100.0%` | `97.5%` | `0.0%` | `0.0%` | `0.00` | `2.5%` | `0.02185` | `384.24` |
| `A2_NOISE_5MM` | `95.0%` | `-2.5` | `A - ROBUST` | `100.0%` | `100.0%` | `100.0%` | `95.0%` | `0.0%` | `0.0%` | `0.00` | `5.0%` | `0.03041` | `363.21` |
| `B1_OBS_DELAY_1` | `72.5%` | `-25.0` | `C - FRAGILE` | `100.0%` | `97.5%` | `92.5%` | `72.5%` | `0.0%` | `0.0%` | `0.03` | `27.5%` | `0.15806` | `227.77` |
| `B2_OBS_DELAY_2` | `0.0%` | `-97.5` | `D - FAILED` | `75.0%` | `30.0%` | `22.5%` | `5.0%` | `0.0%` | `5.0%` | `0.33` | `100.0%` | `0.65037` | `-73.37` |
| `B3_OBS_DELAY_4` | `0.0%` | `-97.5` | `D - FAILED` | `45.0%` | `5.0%` | `2.5%` | `0.0%` | `0.0%` | `0.0%` | `0.00` | `100.0%` | `0.58332` | `-71.78` |
| `B4_COMBINED_DELAY_1_1` | `0.0%` | `-97.5` | `D - FAILED` | `80.0%` | `35.0%` | `32.5%` | `2.5%` | `0.0%` | `2.5%` | `0.05` | `100.0%` | `0.61610` | `-73.89` |
| `C1_UNPRIVILEGED` | `42.5%` | `-55.0` | `D - FAILED` | `100.0%` | `100.0%` | `N/A (Unpriv)` | `42.5%` | `0.0%` | `0.0%` | `0.00` | `57.5%` | `0.33001` | `143.30` |

![Phase 15.1 Sim-to-Real Readiness Chart](file:///C:/Users/Welcome/.gemini/antigravity/brain/4742db0c-dd40-42a9-8b0e-4d26031388ef/phase15_1_readiness_chart.png)

---

## 5. Failure Taxonomy & Mechanistic Diagnosis

### 5.1 Failure Counts (`results/phase15_1/phase15_1_episodes.csv`, `N = 40` Combined / `N = 20` per Seed)

| Condition | Seed 42 (`S / GBNL / RBG / NR / WD`) | Seed 43 (`S / GBNL / RBG / NR / WD`) | Combined (`S / GBNL / RBG / NR / WD`) |
| :--- | :---: | :---: | :---: |
| `A0_CLEAN_BASELINE` | `19 / 1 / 0 / 0 / 0` | `20 / 0 / 0 / 0 / 0` | `39 / 1 / 0 / 0 / 0` |
| `A1_NOISE_3MM` | `20 / 0 / 0 / 0 / 0` | `19 / 1 / 0 / 0 / 0` | `39 / 1 / 0 / 0 / 0` |
| `A2_NOISE_5MM` | `20 / 0 / 0 / 0 / 0` | `18 / 2 / 0 / 0 / 0` | `38 / 2 / 0 / 0 / 0` |
| `B1_OBS_DELAY_1` | `11 / 8 / 1 / 0 / 0` | `18 / 2 / 0 / 0 / 0` | `29 / 10 / 1 / 0 / 0` |
| `B2_OBS_DELAY_2` | `0 / 4 / 9 / 6 / 1` | `0 / 6 / 9 / 4 / 1` | `0 / 10 / 18 / 10 / 2` |
| `B3_OBS_DELAY_4` | `0 / 0 / 6 / 14 / 0` | `0 / 2 / 10 / 8 / 0` | `0 / 2 / 16 / 22 / 0` |
| `B4_COMBINED_DELAY_1_1` | `0 / 8 / 8 / 4 / 0` | `0 / 5 / 10 / 4 / 1` | `0 / 13 / 18 / 8 / 1` |
| `C1_UNPRIVILEGED` | `9 / 11 / 0 / 0 / 0` | `8 / 12 / 0 / 0 / 0` | `17 / 23 / 0 / 0 / 0` |

*(Legend: `S` = `SUCCESS`, `GBNL` = `GRASP_BUT_NO_LIFT`, `RBG` = `REACH_BUT_NO_GRASP`, `NR` = `NO_REACH`, `WD` = `LIFT_AND_HELD_BUT_WRONG_DIRECTION`.)*

### 5.2 Root-Cause Findings
- **Observation Noise Resilience (`A1`, `A2`)**: `±3 mm` and `±5 mm` position jitter sits comfortably inside the `4 cm` object half-aperture / `5 cm` `distance_threshold`, `±3 cm` fine alignment radius, and velocity-damped settling check (`|cube_vel| < 0.08`). Small noise on Seed 42 (`100%` vs `95%` clean) appears to break a single marginal corner-grasp stall without disrupting proportional transport braking.
- **Latency Instability (`B1`–`B4`)**: With `action_scale = 0.05` (`5 cm/step` max displacement) over an `~83–104 ms` step, both the frozen SAC actor and `E_ADAPTIVE_TRANSPORT`'s proportional edge-approach and fine-alignment terms act as high-gain feedback loops. Even `1` delayed step (`B1`) induces `GRASP_BUT_NO_LIFT` (`10/40`, disproportionately `8/20` on Seed 42 vs `2/20` on Seed 43), while `2+` steps (`B2`, `B4`, `B3`) violate closed-loop phase margin—causing orbital approach oscillations (`NO_REACH`: `10/40` in `B2`, `22/40` in `B3`), grasping air (`REACH_BUT_NO_GRASP`: `18/40` in `B2` & `B4`), and complete `0.0%` collapse.
- **Unprivileged-Contact Bottleneck (`C1`)**: In [`UnprivilegedAdaptiveController.step_action`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/rl/unprivileged_contact_wrapper.py#L122-L185), removing PyBullet normal-force feedback eliminates true bilateral seating verification. Because `Stage 2` jerk/acceleration limiting only activates once `self.grasp_acquired` becomes `True` via the `2`-step `gripper_width ∈ [0.042, 0.066]` stall dwell, subtle edge mis-seating or asynchronous closure misses the kinematic trigger or prematurely zeroes `cmd[2]`, yielding `23/40` `GRASP_BUT_NO_LIFT` failures (`42.5%` overall) with `0` drops and `0` transport failures once lifted.

---

## 6. Environment, Hardware & Runtime Log

- **OS**: `Windows-10-10.0.26300-SP0`
- **CPU**: `Intel(R) Core(TM) Ultra 7 155H` (`16` physical cores / `22` logical processors)
- **RAM**: `16.66 GB` (`16,660,357,120` bytes)
- **Software Versions**: Python `3.11.16`, PyTorch `2.14.1+cpu`, Stable-Baselines3 `2.9.0`, Gymnasium `1.3.0`, PyBullet API `202010061`, NumPy `2.4.6`, Pandas `3.0.6`
- **Execution Wall Time & Memory**:
  - Seed 42 worker (`run_phase15_1.py --seed 42`): `318.3 s`, peak RSS `459.0 MB`, exit code `0`
  - Seed 43 worker (`run_phase15_1.py --seed 43`): `356.4 s`, peak RSS `459.1 MB`, exit code `0`
  - Combined CPU time: `674.7 s` (`~6 min` concurrent wall time; total Phase 15.1 session ~`12 min`, well under `45 min` hard limit)

---

## 7. Known Limitations

1. **Sample Size**: `N = 20` held-out episodes per seed per condition (`40` combined) provides strong signal for catastrophic collapses (`B2..B4`, `C1`) and clear latency trends, though `±5–10 pp` differences (`A1`, `A2`) sit near binomial sampling variance.
2. **Observation Noise Simplifications**: Noise was modeled as i.i.d. per-step clipped Gaussian on `ee_pos`, `obj_pos`, and `desired_goal`; correlated camera drift, dropped frames, quaternion orientation noise, and velocity estimation lag were not modeled in Phase 15.1.
3. **Unprivileged Mode Heuristic**: `C1_UNPRIVILEGED` reflects one conservative fixed jaw-width dwell heuristic (`[0.042, 0.066] m`, `2` steps); motor current / effort proxy sensing (if hardware exposes gripper joint torque rather than PyBullet contact points) was excluded to enforce strict kinematic unprivileged bounds.

---

## 8. Phase 15.2 & Phase 15.3 Implementation Plan (Prepared — NOT Executed)

> [!IMPORTANT]
> Per instructions, **neither Phase 15.2 nor Phase 15.3 has been implemented or executed**. Before running Phase 15.2/15.3 robustness benchmarks, inspect-verified evaluations in `experiments/multi_sort/models/evaluations.npz` and `experiments/assembly/models/evaluations.npz` show both existing `10k`-step sparse smoke checkpoints currently score **`0.0%` success** across `[2500, 5000, 7500, 10000]` timesteps. Thus, a valid baseline policy must exist before frozen-policy sim-to-real perturbation sweeps can yield meaningful degradation deltas.

### 8.1 Phase 15.2 Plan — Two-Object Sorting (`MultiObjectSortEnv`)
- **Observed Target Environment** ([`multi_sort_env.py`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/environments/multi_sort_env.py)):
  - `num_objects = 2`, `obs_dim = 31` (`4` robot state + `2×9` object states + `9` relational features), `goal_dim = 6` (`2 × 3D` bin slots), `action_space` 4D `[dx, dy, dz, gripper]`, `workspace = [0.3..0.6, -0.25..0.25]`, `bins = [0.65..0.75, -0.35..0.35]`, `max_episode_steps = 300`, sparse `-1` step reward, bin placement requires `gripper_width > 0.04` and `pos[2] < 0.05`.
- **Implementation Roadmap**:
  1. **Baseline Readiness Gate**: Train/verify a dense-reward or curriculum Multi-Sort SAC+HER baseline (avoiding reliance on the `0%` `10k` sparse smoke model in `experiments/multi_sort/models`).
  2. **Multi-Stage Sequential Adaptive Controller** (extension outside frozen policy): Add an active-object index tracker (`obj_0` vs `obj_1`), sequential Phase 14.3/14.4 pick-lift-transport state machine resetting between object 1 bin release (`width > 0.04`, `pos[2] < 0.05`) and object 2 acquisition, plus bin-clearance collision guardrails (`z >= bin_wall_clearance`).
  3. **Phase 15.2 Sim-to-Real Benchmark Matrix**:
     - Extend `ObservationNoiseWrapper` to perturb `obs[4:22]` (both objects' positions) cleanly.
     - Evaluate sequential re-grasp recovery under `±3 mm / ±5 mm` noise, `1-step` latency mitigation (predictive Smith/state extrapolation or reduced `action_scale`), and unprivileged contact vs motor-effort proxy gating across `N = 20+` seeds per DR regime.

### 8.2 Phase 15.3 Plan — Orientation-Aware Peg-in-Hole Assembly (`AssemblyEnv`)
- **Observed Target Environment** ([`assembly_env.py`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/environments/assembly_env.py) & [`parts.py`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/objects/parts.py)):
  - `obs_dim = 22`, `goal_dim = 7` (`pos(3) + quat(4)`), **5D action space** `[dx, dy, dz, dyaw, gripper]`, `distance_threshold = 0.02 m`, `orientation_threshold = 0.1` (`1 - <q1, q2>^2`), `max_episode_steps = 200`, sparse `0 / -1` reward.
  - Geometry: `Peg` (`0.06 × 0.06 × 0.06 m`, mass `0.1`, friction `1.0`) inserted into `Receptacle` (`0.08 × 0.08 m` inner hole, `0.05 m` depth -> `±10 mm` radial clearance!).
- **Implementation Roadmap**:
  1. **Baseline Readiness Gate**: Establish a converging dense-reward / curriculum Assembly SAC+HER baseline (`experiments/assembly/models` is currently `0%` at `10k` sparse steps).
  2. **5-DoF Pose & Insertion Controller**: Extend Phase 14.4 target alignment to include yaw alignment (`dyaw` proportional braking using quaternion inner product `<q_obj, q_goal>`) above receptacle entry (`z > hole_top + clearance`) prior to vertical compliant descent into the `0.08 m` cavity (`±10 mm` radial tolerance).
  3. **Phase 15.3 Sim-to-Real Tight-Tolerance Suite**:
     - Introduce rotational perception noise (`±2°`, `±5°` yaw/quaternion jitter) alongside `±3 mm / ±5 mm` translation noise (critical because `±5 mm` consumes `50%` of the `10 mm` peg-hole radial clearance).
     - Benchmark unprivileged insertion jam detection (monitoring `|Δee_z|` stalling during commanded `-dz` descent as an impedance/jam proxy without PyBullet contact forces) and closed-loop `1-step` latency compensation before insertion commit.

---

**FINAL STATUS: PHASE 15.1 COMPLETE**

