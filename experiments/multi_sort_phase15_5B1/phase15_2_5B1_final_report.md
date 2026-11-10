# Phase 15.2.5B.1 Final Report — Vertical Approach Learning & Demonstration-Guided SAC+HER

**Project:** Robotic Manipulation using SAC + HER  
**Parent Phase:** 15.2 Multi-Object Sorting  
**Environment:** [`PandaSinglePreGraspEnv`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/environments/single_pregrasp_env.py) (`configs/multi_sort_phase15_5B1.yaml`)  
**Hardware:** CPU-only PyTorch (`device: "cpu"`), single training process  
**Classification:** **PARTIALLY COMPLETE** (Vertical approach solved; full grasp-and-lift target not yet reached; stopped at `20k` Gate 2; Phase 15.2.5C NOT started)

---

## 1. Executive Summary & Preservation Status

All Phase 14 controllers, Phase 15.2.2/15.2.4/15.2.5A environments and checkpoints, and Phase 15.2.5B models/buffers (`experiments/multi_sort_phase15_5B/`) were preserved untouched.

Phase 15.2.5B.1 successfully solved the Phase B `0.50 m` hovering failure:
- **Vertical Action Pathway**: Verified `negative action[2]` commands smooth downward EE motion (`0.4853 m -> 0.0216 m` over `11` steps, `0` joint limits, `0` table collisions, open/close gripper independence).
- **Continuous Vertical Potential ($\Phi_z, w_z=2.0$) & HER Tests**: Implemented `R_{z,\text{progress}} = \Phi_z(s_{t+1}) - \Phi_z(s_t)` cleanly with safety/grasp-height deadband gating (`tests/`: **`16 passed in 12.13s`**).
- **Scripted Demonstration Dataset & HER Buffer Warm-Start**: Generated **`50 / 50`** validated episodes (`4,000` transitions, `0` rejected, `mean_return = +9.4439`) and seeded `HerReplayBuffer` via `Method A`.
- **10k -> 20k Conditional Execution**:
  - `10k` passed Part 9 Gate 1 (`meanZ`: `0.4995 m -> 0.1140 m`, `minZ`: `0.4853 m -> 0.0200 m` at `7.5k` / `0.0720 m` at `10k`, `Reach` jumped `0% -> 60%` at `7.5k`, `Bilateral Grasp` jumped `0% -> 40%` at `7.5k`).
  - Conditional extension `10k -> 20k` drove vertical error even lower (`mean_vert_err = 0.0424 m`, `minZ = 0.0230 m` at `20k`), **but bilateral grasping did not stabilize** (`40%` at `17.5k` -> `0%` at `20k`, `Lift|Grasp = 0.0%`, `eval/mean_reward` dropped `-4.71 -> -9.37`, `ent_coef = 0.00804`). Training stopped cleanly at `20,000` steps per Part 9 Gate 2.

---

## 2. Summaries of Pre-Training Audits (See Companion Reports)

1. **Vertical Action-Pathway Audit** ([`phase15_2_5B1_action_pathway_audit.md`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/experiments/multi_sort_phase15_5B1/phase15_2_5B1_action_pathway_audit.md)): All 10 checklist items passed; `neg_full_11steps` (`action[2]=-1.0`) moved `ee_z` from `0.4853 m -> 0.0216 m` (`max_ik_err = 0.0064 m`, `jlim=0`, `tcoll=0`), whereas positive `action[2]` from reset hit the `0.5000 m` workspace ceiling.
2. **Reward Equations, HER Compatibility & Magnitude Audit** ([`phase15_2_5B1_reward_audit.md`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/experiments/multi_sort_phase15_5B1/phase15_2_5B1_reward_audit.md)):
   - $\Phi_z(s) = -2.0 \cdot \operatorname{clip}(|z_{EE} - z_{\text{grasp}}|, 0, 1)$, $z_{\text{grasp}} = \max(0.0205, o_z)$, `R_z_progress = Phi_z(s_{t+1}) - Phi_z(s_t)` gated against premature closure, table collision, floor overshoot, and valid grasp-height dwell (`<= 0.008 m`).
   - `80`-step calibration ordering: `idle (-9.5211) < premature_close (-6.2401) < xy_align_only (-1.0421) < vertical_descent (+0.3686) < scripted_full (+9.2098)`.
3. **Demonstration & Warm-Start Audit** ([`phase15_2_5B1_demonstration_audit.md`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/experiments/multi_sort_phase15_5B1/phase15_2_5B1_demonstration_audit.md)):
   - `50 / 50` valid trajectories (`4,000` transitions; `APPROACH=450, ALIGN=156, DESCENT=158, AUTH=50, GRASP=50, LIFT=255, ACQUIRED=50, HOLD_AT_GOAL=2831`).
   - `Method A` (`HerReplayBuffer.add` preserving `ep_length=80` and `is_demo=True`) selected over `Method B` (untuned actor baseline `MSE = 0.3050` reported for comparison).

---

## 3. Deterministic Checkpoint Evaluation (`N = 20` Episodes Each, `seeds = 1001..1020`)

Full telemetry saved in [`phase15_2_5B1_training_results.json`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/experiments/multi_sort_phase15_5B1/phase15_2_5B1_training_results.json).

| Checkpoint | Reach | Align | Safe Desc | Auth | Bilateral Grasp | Lift\|Grasp | Mean `ee_z` | Min `ee_z` | Vertical Err | Min XY Err | Down Act Freq | Table Coll Ep (`Steps`) | Prem Close Ep | Mean Return |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0_initial` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | `0.4995 m` | `0.4853 m` | `0.4790 m` | `0.1331 m` | `0.0%` | `0%` (`0.0`) | `100%` | **`-10.0586`** |
| `2,500` | 0.0% | 5.0% | 0.0% | 0.0% | 0.0% | 0.0% | `0.4220 m` | `0.3957 m` | `0.4015 m` | `0.1281 m` | `16.0%` | `0%` (`0.0`) | **`0%`** | **`-4.4109`** |
| `5,000` | 15.0% | 30.0% | 0.0% | 0.0% | 15.0% | 0.0% | `0.1490 m` | `0.0317 m` | `0.1286 m` | `0.0354 m` | `83.7%` | `0%` (`0.0`) | `90%` | **`-4.6584`** |
| **`7,500` (Peak Grasp)** | **`60.0%`** | `30.0%` | **`35.0%`** | **`20.0%`** | **`40.0%`** | `0.0%` | `0.0736 m` | **`0.0200 m`** | `0.0537 m` | `0.0435 m` | **`98.8%`** | `50%` (`8.4`) | `80%` | **`-4.2039`** |
| `10,000` | 40.0% | **`40.0%`** | 10.0% | 0.0% | 30.0% | 0.0% | `0.1140 m` | `0.0720 m` | `0.0937 m` | **`0.0183 m`** | `54.5%` | `25%` (`10.1`) | `95%` | **`-5.5828`** |
| `12,500` | 10.0% | 30.0% | 0.0% | 0.0% | 0.0% | 0.0% | `0.1041 m` | `0.0722 m` | `0.0837 m` | `0.0291 m` | `42.7%` | `5%` (`0.1`) | `100%` | **`-5.7908`** |
| `15,000` | 5.0% | 15.0% | 0.0% | 0.0% | 0.0% | 0.0% | `0.0844 m` | `0.0462 m` | `0.0639 m` | `0.0390 m` | `56.9%` | `5%` (`1.0`) | `100%` | **`-7.8333`** |
| `17,500` | **50.0%** | 25.0% | 0.0% | 0.0% | **40.0%** | 0.0% | `0.0635 m` | `0.0237 m` | `0.0431 m` | `0.0252 m` | `61.8%` | `20%` (`3.1`) | `100%` | **`-7.4138`** |
| `20,000` | 0.0% | 35.0% | 0.0% | 0.0% | 0.0% | 0.0% | **`0.0629 m`** | `0.0230 m` | **`0.0424 m`** | `0.0267 m` | `60.1%` | **`0%` (`0.0`)** | `100%` | **`-9.5578`** |

- **Replay Buffer & EvalCallback Telemetry**:
  - `evaluations_0_10k.npz` (`n=10`): `[-4.7013, -3.7827, -3.9802, -4.7066]` at `[2500, 5000, 7500, 10000]`.
  - `evaluations.npz` (`12.5k–20k`, `n=10`): `[-6.1800, -7.8207, -6.8822, -9.3696]`.
  - `replay_buffer.pkl`: `size = 24,000`, `sample(128) finite = True`, `min/mean/max rew = -1.7450 / -0.1127 / +0.2274`.
  - TensorBoard: `rollout/ep_rew_mean` rose `-12.22 -> -7.63`; `ent_coef` decayed `0.9595 -> 0.0568 (10k) -> 0.00804 (20k)`; `actor_loss = -9.71`, `critic_loss = 0.156` at `20k`.

---

## 4. Runtime, Hardware & Failure Taxonomy

- **Runtime & System Resources**: CPU-only single process (`15.52 GB` total RAM, `3.03 GB` free at finish):
  - `0 -> 10,000` steps: **`646.00 s` (`10.77 min`)** (`time/fps ≈ 15–16`).
  - `10,000 -> 20,000` steps: **`678.78 s` (`11.31 min`)** (`1,324.78 s` = `22.08 min` total training).
  - `0 -> 20,000` 9-checkpoint evaluation (`N=20` eps each): **`138.48 s` (`2.31 min`)**.
- **Failure Taxonomy (`Lift = 0%`, Regression `7.5k -> 20k`)**:
  1. **Closed-Gripper Bias in Warm-Start Buffer (`71–78%` Closed Transitions)**: In the `4,000`-transition demo dataset, `HOLD_AT_GOAL` (`2,831` steps) + `LIFT` (`255`) + `GRASP` (`50`) accounted for **`78.4%`** of transitions with `action[3] = -1.0`, whereas `ALIGNMENT` + `CONTROLLED_DESCENT` accounted for only `314` (`7.85%`). Between `10k` and `20k`, the policy converged to `|act[3]| ≈ 0.86` (closed gripper during approach), dropping `premEp` to `100%` (`r_open_comp = -3.98` to `-4.14/ep`) and blocking `Safe Descent` / `Grasp Auth` (`0%`).
  2. **Open-Mask / Negative-Rebound Coupling**: With `curr_open = False` during approach, `safe_for_pos_descent` suppressed positive `r_vert_progress` while minor upward vertical jitter still recorded negative `r_vert_progress` (`-0.28` to `-0.77` over `15k–20k`).
  3. **Post-Grasp Lift Absence**: Even at `7.5k` and `17.5k` when `40%` bilateral grasps occurred, `action[2]` remained predominantly downward (`61.8–98.8%`) without reversing upward cleanly under low entropy (`0.0080` at `20k`).
- **Preserved Artifact Paths**:
  - Peak grasping/pre-grasp checkpoint (`7.5k`, `Reach=60%, Grasp=40%, SafeD=35%`): [`experiments/multi_sort_phase15_5B1/seed42/models/checkpoint_7500_steps.zip`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/experiments/multi_sort_phase15_5B1/seed42/models/checkpoint_7500_steps.zip)
  - Best `EvalCallback` resumed checkpoint (`12.5k`): [`experiments/multi_sort_phase15_5B1/seed42/models/best_model.zip`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/experiments/multi_sort_phase15_5B1/seed42/models/best_model.zip)
  - Final `20k` model & replay buffer: [`final_model.zip`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/experiments/multi_sort_phase15_5B1/seed42/models/final_model.zip), [`replay_buffer.pkl`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/experiments/multi_sort_phase15_5B1/seed42/models/replay_buffer.pkl)

---

## 5. Recommendation for Next Phase (**Phase 15.2.5B.2 — NOT Phase 15.2.5C**)

Do **NOT** launch Phase 15.2.5C yet (`Lift|Grasp = 0%`). Recommend **Phase 15.2.5B.2**:
1. **Re-balance Scripted Demonstrations**: Cap `HOLD_AT_GOAL` to `5–10` steps per episode (or downsample held tail) so open-gripper `APPROACH / ALIGNMENT / DESCENT` transitions dominate (`>60%`) instead of `78%` closed-gripper steps.
2. **Method B Decaying BC Auxiliary Loss + Entropy Floor**: Combine balanced buffer initialization with a decaying supervised actor BC loss (`action[3] = +1.0` strictly preserved on open descent states -> decay weight `1.0 -> 0.0` by step `5k`) and an entropy floor (`>= 0.05`) so once `Bilateral Grasp` hits `>= 40%`, upward `Lift` policy exploration remains active.

**FINAL STATUS: PARTIALLY COMPLETE — Stopped at 20k Gate; awaiting manual approval for Phase 15.2.5B.2.**

