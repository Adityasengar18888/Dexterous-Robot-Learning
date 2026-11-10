# Phase 15.2.5B Final Report — Single-Object Grasp Acquisition (SAC + HER)

**Project:** Robotic Manipulation using SAC + HER  
**Parent Phase:** 15.2 Multi-Object Sorting  
**Execution Mode:** CPU-only PyTorch (`device: "cpu"`), single training process  
**Gate Status:** **STOPPED / BLOCKED AT PHASE B GATE** (Phase C training **NOT started**)

---

## 1. Environment, Registration & Pre-Training Calibration

- **Environment Class**: [`PandaSinglePreGraspEnv`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/environments/single_pregrasp_env.py) (registered in [`src/rl/train.py`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/rl/train.py) and [`src/rl/evaluate.py`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/rl/evaluate.py) as `"SinglePreGrasp"`).
- **Spaces**: `observation=(22,)`, `achieved_goal=(3,)`, `desired_goal=(3,)` (flattened `28D` input to `MultiInputPolicy`), `action=(4,)`, `max_episode_steps=80`, `action_scale=0.05`.
- **Configuration**: [`configs/multi_sort_phase15_5B.yaml`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/configs/multi_sort_phase15_5B.yaml) (`SAC`, `MultiInputPolicy`, `seed=42`, `total_timesteps=20000`, `buffer_size=200000`, `batch_size=128`, `learning_rate=0.0003`, `learning_starts=500`, `gamma=0.98`, `tau=0.005`, `HerReplayBuffer` with `future`, `n_sampled_goal=4`, `copy_info_dict=true`, `eval_freq=5000`, `save_freq=5000`).
- **Pre-Training Unit & Calibration Verification**:
  - `pytest`: **`9 passed in 9.08s`** across `tests/test_single_pregrasp_and_transfer.py`, `tests/test_pregrasp_controller.py`, `tests/test_pregrasp_reward.py`, `tests/test_pregrasp_her_compatibility.py`.
  - Idle episode (`80` steps): **`-9.521`**.
  - Scripted `PreGraspController` hold (`80` steps): **`+8.880`** (`is_success: True`, `r_align=0.561`, `r_vert_approach=0.682`, `r_open_comp=0.0`, `r_collision=0.0`, `r_disp=0.0`, `r_auth=0.5`, `r_grasp=1.5`, `r_lift=1.5`, `r_goal_lift=0.616`, `r_hold=4.0`, `r_step=-0.48`).

---

## 2. Training Execution & Saved Artifacts (`20,000` Steps)

- **Wall-Clock Duration**: **`892.62 s` (`14.88 min`)** CPU-only single process (`time/fps`: `248.0 -> 22.0`).
- **Memory Snapshot** (`13,440` steps): `WorkingSet64 = 555,438,080` bytes (`529.7 MB`), `FreePhysicalMemory = 2,432,012 KB` (`2.32 GB`).
- **Training Telemetry**:
  - `rollout/ep_rew_mean`: `-11.08487` (`320` steps) -> `-2.15074` (`19,840` steps)
  - `train/actor_loss`: `-5.87241` (`640` steps) -> `-6.32688` (`20,000` steps)
  - `train/critic_loss`: `0.07794` (`640` steps) -> `0.00249` (`20,000` steps)
  - `train/ent_coef`: `0.95944` (`640` steps) -> `0.00376` (`20,000` steps)
  - `evaluations.npz` (`n_eval_episodes=10`): `[-2.583, -1.613, -1.263, -1.288]` at `[5000, 10000, 15000, 20000]`.
  - `replay_buffer.pkl`: `size = 20000`, `sample(128) finite = True`, `min/mean/max rew = -0.3149 / -0.0654 / 0.6078`.
- **Preserved Checkpoints (`experiments/multi_sort_phase15_5B/seed42/models/`)**:
  - `initial_model.zip` (`1,522,147 B`), `checkpoint_5000_steps.zip` (`3,337,763 B`), `checkpoint_10000_steps.zip` (`3,340,037 B`), `checkpoint_15000_steps.zip` (`3,340,044 B`), `checkpoint_20000_steps.zip` (`3,340,064 B`), `best_model.zip` (`3,340,044 B`), `final_model.zip` (`3,340,031 B`), `evaluations.npz` (`1,732 B`), `replay_buffer.pkl` (`62,349,696 B`), plus TensorBoard logs in `logs/multi_sort_phase15_5B/seed42/SAC_1`.

---

## 3. Deterministic Checkpoint Evaluation (`N = 20` Episodes Each, Seeds `1001..1020`)

| Checkpoint | Reach | Align | Safe Desc | Auth | Bilateral Grasp | Lift\|Grasp | Hold | Drop | Min EE Dist | Min XY Dist | Mean Return | `r_align_prog` | `r_align_prox` | `r_vert_approach` | `r_open_comp` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0_initial` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | `0.5185 m` | `0.1331 m` | **`-10.0292`** | `-0.0957` | `-3.5335` | `0.0000` | `-4.8000` |
| `5000` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | `0.5005 m` | `0.0277 m` | **`-2.4860`** | `+0.5255` | `-1.4115` | `0.0000` | `0.0000` |
| `10000` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | `0.4997 m` | `0.0232 m` | **`-1.6006`** | `+0.7130` | `-0.7129` | `-0.0007` | `0.0000` |
| `15000` | 0.0% | **5.0%** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | `0.4569 m` | `0.0265 m` | **`-1.5613`** | `+0.7042` | `-0.7056` | `+0.0401` | `0.0000` |
| `20000` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | `0.5000 m` | **`0.0110 m`** | **`-1.2179`** | `+0.7430` | **`-0.3615`** | `+0.0006` | `0.0000` |

- `mean_table_coll_steps = 0.0` and `mean_max_pregrasp_disp_m = 0.0` across all 5 checkpoints (`Sat = 0.53%` at `20k`, `mean_abs_action = [0.0432, 0.0410, 0.4971, 0.6789]`).

---

## 4. Gate Evaluation & Root-Cause Bottleneck Diagnosis

### Gate Checks
1. **Conditional `50k` Extension Gate**: Extension to `50,000` steps was authorized *only if reach, grasp, or lift metrics demonstrate measurable improvement*. Across `0 -> 20,000`, `Reach = 0.0%`, `Bilateral Grasp = 0.0%`, `Lift = 0.0%`, and `evaluations.npz` plateaued between `15k` (`-1.263`) and `20k` (`-1.288`) while `ent_coef` decayed to `0.00376`. **Extension Gate: FAILED.**
2. **Phase B -> C Transfer Gate (`B7`)**: Requires `Reach >= 90%`, `Align >= 90%`, `Bilateral Grasp >= 85%`, `Lift | Grasp >= 85%`, `Drop <= 10%`. Actual at `20k`: `0% / 0% / 0% / 0%`. **Transfer Gate: FAILED.**
3. **Rule Compliance**: Per Global Execution Policy & Gate Decision Rules, Phase B was **STOPPED** and **Phase C training was NOT started**.

### Exact Reward-Landscape Bottleneck
1. `r_align_prox = -0.25 * clip(xy_d_curr, 0, 1)` only penalized **horizontal XY error** (`xy_d_curr`), with **no continuous vertical Z proximity term** (only the gated delta `2.0 * clip(Δz_err, -0.05, 0.05)` active inside `xy <= 0.030`).
2. `r_open_comp` imposed `-0.06/step` (`-4.80/ep`) when closed below `0.065 m`.
3. Consequently, SAC rapidly solved two objectives — opening the gripper (`r_open_comp: -4.80 -> 0.0000`) and hovering directly over the cube in XY (`min_xy_dist_m: 0.1331 m -> 0.0110 m`, boosting episode return from `-10.0292` to `-1.2179`) — **while remaining at `z ≈ 0.50 m`** (`mean_min_ee_dist_m = 0.5000 m`, `r_vert_approach = +0.0006`). Because auto-entropy collapsed to `0.00376`, exploration vanished before sustained descent ever entered the replay buffer.

### Recommended Next Remediation (Before Re-Running Phase B)
1. Add a continuous vertical Z proximity term to [`PandaSinglePreGraspEnv`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/environments/single_pregrasp_env.py) (e.g., `-0.25 * clip(|ee_z - target_z|, 0, 1)` ungrasped) so hovering at `z = 0.50 m` incurs a persistent negative Z gradient alongside XY centering.
2. Seed `HerReplayBuffer` with a small set of scripted [`PreGraspController`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/controllers/pregrasp_controller.py) rollouts on `SinglePreGrasp` (or maintain an entropy floor during early training) before launching a fresh Phase B run.

**STATUS: B BLOCKED (Phase C training not started)**

