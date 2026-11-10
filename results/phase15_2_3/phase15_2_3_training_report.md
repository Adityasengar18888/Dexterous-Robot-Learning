# Phase 15.2.3 — Smoke Training & Evaluation Report (`10,000` Timesteps)

**Configuration:** [`configs/multi_sort_phase15_dense.yaml`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/configs/multi_sort_phase15_dense.yaml)  
**Environment:** [`PandaMultiSortDenseRewardEnv`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/environments/dense_multi_sort_env.py)  
**Algorithm:** `SAC("MultiInputPolicy", device="cpu")` + `HerReplayBuffer(goal_selection_strategy="future", n_sampled_goal=4, copy_info_dict=True)`  
**Budget:** Exactly `10,000` timesteps (`seed = 42`, `batch_size = 128`, `lr = 0.0003`, `gamma = 0.98`, `tau = 0.005`, `learning_starts = 1000`)

---

## 1. Saved Checkpoints & Artifacts (`experiments/multi_sort_phase15_dense/seed42/models`)

| Artifact | Path / Size | Notes |
| :--- | :---: | :--- |
| Initial model | `initial_model.zip` (`1,599,111` bytes) | Saved pre-training (`0` steps) |
| 2,500-step checkpoint | `checkpoint_2500_steps.zip` (`3,504,267` bytes) | `CheckpointCallback` |
| 5,000-step checkpoint | `checkpoint_5000_steps.zip` (`3,504,834` bytes) | Also saved as `best_model.zip` |
| 7,500-step checkpoint | `checkpoint_7500_steps.zip` (`3,505,248` bytes) | `CheckpointCallback` |
| 10,000-step checkpoint | `checkpoint_10000_steps.zip` (`3,505,816` bytes) | Also saved as `final_model.zip` (`3,505,798` bytes) |
| HER Replay Buffer | `replay_buffer.pkl` (`122,486,817` bytes) | Size `10,000` transitions; verified loadable & finite (`sample(64)`) |
| Eval & TensorBoard Logs | `evaluations.npz`, `logs/multi_sort_phase15_dense/seed42/SAC_1` | `12` scalar steps logged (`1200..10000`) |

---

## 2. Training Curves (Rollout Exploration vs Deterministic EvalCallback)

| Timestep | Rollout `ep_rew_mean` | Eval `mean_reward` (`20` eps) | Eval `success` | `train/actor_loss` | `train/critic_loss` | `train/ent_coef` | `time/fps` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1,200` | `-5.845` | — | `0%` | `-6.444` | `0.0674` | `0.9423` | `10` |
| `2,400` / `2,500` | `-5.632` | `-5.95 ± 0.46` | `0%` | `-15.031` | `0.0353` | `0.6380` | `15` |
| `4,800` / `5,000` | `-5.646` | `-5.50 ± 0.56` | `0%` | `-23.134` | `0.0801` | `0.3017` | `14` |
| `7,200` / `7,500` | `-5.568` | `-5.50 ± 0.52` | `0%` | `-23.104` | `0.0432` | `0.1433` | `14` |
| `9,600` / `10,000` | `-5.556` | `-5.50 ± 0.65` | `0%` | `-20.980` | `2.9208` | `0.0715` | `14` |

*(Rollout `ep_len_mean = 300.0` throughout; zero NaN/Inf across all updates.)*

---

## 3. Deterministic Checkpoint Evaluation Funnel (`20` Fixed Seeds: `1001..1020`)

Full per-episode metrics stored in [`results/phase15_2_3/phase15_2_3_results.json`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/results/phase15_2_3/phase15_2_3_results.json):

| Checkpoint | O1 Reach / Grasp / Lift / Place | O2 Reach / Grasp / Lift / Place | Both Success | Wrong Bins | Drops | Mean Collisions | Mean Steps | Mean Final Dist (m) | Mean Return | Action Sat (`|a|>0.95`) | Mean `|action|` (`dx, dy, dz, g`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0_initial` | `0% / 0% / 0% / 0%` | `0% / 0% / 0% / 0%` | `0.0%` | `0.0` | `0.0` | `0.0` | `300.0` | `0.3682` | `-6.11` | `0.0%` | `[0.007, 0.059, 0.015, 0.025]` |
| `2500` | `0% / 0% / 0% / 0%` | `0% / 0% / 0% / 0%` | `0.0%` | `0.0` | `0.0` | `0.0` | `300.0` | `0.3682` | `-6.15` | `0.0%` | `[0.036, 0.033, 0.014, 0.039]` |
| `5000` | `0% / 0% / 0% / 0%` | `0% / 0% / 0% / 0%` | `0.0%` | `0.0` | `0.0` | `0.0` | `300.0` | `0.3682` | `-5.64` | `0.0%` | `[0.052, 0.039, 0.034, 0.040]` |
| `7500` | `0% / 0% / 0% / 0%` | `0% / 0% / 0% / 0%` | `0.0%` | `0.0` | `0.0` | `0.0` | `300.0` | `0.3682` | `-5.74` | `0.0%` | `[0.162, 0.073, 0.053, 0.159]` |
| `10000` | `0% / 0% / 0% / 0%` | **`10%`** `/ 0% / 0% / 0%` | `0.0%` | `0.0` | `0.0` | `21.9` | `300.0` | `0.3733` | `-5.61` | `0.0%` | `[0.385, 0.196, 0.267, 0.290]` |

---

## 4. HER Replay-Buffer Verification & Transition Diagnostics

Loaded `experiments/multi_sort_phase15_dense/seed42/models/replay_buffer.pkl` (`size = 10,000` transitions):
- **`64`-sample HER virtual batch**: `100%` finite rewards (`min = -0.1835`, `mean = +0.1860`, `max = +11.9351`).
- **`256`-sample virtual inspection**: Found `1/256` high-reward virtual HER sample (`rew = +11.935`, `prev_gw=0.0543 -> curr_gw=0.0562`, `min(ee, obj) = 0.0953 m`, `dist(ag, dg_virtual) = 0.0256 m`), confirming that physical cube displacements inside exploration cross HER relabeled goal radii cleanly without NaN/Inf or empty-air exploitation.

---

## 5. Runtime & Hardware Utilization

- **Total Training Wall Time** (including `4 × 20` episode `EvalCallback` runs = `24,000` eval steps): **`749.87 s` (`12.50 min`)**, exit code `0`, well inside `15 min` limit.
- **Evaluation Wall Time (`5 × 20` eps)**: `228.67 s` (`3.81 min`).
- **Memory & CPU**: Python RSS = `454.2 MB` (`476,237,824` bytes); system free RAM = `2.78 GB` / `15.52 GB` (`CPU` device).
- **Regression Verification**: `pytest tests\test_multi_sort_env.py tests\test_dense_multi_sort_env.py -v` passed (`12 passed in 17.52s`).

