# Phase 15.2.3 Final Report — Dense Multi-Object Sorting Smoke Training

**Project:** Robotic Manipulation using SAC + HER  
**Phase:** 15.2.3 (`10,000`-timestep smoke run on [`PandaMultiSortDenseRewardEnv`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/environments/dense_multi_sort_env.py))  
**Status:** **COMPLETE**  
**Diagnostic Classification:** **`B. PARTIAL LEARNING`** (early Object 2 reach emergence `0% -> 10%` and deterministic eval return improvement `-6.11 -> -5.50/-5.61`, while Object 1 placement & full sorting remain `0.0%` at `10k` steps)

---

## 1. Summary of Changes & Execution Integrity

- **Minimal Launcher Integration**:
  - Updated [`src/rl/train.py`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/rl/train.py) (`MultiSortDense` factory registration, `_validate_config()`, `her_copy_info_dict -> copy_info_dict` forwarding, explicit `device`, `initial_model.zip`, `CheckpointCallback` at `2500` steps, and guarded `replay_buffer.pkl` save).
  - Updated [`src/rl/evaluate.py`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/rl/evaluate.py) (`MultiSortDense` factory registration).
  - Aligned [`configs/multi_sort_phase15_dense.yaml`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/configs/multi_sort_phase15_dense.yaml) paths to `experiments/multi_sort_phase15_dense/seed42/models` and `logs/multi_sort_phase15_dense/seed42`.
- **Preservation Verified**: Original `multi_sort_env.py`, sparse `experiments/multi_sort/models`, `adaptive_controllers.py`, reward weights, and all Phase 14/15.1/15.2.1/15.2.2 artifacts remained untouched.
- **Training Budget & Compute**: Exactly `10,000` timesteps executed single-process on CPU in **`749.87 s` (`12.50 min`)**, peak Python RSS `454.2 MB`.

---

## 2. Smoke Training Results & Comparison Against Sparse Baseline

| Metric | Old Sparse `final_model` (`10k`) | Dense `initial_model` (`0`) | Dense `checkpoint_5000` | Dense `checkpoint_10000` |
| :--- | :---: | :---: | :---: | :---: |
| Eval Mean Return (`20` eps) | `-600.0` (sparse) | `-6.11` | `-5.64` (`EvalCB: -5.50`) | `-5.61` (`EvalCB: -5.50`) |
| `train/ent_coef` | `0.0035` (collapsed) | — | `0.3017` | **`0.0715`** (healthy) |
| Action Saturation (`|a| > 0.95`) | High (corner bang-bang) | `0.0%` | `0.0%` | **`0.0%`** |
| O1 Reach / Grasp / Lift / Place | `0% / 0% / 0% / 0%` | `0% / 0% / 0% / 0%` | `0% / 0% / 0% / 0%` | `0% / 0% / 0% / 0%` |
| O2 Reach / Grasp / Lift / Place | `0% / 0% / 0% / 0%` | `0% / 0% / 0% / 0%` | `0% / 0% / 0% / 0%` | **`10%`** `/ 0% / 0% / 0%` |
| Mean Table Collisions | `~219–275` (`4/10` eps) | `0.0` | `0.0` | `21.9` (descending toward workspace) |
| Full Both-Object Success | `0.0%` | `0.0%` | `0.0%` | `0.0%` |

- **Pre-training & Post-training Validation**: Pre-training 8-point checks all passed; `pytest tests\test_multi_sort_env.py tests\test_dense_multi_sort_env.py -v` passed (`12 passed in 17.52s`); saved `replay_buffer.pkl` (`10,000` transitions) verified valid and finite under HER virtual sampling (`mean = +0.1860`, `max = +11.9351`).

---

## 3. Failure Analysis (Why `10k` Reaches Early Motion Only)

1. **Sample Budget Horizon Mismatch (`33` Episodes Total)**: With `max_episode_steps = 300` and `learning_starts = 1,000`, the `10k` run comprises only `33` full episodes (`30` post-warmup episodes, `8,999` updates) — compared to `100`-step single-object PickPlace (`100` episodes per `10k` steps).
2. **Low Reach Signal Magnitude (`w_reach * Δd` vs Continuous Proximity)**: Across `300` steps, moving toward a cube adds only `~+0.5` total potential difference (`-5.50` vs `-6.00` idle `step_penalty`), giving a modest per-step gradient compared to continuous `-||ee - obj||` shaping (as in `PandaPickPlaceDenseRewardEnv`), causing action magnitudes (`|a|`) to only finish ramping from `0.03` (`2.5k`) -> `0.16` (`7.5k`) -> `0.39` (`10k`) near the very end of the smoke budget.
3. **Healthy Optimization Dynamics**: Unlike the collapsed sparse baseline (`ent_coef = 0.0035`, `100%` saturation, worsening EE distance), SAC+HER with `lr = 0.0003, gamma = 0.98, tau = 0.005, batch_size = 128, copy_info_dict = True` remains numerically stable (`0` NaNs, `0.0%` saturation, `ent_coef = 0.0715`) and shows emerging table-height approach & `10%` reach contacts at step `10,000`.

---

## 4. Deliverables Produced

- [`results/phase15_2_3/phase15_2_3_integration_report.md`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/results/phase15_2_3/phase15_2_3_integration_report.md)
- [`results/phase15_2_3/phase15_2_3_training_report.md`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/results/phase15_2_3/phase15_2_3_training_report.md)
- [`results/phase15_2_3/phase15_2_3_results.json`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/results/phase15_2_3/phase15_2_3_results.json)
- [`results/phase15_2_3/phase15_2_3_final_report.md`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/results/phase15_2_3/phase15_2_3_final_report.md)

---

## 5. Recommended Next Step (Awaiting Manual Approval)

Because `10k` steps (~`33` episodes) only reaches early downward approach (`10%` reach, `0%` grasp), **Phase 15.2.4** should choose between:
1. **(Recommended)** Adding a gentle continuous EE-to-active-cube proximity term (`-0.25 * clip(||ee - obj_k||, 0, 1)`) OR single-object pretraining (`PandaSortEnv` -> `MultiSortDense`) before a medium **`50k–100k` validation run** (`~35–70 min` at `~14 FPS` with `eval_freq=10000`, `n_eval_episodes=10` to cut eval overhead), OR
2. Directly running a `50k` continuation/run on CPU (`eval_freq=10000`, `n_eval_episodes=10`) to test whether `O1/O2` grasp & lift emerge cleanly with more episodes without changing weights yet.

**FINAL STATUS: COMPLETE**  
**STOPPED — Awaiting manual approval before Phase 15.2.4.**

