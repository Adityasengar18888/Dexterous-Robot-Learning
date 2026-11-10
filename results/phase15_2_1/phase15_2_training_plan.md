# Phase 15.2 Training & Controller Adaptation Plan (`PandaMultiSortEnv`)

**Configuration Prepared:** [`configs/multi_sort_phase15.yaml`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/configs/multi_sort_phase15.yaml)  
**Dry-Run Init Status:** Verified (`SAC("MultiInputPolicy", env, batch_size=128, lr=0.0003, gamma=0.98, tau=0.005, HerReplayBuffer(future, n=4))`, `0` training steps run)

---

## 1. Proposed SAC + HER Configuration ([`multi_sort_phase15.yaml`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/configs/multi_sort_phase15.yaml))

| Parameter | Baseline `multi_sort.yaml` | Proposed `multi_sort_phase15.yaml` | Rationale |
| :--- | :---: | :---: | :--- |
| Algorithm & Policy | `SAC` + `MultiInputPolicy` | `SAC` + `MultiInputPolicy` | Compatible with `spaces.Dict` (`31D obs`, `6D ag`, `6D dg`) |
| Replay Buffer | `HerReplayBuffer` | `HerReplayBuffer` | Standard HER trajectory relabeling |
| `goal_selection_strategy` | `"future"` | `"future"` | Validated with vectorized `compute_reward()` |
| `n_sampled_goal` | `4` | `4` | `4:1` HER relabel ratio |
| `buffer_size` | `1,000,000` | `300,000` | Holds `1,000` full `300`-step episodes comfortably in CPU RAM |
| `batch_size` | `256` | `128` | Higher CPU update throughput (`~35-45 FPS`) |
| `learning_rate` | `0.001` | `0.0003` | Prevents rapid `ent_coef` collapse observed at `10k` (`0.428 -> 0.0035`) |
| `learning_starts` | `350` | `1000` | Guarantees `>= 3` complete `300`-step HER episodes before first update |
| `gamma` | `0.95` | `0.98` | Longer credit horizon needed across `~74..150+` step two-object episodes |
| `tau` | `0.05` | `0.005` | Smooth target critic updates, preventing Q-target oscillation |
| `seed` | `20` | `42` | Reproducible seed baseline |
| Initial Smoke Budget | `10,000` | `10,000` | Smoke gate only; full run requires dense reward/curriculum (`300k-500k`) |

---

## 2. Phase 14 Controller Reuse Architecture (`E_ADAPTIVE_TRANSPORT` -> Sequential Sorting)

[`src/rl/adaptive_controllers.py`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/rl/adaptive_controllers.py) (`E_ADAPTIVE_TRANSPORT`) remains **unmodified**. A separate multi-object wrapper tracks:
- `active_obj_idx ∈ {0, 1}` (determined dynamically by first unplaced object whose `||cube_k_pos - goal_k_pos|| > 0.05` or settled state)
- `active_color_idx = active_color_indices[active_obj_idx]` and `assigned_bin = desired_goal[active_obj_idx*3:(active_obj_idx+1)*3]`
- Per-object `grasp_acquired`, `transport_active`, and `placed[k]` state flags

### Sequential Control State Machine (Validated via 20/20 Scripted Rollouts)

```text
IDLE
  └─> SELECT_OBJECT (pick lowest-index unplaced k in {0, 1})
        └─> APPROACH (pregrasp overhead z=0.14m -> align XY < 0.015m -> descend z=0.025m, grip open)
              └─> GRASP (cmd[3]=-1.0 while holding XY/Z)
                    └─> VERIFY_CONTACT (f9 & f10 on cubes[k].body_id >= 10N; retry <= 2 if missed)
                          └─> LIFT (Phase 14-Lite C_SLIP_AWARE accel/jerk smoothing + slip guard up to z=0.13m)
                                └─> TRANSPORT (Phase 14.4 kp + overshoot_brake toward assigned_bin[:2] at z=0.15m)
                                      └─> ALIGN_WITH_BIN (servo cube_k XY < 0.028m, descend z=0.032m)
                                            └─> RELEASE (cmd[3]=+1.0 until gw > 0.068m)
                                                  └─> VERIFY_PLACEMENT (retract z>=0.13m; dist<=0.05 & z<0.05)
                                                        ├─ if k==0 ─> SELECT_NEXT_OBJECT (reset per-obj state, k=1)
                                                        └─ if k==1 ─> COMPLETE
```

---

## 3. Reproducible Commands, Checkpoint Naming & Logging

### 3.1 Proposed Scripts & Paths (Phase 15.2.2)
- **Reward Wrapper (to add in Phase 15.2.2)**: `PandaMultiSortDenseRewardEnv` (subclasses `PandaMultiSortEnv` cleanly without mutating the sparse benchmark, adding reach + lift + dual-object transport + per-object/full placement bonuses; optional `width > 0.065` release gate).
- **Checkpoint Naming Scheme**: `experiments/phase15_2/multi_sort/seed42/models/multi_sort_p15_seed42_step{timesteps}.zip` (via `CheckpointCallback` if added to training script) alongside `best_model.zip` and `final_model.zip`. *(Note: current `src/rl/train.py` saves `best_model.zip`, `final_model.zip`, and `evaluations.npz`, and reads `MultiSort` directly via `make_env`.)*
- **Logging**: TensorBoard (`logs/phase15_2/multi_sort/seed42`) + `EvalCallback` (`evaluations.npz` every `2,500` steps) + `Monitor(..., info_keywords=("is_success", "distance"))`.

### 3.2 Commands (DO NOT RUN WITHOUT APPROVAL)
1. **10k Smoke Sanity Run (current `train.py` + `configs/multi_sort_phase15.yaml`)**:
   ```powershell
   $env:PYTHONPATH="."; .conda-env\python.exe src\rl\train.py --config configs\multi_sort_phase15.yaml
   ```
2. **Evaluation Command**:
   ```powershell
   $env:PYTHONPATH="."; .conda-env\python.exe src\rl\evaluate.py --model experiments\phase15_2\multi_sort\seed42\models\final_model.zip --config configs\multi_sort_phase15.yaml --episodes 20 --headless
   ```

---

## 4. Expected Compute Requirements & Risks

### 4.1 Empirically Calibrated CPU Compute Estimates
From historical `logs/multi_sort/SAC_1` TensorBoard `time/fps` on this machine (`Intel Core Ultra 7 155H`, `16.66 GB` RAM):
- Pre-update collection FPS: `56–69 FPS`; steady-state SAC+HER training FPS (`batch_size=256`): **`31–33 FPS`** (`~35–45 FPS` expected at `batch_size=128`).
- **`10,000` steps (smoke)**: `~4–6 minutes` (`~500 MB` RAM).
- **`100,000` steps (stage validation)**: `~40–55 minutes`.
- **`300,000–500,000` steps (full convergence)**: `~2.2–4.2 hours` per seed on CPU.

### 4.2 Identified Risks
1. **Sparse `-2/-1/0` Reward Collapse**: Training `300k–500k` steps on unmodified sparse `PandaMultiSortEnv` risks repeating the `10k` non-contact HER degeneracy unless staged dense reward / 1->2 object curriculum is introduced first.
2. **Premature `is_released` Telemetry (`width > 0.04`)**: Pinching a `0.05 m` cube yields `gw = 0.0499..0.0506 m > 0.04`, falsely counting `objects_placed = 1` before opening the gripper; batched `_get_info()` also omits release and `z < 0.05` checks.
3. **Close-Pair Cube Grasp Collisions**: `min_cube_separation` can be `0.064 m` (`~1.4 cm` edge-to-edge gap for `0.05 m` cubes); RL policies must approach vertically overhead (`z >= 0.13 m`) like the scripted controller to avoid knocking the neighboring cube.

