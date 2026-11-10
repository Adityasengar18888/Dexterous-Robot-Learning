# Phase 15.2.1 — Environment Validation Tests (`PandaMultiSortEnv`)

**Target:** [`PandaMultiSortEnv`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/environments/multi_sort_env.py#L13-L292)  
**Test Suite Run:** `pytest tests/test_multi_sort_env.py -v` -> **`3 passed in 0.85s`**  
**Extended Validation Script:** `phase15_2_1_env_validation.py` (`30` multi-seed resets `100..129`, `300`-step horizon check, release threshold probe, batched vs single `_get_info` parity check, HER batch check)

---

## 1. Space & Dimension Verification (Directly Verified Against Source)

| Property | Expected / Previous Docs | Verified Source Value | Status |
| :--- | :---: | :---: | :--- |
| Object count (`num_objects`) | `2` | `2` (clamped `1..3`) | **PASS** |
| `observation` dimension | `31` | `(31,)` (`float32`, finite): `3` ee + `1` gw + `2×(3 pos + 3 vel + 3 color_oh)` + `9` bins | **PASS** |
| `achieved_goal` dimension | `6` | `(6,)` (`float32`): concatenated `[cube0_pos, cube1_pos]` | **PASS** |
| `desired_goal` dimension | `6` | `(6,)` (`float32`): concatenated `[bin(color0)_xy + [0.025], bin(color1)_xy + [0.025]]` | **PASS** |
| `action_space` | `4` (`[dx, dy, dz, gripper]`) | `(4,)` `Box(-1.0, 1.0, float32)` inherited from [`BasePandaEnv`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/environments/base_env.py#L48) | **PASS** |
| `max_episode_steps` | `300` | `300` | **PASS** |
| `distance_threshold` | `0.05 m` | `0.05 m` | **PASS** |
| `obs_in_space` & finite | `True` | `True` (`obs_in_space=true`, `obs_finite=true`) | **PASS** |

---

## 2. Multi-Seed Geometry, Identity & Kinematics Validation (`30` Seeds: `100..129`)

| Validation Check | Measured Outcome | Notes / Verification |
| :--- | :---: | :--- |
| Unique object color identities (`obs[10:13]`, `obs[19:22]`) | `0` mismatches | Valid orthogonal one-hot indices matching `env.active_color_indices` |
| Object-to-bin goal assignments (`desired_goal`) | `0` mismatches | Exact match to `bin_positions[active_color_indices[i]]` XY with `z=0.025` |
| Minimum spawned inter-cube separation | `0.0642 m` | Enforces `> 0.06 m` center-to-center (note: `0.05 m` cube width leaves `>= 1.4 cm` edge gap) |
| Minimum inter-bin separation | `0.2500 m` | Bins spaced `0.25 m` apart in Y across `[-0.3486, +0.3486]` |
| Bin X-coordinate span | `[0.6502, 0.7489] m` | Inside `FrankaPanda` IK bounds `ws_x=[0.25, 0.75]` (`max_ws_clip_error = 0.0 m`) |
| Cube-region / bin-region overlap | `0` overlaps | Cubes spawn in `x ∈ [0.30, 0.60]`; bins spawn in `x ∈ [0.65, 0.75]` |
| `SortingBin` physical interference | None | Visual-only zone (`baseCollisionShapeIndex=-1` in [`bin.py`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/objects/bin.py#L23)) |
| Independent graspability & non-interference | `20 / 20` (`100%`) | Verified in scripted feasibility rollout (`0` Object 1 disturbances, `0` drops, `0` collisions) |

---

## 3. Horizon, Release & Telemetry Consistency Checks (Defects Documented — Env Unmodified)

| Check | Measured Result | Verdict |
| :--- | :---: | :--- |
| Idle step reward (`t=1..300`) | `-2.0` per step (`-600.0` total) | Exact `compute_reward()` behavior (`-1` per unplaced object) |
| Termination / Truncation timing | `terminated = False`, `truncated = True` strictly at `step == 300` | **PASS** |
| **Defect 1: `is_released` width threshold** | Pinched `0.05 m` cube inside Bin 0 with closed action `-1.0` gives `gw = 0.04988 m > 0.04` -> `objects_placed = 1` **before opening gripper** | **MINOR TELEMETRY DEFECT** (`width > 0.04` matches grasping width `~0.0506 m`; open gripper reaches `~0.08 m`, so threshold should be `> 0.065`) |
| **Defect 2: Single vs Batched `_get_info` parity** | Cube hovering at `z = 0.06 m, dist = 0.035 m` -> `single["is_success"] = False`, **`batched["is_success"] = True`**, **`compute_reward = 0.0`** | **MINOR METRIC MISMATCH** (Batched `_get_info` omits `is_released and pos[2] < 0.05`, matching `compute_reward` instead of single-env `is_success`) |

> [!NOTE]
> Per Phase 15.2.1 instructions ("Do not rewrite working environment components", "Do not silently change the existing benchmark"), [`multi_sort_env.py`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/environments/multi_sort_env.py) was **not** modified. Both telemetry defects should be gated on explicit approval if Phase 15.2.2 revises the environment.

