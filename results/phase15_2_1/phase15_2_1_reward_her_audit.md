# Phase 15.2.1 — Reward Function & HER Compatibility Audit

**Inspected Components:**
- [`PandaMultiSortEnv.compute_reward`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/environments/multi_sort_env.py#L281-L292)
- [`PandaMultiSortEnv._get_info`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/environments/multi_sort_env.py#L182-L246)
- [`PandaMultiSortEnv.step`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/environments/multi_sort_env.py#L248-L279)
- [`HerReplayBuffer` usage in `train.py`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/rl/train.py#L117-L143)
- Reference dense formulation: [`PandaPickPlaceDenseRewardEnv`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/environments/dense_pick_place_env.py#L5-L108)

---

## 1. Goal Space & Reward Semantics

1. **Goal Representation (`goal_dim = 6`)**:
   - `achieved_goal`: `[cube0_x, cube0_y, cube0_z, cube1_x, cube1_y, cube1_z]` (`6D`).
   - `desired_goal`: `[bin(color0)_x, bin(color0)_y, 0.025, bin(color1)_x, bin(color1)_y, 0.025]` (`6D`).
   - Color-to-bin mapping IS preserved inside `desired_goal`, and each object's color + all three bin positions are exposed in `observation` (`31D`), so the Markov state is fully observable for goal-conditioned RL.
2. **Completion Semantics (Simultaneous vs Sequential)**:
   - `compute_reward(ag, dg, info)` computes `distances = ||ag_k - dg_k||_2` for `k ∈ {0, 1}` and returns `-1.0` per object with `distance > 0.05 m` (`-2.0` initially, `-1.0` when 1 object is within `0.05 m` of its goal, `0.0` when **both** are simultaneously within `0.05 m`).
   - In single-env `_get_info()`, `episode_success = (objects_placed_correctly == 2)` requires **both** objects simultaneously inside `0.05 m` with `pos[2] < 0.05` and `width > 0.04`. Because picking Object 2 later closes the gripper (`width < 0.04` only when empty), both objects are evaluated simultaneously once the second object is released.
   - However, **neither** `compute_reward()` **nor** batched `_get_info()` checks `pos[2] < 0.05` or `is_released`, meaning hovering a grasped cube `3.5 cm` above the bin (`z = 0.06 m`) yields maximum `0.0` reward and batched `is_success = True` while single-env `is_success = False`.

---

## 2. HER Compatibility & Why Sparse Training Collapsed to `0%`

Running `HerReplayBuffer(goal_selection_strategy="future", n_sampled_goal=4)` with `SAC("MultiInputPolicy", env)` executes cleanly at the API level (`batch=(16,)` vectorized `compute_reward` returned `[0.0, -1.0, -2.0]` as expected). **There is no tensor shape crash or HER API error.**

Instead, the existing `10k`-step baseline in `experiments/multi_sort/models` failed (`0/10` reach, `0/10` grasp, `-600.0` reward across `[2500, 5000, 7500, 10000]`) due to three compounding **learning-design bottlenecks**:

1. **HER Degeneracy Under Non-Contact Trajectories**:
   - In early random exploration (`0..10k` steps), the robot rarely makes sustained contact with either `0.05 m` cube, so `achieved_goal(t) == [p0(0), p1(0)]` remains stationary throughout the `300`-step episode.
   - When HER `"future"` relabels `desired_goal = achieved_goal(t_future) = [p0(0), p1(0)]`, **every transition receives `reward = 0.0` regardless of EE motion**.
   - Conversely, non-relabeled goals give a flat **`-2.0`** reward everywhere in the `300`-step rollout. Neither original nor HER goals provide any spatial gradient for `ee_pos` to approach Cube `0` or Cube `1`.
2. **Joint 6D Goal Coupling Without Reach/Lift Shaping**:
   - Even if one cube is pushed near its bin (`< 1e-4` probability in 10k steps), joint relabeling only hits `0.0` when `p0` AND `p1` match future targets, and `compute_reward()` contains zero shaping for reaching (`-||ee - active_obj||`), grasping, lifting, or object-to-bin transport distance.
3. **Aggressive `lr=0.001, tau=0.05, gamma=0.95` Entropy Collapse**:
   - TensorBoard scalars (`logs/multi_sort/SAC_1`) show `train/ent_coef` collapsed from `0.4282` (step `1200`) -> `0.1178` (`2500`) -> `0.0129` (`5000`) -> `0.0035` (`10000`), locking `final_model.zip` into corner-saturated `tanh(±1)` bang-bang actions hitting the table `219–275` times per episode (`mean_min_ee_dist = 0.346 m`, worse than `0.210 m` at step `2500`).

---

## 3. Proposed Dense-Reward & Curriculum Design (Separate — Existing Benchmark Unchanged)

Mirroring how Phase 12/13 solved `PandaPickPlaceEnv` via [`PandaPickPlaceDenseRewardEnv`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/environments/dense_pick_place_env.py), a trainable Multi-Sort environment (`PandaMultiSortDenseRewardEnv`, keeping `PandaMultiSortEnv` intact) should provide:
1. **Staged Sequential Dense Reward** (preserving HER compatibility via `info`-augmented `compute_reward(ag, dg, info)`):
   - Determine active target index `k ∈ {0, 1}` as the first unplaced object (`||ag_k - dg_k|| > 0.05`).
   - **Reach term**: `-w_reach * ||ee_pos - ag_k||` (with reduced weight or zeroed once object `k` is grasped so the arm doesn't hover passively).
   - **Lift term**: `+w_lift` when active object `ag_k[2] > 0.05 m`.
   - **Transport term**: `-w_trans * sum_{i=0..1} ||ag_i - dg_i||` (shapes moving *both* objects smoothly into their assigned bins).
   - **Per-object & Full Placement Bonus**: `+2.0` per object placed & released (`dist_i <= 0.05` and `pos_i[2] < 0.05` and `gw > 0.06`), `+5.0` full sort bonus when both sorted.
2. **2-Stage Curriculum Option**:
   - **Stage A (`num_objects=1` / `PandaSortEnv` pretraining or single-active sort)** for `100k–150k` steps to learn reach-grasp-lift-bin-release skills quickly.
   - **Stage B (`num_objects=2`, `max_episode_steps=300`)** for `300k–500k` steps (with `lr=0.0003`, `gamma=0.98`, `tau=0.005`, `learning_starts=1000`).
3. **Optional Release/Parity Hygiene Fix** (if approved in Phase 15.2.2):
   - Raise `is_released` threshold in `_get_info()` from `width > 0.04` to `width > 0.065` (since pinching a `0.05 m` cube yields `width = 0.0499..0.0506 m`) and align batched `_get_info` / terminal checks cleanly.

