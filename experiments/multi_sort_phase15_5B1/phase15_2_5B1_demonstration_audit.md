# Phase 15.2.5B.1 — Scripted Demonstration & Warm-Start Audit

**Project:** Robotic Manipulation using SAC + HER  
**Environment:** [`PandaSinglePreGraspEnv`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/environments/single_pregrasp_env.py) (`use_vertical_potential=True`, `w_vertical_potential=2.0`)  
**Generator Script:** [`scripts/generate_phase15_5B1_demos.py`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/scripts/generate_phase15_5B1_demos.py)  
**Verification Tests:** [`tests/test_b1_demos_and_warmstart.py`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/tests/test_b1_demos_and_warmstart.py) (`2 passed in 40.81s`)

---

## 1. Scripted Demonstration Dataset Statistics (`50` Episodes)

Generated using the validated Phase 15.2.5A [`PreGraspController`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/controllers/pregrasp_controller.py) (`lift_verify_z_min=0.11`) over fixed reproducible seeds `5000..5049` (disjoint from evaluation seeds `1001..1020`):

- **Accepted Episodes**: **`50 / 50`** (`rejected_episodes = 0`)
- **Total Transitions**: **`4,000`** (`mean_episode_length = 80.0`)
- **Episode Return & Holding**: `mean_episode_return = +9.4439`, `mean_hold_steps = 58.7` steps/episode
- **Safety Invariants**: `max_pregrasp_disp_m = 0.0 m`, `total_table_collisions = 0`, `total_premature_closures = 0`
- **All 8 Trajectory Phases Represented (`stage_step_counts`)**:
  - `APPROACH`: `450` steps
  - `ALIGNMENT`: `156` steps
  - `CONTROLLED_DESCENT`: `158` steps
  - `GRASP_AUTHORIZATION`: `50` steps
  - `GRASP`: `50` steps
  - `LIFT_VERIFICATION`: `255` steps
  - `ACQUIRED`: `50` steps
  - `HOLD_AT_GOAL`: `2,831` steps
- **Mean Scripted Return Breakdown**:
  - `r_align = +0.6999` (`r_align_prog = +0.8028`, `r_align_prox = -0.1029`)
  - `r_vert_progress = +0.9172` (`r_vert_potential = -5.3644` logged)
  - `r_auth = +0.5000`, `r_grasp = +1.5000`, `r_lift = +1.5000`, `r_goal_lift = +0.7527`, `r_hold = +4.0000`, `r_step = -0.4260`
- **Stored Dataset Artifacts**:
  - [`experiments/multi_sort_phase15_5B1/seed42/demonstrations/single_pregrasp_demos_50ep.pkl`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/experiments/multi_sort_phase15_5B1/seed42/demonstrations/single_pregrasp_demos_50ep.pkl)
  - [`experiments/multi_sort_phase15_5B1/seed42/demonstrations/demo_dataset_summary.json`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/experiments/multi_sort_phase15_5B1/seed42/demonstrations/demo_dataset_summary.json)

---

## 2. Evaluation of Warm-Start Methods (Option A vs Option B)

1. **Option A — Safe `HerReplayBuffer` Demonstration Initialization (SELECTED)**:
   - Verified directly against SB3 `HerReplayBuffer.add(obs, next_obs, action, reward, done, infos)` (`her_copy_info_dict=true`, `goal_selection_strategy="future"`, `n_sampled_goal=4`).
   - Preserves exact episode boundaries (`ep_length == 80`), stores `is_demo = True` provenance inside `info` dicts without modifying `SAC.actor` weights or introducing supervised BC artifacts at evaluation step `0`.
   - Smoke-verified in [`test_her_buffer_demo_warmstart_and_short_rl_step`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/tests/test_b1_demos_and_warmstart.py): buffer size/pos (`400 -> 560` after online rollouts), uniform `ep_length == 80`, and finite HER-relabeled `sample()` rewards (`mean = +0.1109`, `max = +4.7855` on 5-ep probe).
   - Demonstration weight naturally dilutes (`4,000` demos out of `4,000 + N_online` transitions -> `28.6%` at `10k` online steps) as genuine online SAC rollouts accumulate.

2. **Option B — Behavior-Cloning Auxiliary Initialization (Audited, Not Applied in Primary B.1 Run)**:
   - Baseline untuned SAC actor (`initial_model`) action MSE measured on the validated demo manifold is **`0.3050`** (`pred min/max = [-0.0612, +0.1147]` vs normalized demo actions in `[-1.0, +1.0]`).
   - Kept unapplied in the primary B.1 run so checkpoint success reflects genuine `SAC + HER` Q-learning over `Phi_z` and buffer transitions rather than direct supervised cloning. If Critic bootstrapping alone proves too slow to trigger grasping within `10k–20k` steps, decaying BC actor initialization (`0.3050 -> <0.02`) is the clean Option B escalation path.

