# Phase 1 Report: Single-Object Grasp Acquisition (`5B2–5B8`) & B.3 Unseen Generalization

**Environment:** `PandaSinglePreGraspEnv` (`src/environments/single_pregrasp_env.py`, 22D obs, 3D goal, 4D action)  
**Controller:** `ScriptedPreGraspController` (`src/controllers/pregrasp_controller.py`)  
**Algorithm:** `BCDecayingSAC` (`src/rl/bc_sac.py`) + HER (`HerReplayBuffer`)  

## 1. Balanced Demonstration Distribution (`5B2`)

Prior Phase `15.2.5B.1` demonstrations over-weighted post-grasp `HOLD` states (`56.7%`). Generating `50` demonstrations via `scripts/generate_phase15_5B2_balanced_demos.py` (`target_hold_steps=5`, `max_approach_cmd=0.55`, `max_align_cmd=0.45`, `max_descent_dz=0.22`) yielded:

- **Accepted / Rejected:** `50 / 50` (`0` rejected), `1,672` transitions
- **Mean Length / Mean Return:** `33.44` steps / `+9.3337`
- **Stage Distribution:** `APPROACH=717`, `ALIGNMENT=200`, `CONTROLLED_DESCENT=205`, `GRASP_AUTHORIZATION=50`, `GRASP=50`, `LIFT_VERIFICATION=252`, `ACQUIRED=50`, `HOLD_AT_GOAL=148`
- **Phase Breakdown:** Open Approach/Align/Descent `67.11%`, Authorization + Close `5.98%`, Lift `15.07%`, Hold `11.84%`, Open Gripper Command `70.10%`
- **Safety:** `0` table collisions, `0` premature closures (`experiments/multi_sort_phase15_5B2/seed42/demonstrations/balanced_demo_summary.json`).

## 2. Controlled Ablations (`5B2–5B8`, 20-ep Evaluation per Checkpoint)

| Run | Config Highlights | BC Warmstart Grasp / Lift | Peak Post-Actor SAC Grasp / Lift | Final Online SAC Grasp / Lift |
|---|---|---|---|---|
| `5B2` (`0..20k`) | `min_ent_coef=0.02`, `bc_lambda=0.305->0.02` (5k decay) | N/A (`0_initial: 0% / 0%`) | `75.0%` (`17.5k`) / `5.0%` (`7.5k,12.5k,17.5k,20k`) | `35.0% / 5.0%` (`20k`) |
| `5B3` (`0..5k`) | `1k` BC warmstart, `5k` SAC | `50.0% / 50.0%` | `0.0% / 0.0%` (`2.5k–5k`) | `0.0% / 0.0%` (`5k`) |
| `5B4` (`0..10k`) | `2k` BC + `1.5k` critic warmup | `90.0% / 90.0%` | `40.0%` (`10k`) / `0.0%` | `40.0% / 0.0%` (`10k`) |
| `5B5` (`0..10k`) | Balanced 50-ep demos + `1.6k` online + `actor_start=2600` | `20.0% / 15.0%` | `45.0%` (`5k`) / `25.0%` (`5k`) | `25.0% / 5.0%` (`10k`) |
| `5B6` (`0..10k`) | `gate_goal_lift_on_lifted_dg=true`, `her_n_sampled_goal=1` | `20.0% / 15.0%` | `10.0%` (`7.5k`) / `0.0%` | `5.0% / 0.0%` (`10k`) |
| `5B7` (`0..10k`) | `2.5k` BC + `bc_normalize_q=true` (`actor_start=2600`) | `80.0% / 70.0%` | `15.0%` (`7.5k`) / `0.0%` | `5.0% / 0.0%` (`10k`) |
| `5B8` (`0..10k`) | `lr=1e-4`, `learning_starts=2500`, `actor_start=4500` | `80.0% / 70.0%` | `10.0%` (`5k,10k`) / `10.0%` (`5k,10k`) | `10.0% / 10.0%` (`10k`) |

> **Key Stability Finding:** Notice that in `5B5`–`5B8`, the `2500`-step checkpoint precedes `actor_update_start_step` (`2600`/`4500`) and therefore evaluates the frozen `BC` warm-start weights, not post-actor SAC updates. Once SAC actor updates begin against HER-relabeled off-policy transitions without explicit bilateral contact state bits in the 22D observation, pure-RL lift consistently regresses from `70–90%` (`BEHAVIOR_CLONING`) down to `0–25%` (`LEARNED_SAC_HER`).

## 3. B.3 100-Episode Unseen Seed Generalization (`results/phase15_2_5B3/phase15_2_5B3_100ep_results.json`)

| Policy Model | Policy Source Label | N | Reach | Align | Bilateral Grasp | Lift | Lift \| Grasp | Hold Success | Drop \| Grasp | Table Collision | Mean Return |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `5B7 bc_warmstart_model.zip` | `BEHAVIOR_CLONING` | 100 | `97.0%` | `58.0%` | `75.0%` | `45.0%` | `60.0%` | `45.0%` | `5.33%` | `0.0%` | `-0.132` |
| `5B2 checkpoint_17500_steps.zip` | `LEARNED_SAC_HER` | 100 | `92.0%` | `66.0%` | `62.0%` | `7.0%` | `11.29%` | `4.0%` | `8.06%` | `31.0%` | `-4.153` |

## 4. Phase 2 Two-Object RL Transfer Decision

**Status:** `BLOCKED` on the Phase 1 `>=85%` sustained pure-RL grasp+lift gate (`5B2` reached `62%` grasp / `7%` lift on 100 unseen episodes; `5B5` peaked at `25%` online lift and dropped to `5%` at `10k`). In accordance with strict gate governance, two-object RL fine-tuning was **not** forced onto a non-lifting policy. Meanwhile:
- The weight transfer utility `transfer_single_to_multi_sac()` (`src/rl/transfer_utils.py`, `28D -> 43D` input expansion) is implemented and verified (`373,260 -> 392,460` parameters, action-equivalent under `pytest tests/test_phase15_integration.py`).
- The Phase 15.2.5A `SCRIPTED` two-object sorting controller (`results/phase15_2_5A/phase15_2_5A_diagnostics.json`) stands verified at `50/50 (100.0%)` full sorting success (`76.4` mean steps, `0` drops, `0` wrong bins, `0` table collisions).

