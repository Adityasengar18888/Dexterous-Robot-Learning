# Phase 4 Orientation-Aware Assembly Benchmark Report

**Environment:** `PandaAssemblyDenseEnv` (`src/environments/assembly_dense_env.py`, 22D obs, 7D goal, 5D yaw+grip action)  
**Controller:** `ScriptedAssemblyController` (`src/controllers/assembly_controller.py`)  
**Config:** `configs/assembly_phase15_3.yaml`  
**Chart:** `results/phase15_master/phase15_orientation_assembly_benchmark.png`  

## 1. Task Geometry & Reward Mechanics

`PandaAssemblyEnv` places the table top at `z = -0.025 m`. The `0.060 m`-tall peg (`half=0.030 m`) spawns resting at `z = 0.005 m`, and the receptacle base sits at `z = 0.005 m` with `0.010 m` bottom thickness, giving a target seated center at `z = 0.045 m` and an orientation-aligned `desired_goal` of `(7,)` (`[x, y, z, qx, qy, qz, qw]`).  
`PandaAssemblyDenseEnv` computes:
- Geodesic quaternion orientation error `quat_angle_error_rad()` (yaw-agnostic `180°` sign flip handled via `|dot(q1, q2)|`)
- Lateral peg-to-receptacle contact force (`wall_force`) and jam detection
- Smooth progress-based HER-compatible `compute_reward(achieved_goal, desired_goal, info)` matching single and batched calls (`tests/test_assembly_dense_and_orientation.py`).

## 2. Benchmark Results (`N = 30` Episodes Each, Unseen Seeds)

> **Terminology Note:** The JSON metadata inside `results/phase15_3_assembly/phase15_3_assembly_learned_results.json` stamped `"policy_type": "LEARNED_SAC_HER"` on `bc_warmstart_only` due to a shared evaluation loop; this report classifies `bc_warmstart_model.zip` accurately as `BEHAVIOR_CLONING` and online updates (`checkpoint_2500_steps.zip`, `final_model.zip`) as `LEARNED_SAC_HER`.

| Policy | Classification | Grasp | Lift | Pre-Insert Align | Assembly Success | Jam Rate | Table Coll | Mean Final Pos Err | Mean Final Angle Err | Mean Steps (Succ) | Mean Return |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `ScriptedAssemblyController` | `SCRIPTED` | `100.0%` | `100.0%` | `100.0%` | `56.67%` | `13.33%` | `0.0%` | `0.0391 m` | `1.41°` | `47.06` | `+6.561` |
| `bc_warmstart_model.zip` (`28` demos) | `BEHAVIOR_CLONING` | `23.33%` | `6.67%` | `6.67%` | `0.0%` | `10.0%` | `0.0%` | `0.2689 m` | `84.60°` | `-` | `-1.864` |
| `checkpoint_2500_steps.zip` | `LEARNED_SAC_HER` | `6.67%` | `0.0%` | `0.0%` | `0.0%` | `3.33%` | `0.0%` | `0.2647 m` | `84.57°` | `-` | `-8.849` |
| `final_model.zip` (`5,000` steps) | `LEARNED_SAC_HER` | `3.33%` | `0.0%` | `0.0%` | `0.0%` | `0.0%` | `0.0%` | `0.2675 m` | `80.92°` | `-` | `-6.621` |

## 3. Diagnosis & Status

- **Status:** **PARTIALLY COMPLETE** (`SCRIPTED` orientation-aware insertion achieved `56.67%` tight-clearance peg-in-hole completion and `1.41°` final yaw error with `100%` pre-insert alignment and `0%` table collisions; however, `BEHAVIOR_CLONING` on `28` accepted demos achieved `23.33%` grasp / `6.67%` lift / `0%` insertion, and `5k` online `LEARNED_SAC_HER` collapsed to `3.33%` grasp / `0%` assembly).
- **Bottleneck:** The `5D` action space (jointly controlling 3D translation, yaw rotation, and gripper width) requires far more than `28` demonstrations (`2,510` transitions) and `5k` CPU SAC+HER steps to learn tight-clearance (`~2 mm` lateral / yaw tolerance) insertion without chamfer compliance or force-torque feedback in the observation vector.

