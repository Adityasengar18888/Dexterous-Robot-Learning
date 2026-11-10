# Phase 15.2.5B.1 — Vertical Action-Pathway Diagnostic Audit

**Project:** Robotic Manipulation using SAC + HER  
**Environment:** [`PandaSinglePreGraspEnv`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/environments/single_pregrasp_env.py)  
**Test Suite:** [`tests/test_vertical_action_pathway.py`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/tests/test_vertical_action_pathway.py) (`2 passed in 23.06s`)  
**Verdict:** **PASS** — Negative `action[2]` cleanly produces downward EE motion; IK, joint limits, action scaling, and train/eval parity are all valid.

---

## 1. Ten-Point Verification Checklist

1. **EE Vertical Coordinate in Observation**: Verified. `obs["observation"][0:3]` directly reports `controller.get_ee_position()`; `obs["observation"][2]` matches PyBullet EE `z` within `1e-5 m` (`0.4853 m` at home reset).
2. **Object Position in Observation**: Verified. `obs["observation"][4:7]` and `obs["achieved_goal"][0:3]` directly report `cubes[0].get_position()`; `obs["observation"][6]` matches cube `z` (`0.0250 m`) within `1e-5 m`.
3. **Vertical Movement Action Dimension**: Verified. `action_space = Box(-1.0, 1.0, (4,), float32)` corresponding to `[dx, dy, dz, gripper]`, so `action[2]` commands vertical EE displacement.
4. **Action Normalization & Scaling**: Verified. `action` is clipped to `[-1.0, 1.0]` and scaled by `action_scale = 0.05 m/step` (`±50 mm` max per environment step).
5. **Positive vs Negative Vertical Signs**: Verified. **Negative `action[2] < 0` produces downward motion** (`Δz < 0`), while **positive `action[2] > 0` produces upward motion** (`Δz > 0`), bounded by [`FrankaPanda.clip_to_workspace`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/robots/panda.py#L191-L197) (`ws_z = [0.02, 0.50]`). Note: from reset (`z = 0.4853 m`), positive `action[2]` quickly hits the `0.5000 m` ceiling, which explains why SAC's `|act[2]| = 0.4971` upward hover in Phase B stayed harmlessly pinned at `z = 0.50 m`.
6. **Gripper Independence**: Verified. Running `action = [0, 0, -1.0, +1.0]` vs `[0, 0, -1.0, -1.0]` over `8` steps yielded identical vertical displacement (`-0.3598 m`, end `z = 0.1255 m`, `|Δz_diff| <= 0.005 m`) while gripper width changed independently (`0.0800 m` open vs `0.0000 m` closed).
7. **Physical Simulator Motion**: Verified across `11` downward steps (`z: 0.4853 m -> 0.0216 m`, `max_ik_err = 0.0064 m`), using `PandaController.move_ee(steps=15, tolerance=0.005)`.
8. **Joint Limits, IK Limits & Workspace Bounds**: Verified. `joint_limit_events = 0` and `table_collisions = 0` across full descent down to `0.0216 m`.
9. **Action Clipping Effects**: Clipping only enforces `[-1.0, 1.0]` on actions and `ws_z = [0.02, 0.50]` on target Cartesian poses; neither suppresses valid negative vertical commands above `0.02 m`.
10. **Train vs Eval Parity**: Verified [`train.py`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/rl/train.py) and [`evaluate.py`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/rl/evaluate.py) instantiate identical [`PandaSinglePreGraspEnv`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/environments/single_pregrasp_env.py) instances with matching `action_space`, `action_scale = 0.05`, and `seed = 42` reset observations (`atol = 1e-5`).

---

## 2. Controlled Vertical Command Probe Results (`seed = 42`)

| Probe Case | Steps | Command `action[2]` | Gripper Cmd `action[3]` | Start `ee_z` | End `ee_z` | Total `Δee_z` | Max IK Err | Joint Limits | Table Coll | End Gripper Width |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `pos_max_from_home` | `8` | `+1.0` | `+1.0` (open) | `0.4853 m` | `0.5000 m` | `+0.0147 m` *(ws_z ceiling)* | `0.0013 m` | `0` | `0` | `0.0800 m` |
| `pos_from_low_z` | `6` | `+0.5` | `+1.0` (open) | `0.1694 m` | `0.3051 m` | **`+0.1357 m`** | `0.0029 m` | `—` | `—` | `—` |
| `zero_open` | `8` | `0.0` | `+1.0` (open) | `0.4853 m` | `0.4853 m` | **`+0.0000 m`** | `0.0001 m` | `0` | `0` | `0.0800 m` |
| `neg_small_open` | `8` | `-0.1` | `+1.0` (open) | `0.4853 m` | `0.4510 m` | **`-0.0342 m`** | `0.0007 m` | `0` | `0` | `0.0800 m` |
| `neg_max_open` | `8` | `-1.0` | `+1.0` (open) | `0.4853 m` | `0.1255 m` | **`-0.3598 m`** | `0.0061 m` | `0` | `0` | `0.0800 m` |
| `neg_max_close` | `8` | `-1.0` | `-1.0` (close) | `0.4853 m` | `0.1255 m` | **`-0.3598 m`** | `0.0061 m` | `0` | `0` | `0.0000 m` |
| `neg_full_11steps` | `11` | `-1.0` | `+1.0` (open) | `0.4853 m` | `0.0216 m` | **`-0.4637 m`** | `0.0064 m` | `0` | `0` | `0.0800 m` |

**Conclusion:** The physical and kinematic vertical pathway is completely healthy; Phase 15.2.5B's `0.50 m` hovering failure was strictly caused by the absence of an un-gated vertical reward shaping gradient prior to achieving tight XY alignment (`xy <= 0.030 m`).

