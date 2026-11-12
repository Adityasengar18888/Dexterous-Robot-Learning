# Backend REST API & WebSocket Documentation

## 1. REST Endpoints (`/api/*`)

| Method & Route | Description & Validation |
| :--- | :--- |
| `GET /api/health` | Health status (`physics_engine="pybullet"`, `physical_robot="NOT_ATTEMPTED"`) |
| `GET /api/system/status` | Active environment, Python version, asset mount flag, Phase 0–18 matrix |
| `GET /api/robot/info` | Franka Panda URDF URL, `/assets/franka_panda/meshes` path, joint limits, 11 link specs |
| `GET /api/simulation/status` | Full authoritative snapshot (joint positions/velocities, EE pose, links, objects, task status) |
| `POST /api/simulation/start` | Starts simulation loop and initializes `PandaPickPlaceEnv` if uninitialized |
| `POST /api/simulation/pause` / `resume` | Pauses or resumes physics stepping (`409` on resume if E-Stop active) |
| `POST /api/simulation/reset` | Resets simulation, objects, task state, and clears Emergency Stop latch |
| `POST /api/robot/joints` | Sets 7-DOF Panda joint targets (`422` if outside Panda URDF joint limits) |
| `POST /api/robot/cartesian-target` | Resolves Cartesian `(x, y, z, roll?, pitch?, yaw?)` via DLS IK (`FrankaPanda.calculate_ik`) |
| `POST /api/robot/gripper` | Commands `"open"`, `"close"`, or `"set_width"` `[0.0, 0.08] m` |
| `POST /api/robot/emergency-stop` | Immediately halts active tasks and clamps Panda motor targets to current joint pose |
| `GET /api/tasks` | Lists implemented tasks + explicit `NOT IMPLEMENTED` physical FCI item |
| `POST /api/tasks/execute` / `stop` | Executes `SCRIPTED` or loaded `POLICY` task inside background worker (`501` if unimplemented) |
| `GET /api/policies` | Lists scripted controllers and SB3 checkpoints stratified by provenance (`SCRIPTED`, `BEHAVIOR_CLONING`, `LEARNED_SAC_HER`, `HYBRID_BC_SAC`) |
| `POST /api/policies/load` | Loads and validates checkpoint existence/architecture (`custom_objects` HER safe load) |
| `GET /api/experiments` | Returns all 27 registered experiments (`experiment_inventory.csv`) and 12 historical bugs |
| `GET /api/evaluation/{experiment_id}` | Returns empirical curve series (`checkpoint_metrics.csv`, JSON summaries, or TB archive) |
| `GET/POST /api/environment/config` | Queries or hot-swaps active Gymnasium environment, seed, and DR tier (`DR0–DR4`) |

## 2. WebSocket Schemas

- **`/ws/simulation` (20 Hz JSON):** `{ "type": "simulation_state", "payload": { timestamp, status, step_count, sim_time_s, active_env, emergency_stopped, joint_positions, joint_velocities, ee_position, ee_orientation, ee_rpy, gripper, link_transforms[11], objects, targets, bins, ee_trajectory, task, policy_id, metrics, last_ik_result, last_error } }`
- **`/ws/training`:** Streams real archived TensorBoard summary snapshots (`runs[]`) without starting heavy jobs automatically.
- **`/ws/events`:** Streams live backend event log entries (`{ timestamp, level, source, message }`).

