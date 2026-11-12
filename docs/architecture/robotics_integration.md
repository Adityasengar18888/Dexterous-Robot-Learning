# Robotics Integration & Compatibility Architecture (`src/` <-> `robotics/` <-> `backend/`)

## 1. Module Reuse Strategy

To satisfy **Phase 1 clean architecture** without breaking existing `Stable-Baselines3` checkpoints (`*.zip` pickles referencing `src.*`) or any of the **122 regression assertions** in `tests/`, `robotics/` implements a **non-destructive facade & asset layer** over the validated `src/` modules:

| Domain | Restructured Facade | Underlying Validated Module(s) | Adaptation / Notes |
| :--- | :--- | :--- | :--- |
| **Robot URDF & FK** | `robotics/robots/franka_panda/robot_model.py` | `src/robots/panda.py` (`FrankaPanda`) | Extracted real `panda.urdf`, visual `.obj`/`.mtl`, and collision `.obj` meshes from `pybullet_data` into repo; adds `get_link_world_transforms()` returning authoritative PyBullet `worldLinkFramePosition/Orientation` for all 11 links (`base`, `0..6`, `8`, `9`, `10`). |
| **PyBullet Client** | `robotics/simulation/pybullet_client.py` | `src/utils/simulation.py`, `src/utils/randomization.py`, `src/utils/config.py` | Direct reuse of 240 Hz PyBullet stepping (`PyBulletSimulation`) & DR tiers. |
| **Environments** | `robotics/environments/__init__.py` | `src/environments/{reach,pick,pick_place,dense_pick_place,sort,multi_sort,pregrasp_multi_sort,single_pregrasp,assembly,assembly_dense,contact_assembly,dynamic_pick_place}_env.py` | Unified export preserving all Gym Dict (`observation`, `achieved_goal`, `desired_goal`) contracts. |
| **Controllers** | `robotics/controllers/__init__.py` | `src/robots/controller.py`, `src/controllers/*`, `src/rl/adaptive_controllers.py` | Exposes `PandaController`, `ScriptedPreGraspController`, `ScriptedAssemblyController`, `CompliantAssemblyController`, and Phase 14 `ControllerWrapper`. |
| **Policies & Perception** | `robotics/policies/__init__.py` | `src/rl/bc_sac.py`, `src/hrl/*`, `src/perception/*` | Exposes `BCDecayingSAC`, `HRLMetaAgent`, `SkillLibrary`, `SimulatedRGBDCamera`, and `DynamicTargetTracker`. |

## 2. Authoritative Physics Contract

The PyBullet process inside `backend/app/managers/simulation_manager.py` is the **single source of truth**:
- Joint limits (`[-2.8973..2.8973]`, ..., `[-2.8973..2.8973]` rad) and workspace bounds (`x:[0.25,0.75]`, `y:[-0.35,0.35]`, `z:[0.02,0.50]` m) are enforced server-side inside `FrankaPanda` and Pydantic validators.
- Cartesian targets resolve via PyBullet's existing Damped Least-Squares IK (`FrankaPanda.calculate_ik`).
- Three.js (`frontend/`) renders `panda_link0`..`panda_rightfinger` `.obj` meshes directly from live PyBullet link poses streamed over `/ws/simulation`.

