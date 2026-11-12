# Dexterous-Robot-Learning — Architecture Refactoring & Robotics Workstation Report

## 1. Initial Repository Condition
Before refactoring (commit `33df652`), the repository held 17 phases of research artifacts across `src/`, `tests/` (122 passing cases), `configs/`, `experiments/`, `results/`, `publication/`, plus 42 unorganized Phase 10–15 Python scripts and 2 empty zero-byte temporary plot files (`plot_err.txt`, `test_fig.svg`) cluttering the repository root.

## 2. Files Reorganized & Archived
- **Archived (`44` files):** Relocated 41 historical root runner/diagnostic/plot scripts cleanly into `archive/historical/root_scripts/` and 3 markdown adapters into `archive/historical/adapters/`. Verified zero imports broken in `tests/`.
- **Deleted (`2` files):** `plot_err.txt` (0 bytes) and `test_fig.svg` (0 bytes).
- **Non-Destructive `robotics/` Facade:** Created `robotics/robots/franka_panda/`, `robotics/simulation/`, `robotics/environments/`, `robotics/controllers/`, `robotics/policies/` re-exporting validated `src.*` modules so every historical SB3 `.zip` checkpoint and all `122` regression tests remain 100% functional without breaking module pickle paths.

## 3. Final Directory Structure
```text
Dexterous-Robot-Learning/
├── robotics/            # Unified Franka Panda URDF/meshes + domain facades over src/
├── backend/             # FastAPI server, Pydantic schemas, authoritative SimulationManager, tests
├── frontend/            # React 18 + TypeScript + Vite + Tailwind + R3F Three.js + Recharts
├── src/, configs/, tests/, experiments/, checkpoints/, logs/, results/, publication/
├── archive/historical/  # Cleanly archived root research scripts & adapters
├── scripts/startup/     # Local FastAPI & Vite launch scripts
└── docs/                # Audit, manifest, integration, API/WS docs, status, report
```

## 4. Backend & Simulation Architecture (`backend/`)
- **Authoritative Physics:** `SimulationManager` (`backend/app/managers/simulation_manager.py`) owns a thread-safe `RLock` and 20 Hz PyBullet stepping daemon thread. The browser never steps physics locally.
- **Endpoints:** 21 REST routes under `/api/*` (`health`, `system/status`, `robot/info`, `simulation/{start,pause,resume,reset,status}`, `robot/{joints,cartesian-target,gripper,emergency-stop}`, `tasks`, `tasks/{execute,stop}`, `policies`, `policies/load`, `experiments`, `evaluation/{id}`, `environment/config`) + static mount `/assets/franka_panda`.
- **WebSockets:** `/ws/simulation` (20 Hz authoritative state + 11 URDF link transforms), `/ws/training` (archived TensorBoard snapshots), `/ws/events` (live diagnostic stream).

## 5. Frontend Architecture & 3D Franka Panda Simulation (`frontend/`)
- **Design System:** Dark industrial `#111315` / `#1B1F23` / `#22272C` workstation (`Inter` UI + `JetBrains Mono` telemetry), thin `#343B42` borders, 10 functional navigation tabs (`Overview`, `Simulation`, `Robot Control`, `Task Execution`, `Reinforcement Learning`, `Environment Config`, `Experiments`, `Evaluation`, `Logs & Diagnostics`, `Settings`).
- **3D Rendering:** Loads real `panda.urdf` `.obj`/`.mtl` meshes (`link0..hand, finger` rotated identically to PyBullet `worldLinkFramePosition/Orientation`), dynamically displays cubes, peg/slot, sorting bins, target spheres, and 3D EE trajectory with smooth `useFrame` interpolation, `OrbitControls`, and visibility toggles.

## 6. Implemented Tasks & Policies
- **Tasks (`IMPLEMENTED`):** `Reach Target`, `Pick and Lift`, `Pick and Place`, `Multi-Object Sorting` (5-Check PreGrasp gate), `Orientation-Aware Peg-in-Slot Assembly` (`CompliantImpedanceAssemblyController`), `Dynamic Moving Target Manipulation`.
- **Explicitly Unimplemented:** `Physical Robot FCI Control (Phase 17.3)` -> tagged `NOT IMPLEMENTED`, returns `HTTP 501` if invoked.
- **Policies:** Cleanly stratified (`SCRIPTED`, `BEHAVIOR_CLONING`, `LEARNED_SAC_HER`, `HYBRID_BC_SAC`) with checkpoint file + env compatibility checks and `CustomObjects` safe HER load.

## 7. Startup Instructions (Two Terminals)
- **Terminal 1 (Backend):** `./scripts/startup/start_backend.ps1` (runs `uvicorn backend.app.main:app --host 127.0.0.1 --port 8000` with `KMP_DUPLICATE_LIB_OK=TRUE`)
- **Terminal 2 (Frontend):** `./scripts/startup/start_frontend.ps1` (runs Vite dev server on `http://localhost:5173` proxying `/api`, `/ws`, and `/assets` seamlessly to `:8000`)

