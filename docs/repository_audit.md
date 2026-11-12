# Phase 0 Repository Safety Audit & Cleanup Plan

- **Recovery Point Created:** Git branch `backup/pre-restructure-ph17` pinned at commit `33df652`
- **Total Audited Files (excluding `.git`, `.conda-env`, caches):** `1026`
- **Action Breakdown:** KEEP=`980`, ARCHIVE=`44`, REMOVE=`2`, REORGANIZE=`0`

## 1. Original Directory Distribution

| Top Directory | File Count | Role |
| :--- | ---: | :--- |
| `<root>` | 52 | Historical / Active workspace component |
| `adapters` | 3 | Historical / Active workspace component |
| `checkpoints` | 7 | Historical / Active workspace component |
| `configs` | 31 | Historical / Active workspace component |
| `docs` | 26 | Historical / Active workspace component |
| `experiments` | 540 | Historical / Active workspace component |
| `logs` | 25 | Historical / Active workspace component |
| `notebooks` | 1 | Historical / Active workspace component |
| `publication` | 66 | Historical / Active workspace component |
| `results` | 154 | Historical / Active workspace component |
| `scripts` | 32 | Historical / Active workspace component |
| `src` | 60 | Historical / Active workspace component |
| `tests` | 28 | Historical / Active workspace component |
| `videos` | 1 | Historical / Active workspace component |

## 2. Cleanup Classifications

### Safe to Remove (`REMOVE`)
- `plot_err.txt` (0 bytes) — empty temporary matplotlib debug output.
- `test_fig.svg` (0 bytes) — empty SVG render check output.

### Historical Archival (`ARCHIVE` -> `archive/historical/root_scripts/`)
- 37 loose Phase 10–15 root `.py` diagnostic, launch, and plotting scripts (`audit_*.py`, `diagnostic_*.py`, `launch_*.py`, `orchestrate_phase11.py`, `run_phase13_5_diagnostics.py`, `run_phase14_*.py`, `run_phase15_1.py`, `train_*.py`, `verify_500k.py`, `draw_plots_pillow.py`, `test_debug.py`, `test_svg.py`, `debug_init.py`, etc.). Verified zero imports from `tests/`.
- `adapters/*` (`CLAUDE.md`, `GEMINI.md`, `GPT_OSS.md`) archived under `archive/historical/adapters/`.

### Preserved In-Place (`KEEP` — Never Removed)
- All `src/**` modules (imported directly by SB3 `.zip` checkpoints and 122 pytest assertions).
- All `experiments/**`, `checkpoints/**`, `results/**`, `logs/**`, `configs/**`, `tests/**`, and `publication/**` deliverables.
- Extracted official PyBullet `franka_panda` URDF & `.obj`/`.mtl` meshes placed under `robotics/robots/franka_panda/` and served by FastAPI for live Three.js rendering.

## 3. Target Restructured Architecture
- `robotics/`: Unified robotics facade wrapping `src/` (`robots/franka_panda/{urdf,meshes,robot_model.py}`, `simulation/`, `environments/`, `controllers/`, `policies/`, `perception/`) preserving 100% backwards compatibility with existing `src.*` checkpoints & tests.
- `backend/`: FastAPI server, Pydantic schemas, authoritative PyBullet `SimulationManager`, REST API & WebSockets (`/ws/simulation`, `/ws/training`, `/ws/events`), and backend pytest suite.
- `frontend/`: React + TypeScript + Vite + Tailwind + Three.js (`@react-three/fiber`, `@react-three/drei`) + Recharts restrained dark industrial workstation.
- `archive/historical/`: Historical loose scripts and legacy markdown adapters cleanly relocated off repository root.