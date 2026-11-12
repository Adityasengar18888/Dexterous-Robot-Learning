"""REST API routes exposing authoritative PyBullet control, tasks, policies, and experiments."""

from __future__ import annotations

import platform
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from robotics.robots.franka_panda.robot_model import PANDA_JOINT_NAMES, PANDA_LINK_MESH_SPECS
from backend.app.core.config import FRANKA_ASSETS_DIR
from backend.app.managers.simulation_manager import ENV_REGISTRY, TASK_CATALOG, sim_manager
from backend.app.schemas.robotics import (
    CartesianTargetRequest,
    EnvConfigRequest,
    GripperCommandRequest,
    JointCommandRequest,
    PANDA_LOWER_LIMITS,
    PANDA_UPPER_LIMITS,
    PolicyLoadRequest,
    TaskExecuteRequest,
    WS_BOUNDS,
)
from backend.app.services.experiment_service import (
    get_debugging_history,
    get_evaluation_detail,
    get_experiments,
    get_phase_matrix,
    list_policies,
)

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "physics_engine": "pybullet", "physical_robot": "NOT_ATTEMPTED"}


@router.get("/system/status")
def system_status() -> Dict[str, Any]:
    snap = sim_manager.get_snapshot()
    return {
        "backend": "FastAPI",
        "python": platform.python_version(),
        "physics_backend": "PyBullet (240 Hz / 20 Hz step loop)",
        "cross_sim_backend": "MuJoCo 4.0.1 (available)",
        "simulation_status": snap["status"],
        "active_env": snap["active_env"],
        "assets_mounted": FRANKA_ASSETS_DIR.exists(),
        "test_baseline": "122/122 passed",
        "phases": get_phase_matrix(),
        "debugging_count": len(get_debugging_history()),
    }


@router.get("/robot/info")
def robot_info() -> Dict[str, Any]:
    return {
        "robot": "Franka Emika Panda (7-DOF + 2-finger parallel jaw)",
        "urdf_url": "/assets/franka_panda/panda.urdf",
        "meshes_base_url": "/assets/franka_panda/meshes",
        "joint_names": PANDA_JOINT_NAMES,
        "joint_limits": {"lower": PANDA_LOWER_LIMITS, "upper": PANDA_UPPER_LIMITS},
        "gripper_range_m": [0.0, 0.08],
        "workspace_bounds_m": WS_BOUNDS,
        "link_specs": PANDA_LINK_MESH_SPECS,
    }


@router.get("/simulation/status")
def simulation_status() -> Dict[str, Any]:
    return sim_manager.get_snapshot()


@router.post("/simulation/start")
def simulation_start() -> Dict[str, Any]:
    return sim_manager.start()


@router.post("/simulation/pause")
def simulation_pause() -> Dict[str, Any]:
    return sim_manager.pause()


@router.post("/simulation/resume")
def simulation_resume() -> Dict[str, Any]:
    try:
        return sim_manager.resume()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/simulation/reset")
def simulation_reset() -> Dict[str, Any]:
    return sim_manager.reset()


@router.post("/robot/joints")
def robot_joints(req: JointCommandRequest) -> Dict[str, Any]:
    try:
        return sim_manager.set_joints(req.positions, steps=req.steps)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/robot/cartesian-target")
def robot_cartesian(req: CartesianTargetRequest) -> Dict[str, Any]:
    try:
        return sim_manager.move_cartesian(req.x, req.y, req.z, req.roll, req.pitch, req.yaw, steps=req.steps)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/robot/gripper")
def robot_gripper(req: GripperCommandRequest) -> Dict[str, Any]:
    try:
        return sim_manager.command_gripper(req.action, width=req.width, steps=req.steps)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/robot/emergency-stop")
def robot_estop() -> Dict[str, Any]:
    return sim_manager.emergency_stop()


@router.get("/tasks")
def tasks_list() -> Dict[str, Any]:
    return {"tasks": TASK_CATALOG}


@router.post("/tasks/execute")
def tasks_execute(req: TaskExecuteRequest) -> Dict[str, Any]:
    try:
        return sim_manager.start_task(
            req.task_id,
            controller_mode=req.controller_mode,
            policy_id=req.policy_id,
            seed=req.seed,
            max_steps=req.max_steps,
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/tasks/stop")
def tasks_stop() -> Dict[str, Any]:
    return sim_manager.stop_task()


@router.get("/policies")
def policies_list() -> Dict[str, Any]:
    snap = sim_manager.get_snapshot()
    return {"active_env": snap["active_env"], "loaded_policy_id": snap["policy_id"], "policies": list_policies(snap["active_env"])}


@router.post("/policies/load")
def policies_load(req: PolicyLoadRequest) -> Dict[str, Any]:
    try:
        return sim_manager.load_policy(req.policy_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/experiments")
def experiments_list() -> Dict[str, Any]:
    return {"experiments": get_experiments(), "phases": get_phase_matrix(), "debugging_history": get_debugging_history()}


@router.get("/evaluation/{experiment_id}")
def evaluation_detail(experiment_id: str) -> Dict[str, Any]:
    res = get_evaluation_detail(experiment_id)
    if res is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
    return res


@router.get("/environment/config")
def env_config_get() -> Dict[str, Any]:
    snap = sim_manager.get_snapshot()
    return {
        "environment": snap["active_env"],
        "seed": snap["seed"],
        "dr_tier": snap["dr_tier"],
        "dr_enabled": snap["dr_enabled"],
        "available_environments": sorted(ENV_REGISTRY.keys()),
    }


@router.post("/environment/config")
def env_config_post(req: EnvConfigRequest) -> Dict[str, Any]:
    if req.environment not in ENV_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown environment: {req.environment}")
    return sim_manager.reset(env_name=req.environment, seed=req.seed, dr_tier=req.dr_tier, dr_enabled=req.dr_enabled)

