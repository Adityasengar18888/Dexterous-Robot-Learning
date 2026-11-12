"""Authoritative PyBullet Simulation Manager for FastAPI & WebSocket Streaming."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import pybullet as p

from robotics.controllers import (
    AdaptiveTransportController,
    CompliantImpedanceAssemblyController,
    PreGraspController,
    ScriptedAssemblyController,
    get_step_telemetry,
)
from robotics.environments import (
    PandaAssemblyDenseEnv,
    PandaContactAssemblyEnv,
    PandaDynamicPickPlaceEnv,
    PandaMultiSortPreGraspEnv,
    PandaPickEnv,
    PandaPickPlaceDenseRewardEnv,
    PandaPickPlaceEnv,
    PandaReachEnv,
    PandaSinglePreGraspEnv,
    PandaSortEnv,
)
from robotics.robots.franka_panda.robot_model import PANDA_JOINT_NAMES, get_link_world_transforms
from src.controllers.pregrasp_controller import query_pybullet_telemetry
from backend.app.core.config import ROOT_DIR, SIM_RATE_HZ
from backend.app.schemas.robotics import (
    PANDA_LOWER_LIMITS,
    PANDA_UPPER_LIMITS,
    WS_BOUNDS,
)
from backend.app.services.experiment_service import _resolve_best_checkpoint, POLICY_CATALOG


ENV_REGISTRY = {
    "PandaReachEnv": PandaReachEnv,
    "PandaPickEnv": PandaPickEnv,
    "PandaPickPlaceEnv": PandaPickPlaceEnv,
    "PandaPickPlaceDenseRewardEnv": PandaPickPlaceDenseRewardEnv,
    "PandaSortEnv": PandaSortEnv,
    "PandaMultiSortPreGraspEnv": PandaMultiSortPreGraspEnv,
    "PandaSinglePreGraspEnv": PandaSinglePreGraspEnv,
    "PandaAssemblyDenseEnv": PandaAssemblyDenseEnv,
    "PandaContactAssemblyEnv": PandaContactAssemblyEnv,
    "PandaDynamicPickPlaceEnv": PandaDynamicPickPlaceEnv,
}

TASK_CATALOG: List[Dict[str, Any]] = [
    {
        "task_id": "reach_target",
        "name": "Reach Target",
        "env": "PandaReachEnv",
        "status": "IMPLEMENTED",
        "controllers": ["SCRIPTED"],
        "description": "Move Franka Panda end-effector within 5 cm of a randomized 3D target.",
    },
    {
        "task_id": "pick_lift",
        "name": "Pick and Lift",
        "env": "PandaPickEnv",
        "status": "IMPLEMENTED",
        "controllers": ["SCRIPTED"],
        "description": "Grasp tabletop cube and elevate above z >= 0.20 m.",
    },
    {
        "task_id": "pick_place",
        "name": "Pick and Place",
        "env": "PandaPickPlaceEnv",
        "status": "IMPLEMENTED",
        "controllers": ["SCRIPTED", "POLICY"],
        "description": "Autonomous or scripted cube grasp, transport, and goal dwell (Phase 12/14 benchmark).",
    },
    {
        "task_id": "multi_sort",
        "name": "Multi-Object Sorting",
        "env": "PandaMultiSortPreGraspEnv",
        "status": "IMPLEMENTED",
        "controllers": ["SCRIPTED"],
        "description": "Phase 15.2.5A 5-Check bilateral-gated sequential sorting into color bins (N=2 cubes).",
    },
    {
        "task_id": "assembly_insert",
        "name": "Orientation-Aware Peg-in-Slot Assembly",
        "env": "PandaContactAssemblyEnv",
        "status": "IMPLEMENTED",
        "controllers": ["SCRIPTED"],
        "description": "Phase 15.3 / 16.4 SE(2)+Z compliant force-feedback peg insertion.",
    },
    {
        "task_id": "dynamic_manip",
        "name": "Dynamic Moving Target Manipulation",
        "env": "PandaDynamicPickPlaceEnv",
        "status": "IMPLEMENTED",
        "controllers": ["SCRIPTED"],
        "description": "Phase 16.3 predictive interception of moving cube & goal.",
    },
    {
        "task_id": "physical_fci_execution",
        "name": "Physical Robot FCI Control (Phase 17.3)",
        "env": "N/A",
        "status": "NOT IMPLEMENTED",
        "controllers": [],
        "description": "Simulation-only repository: real libfranka hardware execution is NOT_ATTEMPTED.",
    },
]


class SimulationManager:
    """Thread-safe single authoritative PyBullet worker managing stepping, controls, tasks, and policies."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._env: Any = None
        self._env_name: str = "PandaPickPlaceEnv"
        self._seed: int = 42
        self._dr_tier: str = "DR0"
        self._dr_enabled: bool = False

        self._status: str = "IDLE"  # IDLE | RUNNING | PAUSED | STOPPED | EMERGENCY_STOPPED
        self._step_count: int = 0
        self._sim_time_s: float = 0.0
        self._last_obs: Any = None
        self._last_info: Dict[str, Any] = {}
        self._last_error: Optional[str] = None

        # Active task execution state
        self._task_active: bool = False
        self._task_id: Optional[str] = None
        self._task_status: str = "IDLE"
        self._task_step: int = 0
        self._task_max_steps: int = 150
        self._task_controller_mode: str = "SCRIPTED"
        self._task_ctrl: Any = None
        self._scripted_substate: int = 0

        # Active loaded policy
        self._loaded_policy_id: Optional[str] = None
        self._loaded_policy_meta: Optional[Dict[str, Any]] = None
        self._loaded_sb3_model: Any = None
        self._adaptive_wrapper: Any = None

        self._ee_traj: Deque[List[float]] = deque(maxlen=120)
        self._events: Deque[Dict[str, Any]] = deque(maxlen=150)

        self._thread: Optional[threading.Thread] = None
        self._stop_worker = threading.Event()
        self._last_ik_result: Dict[str, Any] = {}

    def _emit_event(self, level: str, source: str, message: str) -> None:
        self._events.append({
            "timestamp": time.time(),
            "level": level,
            "source": source,
            "message": message,
        })

    def ensure_worker(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop_worker.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True, name="pybullet-sim-worker")
            self._thread.start()

    def _build_env(self, env_name: str, seed: int, dr_enabled: bool) -> Any:
        if env_name not in ENV_REGISTRY:
            raise ValueError(f"Unsupported environment: {env_name}")
        cls = ENV_REGISTRY[env_name]
        rcfg = {"enabled": bool(dr_enabled)}
        if env_name in ("PandaPickPlaceEnv", "PandaPickPlaceDenseRewardEnv") and not dr_enabled:
            # Avoid half-sunk spawn floor issue (ERR-010): keep fixed z=0.025 when nominal
            rcfg = {"enabled": False}
        kwargs: Dict[str, Any] = {"render_mode": None}
        try:
            env = cls(randomizer_config=rcfg, **kwargs)
        except TypeError:
            env = cls(**kwargs)
        obs, info = env.reset(seed=seed)
        self._last_obs = obs
        self._last_info = dict(info) if isinstance(info, dict) else {}
        return env

    def start(self, env_name: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
        with self._lock:
            if env_name:
                self._env_name = env_name
            if seed is not None:
                self._seed = seed
            if self._env is None or (env_name and env_name != self._env_name):
                if self._env is not None:
                    try:
                        self._env.close()
                    except Exception:
                        pass
                self._env = self._build_env(self._env_name, self._seed, self._dr_enabled)
            elif self._status == "STOPPED":
                obs, info = self._env.reset(seed=self._seed)
                self._last_obs = obs
                self._last_info = dict(info) if isinstance(info, dict) else {}
            self._status = "RUNNING"
            self._last_error = None
            self.ensure_worker()
            self._emit_event("INFO", "SimulationManager", f"Simulation started ({self._env_name}, seed={self._seed})")
            return self.get_snapshot()

    def pause(self) -> Dict[str, Any]:
        with self._lock:
            if self._status == "RUNNING":
                self._status = "PAUSED"
                self._emit_event("INFO", "SimulationManager", "Simulation paused")
            return self.get_snapshot()

    def resume(self) -> Dict[str, Any]:
        with self._lock:
            if self._status == "EMERGENCY_STOPPED":
                raise RuntimeError("Emergency stop active: execute /api/simulation/reset to clear safety latch.")
            if self._env is None:
                return self.start()
            self._status = "RUNNING"
            self.ensure_worker()
            self._emit_event("INFO", "SimulationManager", "Simulation resumed")
            return self.get_snapshot()

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            self._task_active = False
            self._task_status = "STOPPED"
            if self._status != "EMERGENCY_STOPPED":
                self._status = "STOPPED"
            self._emit_event("WARNING", "SimulationManager", "Simulation stopped")
            return self.get_snapshot()

    def reset(self, env_name: Optional[str] = None, seed: Optional[int] = None, dr_tier: Optional[str] = None, dr_enabled: Optional[bool] = None) -> Dict[str, Any]:
        with self._lock:
            self._task_active = False
            self._task_status = "IDLE"
            self._task_step = 0
            self._scripted_substate = 0
            self._step_count = 0
            self._sim_time_s = 0.0
            self._ee_traj.clear()
            self._last_error = None
            if seed is not None:
                self._seed = seed
            if dr_tier is not None:
                self._dr_tier = dr_tier
            if dr_enabled is not None:
                self._dr_enabled = dr_enabled

            target_env = env_name or self._env_name
            recreate = self._env is None or (target_env != self._env_name)
            self._env_name = target_env
            if recreate:
                if self._env is not None:
                    try:
                        self._env.close()
                    except Exception:
                        pass
                self._env = self._build_env(self._env_name, self._seed, self._dr_enabled)
            else:
                obs, info = self._env.reset(seed=self._seed)
                self._last_obs = obs
                self._last_info = dict(info) if isinstance(info, dict) else {}

            self._status = "RUNNING"
            self.ensure_worker()
            self._emit_event("INFO", "SimulationManager", f"Simulation reset ({self._env_name}, seed={self._seed})")
            return self.get_snapshot()

    def emergency_stop(self) -> Dict[str, Any]:
        with self._lock:
            self._task_active = False
            self._task_status = "EMERGENCY_STOPPED"
            self._status = "EMERGENCY_STOPPED"
            self._last_error = "E-STOP engaged by operator: active commands halted."
            if self._env is not None and self._env.controller.robot is not None:
                robot = self._env.controller.robot
                cur_q = robot.get_joint_positions().tolist()
                p.setJointMotorControlArray(
                    bodyUniqueId=robot.body_id,
                    jointIndices=robot.arm_joint_indices,
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=cur_q,
                    targetVelocities=[0.0] * 7,
                    forces=robot.arm_max_forces,
                    physicsClientId=robot.cid,
                )
            self._emit_event("ERROR", "SafetySupervisor", "EMERGENCY STOP triggered")
            return self.get_snapshot()

    def _require_ready_for_command(self) -> None:
        if self._status == "EMERGENCY_STOPPED":
            raise RuntimeError("Emergency stop active. Reset simulation before sending motion commands.")
        if self._env is None:
            self.start()

    def set_joints(self, positions: List[float], steps: int = 60) -> Dict[str, Any]:
        with self._lock:
            self._require_ready_for_command()
            robot = self._env.controller.robot
            p.setJointMotorControlArray(
                bodyUniqueId=robot.body_id,
                jointIndices=robot.arm_joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=[float(x) for x in positions],
                forces=robot.arm_max_forces,
                physicsClientId=robot.cid,
            )
            self._env.sim.step(steps)
            self._step_count += 1
            self._sim_time_s += steps / 240.0
            achieved = robot.get_joint_positions().tolist()
            self._emit_event("INFO", "JointControl", f"Applied 7-DOF target (max_err={max(abs(a-b) for a,b in zip(positions, achieved)):.4f} rad)")
            return {"achieved_joints": [round(float(x), 5) for x in achieved], "snapshot": self.get_snapshot()}

    def move_cartesian(
        self,
        x: float,
        y: float,
        z: float,
        roll: Optional[float] = None,
        pitch: Optional[float] = None,
        yaw: Optional[float] = None,
        steps: int = 90,
    ) -> Dict[str, Any]:
        with self._lock:
            self._require_ready_for_command()
            ctrl = self._env.controller
            target_pos = np.array([x, y, z], dtype=np.float64)
            target_orn = None
            if roll is not None and pitch is not None and yaw is not None:
                target_orn = np.array(p.getQuaternionFromEuler([roll, pitch, yaw]), dtype=np.float64)
            else:
                target_orn = np.array(ctrl.robot.default_ee_orientation, dtype=np.float64)

            achieved_pos = ctrl.move_ee(target_pos, target_orn, steps=steps, tolerance=0.002)
            achieved_orn = ctrl.get_ee_orientation()
            pos_err = float(np.linalg.norm(achieved_pos - target_pos))
            dot_q = abs(float(np.dot(achieved_orn / np.linalg.norm(achieved_orn), target_orn / np.linalg.norm(target_orn))))
            orn_err = float(2.0 * math.acos(min(1.0, dot_q)))

            self._step_count += 1
            self._sim_time_s += steps / 240.0
            res = {
                "requested_pose": {"x": x, "y": y, "z": z, "quat": [round(float(v), 5) for v in target_orn]},
                "achieved_pose": {
                    "x": round(float(achieved_pos[0]), 5),
                    "y": round(float(achieved_pos[1]), 5),
                    "z": round(float(achieved_pos[2]), 5),
                    "quat": [round(float(v), 5) for v in achieved_orn],
                },
                "position_error_m": round(pos_err, 5),
                "orientation_error_rad": round(orn_err, 5),
                "ik_converged": pos_err < 0.025,
                "joint_positions": [round(float(v), 5) for v in ctrl.get_joint_positions()],
            }
            self._last_ik_result = res
            self._emit_event("INFO", "CartesianControl", f"IK target ({x:.3f},{y:.3f},{z:.3f}) -> err={pos_err*1000:.2f} mm")
            return {**res, "snapshot": self.get_snapshot()}

    def command_gripper(self, action: str, width: Optional[float] = None, steps: int = 40) -> Dict[str, Any]:
        with self._lock:
            self._require_ready_for_command()
            ctrl = self._env.controller
            if action == "open":
                actual = ctrl.open_gripper(steps=steps)
            elif action == "close":
                actual = ctrl.close_gripper(steps=steps)
            elif action == "set_width":
                if width is None:
                    raise ValueError("width is required when action=='set_width'")
                ctrl.robot.set_gripper_width(float(width))
                ctrl.sim.step(steps)
                actual = ctrl.get_gripper_state()["width"]
            else:
                raise ValueError(f"Invalid gripper action: {action}")
            self._step_count += 1
            self._sim_time_s += steps / 240.0
            self._emit_event("INFO", "GripperControl", f"Gripper '{action}' -> width={actual*100:.2f} cm")
            return {"actual_width": round(float(actual), 5), "status": "OK", "snapshot": self.get_snapshot()}

    def load_policy(self, policy_id: str) -> Dict[str, Any]:
        with self._lock:
            entry = next((p for p in POLICY_CATALOG if p["policy_id"] == policy_id), None)
            if entry is None:
                raise ValueError(f"Unknown policy_id: {policy_id}")

            resolved = _resolve_best_checkpoint(entry["checkpoint_path"])
            if entry["provenance"] != "SCRIPTED" and not resolved.exists():
                raise FileNotFoundError(f"Checkpoint not found on disk: {entry['checkpoint_path']}")

            if entry["provenance"] == "SCRIPTED":
                self._loaded_policy_id = policy_id
                self._loaded_policy_meta = entry
                self._loaded_sb3_model = None
                self._adaptive_wrapper = None
            else:
                # Validate SB3 load with ERR-008 custom_objects HerReplayBuffer bypass
                from stable_baselines3 import SAC
                model = SAC.load(
                    str(resolved),
                    device="cpu",
                    custom_objects={"replay_buffer_class": None, "replay_buffer_kwargs": {}},
                )
                self._loaded_policy_id = policy_id
                self._loaded_policy_meta = {**entry, "resolved_path": str(resolved.relative_to(ROOT_DIR)).replace("\\", "/")}
                self._loaded_sb3_model = model
                if "adaptive_e" in policy_id:
                    self._adaptive_wrapper = AdaptiveTransportController()
                    self._adaptive_wrapper.reset()
                else:
                    self._adaptive_wrapper = None

            self._emit_event("INFO", "PolicyLoader", f"Loaded policy '{policy_id}' ({entry['provenance']})")
            return {"policy": self._loaded_policy_meta, "snapshot": self.get_snapshot()}

    def start_task(self, task_id: str, controller_mode: str = "SCRIPTED", policy_id: Optional[str] = None, seed: int = 42, max_steps: int = 150) -> Dict[str, Any]:
        task_spec = next((t for t in TASK_CATALOG if t["task_id"] == task_id), None)
        if task_spec is None:
            raise ValueError(f"Unknown task_id: {task_id}")
        if task_spec["status"] != "IMPLEMENTED":
            raise NotImplementedError(f"Task '{task_id}' is NOT IMPLEMENTED ({task_spec['description']})")

        with self._lock:
            if controller_mode == "POLICY":
                pid = policy_id or self._loaded_policy_id
                if pid is None:
                    raise ValueError("No policy_id provided or preloaded for POLICY mode")
                if pid != self._loaded_policy_id:
                    self.load_policy(pid)
                if self._loaded_policy_meta and task_spec["env"] not in self._loaded_policy_meta["compatible_envs"]:
                    # Allow switching to compatible env if policy targets dense variant
                    compat_env = self._loaded_policy_meta["compatible_envs"][0]
                else:
                    compat_env = task_spec["env"]
            else:
                compat_env = task_spec["env"]

            self.reset(env_name=compat_env, seed=seed)
            self._task_id = task_id
            self._task_controller_mode = controller_mode
            self._task_step = 0
            self._task_max_steps = max_steps
            self._task_status = "RUNNING"
            self._task_active = True

            if controller_mode == "SCRIPTED":
                if task_id == "multi_sort":
                    self._task_ctrl = PreGraspController()
                    self._task_ctrl.reset(active_obj_idx=0)
                elif task_id == "assembly_insert":
                    self._task_ctrl = CompliantImpedanceAssemblyController()
                    self._task_ctrl.reset()
                else:
                    self._task_ctrl = None  # Built-in Cartesian PD state machine per step
            self._emit_event("INFO", "TaskRunner", f"Started task '{task_id}' ({controller_mode}, env={compat_env})")
            return self.get_snapshot()

    def stop_task(self) -> Dict[str, Any]:
        with self._lock:
            self._task_active = False
            self._task_status = "STOPPED"
            self._emit_event("WARNING", "TaskRunner", f"Stopped task '{self._task_id}'")
            return self.get_snapshot()

    def _compute_scripted_pickplace_action(self) -> np.ndarray:
        """Simple deterministic 4D Cartesian PD step for Reach/Pick/PickPlace/Dynamic demos."""
        ee = self._env.controller.get_ee_position()
        grip_w = self._env.controller.get_gripper_state()["width"]
        obj = self._env.cube.get_position() if hasattr(self._env, "cube") else np.array([0.45, 0.0, 0.025])
        goal = getattr(self._env, "goal", np.array([0.55, 0.15, 0.15]))

        if self._task_id == "reach_target":
            delta = (goal[:3] - ee) * 6.0
            return np.clip([delta[0], delta[1], delta[2], 1.0], -1.0, 1.0).astype(np.float32)

        # Stage 0: Hover above cube -> 1: Descend -> 2: Close -> 3: Lift/Move -> Hold
        dist_xy = float(np.linalg.norm(ee[:2] - obj[:2]))
        if self._scripted_substate == 0:
            tgt = np.array([obj[0], obj[1], obj[2] + 0.10])
            grip = 1.0
            if dist_xy < 0.02 and abs(ee[2] - tgt[2]) < 0.025:
                self._scripted_substate = 1
        elif self._scripted_substate == 1:
            tgt = np.array([obj[0], obj[1], obj[2] + 0.005])
            grip = 1.0
            if np.linalg.norm(ee - tgt) < 0.02:
                self._scripted_substate = 2
        elif self._scripted_substate == 2:
            tgt = np.array([obj[0], obj[1], obj[2] + 0.005])
            grip = -1.0
            if grip_w < 0.048 or self._task_step > 35:
                self._scripted_substate = 3
        else:
            tgt = np.array([goal[0], goal[1], max(float(goal[2]), 0.12)])
            grip = -1.0

        delta = (tgt - ee) * 7.0
        return np.clip([delta[0], delta[1], delta[2], grip], -1.0, 1.0).astype(np.float32)

    def _step_task_once(self) -> None:
        if not self._task_active or self._env is None:
            return
        self._task_step += 1
        try:
            if self._task_controller_mode == "POLICY" and self._loaded_sb3_model is not None:
                action, _ = self._loaded_sb3_model.predict(self._last_obs, deterministic=True)
                if self._adaptive_wrapper is not None:
                    action, _ = self._adaptive_wrapper.step_action(action, get_step_telemetry(self._env))
            elif self._task_id == "multi_sort" and isinstance(self._task_ctrl, PreGraspController):
                telem = query_pybullet_telemetry(self._env, self._task_ctrl.active_obj_idx)
                action, _ = self._task_ctrl.step(self._task_step, telem)
            elif self._task_id == "assembly_insert" and isinstance(self._task_ctrl, CompliantImpedanceAssemblyController):
                action, _ = self._task_ctrl.step_compliant(self._env, self._last_obs, self._last_info)
            else:
                action = self._compute_scripted_pickplace_action()

            obs, rew, term, trunc, info = self._env.step(action)
            self._last_obs = obs
            self._last_info = dict(info) if isinstance(info, dict) else {}
            if self._last_info.get("is_success"):
                self._task_active = False
                self._task_status = "COMPLETED_SUCCESS"
                self._emit_event("INFO", "TaskRunner", f"Task '{self._task_id}' succeeded at step {self._task_step}")
            elif term or trunc or self._task_step >= self._task_max_steps:
                self._task_active = False
                self._task_status = "COMPLETED_TIMEOUT"
                self._emit_event("WARNING", "TaskRunner", f"Task '{self._task_id}' ended at step {self._task_step} (success=False)")
        except Exception as exc:
            self._task_active = False
            self._task_status = "FAILED"
            self._last_error = str(exc)
            self._emit_event("ERROR", "TaskRunner", f"Task error: {exc}")

    def _loop(self) -> None:
        period = 1.0 / max(1.0, SIM_RATE_HZ)
        while not self._stop_worker.is_set():
            t0 = time.perf_counter()
            with self._lock:
                if self._env is not None and self._status == "RUNNING":
                    if self._task_active:
                        self._step_task_once()
                    else:
                        self._env.sim.step(4)
                    self._step_count += 1
                    self._sim_time_s += period
                    ee = self._env.controller.get_ee_position()
                    self._ee_traj.append([round(float(v), 4) for v in ee])
            dt = time.perf_counter() - t0
            time.sleep(max(0.005, period - dt))

    def _extract_scene_entities(self) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        objects: List[Dict[str, Any]] = []
        targets: List[Dict[str, Any]] = []
        bins: List[Dict[str, Any]] = []

        if hasattr(self._env, "cube") and self._env.cube is not None:
            pos = self._env.cube.get_position()
            orn = p.getBasePositionAndOrientation(self._env.cube.body_id, physicsClientId=self._env.sim.client_id)[1] if hasattr(self._env.cube, "body_id") else [0, 0, 0, 1]
            objects.append({"id": "cube_0", "kind": "cube", "size": 0.04, "color": "#C44E52",
                            "position": [round(float(v), 4) for v in pos],
                            "quaternion": [round(float(v), 4) for v in orn]})
        if hasattr(self._env, "cubes") and isinstance(self._env.cubes, list):
            palette = ["#C44E52", "#4C72B0", "#55A868"]
            for i, cb in enumerate(self._env.cubes):
                pos = cb.get_position()
                orn = p.getBasePositionAndOrientation(cb.body_id, physicsClientId=self._env.sim.client_id)[1] if hasattr(cb, "body_id") else [0, 0, 0, 1]
                objects.append({"id": f"cube_{i}", "kind": "cube", "size": 0.04, "color": palette[i % len(palette)],
                                "position": [round(float(v), 4) for v in pos],
                                "quaternion": [round(float(v), 4) for v in orn]})
        if hasattr(self._env, "peg") and self._env.peg is not None:
            pos = self._env.peg.get_position()
            objects.append({"id": "peg_0", "kind": "peg", "size": 0.045, "color": "#D6A65D",
                            "position": [round(float(v), 4) for v in pos], "quaternion": [0, 0, 0, 1]})
        if hasattr(self._env, "receptacle") and self._env.receptacle is not None:
            pos = self._env.receptacle.get_position()
            targets.append({"id": "receptacle", "kind": "slot", "position": [round(float(v), 4) for v in pos], "color": "#789DC4"})
        if hasattr(self._env, "bins") and isinstance(self._env.bins, list):
            palette = ["#C44E52", "#4C72B0", "#55A868"]
            for i, bn in enumerate(self._env.bins):
                pos = bn.get_position() if hasattr(bn, "get_position") else getattr(bn, "position", [0.5, 0.2 * (i - 1), 0.0])
                bins.append({"id": f"bin_{i}", "position": [round(float(v), 4) for v in pos], "color": palette[i % len(palette)]})
        if hasattr(self._env, "goal") and self._env.goal is not None:
            g = np.asarray(self._env.goal, dtype=float).flatten()
            for idx in range(0, min(len(g), 6), 3):
                if idx + 3 <= len(g):
                    targets.append({"id": f"goal_{idx//3}", "kind": "target",
                                    "position": [round(float(v), 4) for v in g[idx:idx + 3]],
                                    "color": "#78B89A"})
        return objects, targets, bins

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            if self._env is None or self._env.controller.robot is None:
                return {
                    "timestamp": time.time(),
                    "status": self._status,
                    "step_count": 0,
                    "sim_time_s": 0.0,
                    "active_env": self._env_name,
                    "seed": self._seed,
                    "dr_tier": self._dr_tier,
                    "dr_enabled": self._dr_enabled,
                    "emergency_stopped": self._status == "EMERGENCY_STOPPED",
                    "joint_names": PANDA_JOINT_NAMES,
                    "joint_positions": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
                    "joint_velocities": [0.0] * 7,
                    "joint_limits": {"lower": PANDA_LOWER_LIMITS, "upper": PANDA_UPPER_LIMITS},
                    "workspace_bounds": WS_BOUNDS,
                    "ee_position": [0.307, 0.0, 0.485],
                    "ee_orientation": [1.0, 0.0, 0.0, 0.0],
                    "ee_rpy": [3.1416, 0.0, 0.0],
                    "gripper": {"width": 0.08, "is_open": True},
                    "link_transforms": [],
                    "objects": [],
                    "targets": [],
                    "bins": [],
                    "ee_trajectory": [],
                    "task": {"task_id": None, "status": "IDLE", "step": 0, "max_steps": 150, "mode": "SCRIPTED"},
                    "policy_id": self._loaded_policy_id,
                    "metrics": {},
                    "last_ik_result": self._last_ik_result,
                    "last_error": self._last_error,
                }

            ctrl = self._env.controller
            robot = ctrl.robot
            ee_pos = ctrl.get_ee_position()
            ee_orn = ctrl.get_ee_orientation()
            ee_rpy = p.getEulerFromQuaternion(ee_orn.tolist())
            grip = ctrl.get_gripper_state()
            objs, tgts, bins = self._extract_scene_entities()

            return {
                "timestamp": time.time(),
                "status": self._status,
                "step_count": self._step_count,
                "sim_time_s": round(self._sim_time_s, 3),
                "active_env": self._env_name,
                "seed": self._seed,
                "dr_tier": self._dr_tier,
                "dr_enabled": self._dr_enabled,
                "emergency_stopped": self._status == "EMERGENCY_STOPPED",
                "joint_names": PANDA_JOINT_NAMES,
                "joint_positions": [round(float(v), 5) for v in ctrl.get_joint_positions()],
                "joint_velocities": [round(float(v), 5) for v in ctrl.get_joint_velocities()],
                "joint_limits": {"lower": PANDA_LOWER_LIMITS, "upper": PANDA_UPPER_LIMITS},
                "workspace_bounds": WS_BOUNDS,
                "ee_position": [round(float(v), 5) for v in ee_pos],
                "ee_orientation": [round(float(v), 5) for v in ee_orn],
                "ee_rpy": [round(float(v), 5) for v in ee_rpy],
                "gripper": {"width": round(float(grip["width"]), 5), "is_open": bool(grip["is_open"])},
                "link_transforms": get_link_world_transforms(robot),
                "objects": objs,
                "targets": tgts,
                "bins": bins,
                "ee_trajectory": list(self._ee_traj),
                "task": {
                    "task_id": self._task_id,
                    "status": self._task_status,
                    "step": self._task_step,
                    "max_steps": self._task_max_steps,
                    "mode": self._task_controller_mode,
                },
                "policy_id": self._loaded_policy_id,
                "metrics": {k: (v if isinstance(v, (bool, int, float, str)) else str(v)) for k, v in self._last_info.items()},
                "last_ik_result": self._last_ik_result,
                "last_error": self._last_error,
            }

    def get_events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def shutdown(self) -> None:
        self._stop_worker.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        with self._lock:
            if self._env is not None:
                try:
                    self._env.close()
                except Exception:
                    pass
                self._env = None
            self._status = "STOPPED"


sim_manager = SimulationManager()
