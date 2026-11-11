"""Modular simulation deployment architecture (Phase 16.5).

Cleanly separates frozen inference, preprocessing, safety monitoring, controller interfacing,
and metrics logging without importing training frameworks (`src.rl.train`, `HerReplayBuffer`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple
import yaml

import numpy as np
import torch as th
from stable_baselines3 import SAC


@dataclass
class RuntimeConfig:
    """Typed runtime configuration for frozen policy deployment."""

    env_name: str = "PickPlace"
    checkpoint_path: str = ""
    policy_type: str = "LEARNED_SAC_HER"  # LEARNED_SAC_HER | LEARNED_HRL | SCRIPTED | BEHAVIOR_CLONING
    max_episode_steps: int = 100
    distance_threshold: float = 0.05
    action_scale: float = 0.05
    workspace_min: Tuple[float, float, float] = (0.20, -0.35, 0.005)
    workspace_max: Tuple[float, float, float] = (0.75, 0.35, 0.50)
    max_step_action_norm: float = 1.0
    max_contact_force_n: float = 180.0
    enable_safety_monitor: bool = True
    deterministic: bool = True
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "RuntimeConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        valid = {k: v for k, v in raw.items() if hasattr(cls, k)}
        for k in ("workspace_min", "workspace_max"):
            if k in valid and isinstance(valid[k], list):
                valid[k] = tuple(float(x) for x in valid[k])
        return cls(**valid)


class ObsPreprocessor:
    """Normalizes, validates, and optionally injects calibration/sensor noise for inference."""

    def __init__(
        self,
        obs_mean: Optional[np.ndarray] = None,
        obs_std: Optional[np.ndarray] = None,
        clip_range: float = 10.0,
    ) -> None:
        self.obs_mean = None if obs_mean is None else np.asarray(obs_mean, dtype=np.float32)
        self.obs_std = None if obs_std is None else np.maximum(np.asarray(obs_std, dtype=np.float32), 1e-6)
        self.clip_range = float(clip_range)

    def process(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for k, v in obs.items():
            if k in ("rgb", "depth", "segmentation"):
                continue
            arr = np.asarray(v, dtype=np.float32).copy()
            arr = np.nan_to_num(arr, nan=0.0, posinf=self.clip_range, neginf=-self.clip_range)
            if k == "observation" and self.obs_mean is not None and self.obs_std is not None:
                if arr.shape == self.obs_mean.shape:
                    arr = np.clip((arr - self.obs_mean) / self.obs_std, -self.clip_range, self.clip_range)
            out[k] = arr
        return out


class CheckpointLoader:
    """Loads frozen SB3 SAC or PyTorch checkpoints without importing training loops."""

    @staticmethod
    def load_sb3_sac(checkpoint_path: str, device: str = "cpu") -> SAC:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return SAC.load(
            checkpoint_path,
            device=device,
            custom_objects={"replay_buffer_class": None, "replay_buffer_kwargs": {}},
        )

    @staticmethod
    def load_torch_state(checkpoint_path: str, module: th.nn.Module, device: str = "cpu") -> th.nn.Module:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Torch checkpoint not found: {checkpoint_path}")
        payload = th.load(checkpoint_path, map_location=device, weights_only=False)
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        module.load_state_dict(state)
        module.eval()
        return module


class PolicyInferenceInterface:
    """Unified frozen-policy inference wrapper supporting SB3, HRL, or callable policies."""

    def __init__(self, predict_fn: Callable[[Dict[str, np.ndarray], bool], np.ndarray], policy_type: str) -> None:
        self._predict_fn = predict_fn
        self.policy_type = policy_type

    @classmethod
    def from_sb3_checkpoint(cls, checkpoint_path: str, policy_type: str = "LEARNED_SAC_HER", device: str = "cpu") -> "PolicyInferenceInterface":
        model = CheckpointLoader.load_sb3_sac(checkpoint_path, device=device)

        def _pred(obs: Dict[str, np.ndarray], deterministic: bool) -> np.ndarray:
            act, _ = model.predict(obs, deterministic=deterministic)
            return np.asarray(act, dtype=np.float32)

        return cls(_pred, policy_type=policy_type)

    def predict(self, obs: Dict[str, np.ndarray], deterministic: bool = True) -> np.ndarray:
        act = self._predict_fn(obs, deterministic)
        return np.asarray(act, dtype=np.float32)


class ActionPostprocessor:
    """Clips, smooths, and bounds policy output commands before controller execution."""

    def __init__(self, action_low: float = -1.0, action_high: float = 1.0, max_delta: Optional[float] = None) -> None:
        self.low = float(action_low)
        self.high = float(action_high)
        self.max_delta = None if max_delta is None else float(max_delta)
        self._prev_action: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_action = None

    def process(self, action: np.ndarray) -> np.ndarray:
        act = np.nan_to_num(np.asarray(action, dtype=np.float32), nan=0.0, posinf=self.high, neginf=self.low)
        act = np.clip(act, self.low, self.high)
        if self.max_delta is not None and self._prev_action is not None and self._prev_action.shape == act.shape:
            diff = np.clip(act[:3] - self._prev_action[:3], -self.max_delta, self.max_delta)
            act = act.copy()
            act[:3] = self._prev_action[:3] + diff
        self._prev_action = act.copy()
        return act


class SafetyMonitor:
    """Enforces workspace envelopes, contact-force caps, finite checks, and E-stop latching."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.cfg = config
        self.estop_active: bool = False
        self.estop_reason: str = ""
        self.ws_clamps: int = 0
        self.force_violations: int = 0
        self.latency_ms_history: List[float] = []

    def reset(self) -> None:
        self.estop_active = False
        self.estop_reason = ""
        self.ws_clamps = 0
        self.force_violations = 0
        self.latency_ms_history.clear()

    def trigger_estop(self, reason: str) -> None:
        self.estop_active = True
        self.estop_reason = str(reason)

    def clear_estop(self) -> None:
        self.estop_active = False
        self.estop_reason = ""

    def check_and_filter(
        self,
        ee_pos: np.ndarray,
        action: np.ndarray,
        contact_force_n: float = 0.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not np.all(np.isfinite(ee_pos)) or not np.all(np.isfinite(action)):
            self.trigger_estop("non_finite_state_or_action")
            return np.zeros_like(action, dtype=np.float32), {"estop": True, "reason": self.estop_reason}

        if contact_force_n > self.cfg.max_contact_force_n:
            self.force_violations += 1
            self.trigger_estop(f"excess_contact_force:{contact_force_n:.1f}N")

        if self.estop_active:
            safe_zero = np.zeros_like(action, dtype=np.float32)
            if safe_zero.shape[0] >= 4:
                safe_zero[-1] = action[-1]  # hold gripper state safely without arm motion
            return safe_zero, {"estop": True, "reason": self.estop_reason, "clamped": False}

        filtered = np.asarray(action, dtype=np.float32).copy()
        ws_min = np.asarray(self.cfg.workspace_min, dtype=np.float32)
        ws_max = np.asarray(self.cfg.workspace_max, dtype=np.float32)
        pred_next = ee_pos[:3] + filtered[:3] * float(self.cfg.action_scale)
        clipped_next = np.clip(pred_next, ws_min, ws_max)
        clamped = bool(np.any(np.abs(clipped_next - pred_next) > 1e-6))
        if clamped:
            self.ws_clamps += 1
            filtered[:3] = (clipped_next - ee_pos[:3]) / max(float(self.cfg.action_scale), 1e-6)
            filtered[:3] = np.clip(filtered[:3], -1.0, 1.0)

        return filtered, {"estop": False, "reason": "", "clamped": clamped}


class EnvironmentAdapter:
    """Thin adapter standardizing Gymnasium / cross-simulator step, obs, and EE pose extraction."""

    def __init__(self, env: Any) -> None:
        self.env = env

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self.env.reset(seed=seed)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        return self.env.step(action)

    def extract_ee_pos(self, obs: Dict[str, Any]) -> np.ndarray:
        if "observation" in obs and len(obs["observation"]) >= 3:
            return np.asarray(obs["observation"][:3], dtype=np.float32)
        if hasattr(self.env.unwrapped, "controller"):
            return np.asarray(self.env.unwrapped.controller.get_ee_position(), dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    def close(self) -> None:
        self.env.close()


class ControllerInterface(Protocol):
    """Protocol for optional action-refining wrappers placed between inference and environment."""

    def reset(self) -> None: ...
    def wrap_action(self, obs: Dict[str, Any], raw_action: np.ndarray) -> np.ndarray: ...


class MetricsLogger:
    """Structured JSONL / dict episode telemetry collector for deployed policies."""

    def __init__(self) -> None:
        self.episodes: List[Dict[str, Any]] = []

    def log_episode(self, record: Dict[str, Any]) -> None:
        self.episodes.append(dict(record))

    def summarize(self) -> Dict[str, Any]:
        if not self.episodes:
            return {"n_episodes": 0}
        return {
            "n_episodes": len(self.episodes),
            "success_rate": round(float(np.mean([float(e.get("success", False)) for e in self.episodes])), 4),
            "mean_return": round(float(np.mean([float(e.get("return", 0.0)) for e in self.episodes])), 4),
            "mean_steps": round(float(np.mean([float(e.get("steps", 0)) for e in self.episodes])), 2),
            "estop_rate": round(float(np.mean([float(e.get("estop_triggered", False)) for e in self.episodes])), 4),
            "mean_inference_ms": round(float(np.mean([float(e.get("mean_inference_ms", 0.0)) for e in self.episodes])), 4),
        }

    def save(self, out_path: str) -> None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"summary": self.summarize(), "episodes": self.episodes}, f, indent=2)


class DeploymentRuntime:
    """End-to-end frozen runtime orchestrating EnvAdapter -> Preproc -> Policy -> Safety -> Env."""

    def __init__(
        self,
        env_adapter: EnvironmentAdapter,
        policy: PolicyInferenceInterface,
        config: RuntimeConfig,
        preprocessor: Optional[ObsPreprocessor] = None,
        postprocessor: Optional[ActionPostprocessor] = None,
        safety_monitor: Optional[SafetyMonitor] = None,
        controller_hook: Optional[ControllerInterface] = None,
    ) -> None:
        self.env = env_adapter
        self.policy = policy
        self.cfg = config
        self.pre = preprocessor or ObsPreprocessor()
        self.post = postprocessor or ActionPostprocessor()
        self.safety = safety_monitor or SafetyMonitor(config)
        self.ctrl = controller_hook
        self.logger = MetricsLogger()

    def run_episode(self, seed: int, contact_force_fn: Optional[Callable[[], float]] = None) -> Dict[str, Any]:
        obs, info = self.env.reset(seed=seed)
        self.post.reset()
        self.safety.reset()
        if self.ctrl is not None:
            self.ctrl.reset()

        total_reward = 0.0
        steps = 0
        done = False
        trunc = False
        inf_times_ms: List[float] = []
        last_info = dict(info)

        while not (done or trunc) and steps < self.cfg.max_episode_steps:
            clean_obs = self.pre.process(obs)
            t0 = time.perf_counter()
            raw_act = self.policy.predict(clean_obs, deterministic=self.cfg.deterministic)
            inf_times_ms.append((time.perf_counter() - t0) * 1000.0)

            if self.ctrl is not None:
                raw_act = self.ctrl.wrap_action(clean_obs, raw_act)
            post_act = self.post.process(raw_act)

            if self.cfg.enable_safety_monitor:
                ee_pos = self.env.extract_ee_pos(clean_obs)
                force_n = float(contact_force_fn()) if contact_force_fn is not None else 0.0
                exec_act, s_info = self.safety.check_and_filter(ee_pos, post_act, contact_force_n=force_n)
            else:
                exec_act = post_act

            obs, rew, done, trunc, last_info = self.env.step(exec_act)
            total_reward += float(rew)
            steps += 1

        rec = {
            "seed": int(seed),
            "policy_type": self.policy.policy_type,
            "success": bool(last_info.get("is_success", False)),
            "return": round(total_reward, 4),
            "steps": steps,
            "final_distance_m": round(float(last_info.get("distance", 0.0)), 5),
            "ws_clamps": self.safety.ws_clamps,
            "estop_triggered": self.safety.estop_active,
            "estop_reason": self.safety.estop_reason,
            "mean_inference_ms": round(float(np.mean(inf_times_ms)) if inf_times_ms else 0.0, 4),
        }
        self.logger.log_episode(rec)
        return rec

