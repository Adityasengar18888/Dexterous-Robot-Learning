"""Observation-interface noise wrapper for Phase 15.1 Sim-to-Real readiness.

Injects reproducible, zero-mean clipped Gaussian noise onto observed 3D
positions (end-effector, object, target) without corrupting the underlying
PyBullet physical state, object velocities, or orientations.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np


class ObservationNoiseWrapper:
    """Applies bounded Gaussian noise strictly at the perception/telemetry interface."""

    def __init__(
        self,
        std_m: float = 0.0,
        clip_bound_m: float = 0.0,
        seed: int = 2026,
    ) -> None:
        if std_m < 0.0 or clip_bound_m < 0.0:
            raise ValueError("Noise std and clip_bound_m must be non-negative.")
        self.std_m = float(std_m)
        self.clip_bound_m = float(clip_bound_m)
        self.base_seed = int(seed)
        self.rng = np.random.default_rng(self.base_seed)

    def reset(self, episode_seed: int = 0) -> None:
        """Reset RNG deterministically per episode for strict paired repeatability."""
        self.rng = np.random.default_rng(self.base_seed + int(episode_seed))

    def _sample_noise3(self) -> np.ndarray:
        if self.std_m == 0.0 or self.clip_bound_m == 0.0:
            return np.zeros(3, dtype=np.float32)
        raw = self.rng.normal(loc=0.0, scale=self.std_m, size=(3,))
        clipped = np.clip(raw, -self.clip_bound_m, self.clip_bound_m)
        return clipped.astype(np.float32)

    def perturb(
        self,
        obs: Dict[str, np.ndarray],
        telemetry: Dict[str, Any],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Return noisy copies of ``obs`` and ``telemetry`` leaving simulator untouched.

        Consistent noise vectors (ee, obj, goal) are applied identically to both the
        RL Dict observation and the controller telemetry structure.
        """
        noisy_obs = {
            "observation": np.asarray(obs["observation"], dtype=np.float32).copy(),
            "achieved_goal": np.asarray(obs["achieved_goal"], dtype=np.float32).copy(),
            "desired_goal": np.asarray(obs["desired_goal"], dtype=np.float32).copy(),
        }
        noisy_telem = dict(telemetry)

        if self.std_m == 0.0 or self.clip_bound_m == 0.0:
            noisy_telem["ee_pos"] = np.asarray(telemetry["ee_pos"], dtype=np.float32).copy()
            noisy_telem["cube_pos"] = np.asarray(telemetry["cube_pos"], dtype=np.float32).copy()
            if "desired_goal" in telemetry:
                noisy_telem["desired_goal"] = np.asarray(telemetry["desired_goal"], dtype=np.float32).copy()
            return noisy_obs, noisy_telem

        n_ee = self._sample_noise3()
        n_obj = self._sample_noise3()
        n_goal = self._sample_noise3()

        # PandaPickPlaceEnv obs layout: [ee_pos(0:3), gripper_width(3), obj_pos(4:7), obj_vel(7:10)]
        noisy_obs["observation"][0:3] += n_ee
        noisy_obs["observation"][4:7] += n_obj
        noisy_obs["achieved_goal"] += n_obj
        noisy_obs["desired_goal"] += n_goal

        ee_noisy = noisy_obs["observation"][0:3].copy()
        obj_noisy = noisy_obs["observation"][4:7].copy()
        goal_noisy = noisy_obs["desired_goal"].copy()

        noisy_telem["ee_pos"] = ee_noisy
        noisy_telem["cube_pos"] = obj_noisy
        noisy_telem["desired_goal"] = goal_noisy
        noisy_telem["ee_z"] = float(ee_noisy[2])
        noisy_telem["cube_z"] = float(obj_noisy[2])

        rel_vec = ee_noisy - obj_noisy
        noisy_telem["rel_dist"] = float(np.linalg.norm(rel_vec))
        noisy_telem["xy_dist"] = float(np.linalg.norm(rel_vec[:2]))

        return noisy_obs, noisy_telem

