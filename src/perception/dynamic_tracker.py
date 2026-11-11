"""Moving-Object Velocity Estimation, Trajectory Prediction & Replanning (Phase 16.3)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np


@dataclass
class TrackerConfig:
    """Configuration for sliding-window velocity estimation and disturbance replanning."""

    dt_step: float = 0.05               # nominal step duration (s)
    window_size: int = 5                # observations for smoothed velocity regression
    ema_alpha: float = 0.55             # exponential velocity filter gain
    replan_err_thresh_m: float = 0.022  # innovation error triggering intercept replan
    max_pred_horizon_s: float = 0.60


class DynamicObjectTracker:
    """Estimates object velocity from streamed observations, predicts future 3D trajectory,
    detects velocity disturbances / prediction drift, and signals trajectory replanning.
    """

    def __init__(self, config: Optional[TrackerConfig] = None) -> None:
        self.cfg = config or TrackerConfig()
        self.history: Deque[Tuple[float, np.ndarray]] = deque(maxlen=self.cfg.window_size)
        self.vel_est: np.ndarray = np.zeros(3, dtype=np.float32)
        self.last_pred_pos: Optional[np.ndarray] = None
        self.pred_errors: Deque[float] = deque(maxlen=30)
        self.disturbance_count: int = 0
        self.replan_count: int = 0
        self._time_s: float = 0.0

    def reset(self, init_pos: Optional[np.ndarray] = None) -> None:
        self.history.clear()
        self.vel_est = np.zeros(3, dtype=np.float32)
        self.last_pred_pos = None
        self.pred_errors.clear()
        self.disturbance_count = 0
        self.replan_count = 0
        self._time_s = 0.0
        if init_pos is not None:
            self.history.append((0.0, np.asarray(init_pos, dtype=np.float32).copy()))

    def update(self, observed_pos: np.ndarray, dt: Optional[float] = None) -> DictSummary:
        pos = np.asarray(observed_pos, dtype=np.float32).copy()
        step_dt = self.cfg.dt_step if dt is None else float(max(1e-4, dt))
        self._time_s += step_dt

        innovation_err = 0.0
        disturbance = False
        replan_needed = False

        if self.last_pred_pos is not None:
            innovation_err = float(np.linalg.norm(pos - self.last_pred_pos))
            self.pred_errors.append(innovation_err)
            if innovation_err > self.cfg.replan_err_thresh_m:
                disturbance = True
                replan_needed = True
                self.disturbance_count += 1
                self.replan_count += 1

        self.history.append((self._time_s, pos))
        if len(self.history) >= 2:
            # Linear least-squares slope over sliding window rejects single-frame noise
            ts = np.asarray([h[0] for h in self.history], dtype=np.float64)
            ps = np.stack([h[1] for h in self.history], axis=0).astype(np.float64)
            t_mean = np.mean(ts)
            dt_c = ts - t_mean
            denom = float(np.sum(dt_c ** 2))
            if denom > 1e-8:
                raw_vel = ((dt_c[:, None] * (ps - np.mean(ps, axis=0, keepdims=True))).sum(axis=0) / denom).astype(np.float32)
                if disturbance:
                    self.vel_est = raw_vel  # snap velocity estimate quickly after sudden disturbance
                else:
                    a = self.cfg.ema_alpha
                    self.vel_est = (a * raw_vel + (1.0 - a) * self.vel_est).astype(np.float32)

        # Next-step single-step prediction stored for innovation monitoring
        self.last_pred_pos = self.predict_position(horizon_s=step_dt, bounds_min=(0.25, -0.32, 0.02), bounds_max=(0.72, 0.32, 0.35))
        return {
            "pos_est": pos,
            "vel_est": self.vel_est.copy(),
            "innovation_err_m": innovation_err,
            "disturbance_detected": disturbance,
            "replan_triggered": replan_needed,
        }

    def predict_position(
        self,
        horizon_s: float,
        bounds_min: Tuple[float, float, float] = (0.32, -0.28, 0.025),
        bounds_max: Tuple[float, float, float] = (0.68, 0.28, 0.30),
    ) -> np.ndarray:
        if not self.history:
            return np.zeros(3, dtype=np.float32)
        cur_pos = self.history[-1][1]
        h = float(np.clip(horizon_s, 0.0, self.cfg.max_pred_horizon_s))
        pred = cur_pos + self.vel_est * h
        return np.clip(pred, np.asarray(bounds_min, dtype=np.float32), np.asarray(bounds_max, dtype=np.float32)).astype(np.float32)


DictSummary = dict

