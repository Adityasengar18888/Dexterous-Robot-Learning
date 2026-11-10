"""FIFO latency wrapper for Phase 15.1 Sim-to-Real transfer readiness.

Models discrete control-step delays on observations/telemetry, actions, or both
at the agent-environment boundary via bounded FIFO queues, leaving PyBullet
physics and frozen SAC weights unmodified.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Tuple
import numpy as np


class LatencyWrapper:
    """Simulates observation and/or action delivery delays using bounded FIFO queues."""

    def __init__(
        self,
        obs_delay_steps: int = 0,
        action_delay_steps: int = 0,
    ) -> None:
        if obs_delay_steps < 0 or action_delay_steps < 0:
            raise ValueError("Delay steps must be non-negative integers.")
        self.obs_delay_steps = int(obs_delay_steps)
        self.action_delay_steps = int(action_delay_steps)
        self._obs_queue: Deque[Tuple[Dict[str, np.ndarray], Dict[str, Any]]] = deque(maxlen=self.obs_delay_steps + 1)
        self._act_queue: Deque[np.ndarray] = deque(maxlen=self.action_delay_steps + 1)

    def reset(self, initial_obs: Dict[str, np.ndarray], initial_telem: Dict[str, Any]) -> None:
        """Pre-fill observation queue with initial t=0 state and action queue with zero-motion open."""
        self._obs_queue.clear()
        self._act_queue.clear()

        init_pair = (
            {k: np.asarray(v, dtype=np.float32).copy() for k, v in initial_obs.items()},
            dict(initial_telem),
        )
        for _ in range(self.obs_delay_steps + 1):
            self._obs_queue.append(init_pair)

        neutral_act = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        for _ in range(self.action_delay_steps):
            self._act_queue.append(neutral_act.copy())

    def push_and_get_obs(
        self,
        obs: Dict[str, np.ndarray],
        telemetry: Dict[str, Any],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Enqueue current observation/telemetry and return the oldest (k-step delayed) pair."""
        obs_copy = {k: np.asarray(v, dtype=np.float32).copy() for k, v in obs.items()}
        telem_copy = dict(telemetry)
        self._obs_queue.append((obs_copy, telem_copy))
        return self._obs_queue[0]

    def push_and_get_action(self, action: np.ndarray) -> np.ndarray:
        """Enqueue commanded action and return the oldest (m-step delayed) command."""
        act_copy = np.asarray(action, dtype=np.float32).copy()
        self._act_queue.append(act_copy)
        if len(self._act_queue) > self.action_delay_steps:
            return self._act_queue.popleft()
        return act_copy

