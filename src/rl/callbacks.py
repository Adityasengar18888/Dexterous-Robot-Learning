"""Custom callbacks for Stable-Baselines3 training."""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class SuccessRateCallback(BaseCallback):
    """
    Custom callback for logging training success rate to TensorBoard.
    It extracts the 'is_success' metric from the info dictionary.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.success_history = []
        self.window_size = 100

    def _on_step(self) -> bool:
        # Check if episode ended
        for idx, done in enumerate(self.locals.get("dones", [])):
            if done:
                infos = self.locals.get("infos", [])
                info = {}
                if isinstance(infos, list) and len(infos) > idx:
                    info = infos[idx]
                elif isinstance(infos, tuple) and len(infos) > idx:
                    info = infos[idx]
                elif isinstance(infos, dict):
                    info = infos
                
                # In SB3 vectorized envs, the terminal info is often stored in 'terminal_observation'
                # or 'terminal_info' depending on version. Try direct key first.
                if "is_success" in info:
                    self.success_history.append(float(info["is_success"]))
                elif "terminal_info" in info and "is_success" in info["terminal_info"]:
                    self.success_history.append(float(info["terminal_info"]["is_success"]))
                
                # Keep sliding window
                if len(self.success_history) > self.window_size:
                    self.success_history.pop(0)

        # Log running success rate
        if len(self.success_history) > 0:
            self.logger.record("rollout/success_rate", np.mean(self.success_history))

        return True
