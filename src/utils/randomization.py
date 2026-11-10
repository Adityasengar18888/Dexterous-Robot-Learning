"""Domain Randomization utilities for Gymnasium environments."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pybullet as p

class DomainRandomizer:
    """Handles controlled domain randomization based on configuration."""

    def __init__(self, config: Dict[str, Any], np_random: np.random.Generator):
        """
        Initialize the randomizer with a configuration and random number generator.
        
        Args:
            config: A dictionary containing randomization settings.
            np_random: The Gymnasium environment's PRNG.
        """
        self.config = config or {}
        self.np_random = np_random
        self.enabled = self.config.get("enabled", False)

    def is_enabled(self, feature: str) -> bool:
        """Check if a specific randomization feature is enabled."""
        if not self.enabled:
            return False
        feature_cfg = self.config.get(feature, {})
        if isinstance(feature_cfg, bool):
            return feature_cfg
        return feature_cfg.get("enabled", False)

    def sample_position(
        self,
        feature: str,
        default_pos: np.ndarray,
        bounds_min: np.ndarray,
        bounds_max: np.ndarray,
        z_fixed: Optional[float] = None
    ) -> np.ndarray:
        """
        Sample a random position within bounds if enabled, else return default.
        """
        if not self.is_enabled(feature):
            return default_pos.copy()

        # If ranges are explicitly provided in config, use them; else use bounds
        feature_cfg = self.config.get(feature, {})
        ranges = feature_cfg.get("ranges", {})
        
        c_min = bounds_min.copy()
        c_max = bounds_max.copy()
        
        if "x" in ranges:
            c_min[0], c_max[0] = ranges["x"][0], ranges["x"][1]
        if "y" in ranges:
            c_min[1], c_max[1] = ranges["y"][0], ranges["y"][1]
        if "z" in ranges:
            c_min[2], c_max[2] = ranges["z"][0], ranges["z"][1]

        pos = self.np_random.uniform(low=c_min, high=c_max).astype(np.float32)
        if z_fixed is not None:
            pos[2] = z_fixed
            
        return pos

    def sample_orientation(
        self,
        feature: str,
        default_ori: np.ndarray,
        yaw_only: bool = True
    ) -> np.ndarray:
        """
        Sample a random quaternion orientation.
        By default, limits to yaw (Z-axis rotation) for stability in manipulation.
        """
        if not self.is_enabled(feature):
            return default_ori.copy()
            
        if yaw_only:
            yaw = self.np_random.uniform(-np.pi, np.pi)
            return np.array(p.getQuaternionFromEuler([0, 0, yaw]), dtype=np.float32)
        else:
            # Sample a uniform quaternion
            # Random uniform variables in [0, 1]
            u1, u2, u3 = self.np_random.uniform(0, 1, 3)
            # Shoemake method
            w = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
            x = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
            y = np.sqrt(u1) * np.sin(2 * np.pi * u3)
            z = np.sqrt(u1) * np.cos(2 * np.pi * u3)
            return np.array([x, y, z, w], dtype=np.float32)

    def sample_mass(self, default_mass: float) -> float:
        """Sample an object mass."""
        if not self.is_enabled("mass"):
            return default_mass
            
        cfg = self.config.get("mass", {})
        min_m = cfg.get("min", default_mass * 0.5)
        max_m = cfg.get("max", default_mass * 2.0)
        return float(self.np_random.uniform(min_m, max_m))

    def sample_friction(self, default_friction: float) -> float:
        """Sample lateral friction."""
        if not self.is_enabled("friction"):
            return default_friction
            
        cfg = self.config.get("friction", {})
        min_f = cfg.get("min", max(0.01, default_friction - 0.5))
        max_f = cfg.get("max", default_friction + 1.0)
        return float(self.np_random.uniform(min_f, max_f))

    def get_metadata(self) -> Dict[str, Any]:
        """Return the current randomization configuration metadata."""
        return {
            "randomization_enabled": self.enabled,
            "config": self.config
        }
