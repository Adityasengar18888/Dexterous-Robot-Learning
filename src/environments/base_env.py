"""Base Gymnasium environment for PyBullet manipulation tasks."""

from typing import Any, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.utils.simulation import PyBulletSimulation
from src.robots.controller import PandaController
from src.utils.randomization import DomainRandomizer


class BasePandaEnv(gym.Env):
    """
    Base Gymnasium environment for Franka Panda manipulation tasks.
    Manages the simulation and controller instances.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        control_steps: int = 15,
        randomizer_config: dict | None = None
    ):
        """
        Initialize the base environment.

        Args:
            render_mode: Render mode (None, "human", "rgb_array").
            control_steps: Number of simulation steps per environment step.
            randomizer_config: Dict for Domain Randomization parameters.
        """
        super().__init__()
        self.render_mode = render_mode
        self.control_steps = control_steps
        self.randomizer_config = randomizer_config or {}
        self.randomizer = None

        # Initialize simulation and robot controller
        gui = self.render_mode == "human"
        self.sim = PyBulletSimulation(gui=gui)
        self.controller = PandaController(simulation=self.sim)

        # Standard action space for all Panda tasks [dx, dy, dz, gripper]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # Subclasses should define self.observation_space

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[Any, dict]:
        """
        Reset the environment, simulation, and arm joints.
        
        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.
            
        Returns:
            Subclasses should return (observation, info)
        """
        super().reset(seed=seed)
        
        # Initialize DomainRandomizer now that self.np_random is available
        if self.randomizer is None:
            self.randomizer = DomainRandomizer(self.randomizer_config, self.np_random)
        else:
            self.randomizer.np_random = self.np_random
            
        # Ensure we only load the simulation and robot once.
        # controller.reset() will initialize the sim and load the robot on the first call.
        self.controller.reset()
        return None, {}

    def render(self):
        """
        Render the environment.
        PyBullet GUI naturally renders if render_mode="human".
        """
        if self.render_mode == "human":
            pass
        elif self.render_mode == "rgb_array":
            raise NotImplementedError("RGB array rendering not implemented yet.")

    def close(self):
        """Clean up the environment by disconnecting the simulation."""
        self.sim.disconnect()
