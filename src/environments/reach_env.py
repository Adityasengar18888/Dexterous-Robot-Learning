"""Gymnasium environment for the Goal-Conditioned Reach task."""

from typing import Any, Dict, Tuple, Union

import numpy as np
from gymnasium import spaces

from src.environments.base_env import BasePandaEnv


class PandaReachEnv(BasePandaEnv):
    """
    Goal-Conditioned Reach environment.
    
    The agent receives a sparse reward of 0 when the end-effector
    reaches within a distance_threshold of the desired_goal position,
    and -1 otherwise.
    """

    def __init__(
        self,
        render_mode: str | None = None,
        distance_threshold: float = 0.05,
        max_episode_steps: int = 50,
        control_steps: int = 15,
        workspace_min: np.ndarray | None = None,
        workspace_max: np.ndarray | None = None,
        action_scale: float = 0.05,
    ):
        """
        Initialize the Reach environment.
        
        Args:
            render_mode: Render mode (None, "human", "rgb_array").
            distance_threshold: Threshold for success.
            max_episode_steps: Max steps before truncation.
            control_steps: Sim steps per env step.
            workspace_min: Minimum bound for target goals and arm position.
            workspace_max: Maximum bound for target goals and arm position.
            action_scale: Scaling factor for relative translations.
        """
        super().__init__(render_mode=render_mode, control_steps=control_steps)
        
        self.distance_threshold = distance_threshold
        self.max_episode_steps = max_episode_steps
        
        if workspace_min is None:
            self.workspace_min = np.array([0.2, -0.3, 0.05], dtype=np.float32)
        else:
            self.workspace_min = np.asarray(workspace_min, dtype=np.float32)
            
        if workspace_max is None:
            self.workspace_max = np.array([0.7, 0.3, 0.4], dtype=np.float32)
        else:
            self.workspace_max = np.asarray(workspace_max, dtype=np.float32)
            
        self.action_scale = action_scale
        self.current_step = 0

        # Define Observation Space (goal-conditioned)
        # observation: [ee_x, ee_y, ee_z, gripper_width]
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        goal_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        
        self.observation_space = spaces.Dict(
            {
                "observation": obs_space,
                "achieved_goal": goal_space,
                "desired_goal": goal_space,
            }
        )

        self.goal = np.zeros(3, dtype=np.float32)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Construct the goal-conditioned observation dictionary."""
        ee_pos = self.controller.get_ee_position()
        gripper_state = self.controller.get_gripper_state()
        
        obs = np.concatenate([ee_pos, [gripper_state["width"]]]).astype(np.float32)
        achieved_goal = ee_pos.astype(np.float32)

        return {
            "observation": obs,
            "achieved_goal": achieved_goal,
            "desired_goal": self.goal.copy(),
        }

    def _get_info(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Dict[str, Any]:
        """Compute the info dictionary, specifically 'is_success'."""
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        is_success = distance <= self.distance_threshold
        
        # Ensure it returns scalar types if scalar was provided
        return {
            "is_success": bool(is_success) if np.isscalar(is_success) or is_success.ndim == 0 else is_success,
            "distance": float(distance) if np.isscalar(distance) or distance.ndim == 0 else distance,
            "episode_length": self.current_step
        }

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment and generate a new random goal."""
        super().reset(seed=seed, options=options)
        self.current_step = 0

        # Generate a random goal within the workspace boundaries
        self.goal = self.np_random.uniform(
            low=self.workspace_min,
            high=self.workspace_max,
            size=(3,)
        ).astype(np.float32)

        obs = self._get_obs()
        info = self._get_info(obs["achieved_goal"], self.goal)
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Action is [dx, dy, dz, gripper_action]
        """
        self.current_step += 1
        
        # Parse and scale action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        dx, dy, dz = action[:3] * self.action_scale
        gripper_action = action[3]

        # Calculate target Cartesian position and clip to workspace boundaries
        current_ee_pos = self.controller.get_ee_position()
        target_ee_pos = current_ee_pos + np.array([dx, dy, dz])
        target_ee_pos = np.clip(target_ee_pos, self.workspace_min, self.workspace_max)

        # Move the arm to the new target
        self.controller.move_ee(position=target_ee_pos, steps=self.control_steps, tolerance=0.005)

        # Actuate gripper based on continuous action (e.g. > 0 open, <= 0 close)
        if gripper_action > 0.0:
            self.controller.open_gripper(steps=10)
        else:
            self.controller.close_gripper(steps=10)

        # Retrieve new state
        obs = self._get_obs()
        
        # Reward & Info
        reward = self.compute_reward(obs["achieved_goal"], self.goal, {})
        info = self._get_info(obs["achieved_goal"], self.goal)

        # Sparse reach task does not terminate early upon success in standard formulations,
        # but truncates at max steps.
        terminated = False
        truncated = self.current_step >= self.max_episode_steps

        # gym step returns: obs, reward, terminated, truncated, info
        return obs, float(reward), terminated, truncated, info

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> Union[float, np.ndarray]:
        """
        Compute the sparse reward.
        
        Args:
            achieved_goal: The achieved goal (can be batched).
            desired_goal: The target goal (can be batched).
            info: Optional info dict.
            
        Returns:
            0.0 if success (distance <= threshold), -1.0 otherwise.
        """
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        reward = np.where(distance > self.distance_threshold, -1.0, 0.0).astype(np.float32)
        
        # Return scalar if input was unbatched, otherwise return array
        if reward.ndim == 0:
            return float(reward)
        return reward
