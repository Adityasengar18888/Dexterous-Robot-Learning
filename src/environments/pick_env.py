"""Gymnasium environment for the Goal-Conditioned Pick task."""

from typing import Any, Dict, Tuple, Union
import numpy as np
from gymnasium import spaces

from src.environments.base_env import BasePandaEnv
from src.objects.cube import Cube
from src.objects.targets import VisualTarget


class PandaPickEnv(BasePandaEnv):
    """
    Goal-Conditioned Pick environment.
    
    The agent must grasp a randomly spawned cube and lift it to a specific altitude.
    """

    def __init__(
        self,
        render_mode: str | None = None,
        distance_threshold: float = 0.05,
        max_episode_steps: int = 100,
        control_steps: int = 15,
        workspace_min: np.ndarray | None = None,
        workspace_max: np.ndarray | None = None,
        target_altitude: float = 0.2,
        action_scale: float = 0.05,
    ):
        super().__init__(render_mode=render_mode, control_steps=control_steps)
        
        self.distance_threshold = distance_threshold
        self.max_episode_steps = max_episode_steps
        self.target_altitude = target_altitude
        
        # Workspace limits for object spawning on the table
        if workspace_min is None:
            self.workspace_min = np.array([0.4, -0.2, 0.0], dtype=np.float32)
        else:
            self.workspace_min = np.asarray(workspace_min, dtype=np.float32)
            
        if workspace_max is None:
            self.workspace_max = np.array([0.6, 0.2, 0.0], dtype=np.float32)
        else:
            self.workspace_max = np.asarray(workspace_max, dtype=np.float32)
            
        self.action_scale = action_scale
        self.current_step = 0

        # Sub-objects
        self.cube = Cube(client_id=self.sim.client_id)
        self.target = VisualTarget(client_id=self.sim.client_id)

        # 4 (ee) + 3 (obj pos) + 3 (obj vel)
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        goal_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        
        self.observation_space = spaces.Dict(
            {
                "observation": obs_space,
                "achieved_goal": goal_space,
                "desired_goal": goal_space,
            }
        )

        self.goal = np.zeros(3, dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.current_step = 0

        # Check if objects are loaded in pybullet
        if self.cube.body_id < 0:
            self.cube.client_id = self.sim.client_id
            self.cube.load()
            if self.render_mode == "human":
                self.target.client_id = self.sim.client_id
                self.target.load(position=(0,0,0))
                
        # Randomize cube position on the table
        obj_pos = self.np_random.uniform(
            low=self.workspace_min,
            high=self.workspace_max,
            size=(3,)
        ).astype(np.float32)
        obj_pos[2] = 0.025 # Half size of cube (0.05/2) to rest properly on table
        
        self.cube.reset(position=obj_pos)
        
        # Target goal: Lift straight up from initial position
        self.goal = obj_pos.copy()
        self.goal[2] += self.target_altitude
        
        if self.render_mode == "human" and self.target.body_id >= 0:
            self.target.reset(position=self.goal)

        # Settle simulation to allow cube to drop if needed
        self.sim.step(10)

        obs = self._get_obs()
        info = self._get_info(obs["achieved_goal"], self.goal)
        return obs, info

    def _get_obs(self) -> Dict[str, np.ndarray]:
        ee_pos = self.controller.get_ee_position()
        gripper_state = self.controller.get_gripper_state()
        obj_pos = self.cube.get_position()
        obj_vel = self.cube.get_velocity()
        
        obs = np.concatenate([
            ee_pos, 
            [gripper_state["width"]],
            obj_pos,
            obj_vel
        ]).astype(np.float32)
        
        # The achieved goal is the object's current position (we want to manipulate the object)
        achieved_goal = obj_pos.astype(np.float32)

        return {
            "observation": obs,
            "achieved_goal": achieved_goal,
            "desired_goal": self.goal.copy(),
        }

    def _get_info(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Dict[str, Any]:
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        is_success = distance <= self.distance_threshold
        
        return {
            "is_success": bool(is_success) if np.isscalar(is_success) or is_success.ndim == 0 else is_success,
            "distance": float(distance) if np.isscalar(distance) or distance.ndim == 0 else distance,
            "episode_length": self.current_step
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        
        action = np.clip(action, self.action_space.low, self.action_space.high)
        dx, dy, dz = action[:3] * self.action_scale
        gripper_action = action[3]

        current_ee_pos = self.controller.get_ee_position()
        target_ee_pos = current_ee_pos + np.array([dx, dy, dz])
        
        # Arm workspace limits
        arm_workspace_min = np.array([0.2, -0.3, 0.0], dtype=np.float32)
        arm_workspace_max = np.array([0.7, 0.3, 0.5], dtype=np.float32)
        
        target_ee_pos = np.clip(target_ee_pos, arm_workspace_min, arm_workspace_max)

        self.controller.move_ee(position=target_ee_pos, steps=self.control_steps, tolerance=0.005)

        if gripper_action > 0.0:
            self.controller.open_gripper(steps=10)
        else:
            self.controller.close_gripper(steps=10)

        obs = self._get_obs()
        reward = self.compute_reward(obs["achieved_goal"], self.goal, {})
        info = self._get_info(obs["achieved_goal"], self.goal)

        terminated = False
        truncated = self.current_step >= self.max_episode_steps

        return obs, float(reward), terminated, truncated, info

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> Union[float, np.ndarray]:
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        reward = np.where(distance > self.distance_threshold, -1.0, 0.0).astype(np.float32)
        
        if reward.ndim == 0:
            return float(reward)
        return reward
