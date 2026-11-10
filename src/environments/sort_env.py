"""Gymnasium environment for the Goal-Conditioned Sorting task."""

from typing import Any, Dict, Tuple, Union
import numpy as np
from gymnasium import spaces

from src.environments.base_env import BasePandaEnv
from src.objects.cube import Cube
from src.objects.bin import SortingBin


class PandaSortEnv(BasePandaEnv):
    """
    Goal-Conditioned Sorting environment.
    
    The agent must pick a colored cube and place it into the matching colored bin.
    """

    COLORS = {
        "red": [0.8, 0.2, 0.2, 1.0],
        "green": [0.2, 0.8, 0.2, 1.0],
        "blue": [0.2, 0.2, 0.8, 1.0]
    }

    def __init__(
        self,
        render_mode: str | None = None,
        distance_threshold: float = 0.05,
        max_episode_steps: int = 150,
        control_steps: int = 15,
        action_scale: float = 0.05,
    ):
        super().__init__(render_mode=render_mode, control_steps=control_steps)
        self.distance_threshold = distance_threshold
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        
        # Object spawn limits
        self.workspace_min = np.array([0.35, -0.2, 0.0], dtype=np.float32)
        self.workspace_max = np.array([0.55, 0.2, 0.0], dtype=np.float32)
            
        # Target bins placement limits
        self.target_workspace_min = np.array([0.55, -0.35, 0.001], dtype=np.float32)
        self.target_workspace_max = np.array([0.75, 0.35, 0.001], dtype=np.float32)
            
        self.current_step = 0
        self.collision_count = 0
        
        self.active_color_idx = 0

        # Sub-objects
        # Initially one cube
        self.cube = Cube(client_id=self.sim.client_id, color=self.COLORS["red"])
        
        self.bins = {
            "red": SortingBin(client_id=self.sim.client_id, color=self.COLORS["red"]),
            "green": SortingBin(client_id=self.sim.client_id, color=self.COLORS["green"]),
            "blue": SortingBin(client_id=self.sim.client_id, color=self.COLORS["blue"])
        }
        
        self.bin_positions = np.zeros((3, 3), dtype=np.float32)

        # 4 (ee) + 3 (obj pos) + 3 (obj vel) + 3 (one hot color) + 9 (3 bins * 3 pos)
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32)
        goal_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            "observation": obs_space,
            "achieved_goal": goal_space,
            "desired_goal": goal_space,
        })
        self.goal = np.zeros(3, dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.current_step = 0
        self.collision_count = 0

        # Select random color for the object
        color_keys = list(self.COLORS.keys())
        self.active_color_idx = self.np_random.integers(0, 3)
        active_color_name = color_keys[self.active_color_idx]
        
        # Load objects if not loaded
        if self.cube.body_id < 0:
            self.cube.client_id = self.sim.client_id
            self.cube.load()
            
            for b in self.bins.values():
                b.client_id = self.sim.client_id
                b.load()
                
        # Update cube color visually
        import pybullet as p
        p.changeVisualShape(
            self.cube.body_id, 
            -1, 
            rgbaColor=self.COLORS[active_color_name],
            physicsClientId=self.sim.client_id
        )
                
        # Randomize cube position
        obj_pos = self.np_random.uniform(
            low=self.workspace_min,
            high=self.workspace_max,
            size=(3,)
        ).astype(np.float32)
        obj_pos[2] = 0.025
        self.cube.reset(position=obj_pos)
        
        # Randomize bin positions (ensure they don't overlap too much)
        # We'll just place them evenly spaced in Y for simplicity but randomized in X
        bin_spacing = 0.25
        base_y = self.np_random.uniform(-0.1, 0.1)
        
        y_offsets = [base_y - bin_spacing, base_y, base_y + bin_spacing]
        self.np_random.shuffle(y_offsets)
        
        for i, (name, b) in enumerate(self.bins.items()):
            x = self.np_random.uniform(self.target_workspace_min[0], self.target_workspace_max[0])
            pos = np.array([x, y_offsets[i], 0.001], dtype=np.float32)
            self.bin_positions[i] = pos
            b.reset(position=pos)
            
            if name == active_color_name:
                self.goal = pos.copy()
                self.goal[2] = 0.025 # Goal is to rest in the bin

        self.sim.step(10)

        obs = self._get_obs()
        info = self._get_info(obs["achieved_goal"], self.goal)
        return obs, info

    def _get_obs(self) -> Dict[str, np.ndarray]:
        ee_pos = self.controller.get_ee_position()
        gripper_state = self.controller.get_gripper_state()
        obj_pos = self.cube.get_position()
        obj_vel = self.cube.get_velocity()
        
        one_hot = np.zeros(3, dtype=np.float32)
        one_hot[self.active_color_idx] = 1.0
        
        obs = np.concatenate([
            ee_pos, 
            [gripper_state["width"]],
            obj_pos,
            obj_vel,
            one_hot,
            self.bin_positions.flatten()
        ]).astype(np.float32)
        
        achieved_goal = obj_pos.astype(np.float32)

        return {
            "observation": obs,
            "achieved_goal": achieved_goal,
            "desired_goal": self.goal.copy(),
        }

    def _get_info(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Dict[str, Any]:
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        is_success = distance <= self.distance_threshold
        
        # Grasp success
        grasp_success = achieved_goal[2] > 0.05 if achieved_goal.ndim == 1 else achieved_goal[..., 2] > 0.05
        
        # Placement success and wrong bin logic
        gripper_state = self.controller.get_gripper_state()
        is_released = gripper_state["width"] > 0.04
        
        placement_success = is_success and is_released and achieved_goal[2] < 0.05
        
        # Check wrong bin
        wrong_bin = False
        if is_released and achieved_goal[2] < 0.05 and not is_success:
            for i in range(3):
                if i != self.active_color_idx:
                    dist_to_wrong = np.linalg.norm(achieved_goal - (self.bin_positions[i] + [0,0,0.025]), axis=-1)
                    if dist_to_wrong <= self.distance_threshold:
                        wrong_bin = True
                        break
        
        return {
            "is_success": bool(is_success) if np.isscalar(is_success) or is_success.ndim == 0 else is_success,
            "distance": float(distance) if np.isscalar(distance) or distance.ndim == 0 else distance,
            "episode_length": self.current_step,
            "grasp_success": bool(grasp_success),
            "placement_success": bool(placement_success),
            "sorting_success": bool(placement_success), # identical for single object
            "wrong_bin_placement": bool(wrong_bin),
            "collisions": self.collision_count
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        
        action = np.clip(action, self.action_space.low, self.action_space.high)
        dx, dy, dz = action[:3] * self.action_scale
        gripper_action = action[3]

        current_ee_pos = self.controller.get_ee_position()
        target_ee_pos = current_ee_pos + np.array([dx, dy, dz])
        
        arm_workspace_min = np.array([0.2, -0.4, 0.0], dtype=np.float32)
        arm_workspace_max = np.array([0.8, 0.4, 0.5], dtype=np.float32)
        target_ee_pos = np.clip(target_ee_pos, arm_workspace_min, arm_workspace_max)

        # Simple collision heuristic (EE driving hard into table)
        if target_ee_pos[2] <= 0.01 and current_ee_pos[2] <= 0.02 and dz < -0.01:
            self.collision_count += 1

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
