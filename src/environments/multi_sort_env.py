"""Gymnasium environment for Multi-Object Sorting."""

from typing import Any, Dict, Tuple, Union
import numpy as np
from gymnasium import spaces
import pybullet as p

from src.environments.base_env import BasePandaEnv
from src.objects.cube import Cube
from src.objects.bin import SortingBin


class PandaMultiSortEnv(BasePandaEnv):
    """
    Goal-Conditioned Multi-Object Sorting environment.
    
    The agent must pick multiple colored cubes and place them into their matching bins.
    """

    COLORS = {
        "red": [0.8, 0.2, 0.2, 1.0],
        "green": [0.2, 0.8, 0.2, 1.0],
        "blue": [0.2, 0.2, 0.8, 1.0]
    }

    def __init__(
        self,
        render_mode: str | None = None,
        num_objects: int = 2,
        distance_threshold: float = 0.05,
        max_episode_steps: int = 300,
        control_steps: int = 15,
        action_scale: float = 0.05,
    ):
        super().__init__(render_mode=render_mode, control_steps=control_steps)
        self.num_objects = min(max(1, num_objects), 3) # Bound between 1 and 3
        self.distance_threshold = distance_threshold
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        
        # Object spawn limits
        self.workspace_min = np.array([0.3, -0.25, 0.0], dtype=np.float32)
        self.workspace_max = np.array([0.6, 0.25, 0.0], dtype=np.float32)
            
        # Target bins placement limits
        self.target_workspace_min = np.array([0.65, -0.35, 0.001], dtype=np.float32)
        self.target_workspace_max = np.array([0.75, 0.35, 0.001], dtype=np.float32)
            
        self.current_step = 0
        self.collision_count = 0
        self.dropped_objects = 0
        
        self.active_color_indices = []

        # Sub-objects
        self.cubes = [Cube(client_id=self.sim.client_id) for _ in range(self.num_objects)]
        
        self.bins = {
            "red": SortingBin(client_id=self.sim.client_id, color=self.COLORS["red"]),
            "green": SortingBin(client_id=self.sim.client_id, color=self.COLORS["green"]),
            "blue": SortingBin(client_id=self.sim.client_id, color=self.COLORS["blue"])
        }
        
        self.bin_positions = np.zeros((3, 3), dtype=np.float32)
        self.object_last_z = np.zeros(self.num_objects, dtype=np.float32)

        # Observation shape
        obs_dim = 4 + (self.num_objects * 9) + 9
        goal_dim = self.num_objects * 3
        
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        goal_space = spaces.Box(low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            "observation": obs_space,
            "achieved_goal": goal_space,
            "desired_goal": goal_space,
        })
        self.goal = np.zeros(goal_dim, dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.current_step = 0
        self.collision_count = 0
        self.dropped_objects = 0

        # Select unique colors for the objects
        color_keys = list(self.COLORS.keys())
        self.active_color_indices = self.np_random.choice(3, size=self.num_objects, replace=False)
        active_color_names = [color_keys[i] for i in self.active_color_indices]
        
        # Load objects if not loaded
        for idx, cube in enumerate(self.cubes):
            if cube.body_id < 0:
                cube.client_id = self.sim.client_id
                cube.load()
            
            # Update cube color visually
            p.changeVisualShape(
                cube.body_id, 
                -1, 
                rgbaColor=self.COLORS[active_color_names[idx]],
                physicsClientId=self.sim.client_id
            )
            
        for b in self.bins.values():
            if b.body_id < 0:
                b.client_id = self.sim.client_id
                b.load()
                
        # Randomize cube positions without overlap
        placed_positions = []
        for cube in self.cubes:
            for _ in range(100):
                pos = self.np_random.uniform(low=self.workspace_min, high=self.workspace_max, size=(3,)).astype(np.float32)
                pos[2] = 0.025
                if all(np.linalg.norm(pos - p_pos) > 0.06 for p_pos in placed_positions):
                    placed_positions.append(pos)
                    cube.reset(position=pos)
                    break
        
        self.object_last_z = np.array([p[2] for p in placed_positions], dtype=np.float32)
        
        # Randomize bin positions
        bin_spacing = 0.25
        base_y = self.np_random.uniform(-0.1, 0.1)
        y_offsets = [base_y - bin_spacing, base_y, base_y + bin_spacing]
        self.np_random.shuffle(y_offsets)
        
        bin_map = {}
        for i, (name, b) in enumerate(self.bins.items()):
            x = self.np_random.uniform(self.target_workspace_min[0], self.target_workspace_max[0])
            pos = np.array([x, y_offsets[i], 0.001], dtype=np.float32)
            self.bin_positions[i] = pos
            b.reset(position=pos)
            bin_map[name] = pos
            
        # Assign goals
        for idx, name in enumerate(active_color_names):
            goal_pos = bin_map[name].copy()
            goal_pos[2] = 0.025
            self.goal[idx*3:(idx+1)*3] = goal_pos

        self.sim.step(10)

        obs = self._get_obs()
        info = self._get_info(obs["achieved_goal"], self.goal)
        return obs, info

    def _get_obs(self) -> Dict[str, np.ndarray]:
        ee_pos = self.controller.get_ee_position()
        gripper_state = self.controller.get_gripper_state()
        
        obj_features = []
        achieved_goal = []
        
        for idx, cube in enumerate(self.cubes):
            pos = cube.get_position()
            vel = cube.get_velocity()
            
            one_hot = np.zeros(3, dtype=np.float32)
            one_hot[self.active_color_indices[idx]] = 1.0
            
            obj_features.extend([pos, vel, one_hot])
            achieved_goal.extend(pos)
            
        obs = np.concatenate([
            ee_pos, 
            [gripper_state["width"]],
        ] + obj_features + [
            self.bin_positions.flatten()
        ]).astype(np.float32)
        
        achieved_goal = np.array(achieved_goal, dtype=np.float32)

        return {
            "observation": obs,
            "achieved_goal": achieved_goal,
            "desired_goal": self.goal.copy(),
        }

    def _get_info(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Dict[str, Any]:
        # Handle batched (from SB3/HER) vs single env calls
        is_batched = achieved_goal.ndim > 1
        
        if is_batched:
            distances = np.linalg.norm(
                achieved_goal.reshape(-1, self.num_objects, 3) - desired_goal.reshape(-1, self.num_objects, 3), 
                axis=-1
            )
            is_success_per_obj = distances <= self.distance_threshold
            is_success = np.all(is_success_per_obj, axis=-1)
            mean_dist = np.mean(distances, axis=-1)
            
            return {
                "is_success": is_success,
                "distance": mean_dist
            }
        
        # Single environment step telemetry
        distances = np.linalg.norm(achieved_goal.reshape(self.num_objects, 3) - desired_goal.reshape(self.num_objects, 3), axis=-1)
        is_success_per_obj = distances <= self.distance_threshold
        
        gripper_state = self.controller.get_gripper_state()
        is_released = gripper_state["width"] > 0.04
        
        objects_placed_correctly = 0
        objects_in_wrong_bin = 0
        
        ag_reshaped = achieved_goal.reshape(self.num_objects, 3)
        for idx in range(self.num_objects):
            pos = ag_reshaped[idx]
            is_succ = is_success_per_obj[idx]
            
            if is_succ and is_released and pos[2] < 0.05:
                objects_placed_correctly += 1
                
            # Dropped check (went high then low, but not in correct bin)
            if self.object_last_z[idx] > 0.05 and pos[2] < 0.04 and not is_succ:
                # Was it placed in a wrong bin?
                in_wrong = False
                for b_idx in range(3):
                    if b_idx != self.active_color_indices[idx]:
                        b_pos = self.bin_positions[b_idx]
                        if np.linalg.norm(pos[:2] - b_pos[:2]) < self.distance_threshold:
                            in_wrong = True
                            break
                if in_wrong:
                    objects_in_wrong_bin += 1
                else:
                    self.dropped_objects += 1
            
            self.object_last_z[idx] = pos[2]
            
        episode_success = (objects_placed_correctly == self.num_objects)
        
        return {
            "is_success": bool(episode_success),
            "distance": float(np.mean(distances)),
            "episode_length": self.current_step,
            "objects_placed": objects_placed_correctly,
            "wrong_bin_placements": objects_in_wrong_bin,
            "dropped_objects": self.dropped_objects,
            "collisions": self.collision_count,
            "is_success_per_obj": is_success_per_obj.tolist()
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
        # Handle batched computation for HER
        ag = achieved_goal.reshape(-1, self.num_objects, 3) if achieved_goal.ndim > 1 else achieved_goal.reshape(1, self.num_objects, 3)
        dg = desired_goal.reshape(-1, self.num_objects, 3) if desired_goal.ndim > 1 else desired_goal.reshape(1, self.num_objects, 3)
        
        distances = np.linalg.norm(ag - dg, axis=-1)
        # Sparse reward: -1 for every object not in its goal
        rewards = np.sum(np.where(distances > self.distance_threshold, -1.0, 0.0), axis=-1).astype(np.float32)
        
        if achieved_goal.ndim == 1:
            return float(rewards[0])
        return rewards
