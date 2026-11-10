"""Gymnasium environment for the Goal-Conditioned Pick and Place task."""

from typing import Any, Dict, Tuple, Union
import numpy as np
from gymnasium import spaces

from src.environments.base_env import BasePandaEnv
from src.objects.cube import Cube
from src.objects.targets import VisualTarget


class PandaPickPlaceEnv(BasePandaEnv):
    """
    Goal-Conditioned Pick and Place environment.
    
    The agent must grasp a randomly spawned cube and move it to a target 3D spatial position.
    """

    def __init__(
        self,
        render_mode: str | None = None,
        distance_threshold: float = 0.05,
        max_episode_steps: int = 100,
        control_steps: int = 15,
        workspace_min: np.ndarray | None = None,
        workspace_max: np.ndarray | None = None,
        target_workspace_min: np.ndarray | None = None,
        target_workspace_max: np.ndarray | None = None,
        action_scale: float = 0.05,
        randomizer_config: dict | None = None,
    ):
        super().__init__(
            render_mode=render_mode, 
            control_steps=control_steps,
            randomizer_config=randomizer_config
        )
        self.distance_threshold = distance_threshold
        self.max_episode_steps = max_episode_steps
        
        # Object spawn limits
        if workspace_min is None:
            self.workspace_min = np.array([0.4, -0.2, 0.0], dtype=np.float32)
        else:
            self.workspace_min = np.asarray(workspace_min, dtype=np.float32)
            
        if workspace_max is None:
            self.workspace_max = np.array([0.6, 0.2, 0.0], dtype=np.float32)
        else:
            self.workspace_max = np.asarray(workspace_max, dtype=np.float32)
            
        # Target placement limits (can be in air or on table)
        if target_workspace_min is None:
            self.target_workspace_min = np.array([0.3, -0.25, 0.025], dtype=np.float32)
        else:
            self.target_workspace_min = np.asarray(target_workspace_min, dtype=np.float32)
            
        if target_workspace_max is None:
            self.target_workspace_max = np.array([0.7, 0.25, 0.2], dtype=np.float32)
        else:
            self.target_workspace_max = np.asarray(target_workspace_max, dtype=np.float32)
            
        self.action_scale = action_scale
        self.current_step = 0
        self.collision_count = 0

        self.cube = Cube(client_id=self.sim.client_id)
        self.target = VisualTarget(client_id=self.sim.client_id)

        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
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

        if self.cube.body_id < 0:
            self.cube.client_id = self.sim.client_id
            self.cube.load()
            if self.render_mode == "human":
                self.target.client_id = self.sim.client_id
                self.target.load(position=(0,0,0))
                
        # Default position sampling if randomization is not used
        default_obj_pos = self.np_random.uniform(
            low=self.workspace_min,
            high=self.workspace_max,
            size=(3,)
        ).astype(np.float32)
        
        # Sample position (with fixed Z on table)
        obj_pos = self.randomizer.sample_position(
            feature="object_position",
            default_pos=default_obj_pos,
            bounds_min=self.workspace_min,
            bounds_max=self.workspace_max,
            z_fixed=0.025
        )
        
        # Sample pose
        default_ori = np.array([0, 0, 0, 1], dtype=np.float32)
        obj_ori = self.randomizer.sample_orientation(
            feature="object_orientation",
            default_ori=default_ori,
            yaw_only=True
        )
        
        # Physics randomization
        mass = self.randomizer.sample_mass(default_mass=0.1)
        friction = self.randomizer.sample_friction(default_friction=1.0)
        
        if self.randomizer.is_enabled("mass"):
            import pybullet as p
            p.changeDynamics(self.cube.body_id, -1, mass=mass, physicsClientId=self.sim.client_id)
        if self.randomizer.is_enabled("friction"):
            import pybullet as p
            p.changeDynamics(self.cube.body_id, -1, lateralFriction=friction, physicsClientId=self.sim.client_id)

        self.cube.reset(position=obj_pos, orientation=obj_ori)
        
        # Randomize target goal
        default_goal = self.np_random.uniform(
            low=self.target_workspace_min,
            high=self.target_workspace_max,
            size=(3,)
        ).astype(np.float32)
        
        self.goal = self.randomizer.sample_position(
            feature="target_position",
            default_pos=default_goal,
            bounds_min=self.target_workspace_min,
            bounds_max=self.target_workspace_max,
            z_fixed=None
        )
        
        if self.render_mode == "human" and self.target.body_id >= 0:
            self.target.reset(position=self.goal)

        self.sim.step(10)

        obs = self._get_obs()
        info = self._get_info(obs["achieved_goal"], self.goal)
        
        # Add metadata
        if self.randomizer.enabled:
            info["dr_metadata"] = {
                "object_position": obj_pos.tolist(),
                "object_orientation": obj_ori.tolist(),
                "target_position": self.goal.tolist(),
                "mass": mass,
                "friction": friction
            }
            
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
        
        achieved_goal = obj_pos.astype(np.float32)

        return {
            "observation": obs,
            "achieved_goal": achieved_goal,
            "desired_goal": self.goal.copy(),
        }

    def _get_info(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Dict[str, Any]:
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        is_success = distance <= self.distance_threshold
        
        # Grasp success: object z is > 0.05 (lifted off table)
        grasp_success = achieved_goal[2] > 0.05 if achieved_goal.ndim == 1 else achieved_goal[..., 2] > 0.05
        
        return {
            "is_success": bool(is_success) if np.isscalar(is_success) or is_success.ndim == 0 else is_success,
            "distance": float(distance) if np.isscalar(distance) or distance.ndim == 0 else distance,
            "episode_length": self.current_step,
            "grasp_success": bool(grasp_success) if np.isscalar(grasp_success) or grasp_success.ndim == 0 else grasp_success,
            "collisions": self.collision_count
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        
        action = np.clip(action, self.action_space.low, self.action_space.high)
        dx, dy, dz = action[:3] * self.action_scale
        gripper_action = action[3]

        current_ee_pos = self.controller.get_ee_position()
        target_ee_pos = current_ee_pos + np.array([dx, dy, dz])
        
        arm_workspace_min = np.array([0.2, -0.3, 0.0], dtype=np.float32)
        arm_workspace_max = np.array([0.7, 0.3, 0.5], dtype=np.float32)
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
