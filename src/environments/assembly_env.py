"""Gymnasium environment for Multi-Part Assembly (Peg Insertion)."""

from typing import Any, Dict, Tuple, Union
import numpy as np
from gymnasium import spaces
import pybullet as p

from src.environments.base_env import BasePandaEnv
from src.objects.parts import Peg, Receptacle

class PandaAssemblyEnv(BasePandaEnv):
    """
    Goal-Conditioned Assembly environment.
    
    The agent must pick a peg, orient it correctly, and insert it into a receptacle.
    """

    def __init__(
        self,
        render_mode: str | None = None,
        distance_threshold: float = 0.02,
        orientation_threshold: float = 0.1, # 1 - cos(theta)^2 tolerance
        max_episode_steps: int = 200,
        control_steps: int = 15,
        action_scale: float = 0.05,
    ):
        super().__init__(render_mode=render_mode, control_steps=control_steps)
        self.distance_threshold = distance_threshold
        self.orientation_threshold = orientation_threshold
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        
        # Action space: dx, dy, dz, d_yaw, gripper
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        
        self.workspace_min = np.array([0.3, -0.2, 0.0], dtype=np.float32)
        self.workspace_max = np.array([0.5, 0.2, 0.0], dtype=np.float32)
        
        self.target_workspace_min = np.array([0.55, -0.15, 0.0], dtype=np.float32)
        self.target_workspace_max = np.array([0.7, 0.15, 0.0], dtype=np.float32)

        self.peg = Peg(client_id=self.sim.client_id)
        self.base = Receptacle(client_id=self.sim.client_id)
        
        self.current_step = 0
        self.collision_count = 0
        
        # State: ee_pos(3), ee_ori(4), gripper(1), peg_pos(3), peg_ori(4), base_pos(3), base_ori(4)
        obs_dim = 3 + 4 + 1 + 3 + 4 + 3 + 4
        # Goal space: peg_pos(3) + peg_ori(4)
        goal_dim = 7
        
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        goal_space = spaces.Box(low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            "observation": obs_space,
            "achieved_goal": goal_space,
            "desired_goal": goal_space,
        })
        self.goal = np.zeros(goal_dim, dtype=np.float32)
        
        # Orientation state tracking for action
        self.current_yaw = 0.0

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.current_step = 0
        self.collision_count = 0
        
        # Downward pose
        self.current_yaw = 0.0
        down_quat = p.getQuaternionFromEuler([np.pi, 0, self.current_yaw])
        self.controller.move_ee(
            position=[0.4, 0.0, 0.4],
            orientation=down_quat,
            steps=30
        )

        if self.peg.body_id < 0:
            self.peg.client_id = self.sim.client_id
            self.base.client_id = self.sim.client_id
            self.peg.load()
            self.base.load()
            
        # Randomize base
        base_pos = self.np_random.uniform(low=self.target_workspace_min, high=self.target_workspace_max, size=(3,)).astype(np.float32)
        base_pos[2] = 0.005 # sit on table
        base_yaw = self.np_random.uniform(-np.pi/4, np.pi/4)
        base_ori = p.getQuaternionFromEuler([0, 0, base_yaw])
        self.base.reset(position=base_pos, orientation=base_ori)
        
        # Randomize peg
        peg_pos = self.np_random.uniform(low=self.workspace_min, high=self.workspace_max, size=(3,)).astype(np.float32)
        peg_pos[2] = 0.03
        peg_yaw = self.np_random.uniform(-np.pi, np.pi)
        peg_ori = p.getQuaternionFromEuler([0, 0, peg_yaw])
        self.peg.reset(position=peg_pos, orientation=peg_ori)
        
        # Goal is base_pos but offset in Z for the hole, and matching base_ori
        goal_pos = base_pos.copy()
        goal_pos[2] = 0.03 # inside the receptacle
        self.goal = np.concatenate([goal_pos, base_ori]).astype(np.float32)
        
        self.sim.step(20) # settle

        obs = self._get_obs()
        info = self._get_info(obs["achieved_goal"], self.goal)
        return obs, info

    def _get_obs(self) -> Dict[str, np.ndarray]:
        ee_pos = self.controller.get_ee_position()
        ee_ori = self.controller.get_ee_orientation()
        gripper_state = self.controller.get_gripper_state()
        
        peg_pos = self.peg.get_position()
        peg_ori = self.peg.get_orientation()
        
        base_pos = self.base.get_position()
        base_ori = self.base.get_orientation()
        
        obs = np.concatenate([
            ee_pos,
            ee_ori,
            [gripper_state["width"]],
            peg_pos,
            peg_ori,
            base_pos,
            base_ori
        ]).astype(np.float32)
        
        achieved_goal = np.concatenate([peg_pos, peg_ori]).astype(np.float32)

        return {
            "observation": obs,
            "achieved_goal": achieved_goal,
            "desired_goal": self.goal.copy(),
        }

    def _get_info(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Dict[str, Any]:
        is_batched = achieved_goal.ndim > 1
        
        ag = achieved_goal.reshape(-1, 7) if is_batched else achieved_goal.reshape(1, 7)
        dg = desired_goal.reshape(-1, 7) if is_batched else desired_goal.reshape(1, 7)
        
        pos_dist = np.linalg.norm(ag[:, :3] - dg[:, :3], axis=-1)
        
        # Orientation distance: 1 - <q1, q2>^2
        dot_product = np.sum(ag[:, 3:] * dg[:, 3:], axis=-1)
        ori_dist = 1.0 - (dot_product ** 2)
        
        is_success = (pos_dist <= self.distance_threshold) & (ori_dist <= self.orientation_threshold)
        
        if is_batched:
            return {
                "is_success": is_success,
                "distance": pos_dist,
                "ori_distance": ori_dist
            }
            
        pos_dist_val = float(pos_dist[0])
        ori_dist_val = float(ori_dist[0])
        success = bool(is_success[0])
        
        # Advanced assembly state logic for telemetry
        assembly_state = "NOT_ASSEMBLED"
        base_pos = self.base.get_position()
        peg_pos = self.peg.get_position()
        
        # Check alignment (above the base, within xy tolerance, orientation matching)
        xy_dist = np.linalg.norm(peg_pos[:2] - base_pos[:2])
        is_aligned_xy = xy_dist < 0.03
        is_aligned_ori = ori_dist_val <= self.orientation_threshold
        is_above = 0.05 < peg_pos[2] < 0.15
        is_inserted = pos_dist_val <= self.distance_threshold
        
        if is_inserted and is_aligned_ori:
            assembly_state = "ASSEMBLED"
        elif is_aligned_xy and is_aligned_ori and is_above:
            assembly_state = "PART_B_ALIGNED"
        else:
            assembly_state = "NOT_ASSEMBLED"

        return {
            "is_success": success,
            "distance": pos_dist_val,
            "orientation_error": ori_dist_val,
            "episode_length": self.current_step,
            "assembly_state": assembly_state,
            "collisions": self.collision_count
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        
        action = np.clip(action, self.action_space.low, self.action_space.high)
        dx, dy, dz, dyaw = action[:4] * self.action_scale
        gripper_action = action[4]

        current_ee_pos = self.controller.get_ee_position()
        target_ee_pos = current_ee_pos + np.array([dx, dy, dz])
        
        arm_workspace_min = np.array([0.2, -0.4, 0.0], dtype=np.float32)
        arm_workspace_max = np.array([0.8, 0.4, 0.5], dtype=np.float32)
        target_ee_pos = np.clip(target_ee_pos, arm_workspace_min, arm_workspace_max)
        
        self.current_yaw += dyaw
        # keep yaw in [-pi, pi]
        self.current_yaw = (self.current_yaw + np.pi) % (2 * np.pi) - np.pi
        
        target_ori = p.getQuaternionFromEuler([np.pi, 0, self.current_yaw])

        self.controller.move_ee(position=target_ee_pos, orientation=target_ori, steps=self.control_steps, tolerance=0.005)

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
        ag = achieved_goal.reshape(-1, 7) if achieved_goal.ndim > 1 else achieved_goal.reshape(1, 7)
        dg = desired_goal.reshape(-1, 7) if desired_goal.ndim > 1 else desired_goal.reshape(1, 7)
        
        pos_dist = np.linalg.norm(ag[:, :3] - dg[:, :3], axis=-1)
        
        dot_product = np.sum(ag[:, 3:] * dg[:, 3:], axis=-1)
        ori_dist = 1.0 - (dot_product ** 2)
        
        success = (pos_dist <= self.distance_threshold) & (ori_dist <= self.orientation_threshold)
        rewards = np.where(success, 0.0, -1.0).astype(np.float32)
        
        if achieved_goal.ndim == 1:
            return float(rewards[0])
        return rewards
