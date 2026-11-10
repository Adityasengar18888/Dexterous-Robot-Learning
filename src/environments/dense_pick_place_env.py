import numpy as np
from typing import Any, Dict, Tuple, Union
from src.environments.pick_place_env import PandaPickPlaceEnv

class PandaPickPlaceDenseRewardEnv(PandaPickPlaceEnv):
    """
    Diagnostic environment with a dense reward signal.
    Inherits all mechanics, success definitions, and constraints from the original.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_ee_pos = None

    def _get_info(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Dict[str, Any]:
        info = super()._get_info(achieved_goal, desired_goal)
        # Store ee_pos for reward computation
        info["ee_pos"] = self.controller.get_ee_position().copy()
        
        # We will also compute individual reward components for logging
        ee_pos = info["ee_pos"]
        ee_obj_dist = np.linalg.norm(ee_pos - achieved_goal)
        obj_target_dist = np.linalg.norm(achieved_goal - desired_goal)
        is_lifted = achieved_goal[2] > 0.05
        is_success = obj_target_dist <= self.distance_threshold
        
        info["reach_dist"] = float(ee_obj_dist)
        info["transport_dist"] = float(obj_target_dist)
        
        info["reach_reward"] = float(-ee_obj_dist)
        info["lift_reward"] = 1.0 if is_lifted else 0.0
        info["transport_reward"] = float(-obj_target_dist) if is_lifted else float(-obj_target_dist) # Penalize transport distance
        info["success_bonus"] = 5.0 if is_success else 0.0
        
        info["total_reward"] = info["reach_reward"] + info["lift_reward"] + info["transport_reward"] + info["success_bonus"]
        
        return info

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

        if target_ee_pos[2] <= 0.01 and current_ee_pos[2] <= 0.02 and dz < -0.01:
            self.collision_count += 1

        self.controller.move_ee(position=target_ee_pos, steps=self.control_steps, tolerance=0.005)

        if gripper_action > 0.0:
            self.controller.open_gripper(steps=10)
        else:
            self.controller.close_gripper(steps=10)

        obs = self._get_obs()
        info = self._get_info(obs["achieved_goal"], self.goal)
        
        # Correctly pass info to compute_reward
        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)

        terminated = False
        truncated = self.current_step >= self.max_episode_steps

        return obs, float(reward), terminated, truncated, info

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Union[dict, list, tuple]) -> Union[float, np.ndarray]:
        # Handle batched (from HER) or single (from step) calls
        
        if isinstance(info, list) or isinstance(info, tuple):
            ee_pos = np.array([i.get("ee_pos", achieved_goal[idx]) for idx, i in enumerate(info)])
        elif isinstance(info, dict):
            ee_pos = info.get("ee_pos", achieved_goal)
            
            # SB3 HER sometimes passes a dict of arrays instead of array of dicts in newer versions
            # Let's handle the dict of arrays case just in case
            if isinstance(ee_pos, np.ndarray) and ee_pos.ndim > 1 and len(ee_pos) == len(achieved_goal):
                pass # Already correct shape
            elif isinstance(ee_pos, list):
                ee_pos = np.array(ee_pos)
        else:
            ee_pos = achieved_goal # Fallback to object pos if missing, effectively nullifying reach reward
            
        ee_pos = np.atleast_2d(ee_pos) if achieved_goal.ndim > 1 else np.array(ee_pos)
            
        ee_obj_dist = np.linalg.norm(ee_pos - achieved_goal, axis=-1)
        obj_target_dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        
        is_lifted = achieved_goal[..., 2] > 0.05
        is_success = obj_target_dist <= self.distance_threshold
        
        reach_reward = -ee_obj_dist
        transport_reward = -obj_target_dist
        lift_reward = np.where(is_lifted, 1.0, 0.0)
        success_bonus = np.where(is_success, 5.0, 0.0)
        
        reward = reach_reward + transport_reward + lift_reward + success_bonus
        
        if reward.ndim == 0:
            return float(reward)
        return reward.astype(np.float32)

