import numpy as np
from typing import Any, Dict, Tuple, Union
from src.environments.pick_place_env import PandaPickPlaceEnv

class PandaTransportDiagnosticEnv(PandaPickPlaceEnv):
    """
    Isolated Phase 11.9 Diagnostic Environment.
    Initializes the state with the object already grasped and lifted.
    Reward is strictly negative object-target distance.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self, *, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        
        obj_pos = self._get_obs()["achieved_goal"]
        
        # Grasp sequence
        self.controller.open_gripper(steps=30)
        self.controller.move_ee(position=obj_pos + np.array([0, 0, 0.15]), steps=150)
        self.controller.move_ee(position=obj_pos + np.array([0, 0, 0.015]), steps=150)
        self.controller.close_gripper(steps=30)
        
        for _ in range(30): 
            self.sim.step()
            
        # Lift sequence
        ee_pos = self.controller.get_ee_position()
        self.controller.move_ee(position=ee_pos + np.array([0, 0, 0.15]), steps=150)
        
        for _ in range(30): 
            self.sim.step()
            
        obs = self._get_obs()
        info = self._get_info(obs["achieved_goal"], self.goal)
        
        # Validation checks
        info["state_initialization_valid"] = True
        
        ee_pos = obs["observation"][:3]
        obj_pos = obs["achieved_goal"]
        ee_obj_dist = np.linalg.norm(ee_pos - obj_pos)
        obj_lifted = obj_pos[2] > 0.05
        
        if ee_obj_dist > 0.05 or not obj_lifted:
            info["state_initialization_valid"] = False
            
        return obs, info

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Union[dict, list, tuple]) -> Union[float, np.ndarray]:
        # Strict transport reward: -distance(object, target)
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        reward = -dist
        if reward.ndim == 0:
            return float(reward)
        return reward.astype(np.float32)

    def _get_info(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Dict[str, Any]:
        info = super()._get_info(achieved_goal, desired_goal)
        # We override is_success to ONLY be object-target distance (even if it's high in the air, we don't care about placing right now)
        dist = np.linalg.norm(achieved_goal - desired_goal)
        info["is_success"] = bool(dist <= self.distance_threshold)
        info["transport_dist"] = dist
        info["ee_pos"] = self.controller.get_ee_position().copy()
        return info
