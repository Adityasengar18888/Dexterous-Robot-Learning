import numpy as np
from typing import Any, Dict, Tuple
from src.environments.dense_pick_place_env import PandaPickPlaceDenseRewardEnv

class PandaCurriculumEnv(PandaPickPlaceDenseRewardEnv):
    """
    Curriculum environment that samples initial states from a specified distribution.
    States:
    - NORMAL_START
    - LIFTED_AND_HELD
    - NEAR_TARGET_AND_HELD
    """
    def __init__(self, state_distribution: Dict[str, float] = None, **kwargs):
        super().__init__(**kwargs)
        if state_distribution is None:
            state_distribution = {"NORMAL_START": 1.0}
            
        # Normalize just in case
        total = sum(state_distribution.values())
        self.state_distribution = {k: v/total for k, v in state_distribution.items()}
        self.states = list(self.state_distribution.keys())
        self.probs = list(self.state_distribution.values())
        
        self.last_sampled_state = "NORMAL_START"

    def set_distribution(self, distribution: Dict[str, float]):
        total = sum(distribution.values())
        self.state_distribution = {k: v/total for k, v in distribution.items()}
        self.states = list(self.state_distribution.keys())
        self.probs = list(self.state_distribution.values())

    def reset(self, *, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        
        # Sample state based on np_random which is properly seeded
        if self.np_random is None:
            state_type = np.random.choice(self.states, p=self.probs)
        else:
            state_type = self.np_random.choice(self.states, p=self.probs)
            
        self.last_sampled_state = state_type
        
        if state_type == "NORMAL_START":
            info["state_initialization_valid"] = True
            info["init_state_type"] = state_type
            return obs, info
            
        obj_pos = self._get_obs()["achieved_goal"]
        
        if state_type in ["LIFTED_AND_HELD", "NEAR_TARGET_AND_HELD"]:
            # Grasp sequence
            self.controller.open_gripper(steps=30)
            self.controller.move_ee(position=obj_pos + np.array([0, 0, 0.15]), steps=150)
            self.controller.move_ee(position=obj_pos + np.array([0, 0, 0.015]), steps=150)
            self.controller.close_gripper(steps=30)
            
            for _ in range(20): 
                self.sim.step()
                
            # Lift sequence
            ee_pos = self.controller.get_ee_position()
            self.controller.move_ee(position=ee_pos + np.array([0, 0, 0.15]), steps=150)
            
            for _ in range(20): 
                self.sim.step()
                
            if state_type == "NEAR_TARGET_AND_HELD":
                target_pos = self.goal
                self.controller.move_ee(position=target_pos + np.array([0, 0, 0.1]), steps=150)
                for _ in range(20): 
                    self.sim.step()
                    
        obs = self._get_obs()
        info = self._get_info(obs["achieved_goal"], self.goal)
        info["init_state_type"] = state_type
        info["state_initialization_valid"] = True
        
        ee_pos = obs["observation"][:3]
        obj_pos = obs["achieved_goal"]
        target_pos = obs["desired_goal"]
        ee_obj_dist = np.linalg.norm(ee_pos - obj_pos)
        obj_lifted = obj_pos[2] > 0.05
        
        if state_type == "LIFTED_AND_HELD":
            if ee_obj_dist > 0.05 or not obj_lifted:
                info["state_initialization_valid"] = False
                
        elif state_type == "NEAR_TARGET_AND_HELD":
            obj_target_dist = np.linalg.norm(obj_pos - target_pos)
            if ee_obj_dist > 0.05 or not obj_lifted or obj_target_dist > 0.15:
                info["state_initialization_valid"] = False
                
        return obs, info
