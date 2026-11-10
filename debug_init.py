import yaml
import numpy as np
from src.environments.transport_diagnostic_env import PandaTransportDiagnosticEnv

def main():
    with open("configs/pick_place_baseline.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    env_kwargs = config["env"].copy()
    if "name" in env_kwargs:
        del env_kwargs["name"]
    if "randomizer" in env_kwargs:
        env_kwargs["randomizer_config"] = env_kwargs.pop("randomizer")
        
    env = PandaTransportDiagnosticEnv(**env_kwargs, render_mode=None)
    obs, info = env.reset(seed=1000)
    
    ee_pos = obs["observation"][:3]
    obj_pos = obs["achieved_goal"]
    
    ee_obj_dist = np.linalg.norm(ee_pos - obj_pos)
    obj_lifted = obj_pos[2] > 0.05
    gripper_width = obs["observation"][3]
    
    print(f"Valid Init: {info['state_initialization_valid']}")
    print(f"EE pos: {ee_pos}")
    print(f"Obj pos: {obj_pos}")
    print(f"EE-Obj Dist: {ee_obj_dist:.4f} (must be <= 0.05)")
    print(f"Obj Z: {obj_pos[2]:.4f} (must be > 0.05)")
    print(f"Gripper Width: {gripper_width:.4f}")
    
    env.close()
    
if __name__ == "__main__":
    main()
