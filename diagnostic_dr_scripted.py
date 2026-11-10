import numpy as np
from src.environments.pick_place_env import PandaPickPlaceEnv

def run_dr_scripted_evaluation():
    configs = {
        "Deterministic (R0)": {"enabled": False},
        "Position (R1)": {"enabled": True, "object_position": {"enabled": True}},
        "Pos+Target (R2)": {"enabled": True, "object_position": {"enabled": True}, "target_position": {"enabled": True}},
        "Pose (R3)": {"enabled": True, "object_position": {"enabled": True}, "target_position": {"enabled": True}, "object_orientation": {"enabled": True}},
        "Physics (R4)": {"enabled": True, "object_position": {"enabled": True}, "target_position": {"enabled": True}, "object_orientation": {"enabled": True}, "mass": {"enabled": True, "min": 0.05, "max": 0.20}, "friction": {"enabled": True, "min": 0.5, "max": 1.5}},
    }
    
    for name, config in configs.items():
        print(f"\n--- Evaluating {name} ---")
        env = PandaPickPlaceEnv(render_mode=None, randomizer_config=config, max_episode_steps=100)
        
        successes = 0
        total = 20
        
        for ep in range(total):
            obs, info = env.reset(seed=100 + ep)
            
            # Scripted logic for pick and place
            step = 0
            phase = 1
            wait_step = 0
            
            while phase <= 6 and step < 100:
                ee_pos = obs["observation"][:3]
                obj_pos = obs["observation"][4:7]
                target_pos = obs["desired_goal"]
                
                action = np.zeros(4, dtype=np.float32)
                
                if phase == 1:
                    target = obj_pos.copy()
                    target[2] = 0.15
                    delta = target - ee_pos
                    action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
                    action[3] = 1.0
                    if np.linalg.norm(delta) < 0.02:
                        phase = 2
                elif phase == 2:
                    target = obj_pos.copy()
                    target[2] = 0.015
                    delta = target - ee_pos
                    action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
                    action[3] = 1.0
                    if np.linalg.norm(delta) < 0.02:
                        phase = 3
                        wait_step = step
                elif phase == 3:
                    action[3] = -1.0
                    if step - wait_step > 15:
                        phase = 4
                elif phase == 4:
                    target = obj_pos.copy()
                    target[2] = 0.15
                    delta = target - ee_pos
                    action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
                    action[3] = -1.0
                    if np.linalg.norm(delta) < 0.02:
                        phase = 5
                elif phase == 5:
                    target = target_pos.copy()
                    target[2] = 0.15
                    delta = target - ee_pos
                    action[:3] = np.clip(delta / env.action_scale, -1.0, 1.0)
                    action[3] = -1.0
                    if np.linalg.norm(delta) < 0.02:
                        phase = 6
                        wait_step = step
                elif phase == 6:
                    action[3] = 1.0
                    if step - wait_step > 10:
                        break
                
                obs, reward, terminated, truncated, info = env.step(action)
                step += 1
                if info["is_success"]:
                    break
                    
            if info["is_success"]:
                successes += 1
                
        print(f"Success Rate: {successes/total*100:.1f}%")
        env.close()

if __name__ == "__main__":
    run_dr_scripted_evaluation()
