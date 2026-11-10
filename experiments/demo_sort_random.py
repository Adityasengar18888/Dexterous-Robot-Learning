"""Random rollout for Sorting Environment."""

from src.environments.sort_env import PandaSortEnv

def run_random_sort():
    env = PandaSortEnv(render_mode=None, max_episode_steps=50)
    obs, info = env.reset(seed=42)
    
    print("=== Random Sorting Policy Rollout ===")
    
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
            
    print(f"Episode finished after {step+1} steps")
    print(f"Final Info: {info}")
    env.close()

if __name__ == "__main__":
    run_random_sort()
