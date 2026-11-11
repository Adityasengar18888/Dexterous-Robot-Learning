"""Evaluation script for trained SAC models."""

import argparse
import yaml
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from src.environments.reach_env import PandaReachEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained SAC model")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model zip file")
    parser.add_argument("--config", type=str, default="configs/reach.yaml", help="Path to config file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI)")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env(env_cfg, render_mode=None):
    if env_cfg.get("name") == "Reach":
        from src.environments.reach_env import PandaReachEnv
        return PandaReachEnv(
            render_mode=render_mode,
            distance_threshold=env_cfg.get("distance_threshold", 0.05),
            max_episode_steps=env_cfg.get("max_episode_steps", 50),
            action_scale=env_cfg.get("action_scale", 0.05)
        )
    elif env_cfg.get("name") == "Pick":
        from src.environments.pick_env import PandaPickEnv
        return PandaPickEnv(
            render_mode=render_mode,
            distance_threshold=env_cfg.get("distance_threshold", 0.05),
            max_episode_steps=env_cfg.get("max_episode_steps", 100),
            action_scale=env_cfg.get("action_scale", 0.05)
        )
    elif env_cfg.get("name") == "PickPlace":
        from src.environments.pick_place_env import PandaPickPlaceEnv
        return PandaPickPlaceEnv(
            render_mode=render_mode,
            distance_threshold=env_cfg.get("distance_threshold", 0.05),
            max_episode_steps=env_cfg.get("max_episode_steps", 100),
            action_scale=env_cfg.get("action_scale", 0.05),
            randomizer_config=env_cfg.get("randomizer")
        )
    elif env_cfg.get("name") == "Sort":
        from src.environments.sort_env import PandaSortEnv
        return PandaSortEnv(
            render_mode=render_mode,
            distance_threshold=env_cfg.get("distance_threshold", 0.05),
            max_episode_steps=env_cfg.get("max_episode_steps", 150),
            action_scale=env_cfg.get("action_scale", 0.05)
        )
    elif env_cfg.get("name") == "MultiSort":
        from src.environments.multi_sort_env import PandaMultiSortEnv
        return PandaMultiSortEnv(
            render_mode=render_mode,
            num_objects=env_cfg.get("num_objects", 2),
            distance_threshold=env_cfg.get("distance_threshold", 0.05),
            max_episode_steps=env_cfg.get("max_episode_steps", 300),
            action_scale=env_cfg.get("action_scale", 0.05)
        )
    elif env_cfg.get("name") == "MultiSortDense":
        from src.environments.dense_multi_sort_env import PandaMultiSortDenseRewardEnv
        return PandaMultiSortDenseRewardEnv(
            render_mode=render_mode,
            num_objects=env_cfg.get("num_objects", 2),
            distance_threshold=env_cfg.get("distance_threshold", 0.05),
            max_episode_steps=env_cfg.get("max_episode_steps", 300),
            action_scale=env_cfg.get("action_scale", 0.05)
        )
    elif env_cfg.get("name") == "MultiSortDenseV2":
        from src.environments.dense_multi_sort_env import PandaMultiSortDenseV2Env
        return PandaMultiSortDenseV2Env(
            render_mode=render_mode,
            num_objects=env_cfg.get("num_objects", 2),
            distance_threshold=env_cfg.get("distance_threshold", 0.05),
            max_episode_steps=env_cfg.get("max_episode_steps", 300),
            action_scale=env_cfg.get("action_scale", 0.05)
        )
    elif env_cfg.get("name") == "SinglePreGrasp":
        from src.environments.single_pregrasp_env import PandaSinglePreGraspEnv, SinglePreGraspRewardConfig
        rcfg_dict = env_cfg.get("reward_config")
        rcfg = SinglePreGraspRewardConfig(**rcfg_dict) if isinstance(rcfg_dict, dict) else None
        return PandaSinglePreGraspEnv(
            render_mode=render_mode,
            distance_threshold=env_cfg.get("distance_threshold", 0.05),
            max_episode_steps=env_cfg.get("max_episode_steps", 80),
            action_scale=env_cfg.get("action_scale", 0.05),
            reward_config=rcfg,
        )
    elif env_cfg.get("name") == "MultiSortPreGrasp":
        from src.environments.pregrasp_multi_sort_env import PandaMultiSortPreGraspEnv
        return PandaMultiSortPreGraspEnv(
            render_mode=render_mode,
            num_objects=env_cfg.get("num_objects", 2),
            distance_threshold=env_cfg.get("distance_threshold", 0.05),
            max_episode_steps=env_cfg.get("max_episode_steps", 300),
            action_scale=env_cfg.get("action_scale", 0.05)
        )
    elif env_cfg.get("name") == "Assembly":
        from src.environments.assembly_env import PandaAssemblyEnv
        return PandaAssemblyEnv(
            render_mode=render_mode,
            distance_threshold=env_cfg.get("distance_threshold", 0.02),
            orientation_threshold=env_cfg.get("orientation_threshold", 0.1),
            max_episode_steps=env_cfg.get("max_episode_steps", 200),
            action_scale=env_cfg.get("action_scale", 0.05)
        )
    elif env_cfg.get("name") == "AssemblyDense":
        from src.environments.assembly_dense_env import PandaAssemblyDenseEnv
        return PandaAssemblyDenseEnv(
            render_mode=render_mode,
            distance_threshold=env_cfg.get("distance_threshold", 0.02),
            orientation_threshold=env_cfg.get("orientation_threshold", 0.1),
            max_episode_steps=env_cfg.get("max_episode_steps", 120),
            action_scale=env_cfg.get("action_scale", 0.05),
            yaw_scale=env_cfg.get("yaw_scale", 0.25),
        )
    else:
        raise ValueError(f"Unknown environment: {env_cfg.get('name')}")


def main():
    args = parse_args()
    config = load_config(args.config)
    env_cfg = config.get("env", {})

    render_mode = None if args.headless else "human"
    
    env = make_env(env_cfg, render_mode=render_mode)

    print(f"Loading model from {args.model}...")
    model = SAC.load(args.model, env=env)

    successes = []
    distances = []
    episode_lengths = []
    grasp_successes = []
    collisions_list = []
    rewards_list = []
    
    # Multi-Sort metrics
    objects_placed_list = []
    wrong_bin_list = []
    dropped_objects_list = []
    
    # Assembly metrics
    orientation_errors = []
    assembly_states = []

    print(f"Evaluating for {args.episodes} episodes...")
    for ep in range(args.episodes):
        # We seed evaluation starting from a deterministic offset
        obs, info = env.reset(seed=100 + ep)
        done = False
        ep_reward = 0.0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        successes.append(info.get("is_success", False))
        distances.append(info.get("distance", float("inf")))
        episode_lengths.append(info.get("episode_length", env_cfg.get("max_episode_steps", 100)))
        grasp_successes.append(info.get("grasp_success", False))
        collisions_list.append(info.get("collisions", 0))
        rewards_list.append(ep_reward)
        
        if "objects_placed" in info:
            objects_placed_list.append(info["objects_placed"])
            wrong_bin_list.append(info["wrong_bin_placements"])
            dropped_objects_list.append(info["dropped_objects"])
            
        if "orientation_error" in info:
            orientation_errors.append(info["orientation_error"])
            assembly_states.append(info["assembly_state"])
        
        print(f"Episode {ep+1:02d} | Succ: {successes[-1]!s:<5} | Dist: {distances[-1]:.4f} | Collisions: {collisions_list[-1]}")

    success_rate = np.mean(successes)
    avg_distance = np.mean(distances)
    avg_length = np.mean(episode_lengths)
    grasp_rate = np.mean(grasp_successes)
    avg_collisions = np.mean(collisions_list)
    avg_reward = np.mean(rewards_list)

    print("\n=== Evaluation Results ===")
    print(f"Success Rate: {success_rate * 100:.2f}%")
    print(f"Grasp Success Rate: {grasp_rate * 100:.2f}%")
    print(f"Average Final Distance: {avg_distance:.4f}m")
    print(f"Average Episode Length: {avg_length:.1f} steps")
    print(f"Average Collisions: {avg_collisions:.1f}")
    print(f"Average Reward: {avg_reward:.2f}")
    
    if objects_placed_list:
        print(f"Average Objects Placed: {np.mean(objects_placed_list):.2f}")
        print(f"Average Wrong Bins: {np.mean(wrong_bin_list):.2f}")
        print(f"Average Dropped Objects: {np.mean(dropped_objects_list):.2f}")
        
    if orientation_errors:
        print(f"Average Orientation Error: {np.mean(orientation_errors):.4f}")
        from collections import Counter
        print(f"Assembly States: {dict(Counter(assembly_states))}")

    env.close()

if __name__ == "__main__":
    main()
