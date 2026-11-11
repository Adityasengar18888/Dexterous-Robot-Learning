"""Training script for SAC Baseline."""

import argparse
import os
import yaml

from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

from src.environments.reach_env import PandaReachEnv
from src.rl.callbacks import SuccessRateCallback


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC on Reach Task")
    parser.add_argument("--config", type=str, default="configs/reach.yaml", help="Path to YAML config file")
    parser.add_argument("--resume-model", type=str, default=None, help="Optional checkpoint zip path to resume training from")
    parser.add_argument("--resume-buffer", type=str, default=None, help="Optional replay_buffer.pkl path to load when resuming")
    parser.add_argument("--additional-timesteps", type=int, default=None, help="Timesteps to train when resuming (with reset_num_timesteps=False)")
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
            action_scale=env_cfg.get("action_scale", 0.05),
        )
    elif env_cfg.get("name") == "MultiSortDenseV2":
        from src.environments.dense_multi_sort_env import PandaMultiSortDenseV2Env
        return PandaMultiSortDenseV2Env(
            render_mode=render_mode,
            num_objects=env_cfg.get("num_objects", 2),
            distance_threshold=env_cfg.get("distance_threshold", 0.05),
            max_episode_steps=env_cfg.get("max_episode_steps", 300),
            action_scale=env_cfg.get("action_scale", 0.05),
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
            action_scale=env_cfg.get("action_scale", 0.05),
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


def _validate_config(config: dict) -> None:
    """Validate required configuration fields explicitly without silent fallbacks."""
    for top_key in ("env", "rl", "paths"):
        if top_key not in config or not isinstance(config[top_key], dict):
            raise ValueError(f"Missing or invalid config section: '{top_key}'")
    for req_path in ("model_save_path", "tensorboard_log"):
        if not config["paths"].get(req_path):
            raise ValueError(f"Missing required paths config field: '{req_path}'")
    if not config["env"].get("name"):
        raise ValueError("Missing required env.name in config")
    if "total_timesteps" not in config["rl"]:
        raise ValueError("Missing required rl.total_timesteps in config")
    if config["env"]["name"] in ("MultiSortDense", "MultiSortDenseV2", "SinglePreGrasp", "MultiSortPreGrasp"):
        for req_rl in ("seed", "buffer_size", "batch_size", "learning_rate", "learning_starts", "gamma", "tau", "use_her", "her_strategy", "her_n_sampled_goal", "her_copy_info_dict"):
            if req_rl not in config["rl"]:
                raise ValueError(f"Missing required {config['env']['name']} rl field: '{req_rl}'")


def main():
    args = parse_args()
    config = load_config(args.config)
    _validate_config(config)

    env_cfg = config.get("env", {})
    rl_cfg = config.get("rl", {})
    eval_cfg = config.get("eval", {})
    ckpt_cfg = config.get("checkpointing", {})
    paths_cfg = config.get("paths", {})

    os.makedirs(paths_cfg["model_save_path"], exist_ok=True)
    os.makedirs(paths_cfg["tensorboard_log"], exist_ok=True)

    print("Initializing environments...")
    # Training Environment (Headless)
    env = make_env(env_cfg, render_mode=None)
    env = Monitor(env, info_keywords=("is_success", "distance"))

    # Evaluation Environment (Headless)
    eval_env = make_env(env_cfg, render_mode=None)
    eval_env = Monitor(eval_env, info_keywords=("is_success", "distance"))

    print("Configuring callbacks...")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=paths_cfg["model_save_path"],
        log_path=paths_cfg["model_save_path"],
        eval_freq=eval_cfg.get("eval_freq", 2500),
        n_eval_episodes=eval_cfg.get("n_eval_episodes", 20),
        deterministic=eval_cfg.get("deterministic", True),
        render=False
    )
    success_callback = SuccessRateCallback()
    cb_list = [eval_callback, success_callback]
    if ckpt_cfg.get("save_freq"):
        cb_list.append(
            CheckpointCallback(
                save_freq=int(ckpt_cfg["save_freq"]),
                save_path=paths_cfg["model_save_path"],
                name_prefix="checkpoint",
                verbose=1,
            )
        )
    callbacks = CallbackList(cb_list)

    print("Initializing SAC model...")
    if rl_cfg.get("use_her", False):
        replay_buffer_class = HerReplayBuffer
        replay_buffer_kwargs = {
            "n_sampled_goal": rl_cfg.get("her_n_sampled_goal", 4),
            "goal_selection_strategy": rl_cfg.get("her_strategy", "future"),
        }
        if "her_copy_info_dict" in rl_cfg:
            replay_buffer_kwargs["copy_info_dict"] = bool(rl_cfg["her_copy_info_dict"])
    else:
        replay_buffer_class = None
        replay_buffer_kwargs = None

    # MultiInputPolicy handles observation spaces of type spaces.Dict
    use_bc_sac = bool(rl_cfg.get("use_bc_sac", False))
    demo_path = rl_cfg.get("demo_warmstart_path")

    if args.resume_model:
        print(f"Resuming SAC model from {args.resume_model} (use_bc_sac={use_bc_sac})...")
        if use_bc_sac:
            from src.rl.bc_sac import BCDecayingSAC
            model = BCDecayingSAC.load(
                args.resume_model,
                env=env,
                device=rl_cfg.get("device", "auto"),
                tensorboard_log=paths_cfg["tensorboard_log"],
            )
            model.bc_lambda_init = float(rl_cfg.get("bc_lambda_init", 0.305))
            model.bc_lambda_final = float(rl_cfg.get("bc_lambda_final", 0.02))
            model.bc_decay_steps = int(rl_cfg.get("bc_decay_steps", 5000))
            model.bc_batch_size = int(rl_cfg.get("bc_batch_size", 128))
            model.min_ent_coef = float(rl_cfg.get("min_ent_coef", 0.02))
            model.actor_update_start_step = int(rl_cfg.get("actor_update_start_step", 0))
            model.bc_normalize_q = bool(rl_cfg.get("bc_normalize_q", False))
            if demo_path:
                model.load_bc_demonstrations(demo_path)
        else:
            model = SAC.load(
                args.resume_model,
                env=env,
                device=rl_cfg.get("device", "auto"),
                tensorboard_log=paths_cfg["tensorboard_log"],
            )
        if args.resume_buffer:
            print(f"Loading replay buffer from {args.resume_buffer}...")
            model.load_replay_buffer(args.resume_buffer)
            print(f"Loaded replay buffer size: {model.replay_buffer.size()}")
        train_steps = int(args.additional_timesteps if args.additional_timesteps is not None else rl_cfg["total_timesteps"])
        reset_ts = False
    else:
        if use_bc_sac:
            from src.rl.bc_sac import BCDecayingSAC
            model = BCDecayingSAC(
                rl_cfg.get("policy", "MultiInputPolicy"),
                env,
                seed=rl_cfg.get("seed", 42),
                buffer_size=rl_cfg.get("buffer_size", 100000),
                batch_size=rl_cfg.get("batch_size", 256),
                learning_rate=rl_cfg.get("learning_rate", 0.001),
                learning_starts=rl_cfg.get("learning_starts", 100),
                gamma=rl_cfg.get("gamma", 0.95),
                tau=rl_cfg.get("tau", 0.05),
                device=rl_cfg.get("device", "auto"),
                replay_buffer_class=replay_buffer_class,
                replay_buffer_kwargs=replay_buffer_kwargs,
                tensorboard_log=paths_cfg["tensorboard_log"],
                bc_demos_path=demo_path,
                bc_lambda_init=rl_cfg.get("bc_lambda_init", 0.305),
                bc_lambda_final=rl_cfg.get("bc_lambda_final", 0.02),
                bc_decay_steps=rl_cfg.get("bc_decay_steps", 5000),
                bc_batch_size=rl_cfg.get("bc_batch_size", 128),
                min_ent_coef=rl_cfg.get("min_ent_coef", 0.02),
                actor_update_start_step=rl_cfg.get("actor_update_start_step", 0),
                bc_normalize_q=rl_cfg.get("bc_normalize_q", False),
                ent_coef=rl_cfg.get("ent_coef", "auto"),
                verbose=1,
            )
        else:
            model = SAC(
                rl_cfg.get("policy", "MultiInputPolicy"),
                env,
                seed=rl_cfg.get("seed", 42),
                buffer_size=rl_cfg.get("buffer_size", 100000),
                batch_size=rl_cfg.get("batch_size", 256),
                learning_rate=rl_cfg.get("learning_rate", 0.001),
                learning_starts=rl_cfg.get("learning_starts", 100),
                gamma=rl_cfg.get("gamma", 0.95),
                tau=rl_cfg.get("tau", 0.05),
                device=rl_cfg.get("device", "auto"),
                replay_buffer_class=replay_buffer_class,
                replay_buffer_kwargs=replay_buffer_kwargs,
                tensorboard_log=paths_cfg["tensorboard_log"],
                verbose=1,
            )

        initial_path = os.path.join(paths_cfg["model_save_path"], "initial_model")
        model.save(initial_path)
        print(f"Saved initial model to {initial_path}.zip")

        bc_pretrain_steps = int(rl_cfg.get("bc_pretrain_steps", 0))
        if use_bc_sac and bc_pretrain_steps > 0 and demo_path:
            bc_stats = model.run_bc_warmstart(
                steps=bc_pretrain_steps,
                lr=float(rl_cfg.get("bc_pretrain_lr", 0.001)),
                batch_size=int(rl_cfg.get("bc_batch_size", 128)),
            )
            bc_warm_path = os.path.join(paths_cfg["model_save_path"], "bc_warmstart_model")
            model.save(bc_warm_path)
            print(f"BC warmstart completed: {bc_stats} -> saved {bc_warm_path}.zip")

        demo_method = rl_cfg.get("demo_warmstart_method", "her_buffer")
        if demo_path and demo_method == "her_buffer":
            import pickle
            import numpy as np

            with open(demo_path, "rb") as df:
                demos = pickle.load(df)
            inserted = 0
            for ep in demos:
                for t in range(int(ep["length"])):
                    obs_b = {k: np.expand_dims(v, axis=0) for k, v in ep["observations"][t].items()}
                    nobs_b = {k: np.expand_dims(v, axis=0) for k, v in ep["next_observations"][t].items()}
                    act_b = np.expand_dims(ep["actions"][t], axis=0).astype(np.float32)
                    rew_b = np.array([ep["rewards"][t]], dtype=np.float32)
                    done_b = np.array([ep["dones"][t]], dtype=bool)
                    info_rec = dict(ep["infos"][t])
                    info_rec["is_demo"] = True
                    model.replay_buffer.add(obs_b, nobs_b, act_b, rew_b, done_b, [info_rec])
                    inserted += 1
            print(f"Warm-started HerReplayBuffer with {len(demos)} demo episodes ({inserted} transitions, size={model.replay_buffer.size()})")

        critic_warmup_steps = int(rl_cfg.get("critic_warmup_steps", 0))
        if use_bc_sac and critic_warmup_steps > 0 and model.replay_buffer.size() > 0:
            cw_stats = model.run_critic_warmup(
                steps=critic_warmup_steps,
                batch_size=int(rl_cfg.get("batch_size", 128)),
            )
            print(f"Critic warmup completed (actor frozen): {cw_stats}")

        train_steps = int(rl_cfg["total_timesteps"])
        reset_ts = True

    print(f"Starting training for {train_steps} timesteps (reset_num_timesteps={reset_ts})...")
    model.learn(
        total_timesteps=train_steps,
        callback=callbacks,
        reset_num_timesteps=reset_ts,
        progress_bar=False,
    )

    final_path = os.path.join(paths_cfg["model_save_path"], "final_model")
    model.save(final_path)
    print(f"Training finished. Final model saved to {final_path}.zip")

    # Save replay buffer only if safely serializable
    rb_path = os.path.join(paths_cfg["model_save_path"], "replay_buffer.pkl")
    try:
        model.save_replay_buffer(rb_path)
        print(f"Replay buffer saved to {rb_path}")
    except Exception as exc:
        print(f"Replay buffer serialization skipped (unsupported/unsafe): {exc}")
        if os.path.exists(rb_path):
            os.remove(rb_path)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
