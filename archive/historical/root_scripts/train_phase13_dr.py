import os
import sys
import time
import json
import yaml
import hashlib
import argparse
import platform
import numpy as np
import pandas as pd
import psutil
import torch

from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environments.dense_pick_place_env import PandaPickPlaceDenseRewardEnv


class Phase13DiagnosticCallback(BaseCallback):
    """
    Diagnostic callback monitoring:
    - throughput (steps/sec)
    - memory RSS (MB)
    - CPU utilization (%)
    - replay buffer size
    - actor/critic losses & ent_coef
    - NaN/Inf parameter and loss validation
    - 10k resource check logging
    """
    def __init__(self, exp_dir: str, process_name: str, check_10k_callback=None, verbose: int = 0):
        super().__init__(verbose)
        self.exp_dir = exp_dir
        self.process_name = process_name
        self.check_10k_callback = check_10k_callback
        self.metrics_dir = os.path.join(exp_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        self.process = psutil.Process(os.getpid())
        self.start_time = time.time()
        self.last_time = time.time()
        self.last_step = 0
        self.logged_10k = False
        self.progress_file = os.path.join(self.metrics_dir, "training_progress.csv")
        self.progress_records = []

    def _on_step(self) -> bool:
        current_step = self.num_timesteps
        
        # Check at exactly 10,000 steps
        if current_step >= 10000 and not self.logged_10k:
            self.logged_10k = True
            elapsed = time.time() - self.start_time
            steps_sec = current_step / max(elapsed, 1e-4)
            mem_rss = self.process.memory_info().rss / (1024 * 1024)
            cpu_pct = self.process.cpu_percent(interval=None)
            sys_mem = psutil.virtual_memory()
            
            res_10k = {
                "process_name": self.process_name,
                "step": current_step,
                "elapsed_seconds": round(elapsed, 2),
                "steps_per_sec": round(steps_sec, 2),
                "process_ram_mb": round(mem_rss, 2),
                "process_cpu_percent": round(cpu_pct, 2),
                "system_ram_available_gb": round(sys_mem.available / (1024**3), 2),
                "system_ram_total_gb": round(sys_mem.total / (1024**3), 2),
                "system_ram_percent": sys_mem.percent,
                "replay_buffer_size": self.model.replay_buffer.size() if self.model.replay_buffer else 0
            }
            res_path = os.path.join(self.metrics_dir, "resource_check_10k.json")
            with open(res_path, "w") as f:
                json.dump(res_10k, f, indent=4)
            print(f"[{self.process_name}] 10,000-STEP RESOURCE CHECK: {steps_sec:.2f} steps/s | RAM: {mem_rss:.1f} MB | CPU: {cpu_pct:.1f}%")
            if self.check_10k_callback:
                self.check_10k_callback(res_10k)

        # Periodic logging every 2,500 steps
        if current_step % 2500 == 0:
            now = time.time()
            dt = now - self.last_time
            ds = current_step - self.last_step
            steps_sec = ds / max(dt, 1e-4)
            self.last_time = now
            self.last_step = current_step
            
            mem_rss = self.process.memory_info().rss / (1024 * 1024)
            cpu_pct = self.process.cpu_percent(interval=None)
            rb_size = self.model.replay_buffer.size() if self.model.replay_buffer else 0
            
            # Extract loss values from logger if available
            actor_loss = float(self.model.logger.name_to_value.get("train/actor_loss", 0.0))
            critic_loss = float(self.model.logger.name_to_value.get("train/critic_loss", 0.0))
            ent_coef = float(self.model.logger.name_to_value.get("train/ent_coef", 0.0))
            
            # NaN / Inf validation
            for name, param in self.model.policy.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    err_msg = f"FATAL: NaN or Inf detected in parameter {name} at step {current_step}!"
                    print(f"[{self.process_name}] {err_msg}", file=sys.stderr)
                    raise FloatingPointError(err_msg)
            if np.isnan(actor_loss) or np.isinf(actor_loss) or np.isnan(critic_loss) or np.isinf(critic_loss):
                err_msg = f"FATAL: NaN or Inf detected in training losses at step {current_step}!"
                print(f"[{self.process_name}] {err_msg}", file=sys.stderr)
                raise FloatingPointError(err_msg)
            
            record = {
                "step": current_step,
                "elapsed_sec": round(now - self.start_time, 2),
                "steps_per_sec": round(steps_sec, 2),
                "process_ram_mb": round(mem_rss, 2),
                "process_cpu_pct": round(cpu_pct, 2),
                "rb_size": rb_size,
                "actor_loss": round(actor_loss, 4),
                "critic_loss": round(critic_loss, 4),
                "ent_coef": round(ent_coef, 4)
            }
            self.progress_records.append(record)
            df_progress = pd.DataFrame(self.progress_records)
            df_progress.to_csv(self.progress_file, index=False)
            
            if current_step % 25000 == 0:
                print(f"[{self.process_name}] Step {current_step:,} | Speed: {steps_sec:.1f} steps/s | RAM: {mem_rss:.1f} MB | ActorLoss: {actor_loss:.3f} | CriticLoss: {critic_loss:.3f}")

        return True


def evaluate_checkpoint_regime(model, env, regime_name: str, num_episodes: int = 100, seed_offset: int = 1001) -> pd.DataFrame:
    """
    Evaluates policy across `num_episodes` using deterministic evaluation.
    Computes all standard physical metrics and failure taxonomy.
    """
    results = []
    
    for ep in range(num_episodes):
        eval_seed = seed_offset + ep
        obs, info = env.reset(seed=eval_seed)
        
        initial_obj_pos = obs["achieved_goal"].copy()
        
        done = False
        step = 0
        current_hold = 0
        max_hold = 0
        grasped = False
        lifted = False
        placed = False
        dropped = False
        
        max_obj_height = float(initial_obj_pos[2])
        max_displacement = 0.0
        min_ee_obj_dist = 1e6
        total_rew = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info_step = env.step(action)
            done = term or trunc
            step += 1
            total_rew += reward
            
            ee_pos = info_step["ee_pos"]
            obj_pos = obs["achieved_goal"]
            target_pos = obs["desired_goal"]
            
            if obj_pos[2] > max_obj_height:
                max_obj_height = float(obj_pos[2])
                
            ee_obj_dist = float(np.linalg.norm(ee_pos - obj_pos))
            if ee_obj_dist < min_ee_obj_dist:
                min_ee_obj_dist = ee_obj_dist
                
            disp = float(np.linalg.norm(obj_pos - initial_obj_pos))
            if disp > max_displacement:
                max_displacement = disp
                
            is_held = (ee_obj_dist < 0.05) and (action[3] < 0.0)
            is_lifted_step = (obj_pos[2] > 0.05)
            
            if is_held:
                grasped = True
                current_hold += 1
                if current_hold > max_hold:
                    max_hold = current_hold
            else:
                if max_hold > 5:
                    dropped = True
                current_hold = 0
                
            if is_lifted_step and is_held:
                lifted = True
                
            if info_step.get("is_success", False):
                placed = True
        
        final_dist = float(info_step.get("transport_dist", np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])))
        reached = min_ee_obj_dist < 0.05
        transported = final_dist < 0.05
        timeout = (not placed) and (step >= env.unwrapped.max_episode_steps)
        
        # Rigorous mutually exclusive Failure Taxonomy
        if placed:
            failure_type = "SUCCESS"
        elif not reached:
            failure_type = "NO_REACH"
        elif not grasped:
            failure_type = "REACH_BUT_NO_GRASP"
        elif not lifted:
            failure_type = "GRASP_BUT_NO_LIFT"
        elif dropped or (max_hold < 5):
            failure_type = "LIFT_BUT_DROPPED"
        elif final_dist > 0.05:
            failure_type = "LIFT_AND_HELD_BUT_WRONG_DIRECTION"
        else:
            failure_type = "REACHED_TARGET_BUT_NO_PLACE"
            
        results.append({
            "regime": regime_name,
            "episode": ep,
            "eval_seed": eval_seed,
            "success": placed,
            "reach": reached,
            "grasped": grasped,
            "lifted": lifted,
            "transport": transported,
            "placed": placed,
            "final_dist": final_dist,
            "min_ee_obj_dist": min_ee_obj_dist,
            "max_obj_height": max_obj_height,
            "max_displacement": max_displacement,
            "hold_duration": max_hold,
            "dropped": dropped,
            "timeout": timeout,
            "ep_length": step,
            "reward": total_rew,
            "failure_type": failure_type
        })
        
    return pd.DataFrame(results)


def get_checkpoint_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def train_single_run(
    seed: int,
    dr_level: int,
    total_timesteps: int,
    exp_dir: str,
    num_threads: int = 6,
    checkpoints: list = None,
    eval_episodes: int = 100,
    eval_seed_start: int = 1001,
    resume_checkpoint: str = None
):
    if checkpoints is None or len(checkpoints) == 0:
        if total_timesteps >= 1000000:
            checkpoints = [250000, 500000, 750000, 1000000]
        else:
            checkpoints = [total_timesteps]
    # Filter or ensure checkpoints do not exceed total_timesteps
    checkpoints = sorted([cp for cp in checkpoints if cp <= total_timesteps])
    if len(checkpoints) == 0 or checkpoints[-1] != total_timesteps:
        checkpoints.append(total_timesteps)
        
    process_name = f"phase13_dr{dr_level}_seed{seed}"
    print(f"\n=======================================================")
    print(f"Starting {process_name}")
    print(f"Directory: {exp_dir}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"CPU threads: {num_threads}")
    print(f"=======================================================\n")
    
    # Set PyTorch & BLAS threads
    torch.set_num_threads(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    
    # Directory setup
    models_dir = os.path.join(exp_dir, "models")
    logs_dir = os.path.join(exp_dir, "logs")
    eval_dir = os.path.join(exp_dir, "evaluations")
    metrics_dir = os.path.join(exp_dir, "metrics")
    for d in [models_dir, logs_dir, eval_dir, metrics_dir]:
        os.makedirs(d, exist_ok=True)
        
    # DR Configuration strictly validated
    if dr_level == 1:
        dr_config = {
            "enabled": True,
            "object_position": {"enabled": True}
        }
        assert dr_config.get("enabled") is True, "Randomizer must be enabled"
        assert dr_config.get("object_position", {}).get("enabled") is True, "object_position must be enabled"
        assert "target_position" not in dr_config or not dr_config["target_position"].get("enabled", False), "target_position must be disabled"
        assert "object_orientation" not in dr_config or not dr_config["object_orientation"].get("enabled", False), "object_orientation must be disabled"
        assert "mass" not in dr_config or not dr_config["mass"].get("enabled", False), "mass must be disabled"
        assert "friction" not in dr_config or not dr_config["friction"].get("enabled", False), "friction must be disabled"
    elif dr_level == 2:
        dr_config = {
            "enabled": True,
            "object_position": {"enabled": True},
            "target_position": {"enabled": True}
        }
        assert dr_config.get("enabled") is True, "Randomizer must be enabled"
        assert dr_config.get("object_position", {}).get("enabled") is True, "object_position must be enabled"
        assert dr_config.get("target_position", {}).get("enabled") is True, "target_position must be enabled"
        assert "object_orientation" not in dr_config or not dr_config["object_orientation"].get("enabled", False), "object_orientation must be disabled"
        assert "mass" not in dr_config or not dr_config["mass"].get("enabled", False), "mass must be disabled"
        assert "friction" not in dr_config or not dr_config["friction"].get("enabled", False), "friction must be disabled"
    elif dr_level == 4:
        dr_config = {
            "enabled": True,
            "object_position": {"enabled": True},
            "target_position": {"enabled": True},
            "object_orientation": {"enabled": True},
            "mass": {"enabled": True, "min": 0.05, "max": 0.20},
            "friction": {"enabled": True, "min": 0.5, "max": 1.5}
        }
        assert dr_config.get("enabled") is True, "Randomizer must be enabled"
        assert dr_config.get("object_position", {}).get("enabled") is True, "object_position must be enabled"
        assert dr_config.get("target_position", {}).get("enabled") is True, "target_position must be enabled"
        assert dr_config.get("object_orientation", {}).get("enabled") is True, "object_orientation must be enabled"
        assert dr_config.get("mass", {}).get("enabled") is True, "mass must be enabled"
        assert dr_config.get("friction", {}).get("enabled") is True, "friction must be enabled"
    else:
        raise ValueError(f"DR level {dr_level} not supported in Phase 13 Fast-Track Benchmark!")
    
    # Save config snapshot
    config_snapshot = {
        "phase": 13,
        "dr_level": dr_level,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "checkpoints": checkpoints,
        "env": {
            "name": "PandaPickPlaceDenseRewardEnv",
            "distance_threshold": 0.05,
            "max_episode_steps": 100,
            "action_scale": 0.05,
            "dr_config": dr_config
        },
        "rl": {
            "algorithm": "SAC",
            "policy": "MultiInputPolicy",
            "her": True,
            "her_strategy": "future",
            "n_sampled_goal": 4,
            "learning_rate": 0.001,
            "learning_starts": 250,
            "buffer_size": 1000000,
            "batch_size": 256,
            "tau": 0.05,
            "gamma": 0.95
        }
    }
    with open(os.path.join(exp_dir, "config_snapshot.yaml"), "w") as f:
        yaml.dump(config_snapshot, f, indent=4)
        
    # Training metadata record
    start_time_epoch = time.time()
    metadata = {
        "experiment_id": process_name,
        "phase": 13,
        "dr_level": dr_level,
        "seed": seed,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "threads_assigned": num_threads,
        "total_timesteps_target": total_timesteps,
        "training_start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time_epoch)),
        "checkpoints_planned": checkpoints,
        "checkpoint_hashes": {},
        "status": "RUNNING"
    }
    with open(os.path.join(exp_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    # 1. Instantiate training environment
    env_train = PandaPickPlaceDenseRewardEnv(
        distance_threshold=0.05,
        max_episode_steps=100,
        action_scale=0.05,
        randomizer_config=dr_config
    )
    env_train = Monitor(env_train)
    vec_env = DummyVecEnv([lambda: env_train])
    
    # 2. Instantiate SAC + HER model
    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        print(f"[{process_name}] Resuming from checkpoint: {resume_checkpoint}")
        model = SAC.load(resume_checkpoint, env=vec_env, device="cpu")
        model.tensorboard_log = logs_dir
        
        rb_path = resume_checkpoint.replace(".zip", "_replay_buffer.pkl")
        if os.path.exists(rb_path):
            print(f"[{process_name}] Loading replay buffer from {rb_path}")
            model.load_replay_buffer(rb_path)
        else:
            print(f"[{process_name}] WARNING: Replay buffer not found. Delaying learning by 250 steps.")
            model.learning_starts = model.num_timesteps + 250
    else:
        model = SAC(
            policy="MultiInputPolicy",
            env=vec_env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",
            ),
            learning_rate=0.001,
            learning_starts=250,
            buffer_size=1000000,
            batch_size=256,
            tau=0.05,
            gamma=0.95,
            tensorboard_log=logs_dir,
            seed=seed,
            verbose=0
        )
    
    # 3. Instantiate evaluation environments
    # Regime 1: Nominal DR-0
    env_eval_dr0 = PandaPickPlaceDenseRewardEnv(
        distance_threshold=0.05,
        max_episode_steps=100,
        action_scale=0.05,
        randomizer_config={"enabled": False}
    )
    # Regime 2: Matched DR
    matched_regime = f"Matched_DR{dr_level}"
    env_eval_matched = PandaPickPlaceDenseRewardEnv(
        distance_threshold=0.05,
        max_episode_steps=100,
        action_scale=0.05,
        randomizer_config=dr_config
    )
    # Regime 3: Held-out Stress
    with open("configs/pick_place_heldout.yaml", "r") as f:
        heldout_cfg = yaml.safe_load(f)["env"]
    heldout_env_kwargs = heldout_cfg.copy()
    if "name" in heldout_env_kwargs:
        del heldout_env_kwargs["name"]
    if "randomizer" in heldout_env_kwargs:
        heldout_env_kwargs["randomizer_config"] = heldout_env_kwargs.pop("randomizer")
    env_eval_heldout = PandaPickPlaceDenseRewardEnv(**heldout_env_kwargs)
    
    # Diagnostic callback
    diag_callback = Phase13DiagnosticCallback(exp_dir=exp_dir, process_name=process_name)
    
    # 4. Sequential Checkpoint Loop
    all_checkpoint_metrics = []
    all_taxonomy_records = []
    previous_step = model.num_timesteps
    
    for cp in checkpoints:
        if cp <= previous_step:
            continue
            
        steps_to_train = cp - previous_step
        print(f"\n[{process_name}] Training chunk: {steps_to_train:,} steps (Target: {cp:,})...")
        
        chunk_t0 = time.time()
        model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False, callback=diag_callback)
        chunk_time = time.time() - chunk_t0
        chunk_speed = steps_to_train / max(chunk_time, 1e-4)
        print(f"[{process_name}] Chunk complete in {chunk_time:.1f}s ({chunk_speed:.1f} steps/s). Saving checkpoint...")
        
        # Save checkpoint
        cp_filename = f"checkpoint_{cp}.zip"
        cp_path = os.path.join(models_dir, cp_filename)
        model.save(cp_path)
        
        # Save replay buffer for future resumes
        rb_path = cp_path.replace(".zip", "_replay_buffer.pkl")
        model.save_replay_buffer(rb_path)
        
        sha256_hash = get_checkpoint_sha256(cp_path)
        metadata["checkpoint_hashes"][cp_filename] = sha256_hash
        
        # Verify checkpoint is loadable
        test_loaded_model = SAC.load(cp_path, env=vec_env, device="cpu")
        del test_loaded_model
        
        # Run Deterministic Evaluations across all 3 regimes
        print(f"[{process_name}] Evaluating checkpoint {cp:,} on 3 regimes ({eval_episodes} eps each)...")
        df_dr0 = evaluate_checkpoint_regime(model, env_eval_dr0, "Nominal_DR0", num_episodes=eval_episodes, seed_offset=eval_seed_start)
        df_matched = evaluate_checkpoint_regime(model, env_eval_matched, matched_regime, num_episodes=eval_episodes, seed_offset=eval_seed_start)
        df_held = evaluate_checkpoint_regime(model, env_eval_heldout, "Heldout_Stress", num_episodes=eval_episodes, seed_offset=eval_seed_start)
        
        df_combined = pd.concat([df_dr0, df_matched, df_held], ignore_index=True)
        df_combined["checkpoint"] = cp
        df_combined.to_csv(os.path.join(eval_dir, f"eval_checkpoint_{cp}.csv"), index=False)
        
        for regime, df_sub in [("Nominal_DR0", df_dr0), (matched_regime, df_matched), ("Heldout_Stress", df_held)]:
            m = {
                "checkpoint": cp,
                "regime": regime,
                "reach_rate": df_sub["reach"].mean(),
                "grasp_rate": df_sub["grasped"].mean(),
                "lift_rate": df_sub["lifted"].mean(),
                "transport_rate": df_sub["transport"].mean(),
                "placement_rate": df_sub["placed"].mean(),
                "success_rate": df_sub["success"].mean(),
                "mean_final_dist": df_sub["final_dist"].mean(),
                "median_final_dist": df_sub["final_dist"].median(),
                "mean_displacement": df_sub["max_displacement"].mean(),
                "mean_hold_duration": df_sub["hold_duration"].mean(),
                "drop_rate": df_sub["dropped"].mean(),
                "timeout_rate": df_sub["timeout"].mean(),
                "mean_reward": df_sub["reward"].mean(),
                "mean_ep_length": df_sub["ep_length"].mean()
            }
            all_checkpoint_metrics.append(m)
            
            # Taxonomy counts
            tax_counts = df_sub["failure_type"].value_counts().to_dict()
            for tax_type in ["SUCCESS", "NO_REACH", "REACH_BUT_NO_GRASP", "GRASP_BUT_NO_LIFT", "LIFT_BUT_DROPPED", "LIFT_AND_HELD_BUT_WRONG_DIRECTION", "REACHED_TARGET_BUT_NO_PLACE"]:
                all_taxonomy_records.append({
                    "checkpoint": cp,
                    "regime": regime,
                    "failure_type": tax_type,
                    "count": tax_counts.get(tax_type, 0),
                    "fraction": tax_counts.get(tax_type, 0) / len(df_sub)
                })
            
            print(f"[{process_name}] CP {cp:,} [{regime}] Success: {m['success_rate']*100:.1f}% | Reach: {m['reach_rate']*100:.1f}% | Grasp: {m['grasp_rate']*100:.1f}% | Lift: {m['lift_rate']*100:.1f}% | FinalDist: {m['mean_final_dist']:.3f}m")
            
        # Update metrics files
        pd.DataFrame(all_checkpoint_metrics).to_csv(os.path.join(metrics_dir, "checkpoint_metrics.csv"), index=False)
        pd.DataFrame(all_taxonomy_records).to_csv(os.path.join(metrics_dir, "failure_taxonomy.csv"), index=False)
        
        # Update metadata
        with open(os.path.join(exp_dir, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
            
        previous_step = cp
        
    # Save final model
    final_model_path = os.path.join(models_dir, "final_model.zip")
    model.save(final_model_path)
    metadata["checkpoint_hashes"]["final_model.zip"] = get_checkpoint_sha256(final_model_path)
    
    end_time_epoch = time.time()
    total_duration = end_time_epoch - start_time_epoch
    metadata["training_end_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time_epoch))
    metadata["total_duration_seconds"] = round(total_duration, 2)
    metadata["final_replay_buffer_size"] = model.replay_buffer.size()
    metadata["status"] = "COMPLETE"
    
    with open(os.path.join(exp_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    completion_summary = {
        "experiment_id": process_name,
        "seed": seed,
        "total_timesteps": model.num_timesteps,
        "duration_seconds": round(total_duration, 2),
        "duration_hours": round(total_duration / 3600, 2),
        "status": "COMPLETE"
    }
    with open(os.path.join(metrics_dir, "completed.json"), "w") as f:
        json.dump(completion_summary, f, indent=4)
        
    print(f"\n[{process_name}] RUN COMPLETE! Total time: {total_duration/3600:.2f} hours ({total_timesteps:,} steps)\n")
    
    # Close environments
    env_train.close()
    env_eval_dr0.close()
    env_eval_matched.close()
    env_eval_heldout.close()
    return completion_summary


def main():
    parser = argparse.ArgumentParser(description="Phase 13 Dense SAC+HER Domain Randomization Runner")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--dr_level", type=int, default=1, help="Domain randomization level (1-4)")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total environment timesteps")
    parser.add_argument("--exp_dir", type=str, required=True, help="Experiment output directory")
    parser.add_argument("--num_threads", type=int, default=6, help="PyTorch CPU threads")
    parser.add_argument("--checkpoints", type=int, nargs="*", default=None, help="Specific checkpoint timesteps")
    parser.add_argument("--eval_episodes", type=int, default=100, help="Deterministic evaluation episodes")
    parser.add_argument("--eval_seed_start", type=int, default=1001, help="Starting seed for evaluation")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    train_single_run(
        seed=args.seed,
        dr_level=args.dr_level,
        total_timesteps=args.timesteps,
        exp_dir=args.exp_dir,
        num_threads=args.num_threads,
        checkpoints=args.checkpoints,
        eval_episodes=args.eval_episodes,
        eval_seed_start=args.eval_seed_start,
        resume_checkpoint=args.resume_checkpoint
    )


if __name__ == "__main__":
    main()
