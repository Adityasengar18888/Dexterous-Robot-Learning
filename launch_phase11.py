import os
import sys
import subprocess
import yaml

def run_batch(batch_name, config_file, experiment_prefix, seeds, timesteps=100000):
    python_exe = ".conda-env/python"
    
    print(f"\n{'='*50}")
    print(f"Starting {batch_name}: {experiment_prefix} with seeds {seeds}")
    print(f"{'='*50}")
    
    # Temporarily override total_timesteps in config for this run
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    original_timesteps = config_data['rl']['total_timesteps']
    original_seed = config_data['rl']['seed']
    original_model_path = config_data['paths']['model_save_path']
    original_log_path = config_data['paths']['tensorboard_log']
    
    for seed in seeds:
        exp_id = f"{experiment_prefix}_seed{seed}"
        print(f"\n---> Training {exp_id} for {timesteps} timesteps...")
        
        # Override config
        config_data['rl']['total_timesteps'] = timesteps
        config_data['rl']['seed'] = seed
        config_data['paths']['model_save_path'] = f"experiments/{exp_id}/models"
        config_data['paths']['tensorboard_log'] = f"experiments/{exp_id}/logs"
        
        # Create directory and save metadata
        exp_dir = f"experiments/{exp_id}"
        os.makedirs(exp_dir, exist_ok=True)
        
        tmp_config = f"configs/tmp_{exp_id}.yaml"
        with open(tmp_config, 'w') as f:
            yaml.dump(config_data, f)
            
        import shutil
        shutil.copy(tmp_config, f"{exp_dir}/config.yaml")
        
        metadata = {
            "experiment_id": exp_id,
            "seed": seed,
            "algorithm": "SAC+HER",
            "environment": "PandaPickPlaceEnv",
            "training_timesteps": timesteps,
            "git_commit": "unavailable"
        }
        import json
        with open(f"{exp_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
            
        try:
            subprocess.run([python_exe, "-m", "src.rl.train", "--config", tmp_config], check=True)
            print(f"[{exp_id}] Training Complete.")
        except subprocess.CalledProcessError:
            print(f"[{exp_id}] Training FAILED.")
            sys.exit(1)
        finally:
            if os.path.exists(tmp_config):
                os.remove(tmp_config)

def main():
    seeds = [42, 43, 44, 45, 46]
    timesteps = 100000
    
    batches = [
        # ("Batch 1", "configs/pick_place_baseline.yaml", "pnp_dr0"),
        ("Batch 2", "configs/pick_place_dr_position.yaml", "pnp_dr1"),
        # ("Batch 3", "configs/pick_place_dr_position_target.yaml", "pnp_dr2"),
        # ("Batch 4", "configs/pick_place_dr_pose.yaml", "pnp_dr3"),
        # ("Batch 5", "configs/pick_place_dr_physics.yaml", "pnp_dr4"),
    ]
    
    for b_name, b_config, b_prefix in batches:
        run_batch(b_name, b_config, b_prefix, seeds, timesteps)
        print(f"\n{b_name} complete. Please verify metrics before proceeding to the next batch.")
        break # Enforce stopping after batch 1 as per instructions

if __name__ == "__main__":
    main()
