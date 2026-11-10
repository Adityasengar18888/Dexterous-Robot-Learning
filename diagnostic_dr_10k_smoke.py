import os
import sys
import subprocess
import yaml
import glob

def run_10k_smoke():
    python_exe = ".conda-env/python"
    
    config_file = "configs/pick_place_baseline.yaml"
    exp_id = "pnp_dr0_smoke_seed42"
    timesteps = 10000
    seed = 42
    
    print(f"\n{'='*50}")
    print(f"Starting 10k Smoke Test: {exp_id}")
    print(f"{'='*50}")
    
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
        
    config_data['rl']['total_timesteps'] = timesteps
    config_data['rl']['seed'] = seed
    
    # Enable evaluation callback
    config_data['eval']['eval_freq'] = 2500
    config_data['eval']['n_eval_episodes'] = 5
    
    config_data['paths']['model_save_path'] = f"experiments/{exp_id}/models"
    config_data['paths']['tensorboard_log'] = f"experiments/{exp_id}/logs"
    
    tmp_config = f"configs/tmp_{exp_id}.yaml"
    with open(tmp_config, 'w') as f:
        yaml.dump(config_data, f)
        
    # Also save metadata as requested
    os.makedirs(f"experiments/{exp_id}", exist_ok=True)
    metadata = {
        "experiment_id": exp_id,
        "seed": seed,
        "algorithm": "SAC+HER",
        "environment": "PandaPickPlaceEnv",
        "training_timesteps": timesteps
    }
    with open(f"experiments/{exp_id}/metadata.json", "w") as f:
        import json
        json.dump(metadata, f, indent=4)
        
    import shutil
    shutil.copy(tmp_config, f"experiments/{exp_id}/config.yaml")

    try:
        subprocess.run([python_exe, "-m", "src.rl.train", "--config", tmp_config], check=True)
        print(f"[{exp_id}] Training Complete.")
    except subprocess.CalledProcessError:
        print(f"[{exp_id}] Training FAILED.")
        sys.exit(1)
    finally:
        if os.path.exists(tmp_config):
            os.remove(tmp_config)
            
    # Verification
    print("Verifying artifacts...")
    model_path = f"experiments/{exp_id}/models/final_model.zip"
    best_model = f"experiments/{exp_id}/models/best_model.zip"
    logs = glob.glob(f"experiments/{exp_id}/logs/*")
    
    assert os.path.exists(model_path), "Final model missing!"
    assert os.path.exists(best_model), "Best model missing!"
    assert len(logs) > 0, "Tensorboard logs missing!"
    
    print("All artifacts verified. Smoke test PASS.")

if __name__ == "__main__":
    run_10k_smoke()
