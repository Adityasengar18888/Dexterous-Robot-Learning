import subprocess
import sys

def run_smoke_tests():
    configs = [
        "configs/pick_place_baseline.yaml",
        "configs/pick_place_dr_position.yaml",
        "configs/pick_place_dr_position_target.yaml",
        "configs/pick_place_dr_pose.yaml",
        "configs/pick_place_dr_physics.yaml",
    ]
    
    python_exe = ".conda-env/python"
    
    for config in configs:
        print(f"\n{'='*50}\nStarting Smoke Test for: {config}\n{'='*50}")
        try:
            subprocess.run([python_exe, "-m", "src.rl.train", "--config", config], check=True)
            print(f"SUCCESS: {config}")
        except subprocess.CalledProcessError as e:
            print(f"FAILED: {config}")
            sys.exit(1)

if __name__ == "__main__":
    run_smoke_tests()
