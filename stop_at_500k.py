import time
import os
import psutil

def check_log_for_eval(log_path):
    if not os.path.exists(log_path): return False
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[-50:]:
                if "Training chunk: 250,000 steps (Target: 750,000)" in line:
                    return True
    except Exception as e:
        pass
    return False

def main():
    log44 = "experiments/phase13_dense_dr/dr1/seed44/run.log"
    log45 = "experiments/phase13_dense_dr/dr1/seed45/run.log"
    print("Monitoring Seed 44 and 45 for 500k completion...")
    
    while True:
        if check_log_for_eval(log44) and check_log_for_eval(log45):
            print("Both seeds have completed 500k evaluation. Stopping processes.")
            for p in psutil.process_iter(['name', 'cmdline']):
                try:
                    cmd_list = p.info.get('cmdline', [])
                    if cmd_list is None: continue
                    cmd = " ".join(cmd_list)
                    if 'launch_phase13.py' in cmd or 'train_phase13_dr.py' in cmd:
                        if 'stop_at_500k' not in cmd:
                            print(f"Killing process {p.pid}: {cmd}")
                            p.kill()
                except Exception:
                    pass
            break
        time.sleep(10)

if __name__ == "__main__":
    main()
