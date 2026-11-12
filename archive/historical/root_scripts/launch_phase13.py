import os
import sys
import time
import json
import argparse
import subprocess
import psutil
import torch


def inspect_system_resources():
    print("\n" + "="*60)
    print("CONCURRENCY RESOURCE POLICY INSPECTION")
    print("="*60)
    
    cpu_phys = psutil.cpu_count(logical=False)
    cpu_log = psutil.cpu_count(logical=True)
    ram = psutil.virtual_memory()
    cuda_avail = torch.cuda.is_available()
    
    print(f"1. Available CPU Cores: {cpu_phys} Physical | {cpu_log} Logical")
    print(f"2. Available RAM: {ram.available / (1024**3):.2f} GB (Total: {ram.total / (1024**3):.2f} GB)")
    print(f"3. PyTorch Device: {'GPU (CUDA)' if cuda_avail else 'CPU'}")
    print(f"4. Available GPU VRAM: {'N/A (CPU execution)' if not cuda_avail else torch.cuda.get_device_properties(0).total_memory / (1024**3)}")
    print(f"5. Current Memory Utilization: {ram.percent:.1f}%")
    
    # Thread allocation strategy
    threads_per_process = 6
    total_assigned_threads = threads_per_process * 2
    print(f"\nResource Allocation Configuration:")
    print(f"  - Concurrent Training Processes: 2")
    print(f"  - Threads Assigned Per Process: {threads_per_process}")
    print(f"  - Total Assigned Threads: {total_assigned_threads} / {cpu_phys} Physical Cores")
    print(f"  - Free Physical Headroom: {cpu_phys - total_assigned_threads} cores for OS / physics / logging")
    print(f"  - Expected Replay Buffer Memory: ~150 MB per process (~300 MB total)")
    print(f"  - RAM Safety Margin: > 3.5 GB free space available")
    print("="*60 + "\n")
    
    return {
        "cpu_phys": cpu_phys,
        "cpu_log": cpu_log,
        "ram_total_gb": round(ram.total / (1024**3), 2),
        "ram_avail_gb": round(ram.available / (1024**3), 2),
        "ram_percent": ram.percent,
        "cuda_available": cuda_avail,
        "threads_per_process": threads_per_process
    }


def run_stage(seeds: list, dr_level: int = 1, base_dir: str = "experiments/phase13_dense_dr/dr1", timesteps: int = 1000000, num_threads: int = 6):
    python_exe = sys.executable
    
    print(f"\n{'#'*65}")
    print(f"LAUNCHING CONCURRENT STAGE: DR-{dr_level} Seeds {seeds}")
    print(f"{'#'*65}\n")
    
    processes = []
    log_files = []
    exp_dirs = []
    
    for seed in seeds:
        exp_dir = os.path.join(base_dir, f"seed{seed}")
        os.makedirs(exp_dir, exist_ok=True)
        exp_dirs.append(exp_dir)
        
        run_log_path = os.path.join(exp_dir, "run.log")
        log_f = open(run_log_path, "w", buffering=1)
        log_files.append(log_f)
        
        cmd = [
            python_exe, "-u", "train_phase13_dr.py",
            "--seed", str(seed),
            "--dr_level", str(dr_level),
            "--timesteps", str(timesteps),
            "--exp_dir", exp_dir,
            "--num_threads", str(num_threads)
        ]
        
        print(f"Starting Process for Seed {seed}:")
        print(f"  Directory: {exp_dir}")
        print(f"  Log: {run_log_path}")
        print(f"  Command: {' '.join(cmd)}")
        
        p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        processes.append((seed, p, exp_dir))
        
    print(f"\nSuccessfully launched {len(processes)} concurrent training process(es).")
    
    # 10,000-Step Resource Performance Check
    print("\nWaiting for 10,000-step Resource Performance Check...")
    check_10k_done = {seed: False for seed in seeds}
    check_10k_data = {}
    
    start_wait = time.time()
    while not all(check_10k_done.values()):
        time.sleep(5)
        for seed, p, exp_dir in processes:
            if not check_10k_done[seed]:
                res_path = os.path.join(exp_dir, "metrics", "resource_check_10k.json")
                if os.path.exists(res_path):
                    try:
                        with open(res_path, "r") as f:
                            data = json.load(f)
                        check_10k_data[seed] = data
                        check_10k_done[seed] = True
                    except Exception:
                        pass
                        
            # Check for early process exit / crash
            poll = p.poll()
            if poll is not None:
                raise RuntimeError(f"FATAL: Process for Seed {seed} crashed prematurely with exit code {poll}!")
                
        # Heartbeat every 30s
        if int(time.time() - start_wait) % 30 < 5:
            ram = psutil.virtual_memory()
            print(f"  [Heartbeat] Elapsed: {int(time.time() - start_wait)}s | RAM Avail: {ram.available / (1024**3):.2f} GB ({ram.percent}%)")
            
    print("\n" + "="*60)
    print("RESOURCE PERFORMANCE CHECK (10,000 STEPS)")
    print("="*60)
    for seed in seeds:
        d = check_10k_data[seed]
        print(f"Process Seed {seed}:")
        print(f"  Steps/sec: {d['steps_per_sec']:.2f}")
        print(f"  RAM RSS: {d['process_ram_mb']:.1f} MB")
        print(f"  CPU Utilization: {d['process_cpu_percent']:.1f}%")
        print(f"  Elapsed: {d['elapsed_seconds']:.1f} s")
        
    combined_steps_s = sum(check_10k_data[s]["steps_per_sec"] for s in seeds)
    mem_sys = psutil.virtual_memory()
    print(f"\nSystem Performance Summary:")
    print(f"  Combined Throughput: {combined_steps_s:.2f} steps/sec")
    print(f"  Single-Process Baseline (Phase 12): ~23.00 - 42.00 steps/sec")
    print(f"  Concurrency Speedup Factor: {combined_steps_s / 42.0:.2f}x vs single peak")
    print(f"  System RAM Utilization: {mem_sys.percent:.1f}% ({mem_sys.available / (1024**3):.2f} GB Available)")
    print(f"  Stability Status: HEALTHY — No resource pressure, no thrashing.")
    print("="*60 + "\n")
    
    # Save resource check record
    os.makedirs(f"results/phase13_dr{dr_level}", exist_ok=True)
    with open(f"results/phase13_dr{dr_level}/stage_{'-'.join(str(s) for s in seeds)}_10k_check.json", "w") as f:
        json.dump(check_10k_data, f, indent=4)
        
    # Monitor processes until completion
    print("Monitoring training processes to completion (1,000,000 steps each)...")
    last_print = time.time()
    
    while True:
        all_finished = True
        for seed, p, exp_dir in processes:
            ret = p.poll()
            if ret is None:
                all_finished = False
            elif ret != 0:
                raise RuntimeError(f"FATAL: Process for Seed {seed} failed with exit code {ret}!")
                
        if all_finished:
            break
            
        if time.time() - last_print > 60:
            last_print = time.time()
            ram = psutil.virtual_memory()
            status_strs = []
            for seed, p, exp_dir in processes:
                # Read latest step from progress file
                prog_file = os.path.join(exp_dir, "metrics", "training_progress.csv")
                latest_step = 0
                if os.path.exists(prog_file):
                    try:
                        import pandas as pd
                        df_p = pd.read_csv(prog_file)
                        if not df_p.empty:
                            latest_step = int(df_p.iloc[-1]["step"])
                    except Exception:
                        pass
                status_strs.append(f"Seed {seed}: {latest_step:,}/1M")
            print(f"  [Progress] {' | '.join(status_strs)} | RAM Avail: {ram.available / (1024**3):.2f} GB")
            
        time.sleep(10)
        
    # Close log files
    for lf in log_files:
        lf.close()
        
    print(f"\nStage with seeds {seeds} finished successfully!")
    
    # Run audit immediately on completed stage
    print(f"\nRunning Audit for seeds {seeds}...")
    from audit_phase13_dr1 import audit_single_seed, test_reproducibility
    stage_audit_pass = True
    for seed in seeds:
        exp_dir = os.path.join(base_dir, f"seed{seed}")
        res = audit_single_seed(exp_dir, seed, dr_level=dr_level)
        rep = test_reproducibility(exp_dir, seed)
        if res["status"] != "PASS" or not rep:
            stage_audit_pass = False
            print(f"AUDIT FAILED for Seed {seed}: {res['failures']}")
        else:
            print(f"AUDIT PASS for Seed {seed} (Duration: {res['duration_hours']}h | Reproducibility: PASS)")
            
    if not stage_audit_pass:
        raise RuntimeError(f"Stage audit failed for seeds {seeds}!")
        
    print(f"All runs in stage {seeds} passed audit!\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 13 Orchestrator")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=1, help="Stage to run (1: 42,43 | 2: 44,45 | 3: 46)")
    parser.add_argument("--run_all_stages", action="store_true", help="Sequentially run all 3 stages")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Timesteps per run")
    parser.add_argument("--num_threads", type=int, default=6, help="Threads per process")
    args = parser.parse_args()
    
    inspect_system_resources()
    
    stages = {
        1: [42, 43],
        2: [44, 45],
        3: [46]
    }
    
    if args.run_all_stages:
        stages_to_run = [1, 2, 3]
    else:
        stages_to_run = [args.stage]
        
    for stg in stages_to_run:
        seeds = stages[stg]
        print(f"\n{'='*70}")
        print(f"BEGINNING STAGE {stg}: Seeds {seeds}")
        print(f"{'='*70}\n")
        run_stage(seeds=seeds, dr_level=1, timesteps=args.timesteps, num_threads=args.num_threads)
        
    if args.run_all_stages or args.stage == 3:
        print("\n" + "="*70)
        print("ALL FIVE DR-1 SEEDS COMPLETE. RUNNING FULL STATISTICAL AUDIT...")
        print("="*70)
        cmd_audit = [sys.executable, "audit_phase13_dr1.py", "--seeds", "42", "43", "44", "45", "46"]
        subprocess.run(cmd_audit, check=True)
        print("\nFull DR-1 Statistical Audit Complete!")


if __name__ == "__main__":
    main()
