import os
import sys
import time
import json
import yaml
import argparse
import subprocess
import psutil
import torch
import pandas as pd

from audit_fast_track_dr1 import analyze_seed_eval


def inspect_system_resources():
    print("\n" + "="*60)
    print("PHASE 13 FAST-TRACK CONCURRENCY RESOURCE INSPECTION")
    print("="*60)
    
    cpu_phys = psutil.cpu_count(logical=False)
    cpu_log = psutil.cpu_count(logical=True)
    ram = psutil.virtual_memory()
    cuda_avail = torch.cuda.is_available()
    
    print(f"1. Available CPU Cores: {cpu_phys} Physical | {cpu_log} Logical")
    print(f"2. Available RAM: {ram.available / (1024**3):.2f} GB (Total: {ram.total / (1024**3):.2f} GB)")
    print(f"3. PyTorch Device: {'GPU (CUDA)' if cuda_avail else 'CPU'}")
    print(f"4. System RAM Utilization: {ram.percent:.1f}%")
    
    threads_per_process = 6
    total_assigned_threads = threads_per_process * 2
    print(f"\nResource Allocation Configuration:")
    print(f"  - Max Concurrent Training Processes: 2")
    print(f"  - Threads Assigned Per Process: {threads_per_process}")
    print(f"  - Total Assigned Threads: {total_assigned_threads} / {cpu_phys} Physical Cores")
    print(f"  - Free Physical Headroom: {cpu_phys - total_assigned_threads} cores")
    print("="*60 + "\n")
    return threads_per_process


def run_benchmark_stage(dr_level: int, seeds: list = [42, 43], timesteps: int = 500000, num_threads: int = 6):
    python_exe = sys.executable
    base_dir = f"experiments/phase13_dense_dr/dr{dr_level}"
    os.makedirs(base_dir, exist_ok=True)
    
    print(f"\n{'#'*65}")
    print(f"LAUNCHING FAST-TRACK BENCHMARK: DR-{dr_level} Seeds {seeds} ({timesteps:,} steps)")
    print(f"{'#'*65}\n")
    
    processes = []
    log_files = []
    
    checkpoints = [250000, 500000] if timesteps == 500000 else [timesteps]
    
    for seed in seeds:
        exp_dir = os.path.join(base_dir, f"seed{seed}")
        os.makedirs(exp_dir, exist_ok=True)
        
        run_log_path = os.path.join(exp_dir, "run.log")
        log_f = open(run_log_path, "w", buffering=1)
        log_files.append(log_f)
        
        cmd = [
            python_exe, "-u", "train_phase13_dr.py",
            "--seed", str(seed),
            "--dr_level", str(dr_level),
            "--timesteps", str(timesteps),
            "--exp_dir", exp_dir,
            "--num_threads", str(num_threads),
            "--checkpoints"
        ] + [str(cp) for cp in checkpoints]
        
        print(f"Starting Process for DR-{dr_level} Seed {seed}:")
        print(f"  Directory: {exp_dir}")
        print(f"  Log: {run_log_path}")
        print(f"  Command: {' '.join(cmd)}")
        
        p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        processes.append((seed, p, exp_dir))
        
    print(f"\nSuccessfully launched {len(processes)} concurrent training process(es).")
    
    # 10,000-Step Resource Check
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
            poll = p.poll()
            if poll is not None:
                raise RuntimeError(f"FATAL: Process for Seed {seed} crashed prematurely with exit code {poll}!")
                
        if int(time.time() - start_wait) % 30 < 5:
            ram = psutil.virtual_memory()
            print(f"  [Heartbeat] Elapsed: {int(time.time() - start_wait)}s | RAM Avail: {ram.available / (1024**3):.2f} GB ({ram.percent}%)")
            
    print("\n" + "="*60)
    print(f"DR-{dr_level} RESOURCE PERFORMANCE CHECK (10,000 STEPS)")
    print("="*60)
    for seed in seeds:
        d = check_10k_data[seed]
        print(f"Seed {seed}: {d['steps_per_sec']:.2f} steps/s | RAM: {d['process_ram_mb']:.1f} MB | CPU: {d['process_cpu_percent']:.1f}%")
        
    res_dir = f"results/phase13_dr{dr_level}"
    os.makedirs(res_dir, exist_ok=True)
    with open(f"{res_dir}/stage_10k_check.json", "w") as f:
        json.dump(check_10k_data, f, indent=4)
        
    # Monitor to completion (500k)
    print(f"\nMonitoring DR-{dr_level} training to completion ({timesteps:,} steps each)...")
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
                prog_file = os.path.join(exp_dir, "metrics", "training_progress.csv")
                latest_step = 0
                if os.path.exists(prog_file):
                    try:
                        df_p = pd.read_csv(prog_file)
                        if not df_p.empty:
                            latest_step = int(df_p.iloc[-1]["step"])
                    except Exception:
                        pass
                status_strs.append(f"Seed {seed}: {latest_step:,}/{timesteps:,}")
            print(f"  [Progress DR-{dr_level}] {' | '.join(status_strs)} | RAM Avail: {ram.available / (1024**3):.2f} GB")
            
        time.sleep(10)
        
    for lf in log_files:
        lf.close()
        
    print(f"\nDR-{dr_level} training completed successfully for seeds {seeds}!\n")
    
    # Evaluate Gate
    matched_regime = f"Matched_DR{dr_level}"
    gate_passed = True
    failing_seeds = []
    
    print("="*60)
    print(f"EVALUATING GATE FOR DR-{dr_level} (THRESHOLD: >= 80% MATCHED SUCCESS)")
    print("="*60)
    for seed in seeds:
        exp_dir = os.path.join(base_dir, f"seed{seed}")
        eval_csv = os.path.join(exp_dir, "evaluations", f"eval_checkpoint_{timesteps}.csv")
        df_eval = pd.read_csv(eval_csv)
        
        # 30-episode screening
        sub_screen = df_eval[df_eval["regime"] == matched_regime].head(30)
        screen_succ = sub_screen["success"].mean()
        
        # 100-episode full
        sub_full = df_eval[df_eval["regime"] == matched_regime]
        full_succ = sub_full["success"].mean()
        
        print(f"Seed {seed}: Screening (30 eps): {screen_succ*100:.1f}% | Full (100 eps): {full_succ*100:.1f}%")
        if screen_succ < 0.80:
            gate_passed = False
            failing_seeds.append(seed)
            
    if gate_passed:
        print(f"\n>> DR-{dr_level} GATE PASSED! Both seeds achieved >= 80% matched success.")
        print(f">> Condition ACCEPTED for fast-track benchmark.")
    else:
        print(f"\n>> DR-{dr_level} GATE FAILED for seeds: {failing_seeds}")
        print(f">> Extending failing seed(s) to 750k...")
        for fseed in failing_seeds:
            extend_seed(dr_level=dr_level, seed=fseed, target_timesteps=750000, num_threads=num_threads)
            
    return gate_passed


def extend_seed(dr_level: int, seed: int, target_timesteps: int = 750000, num_threads: int = 6):
    python_exe = sys.executable
    base_dir = f"experiments/phase13_dense_dr/dr{dr_level}"
    exp_dir = os.path.join(base_dir, f"seed{seed}")
    resume_cp = os.path.join(exp_dir, "models", "checkpoint_500000.zip")
    
    print(f"\nExtending DR-{dr_level} Seed {seed} to {target_timesteps:,} steps...")
    run_log_path = os.path.join(exp_dir, "run_extension.log")
    with open(run_log_path, "w", buffering=1) as log_f:
        cmd = [
            python_exe, "-u", "train_phase13_dr.py",
            "--seed", str(seed),
            "--dr_level", str(dr_level),
            "--timesteps", str(target_timesteps),
            "--exp_dir", exp_dir,
            "--num_threads", str(num_threads),
            "--checkpoints", str(target_timesteps),
            "--resume_checkpoint", resume_cp
        ]
        p = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        if p.returncode != 0:
            raise RuntimeError(f"Extension for Seed {seed} failed with code {p.returncode}!")
            
    print(f"Extension complete for DR-{dr_level} Seed {seed}!")


def main():
    parser = argparse.ArgumentParser(description="Phase 13 Fast-Track Benchmark Runner")
    parser.add_argument("--dr_level", type=int, choices=[2, 4], default=2, help="DR Level to train")
    parser.add_argument("--run_all", action="store_true", help="Run DR2 then DR4 sequentially")
    parser.add_argument("--timesteps", type=int, default=500000, help="Initial timesteps budget")
    parser.add_argument("--num_threads", type=int, default=6, help="CPU threads per process")
    args = parser.parse_args()
    
    num_threads = inspect_system_resources()
    
    if args.run_all:
        dr_levels = [2, 4]
    else:
        dr_levels = [args.dr_level]
        
    for dr in dr_levels:
        run_benchmark_stage(dr_level=dr, seeds=[42, 43], timesteps=args.timesteps, num_threads=num_threads)
        
    print("\n" + "="*70)
    print("PHASE 13 FAST-TRACK TRAINING RUNS FINISHED.")
    print("PRODUCING FINAL REPORT AND AWAITING MANUAL APPROVAL.")
    print("="*70 + "\n")
    
    from generate_fast_track_report import generate_report
    generate_report()


if __name__ == "__main__":
    main()

