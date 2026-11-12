import os
import sys
import time
import json
import yaml
import numpy as np
import pandas as pd
import pybullet as p
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from stable_baselines3 import SAC

from src.environments.dense_pick_place_env import PandaPickPlaceDenseRewardEnv


def get_step_telemetry(env):
    """
    Extracts physical ground-truth contact, dynamics, and kinematic state from PyBullet.
    Does not invent or fabricate physical measurements.
    """
    cid = env.sim.client_id
    robot_id = env.controller.robot.body_id
    cube_id = env.cube.body_id
    
    # 1. Contact points and forces
    contacts = p.getContactPoints(bodyA=robot_id, bodyB=cube_id, physicsClientId=cid)
    f9_contacts = [c for c in contacts if c[3] == 9]
    f10_contacts = [c for c in contacts if c[3] == 10]
    
    f9_force = float(sum(c[9] for c in f9_contacts))
    f10_force = float(sum(c[9] for c in f10_contacts))
    total_force = f9_force + f10_force
    
    # Lateral friction forces if available (indices 10 and 12 in getContactPoints)
    f9_lateral = float(sum(c[10] + c[12] for c in f9_contacts))
    f10_lateral = float(sum(c[10] + c[12] for c in f10_contacts))
    total_lateral = f9_lateral + f10_lateral
    
    contact_symmetry = bool(len(f9_contacts) > 0 and len(f10_contacts) > 0)
    has_contact = bool(len(contacts) > 0)
    
    # 2. Velocities and Positions
    cube_pos, cube_orn = p.getBasePositionAndOrientation(cube_id, physicsClientId=cid)
    cube_vel, cube_ang_vel = p.getBaseVelocity(cube_id, physicsClientId=cid)
    ee_pos = env.controller.get_ee_position()
    gripper_width = env.controller.robot.get_gripper_width()
    
    rel_dist = float(np.linalg.norm(np.array(ee_pos) - np.array(cube_pos)))
    
    return {
        "ee_pos": np.array(ee_pos, dtype=np.float32),
        "cube_pos": np.array(cube_pos, dtype=np.float32),
        "cube_orn": np.array(cube_orn, dtype=np.float32),
        "cube_vel": np.array(cube_vel, dtype=np.float32),
        "cube_ang_vel": np.array(cube_ang_vel, dtype=np.float32),
        "gripper_width": float(gripper_width),
        "contact_count": len(contacts),
        "f9_contact_count": len(f9_contacts),
        "f10_contact_count": len(f10_contacts),
        "f9_normal_force": f9_force,
        "f10_normal_force": f10_force,
        "total_normal_force": total_force,
        "total_lateral_force": total_lateral,
        "contact_symmetry": contact_symmetry,
        "has_contact": has_contact,
        "rel_dist": rel_dist,
        "cube_z": float(cube_pos[2]),
        "cube_vz": float(cube_vel[2])
    }


def run_episode(env, model, seed: int, track_trajectory: bool = False, mass_override: float = None, friction_override: float = None, yaw_override: float = None):
    obs, info = env.reset(seed=seed)
    cid = env.sim.client_id
    cube_id = env.cube.body_id
    
    # Controlled single-variable overrides
    if mass_override is not None:
        p.changeDynamics(cube_id, -1, mass=float(mass_override), physicsClientId=cid)
    if friction_override is not None:
        p.changeDynamics(cube_id, -1, lateralFriction=float(friction_override), physicsClientId=cid)
    if yaw_override is not None:
        pos, _ = p.getBasePositionAndOrientation(cube_id, physicsClientId=cid)
        orn = p.getQuaternionFromEuler([0, 0, float(yaw_override)])
        p.resetBasePositionAndOrientation(cube_id, pos, orn, physicsClientId=cid)
        obs = env._get_obs() # update observation with modified state
        
    initial_obj_pos = obs["achieved_goal"].copy()
    initial_cube_z = float(initial_obj_pos[2])
    
    done = False
    step = 0
    current_hold = 0
    max_hold = 0
    grasped = False
    lifted = False
    placed = False
    dropped = False
    
    min_ee_obj_dist = 1e6
    max_cube_z = initial_cube_z
    max_vz = 0.0
    max_normal_force = 0.0
    total_slip_events = 0
    time_to_lift = None
    
    trajectory_steps = []
    
    # Baseline telemetry
    t0_data = get_step_telemetry(env)
    if track_trajectory:
        trajectory_steps.append({"step": 0, "action": [0,0,0,1], **t0_data})
        
    prev_rel_dist = t0_data["rel_dist"]
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info_step = env.step(action)
        done = term or trunc
        step += 1
        
        telem = get_step_telemetry(env)
        if track_trajectory:
            trajectory_steps.append({"step": step, "action": [float(a) for a in action], **telem})
            
        ee_pos = telem["ee_pos"]
        cube_pos = telem["cube_pos"]
        ee_obj_dist = telem["rel_dist"]
        cube_z = telem["cube_z"]
        cube_vz = telem["cube_vz"]
        norm_force = telem["total_normal_force"]
        
        if ee_obj_dist < min_ee_obj_dist:
            min_ee_obj_dist = ee_obj_dist
        if cube_z > max_cube_z:
            max_cube_z = cube_z
        if cube_vz > max_vz:
            max_vz = cube_vz
        if norm_force > max_normal_force:
            max_normal_force = norm_force
            
        # Physical hold definition
        is_held = (ee_obj_dist < 0.05) and (action[3] < 0.0) and (telem["has_contact"] or telem["gripper_width"] < 0.05)
        is_lifted_step = (cube_z > 0.05)
        
        # Slip event detection: gripper is closed, but relative distance increases
        if action[3] < 0.0 and grasped and (ee_obj_dist - prev_rel_dist > 0.005) and not is_lifted_step:
            total_slip_events += 1
        prev_rel_dist = ee_obj_dist
        
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
            if not lifted:
                lifted = True
                time_to_lift = step
                
        if info_step.get("is_success", False):
            placed = True
            
    final_dist = float(info_step.get("transport_dist", np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])))
    reached = min_ee_obj_dist < 0.05
    transported = final_dist < 0.05
    timeout = (not placed) and (step >= env.unwrapped.max_episode_steps)
    
    # Rigorous Mutually Exclusive Failure Taxonomy
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
        
    ep_summary = {
        "seed": seed,
        "success": placed,
        "reached": reached,
        "grasped": grasped,
        "lifted": lifted,
        "transported": transported,
        "placed": placed,
        "final_dist": final_dist,
        "min_ee_obj_dist": min_ee_obj_dist,
        "max_cube_z": max_cube_z,
        "vertical_displacement": max_cube_z - initial_cube_z,
        "max_vz": max_vz,
        "max_normal_force": max_normal_force,
        "hold_duration": max_hold,
        "time_to_lift": time_to_lift if time_to_lift is not None else -1,
        "total_slip_events": total_slip_events,
        "dropped": dropped,
        "timeout": timeout,
        "ep_length": step,
        "failure_type": failure_type
    }
    
    return ep_summary, trajectory_steps


def d1_reproduction(seeds=[42, 43], num_episodes=100):
    print("\n" + "="*70)
    print("STEP 1 — D1: BASELINE REPRODUCTION (DR-4 Held-out Stress)")
    print("="*70)
    
    with open("configs/pick_place_heldout.yaml", "r") as f:
        cfg = yaml.safe_load(f)["env"]
    kwargs = cfg.copy()
    if "name" in kwargs: del kwargs["name"]
    if "randomizer" in kwargs: kwargs["randomizer_config"] = kwargs.pop("randomizer")
    
    env = PandaPickPlaceDenseRewardEnv(**kwargs)
    
    d1_dir = "results/phase13_5/d1_baseline_reproduction"
    os.makedirs(d1_dir, exist_ok=True)
    
    all_rows = []
    
    for s in seeds:
        model_path = f"experiments/phase13_dense_dr/dr4/seed{s}/models/checkpoint_500000.zip"
        print(f"\nLoading Model Seed {s} from {model_path}...")
        model = SAC.load(model_path, env=env, device="cpu")
        
        ep_records = []
        for ep in range(num_episodes):
            eval_seed = 1001 + ep
            rec, _ = run_episode(env, model, seed=eval_seed)
            rec["training_seed"] = s
            ep_records.append(rec)
            
        df_s = pd.DataFrame(ep_records)
        df_s.to_csv(f"{d1_dir}/seed{s}_reproduction.csv", index=False)
        all_rows.append(df_s)
        
        print(f"Seed {s} Reproduction Results ({num_episodes} eps):")
        print(f"  Success:   {df_s['success'].mean()*100:.1f}%")
        print(f"  Reach:     {df_s['reached'].mean()*100:.1f}%")
        print(f"  Grasp:     {df_s['grasped'].mean()*100:.1f}%")
        print(f"  Lift:      {df_s['lifted'].mean()*100:.1f}%")
        print(f"  Transport: {df_s['transported'].mean()*100:.1f}%")
        print(f"  FinalDist: {df_s['final_dist'].mean():.4f} m")
        print(f"  Taxonomy:  {df_s['failure_type'].value_counts().to_dict()}")
        
    env.close()
    
    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.to_csv(f"{d1_dir}/combined_reproduction.csv", index=False)
    
    mean_succ = df_all["success"].mean() * 100
    reach_rate = df_all["reached"].mean() * 100
    grasp_rate = df_all["grasped"].mean() * 100
    lift_rate = df_all["lifted"].mean() * 100
    trans_rate = df_all["transported"].mean() * 100
    tax_counts = df_all["failure_type"].value_counts(normalize=True).to_dict()
    
    print("\n" + "-"*60)
    print("D1 REPRODUCTION SUMMARY VS PHASE 13 REPORT:")
    print(f"  Metric              Phase 13 Report    Reproduction (200 eps)")
    print(f"  Success:            61.0%              {mean_succ:.1f}%")
    print(f"  Reach:              96.5%              {reach_rate:.1f}%")
    print(f"  Grasp:              92.0%              {grasp_rate:.1f}%")
    print(f"  Lift:               61.5%              {lift_rate:.1f}%")
    print(f"  Transport:          57.0%              {trans_rate:.1f}%")
    print(f"  GRASP_BUT_NO_LIFT:  30.5%              {tax_counts.get('GRASP_BUT_NO_LIFT', 0.0)*100:.1f}%")
    print(f"  NO_REACH:           3.5%               {tax_counts.get('NO_REACH', 0.0)*100:.1f}%")
    print(f"  REACH_BUT_NO_GRASP: 4.5%               {tax_counts.get('REACH_BUT_NO_GRASP', 0.0)*100:.1f}%")
    print(f"  WRONG_DIRECTION:    0.5%               {tax_counts.get('LIFT_AND_HELD_BUT_WRONG_DIRECTION', 0.0)*100:.1f}%")
    print("-"*60 + "\n")
    return df_all


def d2_mass_sensitivity(seeds=[42, 43], num_episodes=50):
    print("\n" + "="*70)
    print("STEP 2 — D2: MASS SENSITIVITY (Controlled Evaluation)")
    print("="*70)
    
    # Controlled environment: fixed friction = 1.0, positions randomized across common seeds
    # only mass varies independently across [0.10, 0.15, 0.20, 0.25] kg
    env = PandaPickPlaceDenseRewardEnv(
        distance_threshold=0.05,
        max_episode_steps=100,
        action_scale=0.05,
        randomizer_config={
            "enabled": True,
            "object_position": {"enabled": True},
            "target_position": {"enabled": True}
        }
    )
    
    d2_dir = "results/phase13_5/d2_mass_sensitivity"
    os.makedirs(d2_dir, exist_ok=True)
    
    mass_levels = [0.10, 0.15, 0.20, 0.25]
    all_records = []
    
    for s in seeds:
        model_path = f"experiments/phase13_dense_dr/dr4/seed{s}/models/checkpoint_500000.zip"
        model = SAC.load(model_path, env=env, device="cpu")
        
        for m in mass_levels:
            for ep in range(num_episodes):
                eval_seed = 2001 + ep
                rec, _ = run_episode(env, model, seed=eval_seed, mass_override=m, friction_override=1.0)
                rec["mass"] = m
                rec["training_seed"] = s
                all_records.append(rec)
                
    env.close()
    
    df_d2 = pd.DataFrame(all_records)
    df_d2.to_csv(f"{d2_dir}/mass_sensitivity.csv", index=False)
    
    summary = df_d2.groupby("mass").agg(
        success_rate=("success", "mean"),
        reach_rate=("reached", "mean"),
        grasp_rate=("grasped", "mean"),
        lift_rate=("lifted", "mean"),
        mean_lift_height=("max_cube_z", "mean"),
        mean_vert_disp=("vertical_displacement", "mean"),
        mean_slip_events=("total_slip_events", "mean"),
        mean_max_vz=("max_vz", "mean"),
        mean_normal_force=("max_normal_force", "mean")
    ).reset_index()
    summary.to_csv(f"{d2_dir}/mass_summary.csv", index=False)
    
    print("\nD2 Mass Sensitivity Results:")
    print(summary.to_string(index=False))
    return df_d2


def d3_friction_sensitivity(seeds=[42, 43], num_episodes=50):
    print("\n" + "="*70)
    print("STEP 3 — D3: FRICTION SENSITIVITY (Controlled Evaluation)")
    print("="*70)
    
    env = PandaPickPlaceDenseRewardEnv(
        distance_threshold=0.05,
        max_episode_steps=100,
        action_scale=0.05,
        randomizer_config={
            "enabled": True,
            "object_position": {"enabled": True},
            "target_position": {"enabled": True}
        }
    )
    
    d3_dir = "results/phase13_5/d3_friction_sensitivity"
    os.makedirs(d3_dir, exist_ok=True)
    
    friction_levels = [0.3, 0.5, 0.8, 1.0, 1.5]
    all_records = []
    
    for s in seeds:
        model_path = f"experiments/phase13_dense_dr/dr4/seed{s}/models/checkpoint_500000.zip"
        model = SAC.load(model_path, env=env, device="cpu")
        
        for f in friction_levels:
            for ep in range(num_episodes):
                eval_seed = 3001 + ep
                rec, _ = run_episode(env, model, seed=eval_seed, mass_override=0.10, friction_override=f)
                rec["friction"] = f
                rec["training_seed"] = s
                all_records.append(rec)
                
    env.close()
    
    df_d3 = pd.DataFrame(all_records)
    df_d3.to_csv(f"{d3_dir}/friction_sensitivity.csv", index=False)
    
    summary = df_d3.groupby("friction").agg(
        success_rate=("success", "mean"),
        grasp_rate=("grasped", "mean"),
        lift_rate=("lifted", "mean"),
        mean_hold_duration=("hold_duration", "mean"),
        mean_slip_events=("total_slip_events", "mean"),
        mean_lift_height=("max_cube_z", "mean"),
        drop_rate=("dropped", "mean")
    ).reset_index()
    summary.to_csv(f"{d3_dir}/friction_summary.csv", index=False)
    
    print("\nD3 Friction Sensitivity Results:")
    print(summary.to_string(index=False))
    return df_d3


def d4_contact_geometry(seeds=[42, 43], num_episodes=30):
    print("\n" + "="*70)
    print("STEP 4 — D4: CONTACT GEOMETRY & YAW ORIENTATION (Controlled Evaluation)")
    print("="*70)
    
    env = PandaPickPlaceDenseRewardEnv(
        distance_threshold=0.05,
        max_episode_steps=100,
        action_scale=0.05,
        randomizer_config={
            "enabled": True,
            "object_position": {"enabled": True},
            "target_position": {"enabled": True}
        }
    )
    
    d4_dir = "results/phase13_5/d4_contact_geometry"
    os.makedirs(d4_dir, exist_ok=True)
    
    yaw_angles_deg = [0, 15, 30, 45, 60, 75, 90]
    all_records = []
    
    for s in seeds:
        model_path = f"experiments/phase13_dense_dr/dr4/seed{s}/models/checkpoint_500000.zip"
        model = SAC.load(model_path, env=env, device="cpu")
        
        for deg in yaw_angles_deg:
            rad = np.radians(deg)
            for ep in range(num_episodes):
                eval_seed = 4001 + ep
                rec, _ = run_episode(env, model, seed=eval_seed, mass_override=0.10, friction_override=1.0, yaw_override=rad)
                rec["yaw_deg"] = deg
                rec["yaw_rad"] = rad
                rec["training_seed"] = s
                all_records.append(rec)
                
    env.close()
    
    df_d4 = pd.DataFrame(all_records)
    df_d4.to_csv(f"{d4_dir}/contact_geometry.csv", index=False)
    
    summary = df_d4.groupby("yaw_deg").agg(
        success_rate=("success", "mean"),
        reach_rate=("reached", "mean"),
        grasp_rate=("grasped", "mean"),
        lift_rate=("lifted", "mean"),
        mean_normal_force=("max_normal_force", "mean"),
        mean_min_dist=("min_ee_obj_dist", "mean"),
        mean_hold_duration=("hold_duration", "mean"),
        mean_slip_events=("total_slip_events", "mean")
    ).reset_index()
    summary.to_csv(f"{d4_dir}/contact_geometry_summary.csv", index=False)
    
    print("\nD4 Contact Geometry & Yaw Results:")
    print(summary.to_string(index=False))
    return df_d4


def d5_trajectory_analysis(seeds=[42, 43]):
    print("\n" + "="*70)
    print("STEP 5 — D5: LIFT TRAJECTORY TELEMETRY & PLOTTING")
    print("="*70)
    
    with open("configs/pick_place_heldout.yaml", "r") as f:
        cfg = yaml.safe_load(f)["env"]
    kwargs = cfg.copy()
    if "name" in kwargs: del kwargs["name"]
    if "randomizer" in kwargs: kwargs["randomizer_config"] = kwargs.pop("randomizer")
    
    env = PandaPickPlaceDenseRewardEnv(**kwargs)
    
    d5_dir = "results/phase13_5/d5_lift_trajectories"
    plots_dir = "results/phase13_5/plots"
    os.makedirs(d5_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Run episodes and collect sample successful and failed trajectories
    trajectories = {"SUCCESS": [], "GRASP_BUT_NO_LIFT": []}
    
    for s in seeds:
        model_path = f"experiments/phase13_dense_dr/dr4/seed{s}/models/checkpoint_500000.zip"
        model = SAC.load(model_path, env=env, device="cpu")
        
        for ep in range(50):
            eval_seed = 5001 + ep
            rec, traj = run_episode(env, model, seed=eval_seed, track_trajectory=True)
            ft = rec["failure_type"]
            if ft in trajectories and len(trajectories[ft]) < 3:
                trajectories[ft].append({
                    "seed": s,
                    "eval_seed": eval_seed,
                    "summary": rec,
                    "steps": traj
                })
            if len(trajectories["SUCCESS"]) >= 3 and len(trajectories["GRASP_BUT_NO_LIFT"]) >= 3:
                break
        if len(trajectories["SUCCESS"]) >= 3 and len(trajectories["GRASP_BUT_NO_LIFT"]) >= 3:
            break
            
    env.close()
    
    # Save trajectory CSVs
    for category in ["SUCCESS", "GRASP_BUT_NO_LIFT"]:
        for idx, t in enumerate(trajectories[category]):
            df_traj = pd.DataFrame(t["steps"])
            # Flatten array fields for CSV storage
            for col in ["ee_pos", "cube_pos", "cube_orn", "cube_vel", "action"]:
                if col in df_traj.columns:
                    arr = np.array(df_traj[col].tolist())
                    for dim in range(arr.shape[1]):
                        df_traj[f"{col}_{dim}"] = arr[:, dim]
                    df_traj.drop(columns=[col], inplace=True)
            df_traj.to_csv(f"{d5_dir}/{category.lower()}_sample_{idx+1}_seed{t['seed']}_ep{t['eval_seed']}.csv", index=False)
            
    # Plotting Representative Successful vs Failed Trajectories
    print("Generating Comparative Telemetry Plots...")
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    
    # Pick sample 1 of success and sample 1 of failure
    s_traj = trajectories["SUCCESS"][0]["steps"]
    f_traj = trajectories["GRASP_BUT_NO_LIFT"][0]["steps"]
    
    s_steps = [x["step"] for x in s_traj]
    f_steps = [x["step"] for x in f_traj]
    
    # 1. Altitude (Z)
    axes[0, 0].plot(s_steps, [x["ee_pos"][2] for x in s_traj], "b-", label="End-Effector Z")
    axes[0, 0].plot(s_steps, [x["cube_z"] for x in s_traj], "g-", label="Cube Z")
    axes[0, 0].axhline(y=0.05, color="r", linestyle="--", label="Lift Threshold (0.05m)")
    axes[0, 0].set_title(f"Successful Episode (Seed {trajectories['SUCCESS'][0]['seed']}): Altitude Profile")
    axes[0, 0].set_ylabel("Altitude (m)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(f_steps, [x["ee_pos"][2] for x in f_traj], "b-", label="End-Effector Z")
    axes[0, 1].plot(f_steps, [x["cube_z"] for x in f_traj], "r-", label="Cube Z (Slip)")
    axes[0, 1].axhline(y=0.05, color="r", linestyle="--", label="Lift Threshold (0.05m)")
    axes[0, 1].set_title(f"Failed Episode (GRASP_BUT_NO_LIFT, Seed {trajectories['GRASP_BUT_NO_LIFT'][0]['seed']}): Altitude Profile")
    axes[0, 1].set_ylabel("Altitude (m)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 2. Contact Normal Forces
    axes[1, 0].plot(s_steps, [x["f9_normal_force"] for x in s_traj], "c-", label="Finger 9 Force")
    axes[1, 0].plot(s_steps, [x["f10_normal_force"] for x in s_traj], "m-", label="Finger 10 Force")
    axes[1, 0].set_title("Successful Episode: Contact Normal Force")
    axes[1, 0].set_ylabel("Force (N)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(f_steps, [x["f9_normal_force"] for x in f_traj], "c-", label="Finger 9 Force")
    axes[1, 1].plot(f_steps, [x["f10_normal_force"] for x in f_traj], "m-", label="Finger 10 Force")
    axes[1, 1].set_title("Failed Episode: Contact Normal Force (Drop / Loss of Contact)")
    axes[1, 1].set_ylabel("Force (N)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 3. Relative EE-Cube Distance (Slip Signature)
    axes[2, 0].plot(s_steps, [x["rel_dist"] for x in s_traj], "g-", label="EE-Cube Distance")
    axes[2, 0].axhline(y=0.05, color="gray", linestyle=":", label="Grasp Sphere (0.05m)")
    axes[2, 0].set_title("Successful Episode: Relative Distance (Stable Grasp)")
    axes[2, 0].set_xlabel("Episode Step")
    axes[2, 0].set_ylabel("Distance (m)")
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].plot(f_steps, [x["rel_dist"] for x in f_traj], "r-", label="EE-Cube Distance (Divergence)")
    axes[2, 1].axhline(y=0.05, color="gray", linestyle=":", label="Grasp Sphere (0.05m)")
    axes[2, 1].set_title("Failed Episode: Relative Distance (Slip Onset at Vertical Lift)")
    axes[2, 1].set_xlabel("Episode Step")
    axes[2, 1].set_ylabel("Distance (m)")
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plot_path = f"{plots_dir}/lift_trajectory_comparison.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {plot_path}")
    
    return trajectories


def main():
    print("="*70)
    print("STARTING PHASE 13.5: OOD GRASP & LIFT FAILURE DIAGNOSTICS")
    print("="*70)
    
    # 1. Baseline reproduction
    df_d1 = d1_reproduction(seeds=[42, 43], num_episodes=100)
    
    # 2. Mass sensitivity
    df_d2 = d2_mass_sensitivity(seeds=[42, 43], num_episodes=50)
    
    # 3. Friction sensitivity
    df_d3 = d3_friction_sensitivity(seeds=[42, 43], num_episodes=50)
    
    # 4. Contact geometry
    df_d4 = d4_contact_geometry(seeds=[42, 43], num_episodes=30)
    
    # 5. Lift trajectory telemetry
    trajs = d5_trajectory_analysis(seeds=[42, 43])
    
    print("\n" + "="*70)
    print("ALL 5 DIAGNOSTIC EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*70)

if __name__ == "__main__":
    main()

