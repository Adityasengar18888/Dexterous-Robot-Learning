"""Phase 14-Lite: Adaptive Lift Stabilization Evaluation Runner.

Evaluates three controller variants (A: Baseline, B: Acceleration Smoothing,
C: Slip-Aware Lifting) on unmodified DR4 Seed 42 and Seed 43 SAC checkpoints
across Nominal DR0, Matched DR4, and Held-out DR4 regimes.
"""

from __future__ import annotations

import os
import sys
import time
import json
import yaml
import hashlib
import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from stable_baselines3 import SAC

from src.environments.dense_pick_place_env import PandaPickPlaceDenseRewardEnv
from src.rl.adaptive_controllers import (
    ControllerConfig,
    BaselineController,
    AccelerationSmoothingController,
    SlipAwareLiftController,
    get_step_telemetry,
)


DR4_RANDOMIZER_CONFIG = {
    "enabled": True,
    "object_position": {"enabled": True},
    "target_position": {"enabled": True},
    "object_orientation": {"enabled": True},
    "mass": {"enabled": True, "min": 0.05, "max": 0.20},
    "friction": {"enabled": True, "min": 0.5, "max": 1.5},
}


def build_eval_envs() -> Dict[str, PandaPickPlaceDenseRewardEnv]:
    """Build the three deterministic evaluation environments matching Phase 13."""
    env_dr0 = PandaPickPlaceDenseRewardEnv(
        distance_threshold=0.05,
        max_episode_steps=100,
        action_scale=0.05,
        randomizer_config={"enabled": False},
    )
    env_matched = PandaPickPlaceDenseRewardEnv(
        distance_threshold=0.05,
        max_episode_steps=100,
        action_scale=0.05,
        randomizer_config=DR4_RANDOMIZER_CONFIG,
    )
    with open("configs/pick_place_heldout.yaml", "r", encoding="utf-8") as f:
        heldout_cfg = yaml.safe_load(f)["env"]
    kw = {k: v for k, v in heldout_cfg.items() if k != "name"}
    if "randomizer" in kw:
        kw["randomizer_config"] = kw.pop("randomizer")
    env_heldout = PandaPickPlaceDenseRewardEnv(**kw)

    return {
        "Nominal_DR0": env_dr0,
        "Matched_DR4": env_matched,
        "Heldout_DR4": env_heldout,
    }


def make_controller(name: str, cfg: ControllerConfig):
    if name == "A_BASELINE":
        return BaselineController()
    if name == "B_ACCEL_SMOOTHING":
        return AccelerationSmoothingController(cfg)
    if name == "C_SLIP_AWARE":
        return SlipAwareLiftController(cfg)
    raise ValueError(f"Unknown controller: {name}")


def run_single_episode(
    env: PandaPickPlaceDenseRewardEnv,
    model: SAC,
    controller,
    eval_seed: int,
    log_step_telemetry: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run one deterministic episode and record comprehensive Rule 4 metrics."""
    obs, _ = env.reset(seed=eval_seed)
    controller.reset()

    # Physical control cycle dt = (15 move + 10 gripper) / 240 Hz
    dt = (float(env.unwrapped.control_steps) + 10.0) / 240.0

    t0 = get_step_telemetry(env)
    initial_obj_z = float(t0["cube_z"])
    prev_rel_dist = float(t0["rel_dist"])
    prev_ee_vz = 0.0
    prev_ee_az_phys = 0.0
    prev_cmd_az = 0.0
    prev_cmd_daz = 0.0
    prev_ee_z = float(t0["ee_z"])

    done = False
    step = 0
    total_reward = 0.0
    current_hold = 0
    max_hold = 0
    grasped = False
    lifted = False
    placed = False
    dropped = False

    first_grasp_step = None
    time_to_lift = None
    time_to_first_success = None

    min_ee_obj_dist = float(t0["rel_dist"])
    max_cube_z = initial_obj_z
    max_normal_force = 0.0
    slip_events = 0
    ctrl_slip_triggers = 0
    rel_dists: List[float] = []

    peak_ee_vert_accel = 0.0
    peak_ee_vert_jerk = 0.0
    peak_cmd_vert_accel = 0.0
    peak_cmd_vert_jerk = 0.0

    step_rows: List[Dict[str, Any]] = []
    info_step: Dict[str, Any] = {}

    while not done:
        telem_before = get_step_telemetry(env)
        raw_action, _ = model.predict(obs, deterministic=True)
        cmd_action, ctrl_info = controller.step_action(raw_action, telem_before)

        if ctrl_info.get("slip_detected", False):
            ctrl_slip_triggers += 1

        # Command-space vertical acceleration and jerk
        cmd_daz = float(cmd_action[2]) - prev_cmd_az
        cmd_jerk = cmd_daz - prev_cmd_daz
        peak_cmd_vert_accel = max(peak_cmd_vert_accel, abs(cmd_daz))
        peak_cmd_vert_jerk = max(peak_cmd_vert_jerk, abs(cmd_jerk))
        prev_cmd_daz = cmd_daz
        prev_cmd_az = float(cmd_action[2])

        obs, reward, terminated, truncated, info_step = env.step(cmd_action)
        done = terminated or truncated
        step += 1
        total_reward += float(reward)

        telem_after = get_step_telemetry(env)
        ee_z = float(telem_after["ee_z"])
        ee_vz = (ee_z - prev_ee_z) / dt
        ee_az_phys = (ee_vz - prev_ee_vz) / dt
        ee_jerk_phys = (ee_az_phys - prev_ee_az_phys) / dt
        if step >= 2:
            peak_ee_vert_accel = max(peak_ee_vert_accel, abs(ee_az_phys))
        if step >= 3:
            peak_ee_vert_jerk = max(peak_ee_vert_jerk, abs(ee_jerk_phys))
        prev_ee_z = ee_z
        prev_ee_vz = ee_vz
        prev_ee_az_phys = ee_az_phys

        rel_dist = float(telem_after["rel_dist"])
        rel_dists.append(rel_dist)
        cube_z = float(telem_after["cube_z"])
        norm_force = float(telem_after["total_normal_force"])

        if rel_dist < min_ee_obj_dist:
            min_ee_obj_dist = rel_dist
        if cube_z > max_cube_z:
            max_cube_z = cube_z
        if norm_force > max_normal_force:
            max_normal_force = norm_force

        is_held = (
            (rel_dist < 0.05)
            and (float(cmd_action[3]) < 0.0)
            and (telem_after["has_contact"] or telem_after["gripper_width"] < 0.05)
        )
        is_lifted_step = cube_z > 0.05

        if (float(cmd_action[3]) < 0.0) and grasped and (rel_dist - prev_rel_dist > 0.005) and not is_lifted_step:
            slip_events += 1
        prev_rel_dist = rel_dist

        if is_held:
            if not grasped:
                first_grasp_step = step
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
            if time_to_first_success is None:
                time_to_first_success = step

        if log_step_telemetry:
            step_rows.append(
                {
                    "eval_seed": eval_seed,
                    "step": step,
                    "ee_z": ee_z,
                    "cube_z": cube_z,
                    "cube_vz": float(telem_after["cube_vz"]),
                    "rel_dist": rel_dist,
                    "gripper_width": float(telem_after["gripper_width"]),
                    "f9_normal_force": float(telem_after["f9_normal_force"]),
                    "f10_normal_force": float(telem_after["f10_normal_force"]),
                    "total_normal_force": norm_force,
                    "contact_symmetry": bool(telem_after["contact_symmetry"]),
                    "raw_az": float(raw_action[2]),
                    "cmd_az": float(cmd_action[2]),
                    "raw_grip": float(raw_action[3]),
                    "cmd_grip": float(cmd_action[3]),
                    "ee_vert_accel": float(ee_az_phys),
                    "ee_vert_jerk": float(ee_jerk_phys),
                    "slip_detected": bool(ctrl_info.get("slip_detected", False)),
                    "controller_mode": str(ctrl_info.get("controller_mode", "")),
                }
            )

    final_dist = float(info_step.get("transport_dist", np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])))
    reached = bool(min_ee_obj_dist < 0.05)
    transported = bool(final_dist < 0.05)
    # Use final step success matching Phase 13 evaluation semantics
    final_success = bool(info_step.get("is_success", False))
    timeout = bool((not final_success) and (step >= env.unwrapped.max_episode_steps))

    if final_success:
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

    lift_duration = (time_to_lift - first_grasp_step) if (time_to_lift is not None and first_grasp_step is not None) else -1

    summary = {
        "eval_seed": eval_seed,
        "reach_success": reached,
        "grasp_success": grasped,
        "lift_success": lifted,
        "transport_success": transported,
        "placement_success": final_success,
        "success": final_success,
        "final_dist": final_dist,
        "min_ee_obj_dist": min_ee_obj_dist,
        "mean_rel_dist": float(np.mean(rel_dists)) if rel_dists else min_ee_obj_dist,
        "max_cube_z": max_cube_z,
        "vertical_displacement": max_cube_z - initial_obj_z,
        "max_normal_force": max_normal_force,
        "hold_duration": max_hold,
        "slip_events": slip_events,
        "ctrl_slip_triggers": ctrl_slip_triggers,
        "peak_vert_accel": peak_ee_vert_accel,
        "peak_vert_jerk": peak_ee_vert_jerk,
        "peak_cmd_vert_accel": peak_cmd_vert_accel,
        "peak_cmd_vert_jerk": peak_cmd_vert_jerk,
        "first_grasp_step": first_grasp_step if first_grasp_step is not None else -1,
        "time_to_lift": time_to_lift if time_to_lift is not None else -1,
        "lift_duration": lift_duration,
        "time_to_first_success": time_to_first_success if time_to_first_success is not None else -1,
        "timeout": timeout,
        "ep_length": step,
        "episode_reward": total_reward,
        "failure_type": failure_type,
    }
    return summary, step_rows


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate episode dataframe into Rule 4 comparison summary."""
    groups = ["controller", "regime"]
    if "training_seed" in df.columns:
        groups = ["training_seed"] + groups

    rows = []
    for keys, sub in df.groupby(groups, sort=False):
        if len(groups) == 3:
            seed_val, ctrl, reg = keys
        else:
            seed_val = "COMBINED"
            ctrl, reg = keys

        n_eps = len(sub)
        n_grasp = int(sub["grasp_success"].sum())
        n_lift = int(sub["lift_success"].sum())
        g2l = float(n_lift / n_grasp) if n_grasp > 0 else 0.0

        valid_lift_dur = sub[sub["lift_duration"] >= 0]["lift_duration"]
        valid_succ_step = sub[sub["time_to_first_success"] >= 0]["time_to_first_success"]

        rows.append(
            {
                "training_seed": seed_val,
                "controller": ctrl,
                "regime": reg,
                "episodes": n_eps,
                "reach_rate": float(sub["reach_success"].mean()),
                "grasp_rate": float(sub["grasp_success"].mean()),
                "lift_rate": float(sub["lift_success"].mean()),
                "transport_rate": float(sub["transport_success"].mean()),
                "placement_rate": float(sub["placement_success"].mean()),
                "success_rate": float(sub["success"].mean()),
                "grasp_to_lift_conversion": g2l,
                "mean_slip_events": float(sub["slip_events"].mean()),
                "mean_ctrl_triggers": float(sub["ctrl_slip_triggers"].mean()),
                "mean_rel_dist": float(sub["mean_rel_dist"].mean()),
                "mean_final_dist": float(sub["final_dist"].mean()),
                "mean_peak_vert_accel": float(sub["peak_vert_accel"].mean()),
                "mean_peak_vert_jerk": float(sub["peak_vert_jerk"].mean()),
                "mean_peak_cmd_accel": float(sub["peak_cmd_vert_accel"].mean()),
                "mean_peak_cmd_jerk": float(sub["peak_cmd_vert_jerk"].mean()),
                "mean_lift_duration": float(valid_lift_dur.mean()) if len(valid_lift_dur) > 0 else -1.0,
                "mean_steps_to_success": float(valid_succ_step.mean()) if len(valid_succ_step) > 0 else -1.0,
                "timeout_rate": float(sub["timeout"].mean()),
                "mean_episode_reward": float(sub["episode_reward"].mean()),
                "grasp_but_no_lift_rate": float((sub["failure_type"] == "GRASP_BUT_NO_LIFT").mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_suite(num_episodes: int, phase_label: str, out_dir: str, seeds: List[int] = [42, 43]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all 3 controllers x 3 regimes x selected seeds for num_episodes."""
    os.makedirs(out_dir, exist_ok=True)
    envs = build_eval_envs()
    cfg = ControllerConfig()
    controllers = ["A_BASELINE", "B_ACCEL_SMOOTHING", "C_SLIP_AWARE"]
    regimes = ["Nominal_DR0", "Matched_DR4", "Heldout_DR4"]

    all_ep_rows: List[Dict[str, Any]] = []
    telemetry_rows: List[Dict[str, Any]] = []

    t_start = time.time()
    for s in seeds:
        cp_path = f"experiments/phase13_dense_dr/dr4/seed{s}/models/checkpoint_500000.zip"
        model = SAC.load(cp_path, env=envs["Heldout_DR4"], device="cpu")
        print(f"\n[{phase_label}] Loaded DR4 Seed {s}: {cp_path}", flush=True)

        for reg in regimes:
            env = envs[reg]
            for cname in controllers:
                ctrl = make_controller(cname, cfg)
                ep_sub = []
                for ep in range(num_episodes):
                    eval_seed = 1001 + ep
                    # Capture step-level lift telemetry for Heldout_DR4 first 10 episodes
                    log_steps = (reg == "Heldout_DR4") and (ep < 10)
                    rec, step_logs = run_single_episode(env, model, ctrl, eval_seed=eval_seed, log_step_telemetry=log_steps)
                    rec["training_seed"] = s
                    rec["regime"] = reg
                    rec["controller"] = cname
                    rec["eval_phase"] = phase_label
                    all_ep_rows.append(rec)
                    ep_sub.append(rec)
                    if log_steps:
                        for sr in step_logs:
                            sr["training_seed"] = s
                            sr["regime"] = reg
                            sr["controller"] = cname
                            sr["eval_phase"] = phase_label
                        telemetry_rows.extend(step_logs)

                df_sub = pd.DataFrame(ep_sub)
                succ_pct = df_sub["success"].mean() * 100.0
                lift_pct = df_sub["lift_success"].mean() * 100.0
                grasp_pct = df_sub["grasp_success"].mean() * 100.0
                g2l_pct = (df_sub["lift_success"].sum() / max(df_sub["grasp_success"].sum(), 1)) * 100.0
                print(
                    f"  Seed {s} | {reg:11s} | {cname:17s} | N={num_episodes:3d} | "
                    f"Succ={succ_pct:5.1f}% | Grasp={grasp_pct:5.1f}% | Lift={lift_pct:5.1f}% | "
                    f"G2L={g2l_pct:5.1f}% | Slip={df_sub['slip_events'].mean():.2f} | "
                    f"PkJerk={df_sub['peak_cmd_vert_jerk'].mean():.2f}",
                    flush=True,
                )

    for env in envs.values():
        env.close()

    df_all = pd.DataFrame(all_ep_rows)
    df_telem = pd.DataFrame(telemetry_rows)

    df_seed42 = df_all[df_all["training_seed"] == 42].copy()
    df_seed43 = df_all[df_all["training_seed"] == 43].copy()

    df_summary_per_seed = aggregate_metrics(df_all)
    # Combined across both seeds
    df_no_seed = df_all.drop(columns=["training_seed"])
    df_summary_comb = aggregate_metrics(df_no_seed)
    df_comparison = pd.concat([df_summary_comb, df_summary_per_seed], ignore_index=True)

    elapsed = time.time() - t_start
    print(f"\n[{phase_label}] Completed in {elapsed:.1f}s.", flush=True)
    return df_all, df_comparison, df_telem


def render_comparison_plots(df_comparison: pd.DataFrame, df_telem: pd.DataFrame, out_dir: str) -> str:
    """Render clean comparison charts using Pillow (no matplotlib backend dependency)."""
    w, h = 1280, 900
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.text((30, 20), "Phase 14-Lite: Adaptive Lift Stabilization Controller Comparison", fill=(0, 0, 0))
    draw.text((30, 42), "Combined DR4 Seeds (42 & 43) | Bar Chart (Top) & Held-Out Peak Command Jerk / Success (Bottom)", fill=(90, 90, 90))

    comb = df_comparison[df_comparison["training_seed"] == "COMBINED"]
    regimes = ["Nominal_DR0", "Matched_DR4", "Heldout_DR4"]
    ctrls = ["A_BASELINE", "B_ACCEL_SMOOTHING", "C_SLIP_AWARE"]
    colors = {"A_BASELINE": (90, 110, 140), "B_ACCEL_SMOOTHING": (30, 140, 210), "C_SLIP_AWARE": (30, 175, 100)}

    # Top-left: Success Rate by Regime & Controller
    def draw_bar_panel(x0, y0, pw, ph, metric_col, title, y_max=100.0, scale=100.0, unit="%"):
        draw.rectangle([x0, y0, x0 + pw, y0 + ph], outline=(200, 200, 200), fill=(250, 250, 250))
        draw.text((x0 + 12, y0 + 8), title, fill=(0, 0, 0))
        pad_l, pad_r, pad_t, pad_b = 55, 20, 35, 38
        gx0, gy0 = x0 + pad_l, y0 + pad_t
        gw, gh = pw - pad_l - pad_r, ph - pad_t - pad_b

        for i in range(5):
            yy = gy0 + gh - int(i * gh / 4)
            val = i * y_max / 4.0
            draw.line([gx0, yy, gx0 + gw, yy], fill=(230, 230, 230))
            draw.text((x0 + 6, yy - 6), f"{val:.0f}{unit}", fill=(110, 110, 110))

        grp_w = gw / len(regimes)
        bar_w = int((grp_w * 0.72) / len(ctrls))
        for ri, reg in enumerate(regimes):
            base_x = gx0 + int(ri * grp_w + grp_w * 0.14)
            draw.text((base_x + 15, gy0 + gh + 8), reg, fill=(30, 30, 30))
            for ci, ctrl in enumerate(ctrls):
                sub = comb[(comb["regime"] == reg) & (comb["controller"] == ctrl)]
                val = float(sub[metric_col].iloc[0]) * scale if len(sub) > 0 else 0.0
                val = max(0.0, val)
                bh = int(min(1.0, val / max(y_max, 1e-6)) * gh)
                bx0 = base_x + ci * (bar_w + 4)
                by0 = gy0 + gh - bh
                draw.rectangle([bx0, by0, bx0 + bar_w, gy0 + gh], fill=colors[ctrl])
                draw.text((bx0 + 2, max(gy0, by0 - 13)), f"{val:.1f}", fill=(0, 0, 0))

    draw_bar_panel(30, 80, 590, 360, "success_rate", "End-to-End Success Rate (%)", y_max=100.0, scale=100.0, unit="%")
    draw_bar_panel(660, 80, 590, 360, "grasp_to_lift_conversion", "Grasp-to-Lift Conversion Rate (%)", y_max=100.0, scale=100.0, unit="%")
    draw_bar_panel(30, 470, 590, 360, "grasp_but_no_lift_rate", "GRASP_BUT_NO_LIFT Failure Rate (%)", y_max=50.0, scale=100.0, unit="%")
    draw_bar_panel(660, 470, 590, 360, "mean_peak_cmd_jerk", "Mean Peak Command Vertical Jerk", y_max=4.0, scale=1.0, unit="")

    # Legend at bottom
    lx = 30
    ly = 855
    for ctrl in ctrls:
        draw.rectangle([lx, ly, lx + 18, ly + 12], fill=colors[ctrl])
        draw.text((lx + 24, ly), ctrl, fill=(0, 0, 0))
        lx += 220

    plot_path = os.path.join(out_dir, "controller_comparison.png")
    img.save(plot_path)
    return plot_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--phase", type=str, default="screening")
    parser.add_argument("--out-dir", type=str, default="results/phase14_lite")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.merge_only:
        s42_path = os.path.join(args.out_dir, "seed42_results.csv")
        s43_path = os.path.join(args.out_dir, "seed43_results.csv")
        df42 = pd.read_csv(s42_path)
        df43 = pd.read_csv(s43_path)
        df_all = pd.concat([df42, df43], ignore_index=True)

        t42_path = os.path.join(args.out_dir, "lift_telemetry_seed42.csv")
        t43_path = os.path.join(args.out_dir, "lift_telemetry_seed43.csv")
        t_parts = []
        if os.path.exists(t42_path):
            t_parts.append(pd.read_csv(t42_path))
        if os.path.exists(t43_path):
            t_parts.append(pd.read_csv(t43_path))
        df_telem = pd.concat(t_parts, ignore_index=True) if t_parts else pd.DataFrame()
        if len(df_telem) > 0:
            df_telem.to_csv(os.path.join(args.out_dir, "lift_telemetry.csv"), index=False)

        df_summary_per_seed = aggregate_metrics(df_all)
        df_summary_comb = aggregate_metrics(df_all.drop(columns=["training_seed"]))
        df_comp = pd.concat([df_summary_comb, df_summary_per_seed], ignore_index=True)
        df_comp.to_csv(os.path.join(args.out_dir, "controller_comparison.csv"), index=False)
        plot_path = render_comparison_plots(df_comp, df_telem, args.out_dir)
        print(f"\nMerged results, comparison CSV, and plot ({plot_path}) into {args.out_dir}/", flush=True)
        return

    df_all, df_comp, df_telem = evaluate_suite(
        num_episodes=args.episodes,
        phase_label=args.phase,
        out_dir=args.out_dir,
        seeds=args.seeds,
    )

    for s in args.seeds:
        df_all[df_all["training_seed"] == s].to_csv(os.path.join(args.out_dir, f"seed{s}_results.csv"), index=False)
        if len(df_telem) > 0:
            df_telem[df_telem["training_seed"] == s].to_csv(os.path.join(args.out_dir, f"lift_telemetry_seed{s}.csv"), index=False)

    if sorted(args.seeds) == [42, 43]:
        df_comp.to_csv(os.path.join(args.out_dir, "controller_comparison.csv"), index=False)
        df_telem.to_csv(os.path.join(args.out_dir, "lift_telemetry.csv"), index=False)
        plot_path = render_comparison_plots(df_comp, df_telem, args.out_dir)
        print(f"\nSaved comparison CSVs and plot ({plot_path}) to {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()

