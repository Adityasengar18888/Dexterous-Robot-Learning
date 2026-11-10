"""Phase 14.3 — Ultra Fast Track: Spatial Grasp Acquisition + Dual-Stage Adaptive Control.

Evaluates DualStageGraspLiftController (Stage 1: Spatial grasp acquisition near workspace
boundaries; Stage 2: Phase 14-Lite C_SLIP_AWARE lifting after bilateral contact) against
A_BASELINE on DR4 Seed 43 and Seed 42 (20 Held-out DR4 episodes per seed, plus 20-episode
DR0 & Matched DR4 non-regression verification).
"""

from __future__ import annotations

import os
import time
import yaml
import psutil
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from stable_baselines3 import SAC

from src.environments.dense_pick_place_env import PandaPickPlaceDenseRewardEnv
from src.rl.adaptive_controllers import (
    ControllerConfig,
    BaselineController,
    SlipAwareLiftController,
    DualStageGraspLiftController,
    get_step_telemetry,
)
from run_phase14_lite import DR4_RANDOMIZER_CONFIG, run_single_episode, aggregate_metrics


def run_eval_regime(env, model, controller, eval_seeds, training_seed: int, regime_name: str):
    rows = []
    edge_corr_counts = []
    lift_block_counts = []
    bilateral_grasp_eps = 0

    for es in eval_seeds:
        obs, _ = env.reset(seed=int(es))
        controller.reset()
        t0 = get_step_telemetry(env)
        initial_obj_z = float(t0["cube_z"])
        prev_rel_dist = float(t0["rel_dist"])
        dt = (float(env.unwrapped.control_steps) + 10.0) / 240.0
        prev_ee_z = float(t0["ee_z"])
        prev_ee_vz = 0.0
        prev_ee_az = 0.0
        prev_cmd_az = 0.0
        prev_cmd_daz = 0.0

        done = False
        step = 0
        total_reward = 0.0
        current_hold = 0
        max_hold = 0
        grasped = False
        lifted = False
        dropped = False
        first_grasp_step = None
        time_to_lift = None
        time_to_first_success = None
        min_ee_obj_dist = float(t0["rel_dist"])
        max_cube_z = initial_obj_z
        max_normal_force = 0.0
        had_bilateral_contact = False
        slip_events = 0
        ctrl_slip_triggers = 0
        edge_corr_steps = 0
        lift_block_steps = 0
        rel_dists = []
        peak_ee_accel = 0.0
        peak_ee_jerk = 0.0
        peak_cmd_accel = 0.0
        peak_cmd_jerk = 0.0
        info_step = {}

        while not done:
            telem_before = get_step_telemetry(env)
            raw_action, _ = model.predict(obs, deterministic=True)
            cmd_action, ctrl_info = controller.step_action(raw_action, telem_before)

            if ctrl_info.get("slip_detected", False):
                ctrl_slip_triggers += 1
            if ctrl_info.get("edge_correction", False):
                edge_corr_steps += 1
            if ctrl_info.get("lift_blocked_pre_contact", False):
                lift_block_steps += 1

            cmd_daz = float(cmd_action[2]) - prev_cmd_az
            cmd_jerk = cmd_daz - prev_cmd_daz
            peak_cmd_accel = max(peak_cmd_accel, abs(cmd_daz))
            peak_cmd_jerk = max(peak_cmd_jerk, abs(cmd_jerk))
            prev_cmd_daz = cmd_daz
            prev_cmd_az = float(cmd_action[2])

            obs, reward, terminated, truncated, info_step = env.step(cmd_action)
            done = terminated or truncated
            step += 1
            total_reward += float(reward)

            telem_after = get_step_telemetry(env)
            ee_z = float(telem_after["ee_z"])
            ee_vz = (ee_z - prev_ee_z) / dt
            ee_az = (ee_vz - prev_ee_vz) / dt
            ee_jk = (ee_az - prev_ee_az) / dt
            if step >= 2:
                peak_ee_accel = max(peak_ee_accel, abs(ee_az))
            if step >= 3:
                peak_ee_jerk = max(peak_ee_jerk, abs(ee_jk))
            prev_ee_z, prev_ee_vz, prev_ee_az = ee_z, ee_vz, ee_az

            rel_dist = float(telem_after["rel_dist"])
            rel_dists.append(rel_dist)
            cube_z = float(telem_after["cube_z"])
            norm_force = float(telem_after["total_normal_force"])

            min_ee_obj_dist = min(min_ee_obj_dist, rel_dist)
            max_cube_z = max(max_cube_z, cube_z)
            max_normal_force = max(max_normal_force, norm_force)
            if telem_after["contact_symmetry"] and norm_force >= 10.0:
                had_bilateral_contact = True

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
                max_hold = max(max_hold, current_hold)
            else:
                if max_hold > 5:
                    dropped = True
                current_hold = 0

            if is_lifted_step and is_held and not lifted:
                lifted = True
                time_to_lift = step

            if info_step.get("is_success", False) and time_to_first_success is None:
                time_to_first_success = step

        final_dist = float(info_step.get("transport_dist", np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])))
        reached = bool(min_ee_obj_dist < 0.05)
        transported = bool(final_dist < 0.05)
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

        # Workspace-edge approach error: failed to establish true bilateral contact (nf < 10N) before lift
        edge_approach_error = bool((not had_bilateral_contact) and (not final_success))
        if had_bilateral_contact:
            bilateral_grasp_eps += 1

        edge_corr_counts.append(edge_corr_steps)
        lift_block_counts.append(lift_block_steps)

        rows.append({
            "training_seed": training_seed,
            "regime": regime_name,
            "controller": controller.name,
            "eval_seed": int(es),
            "reach_success": reached,
            "grasp_success": grasped,
            "bilateral_grasp": had_bilateral_contact,
            "lift_success": lifted,
            "transport_success": transported,
            "placement_success": final_success,
            "success": final_success,
            "final_dist": final_dist,
            "min_ee_obj_dist": min_ee_obj_dist,
            "mean_rel_dist": float(np.mean(rel_dists)),
            "max_cube_z": max_cube_z,
            "max_normal_force": max_normal_force,
            "slip_events": slip_events,
            "ctrl_slip_triggers": ctrl_slip_triggers,
            "edge_correction_steps": edge_corr_steps,
            "lift_blocked_steps": lift_block_steps,
            "edge_approach_error": edge_approach_error,
            "peak_vert_accel": peak_ee_accel,
            "peak_vert_jerk": peak_ee_jerk,
            "peak_cmd_vert_accel": peak_cmd_accel,
            "peak_cmd_vert_jerk": peak_cmd_jerk,
            "lift_duration": (time_to_lift - first_grasp_step) if (time_to_lift and first_grasp_step) else -1,
            "time_to_first_success": time_to_first_success if time_to_first_success else -1,
            "timeout": timeout,
            "ep_length": step,
            "episode_reward": total_reward,
            "failure_type": failure_type,
        })
    return pd.DataFrame(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-seed", type=int, default=None, help="Run 100-ep D_DUAL_STAGE confirmation for one seed (42 or 43)")
    parser.add_argument("--merge-confirm", action="store_true", help="Merge 100-ep confirmation results with cached Phase 14-Lite A_BASELINE")
    args = parser.parse_args()

    t_start = time.time()
    proc = psutil.Process(os.getpid())
    out_dir = "results/phase14_3"
    os.makedirs(out_dir, exist_ok=True)

    with open("configs/pick_place_heldout.yaml", "r", encoding="utf-8") as f:
        h_cfg = yaml.safe_load(f)["env"]
    h_kw = {k: v for k, v in h_cfg.items() if k != "name"}
    if "randomizer" in h_kw:
        h_kw["randomizer_config"] = h_kw.pop("randomizer")

    cfg = ControllerConfig()

    if args.merge_confirm:
        d42 = pd.read_csv(os.path.join(out_dir, "confirm_d_seed42.csv"))
        d43 = pd.read_csv(os.path.join(out_dir, "confirm_d_seed43.csv"))
        df_d = pd.concat([d42, d43], ignore_index=True)

        # Load cached Phase 14-Lite A_BASELINE (100 episodes per seed x 3 regimes)
        a42 = pd.read_csv("results/phase14_lite/seed42_results.csv")
        a43 = pd.read_csv("results/phase14_lite/seed43_results.csv")
        df_a = pd.concat([a42[a42["controller"] == "A_BASELINE"], a43[a43["controller"] == "A_BASELINE"]], ignore_index=True)
        # Align columns cleanly
        for col in ["bilateral_grasp", "edge_approach_error"]:
            if col not in df_a.columns:
                df_a[col] = (df_a["max_normal_force"] >= 10.0) if col == "bilateral_grasp" else ((df_a["max_normal_force"] < 10.0) & (~df_a["success"]))
        for col in ["edge_correction_steps", "lift_blocked_steps"]:
            if col not in df_a.columns:
                df_a[col] = 0

        df_comb = pd.concat([df_a, df_d], ignore_index=True)
        df_comb.to_csv(os.path.join(out_dir, "confirmation_episodes.csv"), index=False)

        sum_rows = []
        for (s, reg, ctrl), sub in df_comb.groupby(["training_seed", "regime", "controller"], sort=False):
            n_grasp = int(sub["grasp_success"].sum())
            n_lift = int(sub["lift_success"].sum())
            sum_rows.append({
                "training_seed": s,
                "regime": reg,
                "controller": ctrl,
                "episodes": len(sub),
                "success_rate": sub["success"].mean(),
                "reach_rate": sub["reach_success"].mean(),
                "grasp_rate": sub["grasp_success"].mean(),
                "bilateral_grasp_rate": sub["bilateral_grasp"].mean(),
                "lift_rate": sub["lift_success"].mean(),
                "grasp_to_lift_conversion": float(n_lift / n_grasp) if n_grasp > 0 else 0.0,
                "grasp_but_no_lift_rate": (sub["failure_type"] == "GRASP_BUT_NO_LIFT").mean(),
                "no_reach_rate": (sub["failure_type"] == "NO_REACH").mean(),
                "edge_approach_error_rate": sub["edge_approach_error"].mean(),
                "mean_slip_events": sub["slip_events"].mean(),
                "mean_steps_to_success": sub[sub["time_to_first_success"] >= 0]["time_to_first_success"].mean() if (sub["time_to_first_success"] >= 0).any() else -1.0,
                "timeout_rate": sub["timeout"].mean(),
                "mean_reward": sub["episode_reward"].mean(),
            })
        df_sum = pd.DataFrame(sum_rows)
        df_sum.to_csv(os.path.join(out_dir, "confirmation_summary.csv"), index=False)
        print("Merged 100-episode confirmation summary:\n" + df_sum.to_string(index=False))
        return

    if args.confirm_seed is not None:
        s = int(args.confirm_seed)
        env_held = PandaPickPlaceDenseRewardEnv(**h_kw)
        env_dr0 = PandaPickPlaceDenseRewardEnv(distance_threshold=0.05, max_episode_steps=100, action_scale=0.05, randomizer_config={"enabled": False})
        env_match = PandaPickPlaceDenseRewardEnv(distance_threshold=0.05, max_episode_steps=100, action_scale=0.05, randomizer_config=DR4_RANDOMIZER_CONFIG)
        eval_seeds_100 = list(range(1001, 1101))
        cp_path = f"experiments/phase13_dense_dr/dr4/seed{s}/models/checkpoint_500000.zip"
        model = SAC.load(cp_path, env=env_held, device="cpu")
        ctrl = DualStageGraspLiftController(cfg)

        parts = []
        for rname, env_r in [("Heldout_DR4", env_held), ("Nominal_DR0", env_dr0), ("Matched_DR4", env_match)]:
            df_r = run_eval_regime(env_r, model, ctrl, eval_seeds_100, s, rname)
            parts.append(df_r)
            g2l = (df_r["lift_success"].sum() / max(df_r["grasp_success"].sum(), 1)) * 100.0
            print(
                f"  [CONFIRM N=100] Seed {s} | {rname:11s} | D_DUAL_STAGE | Succ={df_r['success'].mean()*100:5.1f}% | "
                f"Reach={df_r['reach_success'].mean()*100:5.1f}% | Grasp={df_r['grasp_success'].mean()*100:5.1f}% | "
                f"Bilateral={df_r['bilateral_grasp'].mean()*100:5.1f}% | G2L={g2l:5.1f}% | "
                f"GBNL={(df_r['failure_type']=='GRASP_BUT_NO_LIFT').mean()*100:4.1f}%",
                flush=True,
            )
        pd.concat(parts, ignore_index=True).to_csv(os.path.join(out_dir, f"confirm_d_seed{s}.csv"), index=False)
        env_held.close(); env_dr0.close(); env_match.close()
        elapsed = time.time() - t_start
        rss_mb = proc.memory_info().rss / (1024 * 1024)
        print(f"Seed {s} 100-ep confirmation finished in {elapsed:.1f}s | RAM: {rss_mb:.1f} MB", flush=True)
        return

    env_held = PandaPickPlaceDenseRewardEnv(**h_kw)
    env_dr0 = PandaPickPlaceDenseRewardEnv(distance_threshold=0.05, max_episode_steps=100, action_scale=0.05, randomizer_config={"enabled": False})
    env_match = PandaPickPlaceDenseRewardEnv(distance_threshold=0.05, max_episode_steps=100, action_scale=0.05, randomizer_config=DR4_RANDOMIZER_CONFIG)

    eval_seeds_20 = list(range(1001, 1021))

    all_dfs = []
    for s in [43, 42]:
        cp_path = f"experiments/phase13_dense_dr/dr4/seed{s}/models/checkpoint_500000.zip"
        model = SAC.load(cp_path, env=env_held, device="cpu")
        print(f"\n=== Evaluating DR4 Seed {s} (20 Heldout DR4 + 20 Nominal DR0 + 20 Matched DR4) ===", flush=True)

        for cname, ctrl in [
            ("A_BASELINE", BaselineController()),
            ("D_DUAL_STAGE", DualStageGraspLiftController(cfg)),
        ]:
            # Heldout DR4 (primary target: 20 eps per seed)
            df_h = run_eval_regime(env_held, model, ctrl, eval_seeds_20, s, "Heldout_DR4")
            all_dfs.append(df_h)
            g2l = (df_h["lift_success"].sum() / max(df_h["grasp_success"].sum(), 1)) * 100.0
            print(
                f"  Seed {s} | Heldout_DR4 | {cname:12s} | Succ={df_h['success'].mean()*100:5.1f}% | "
                f"Reach={df_h['reach_success'].mean()*100:5.1f}% | Grasp={df_h['grasp_success'].mean()*100:5.1f}% | "
                f"Bilateral={df_h['bilateral_grasp'].mean()*100:5.1f}% | G2L={g2l:5.1f}% | "
                f"GBNL={(df_h['failure_type']=='GRASP_BUT_NO_LIFT').mean()*100:4.1f}% | "
                f"EdgeErr={df_h['edge_approach_error'].mean()*100:4.1f}%",
                flush=True,
            )

            # Non-regression check on DR0 and Matched DR4 (20 eps each)
            for rname, env_r in [("Nominal_DR0", env_dr0), ("Matched_DR4", env_match)]:
                df_r = run_eval_regime(env_r, model, ctrl, eval_seeds_20, s, rname)
                all_dfs.append(df_r)
                print(f"  Seed {s} | {rname:11s} | {cname:12s} | Succ={df_r['success'].mean()*100:5.1f}%", flush=True)

    env_held.close()
    env_dr0.close()
    env_match.close()

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(os.path.join(out_dir, "phase14_3_episodes.csv"), index=False)

    # Summary table
    summary_rows = []
    for (s, reg, ctrl), sub in df_all.groupby(["training_seed", "regime", "controller"], sort=False):
        n_grasp = int(sub["grasp_success"].sum())
        n_lift = int(sub["lift_success"].sum())
        summary_rows.append({
            "training_seed": s,
            "regime": reg,
            "controller": ctrl,
            "episodes": len(sub),
            "success_rate": sub["success"].mean(),
            "reach_rate": sub["reach_success"].mean(),
            "grasp_rate": sub["grasp_success"].mean(),
            "bilateral_grasp_rate": sub["bilateral_grasp"].mean(),
            "lift_rate": sub["lift_success"].mean(),
            "grasp_to_lift_conversion": float(n_lift / n_grasp) if n_grasp > 0 else 0.0,
            "grasp_but_no_lift_rate": (sub["failure_type"] == "GRASP_BUT_NO_LIFT").mean(),
            "no_reach_rate": (sub["failure_type"] == "NO_REACH").mean(),
            "edge_approach_error_rate": sub["edge_approach_error"].mean(),
            "mean_edge_corr_steps": sub["edge_correction_steps"].mean(),
            "mean_slip_events": sub["slip_events"].mean(),
            "mean_steps_to_success": sub[sub["time_to_first_success"] >= 0]["time_to_first_success"].mean() if (sub["time_to_first_success"] >= 0).any() else -1.0,
            "timeout_rate": sub["timeout"].mean(),
            "mean_reward": sub["episode_reward"].mean(),
        })
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(os.path.join(out_dir, "phase14_3_summary.csv"), index=False)

    elapsed = time.time() - t_start
    rss_mb = proc.memory_info().rss / (1024 * 1024)
    print(f"\nPhase 14.3 screening completed in {elapsed:.1f}s | Process RAM: {rss_mb:.1f} MB", flush=True)


if __name__ == "__main__":
    main()

