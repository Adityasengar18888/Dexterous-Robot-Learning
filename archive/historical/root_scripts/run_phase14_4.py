"""Phase 14.4 — Adaptive Post-Lift Transport & Target Alignment Evaluation.

Evaluates E_ADAPTIVE_TRANSPORT against D_DUAL_STAGE (Phase 14.3) and
A_BASELINE (unmodified DR4) across Heldout_DR4, Nominal_DR0, and Matched_DR4.
"""

from __future__ import annotations

import os
import json
import time
import yaml
import psutil
import hashlib
import argparse
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from stable_baselines3 import SAC

from src.environments.dense_pick_place_env import PandaPickPlaceDenseRewardEnv
from src.rl.adaptive_controllers import (
    ControllerConfig,
    BaselineController,
    DualStageGraspLiftController,
    AdaptiveTransportController,
    get_step_telemetry,
)
from run_phase14_lite import DR4_RANDOMIZER_CONFIG


def make_ctrl(name: str, cfg: ControllerConfig):
    if name == "A_BASELINE":
        return BaselineController()
    if name == "D_DUAL_STAGE":
        return DualStageGraspLiftController(cfg)
    if name == "E_ADAPTIVE_TRANSPORT":
        return AdaptiveTransportController(cfg)
    raise ValueError(f"Unknown controller: {name}")


def run_eval_episodes(env, model, controller, eval_seeds: List[int], training_seed: int, regime_name: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for es in eval_seeds:
        obs, _ = env.reset(seed=int(es))
        controller.reset()
        t0 = get_step_telemetry(env)
        prev_rel_dist = float(t0["rel_dist"])
        desired_goal = obs["desired_goal"].copy()

        done = False
        step = 0
        total_reward = 0.0
        current_hold = 0
        max_hold = 0
        grasped = False
        lifted = False
        dropped = False
        had_bilateral = False
        entered_target_ever = False
        time_to_first_success = None
        min_ee_obj_dist = float(t0["rel_dist"])
        max_cube_z = float(t0["cube_z"])
        max_normal_force = 0.0
        slip_events = 0
        transport_aligns: List[float] = []
        nan_detected = False
        info_step: Dict[str, Any] = {}

        while not done:
            telem_before = get_step_telemetry(env)
            raw_action, _ = model.predict(obs, deterministic=True)
            cmd_action, ctrl_info = controller.step_action(raw_action, telem_before)

            if not (np.all(np.isfinite(cmd_action)) and np.all(np.isfinite(obs["observation"]))):
                nan_detected = True

            # Record transport-direction alignment during post-lift transport
            cube_pos_b = telem_before["cube_pos"]
            if cube_pos_b[2] > 0.05:
                gv = desired_goal - cube_pos_b
                gn = float(np.linalg.norm(gv))
                cn = float(np.linalg.norm(cmd_action[:3]))
                if gn > 0.015 and cn > 1e-4:
                    transport_aligns.append(float(np.dot(cmd_action[:3] / cn, gv / gn)))

            obs, reward, terminated, truncated, info_step = env.step(cmd_action)
            done = terminated or truncated
            step += 1
            total_reward += float(reward)

            ta = get_step_telemetry(env)
            rel_dist = float(ta["rel_dist"])
            cube_z = float(ta["cube_z"])
            norm_force = float(ta["total_normal_force"])

            min_ee_obj_dist = min(min_ee_obj_dist, rel_dist)
            max_cube_z = max(max_cube_z, cube_z)
            max_normal_force = max(max_normal_force, norm_force)
            if ta["contact_symmetry"] and norm_force >= 10.0:
                had_bilateral = True

            is_held = (
                (rel_dist < 0.05)
                and (float(cmd_action[3]) < 0.0)
                and (ta["has_contact"] or ta["gripper_width"] < 0.05)
            )
            is_lifted_step = cube_z > 0.05
            if (float(cmd_action[3]) < 0.0) and grasped and (rel_dist - prev_rel_dist > 0.005) and not is_lifted_step:
                slip_events += 1
            prev_rel_dist = rel_dist

            if is_held:
                grasped = True
                current_hold += 1
                max_hold = max(max_hold, current_hold)
            else:
                if max_hold > 5:
                    dropped = True
                current_hold = 0

            if is_lifted_step and (is_held or ta["has_contact"]):
                lifted = True

            cur_goal_dist = float(np.linalg.norm(obs["achieved_goal"] - desired_goal))
            if cur_goal_dist <= 0.05:
                entered_target_ever = True
            if info_step.get("is_success", False) and time_to_first_success is None:
                time_to_first_success = step

        final_dist = float(info_step.get("transport_dist", np.linalg.norm(obs["achieved_goal"] - desired_goal)))
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

        rows.append({
            "training_seed": training_seed,
            "regime": regime_name,
            "controller": controller.name,
            "eval_seed": int(es),
            "reach_success": reached,
            "grasp_success": grasped,
            "bilateral_success": had_bilateral,
            "lift_success": lifted,
            "transport_success": transported,
            "placement_success": final_success,
            "success": final_success,
            "entered_target_ever": entered_target_ever,
            "final_dist": final_dist,
            "mean_transport_alignment": float(np.mean(transport_aligns)) if transport_aligns else 0.0,
            "slip_events": slip_events,
            "dropped": dropped,
            "timeout": timeout,
            "ep_length": step,
            "time_to_first_success": time_to_first_success if time_to_first_success else -1,
            "episode_reward": total_reward,
            "failure_type": failure_type,
            "nan_detected": nan_detected,
        })
    return pd.DataFrame(rows)


def summarize_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
    summaries = []
    for (s, reg, ctrl), sub in df.groupby(["training_seed", "regime", "controller"], sort=False):
        n = len(sub)
        valid_succ = sub[sub["time_to_first_success"] >= 0]["time_to_first_success"]
        summaries.append({
            "training_seed": int(s) if str(s).isdigit() else str(s),
            "regime": reg,
            "controller": ctrl,
            "episodes": n,
            "reach_rate": round(float(sub["reach_success"].mean()), 4),
            "grasp_rate": round(float(sub["grasp_success"].mean()), 4),
            "bilateral_rate": round(float(sub["bilateral_success"].mean()), 4),
            "lift_rate": round(float(sub["lift_success"].mean()), 4),
            "transport_rate": round(float(sub["transport_success"].mean()), 4),
            "placement_rate": round(float(sub["placement_success"].mean()), 4),
            "success_rate": round(float(sub["success"].mean()), 4),
            "mean_final_dist": round(float(sub["final_dist"].mean()), 5),
            "mean_transport_alignment": round(float(sub["mean_transport_alignment"].mean()), 4),
            "wrong_direction_rate": round(float((sub["failure_type"] == "LIFT_AND_HELD_BUT_WRONG_DIRECTION").mean()), 4),
            "dropped_rate": round(float((sub["failure_type"] == "LIFT_BUT_DROPPED").mean()), 4),
            "grasp_but_no_lift_rate": round(float((sub["failure_type"] == "GRASP_BUT_NO_LIFT").mean()), 4),
            "timeout_rate": round(float(sub["timeout"].mean()), 4),
            "mean_reward": round(float(sub["episode_reward"].mean()), 2),
            "mean_steps_to_success": round(float(valid_succ.mean()), 2) if len(valid_succ) > 0 else -1.0,
            "any_nan": bool(sub["nan_detected"].any()),
        })
    return summaries


def build_heldout_env():
    with open("configs/pick_place_heldout.yaml", "r", encoding="utf-8") as f:
        h_cfg = yaml.safe_load(f)["env"]
    h_kw = {k: v for k, v in h_cfg.items() if k != "name"}
    if "randomizer" in h_kw:
        h_kw["randomizer_config"] = h_kw.pop("randomizer")
    return PandaPickPlaceDenseRewardEnv(**h_kw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["screening", "confirm-worker", "merge-confirm"], required=True)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    t_start = time.time()
    proc = psutil.Process(os.getpid())
    out_dir = "results/phase14_4"
    os.makedirs(out_dir, exist_ok=True)
    cfg = ControllerConfig()

    if args.mode == "screening":
        env_held = build_heldout_env()
        eval_seeds = list(range(1001, 1021))
        parts = []
        for s in [43, 42]:
            cp = f"experiments/phase13_dense_dr/dr4/seed{s}/models/checkpoint_500000.zip"
            model = SAC.load(cp, env=env_held, device="cpu")
            for cname in ["A_BASELINE", "D_DUAL_STAGE", "E_ADAPTIVE_TRANSPORT"]:
                ctrl = make_ctrl(cname, cfg)
                df_r = run_eval_episodes(env_held, model, ctrl, eval_seeds, s, "Heldout_DR4")
                parts.append(df_r)
                print(
                    f"[SCREEN N=20] Seed {s} | {cname:20s} | Succ={df_r['success'].mean()*100:5.1f}% | "
                    f"Lift={df_r['lift_success'].mean()*100:5.1f}% | Trans={df_r['transport_success'].mean()*100:5.1f}% | "
                    f"WrongDir={(df_r['failure_type']=='LIFT_AND_HELD_BUT_WRONG_DIRECTION').mean()*100:4.1f}% | "
                    f"Align={df_r['mean_transport_alignment'].mean():.3f} | FinalDist={df_r['final_dist'].mean():.4f}m",
                    flush=True,
                )
        env_held.close()
        df_s = pd.concat(parts, ignore_index=True)
        df_s.to_csv(os.path.join(out_dir, "screening_episodes.csv"), index=False)

        comb = df_s.copy()
        comb["training_seed"] = "COMBINED"
        summaries = summarize_df(df_s) + summarize_df(comb)
        with open(os.path.join(out_dir, "phase14_4_screening_results.json"), "w", encoding="utf-8") as f:
            json.dump({"mode": "screening_N20", "summaries": summaries}, f, indent=4)
        elapsed = time.time() - t_start
        print(f"Screening finished in {elapsed:.1f}s | RAM: {proc.memory_info().rss/(1024*1024):.1f} MB")

    elif args.mode == "confirm-worker":
        s = int(args.seed)
        env_held = build_heldout_env()
        env_dr0 = PandaPickPlaceDenseRewardEnv(distance_threshold=0.05, max_episode_steps=100, action_scale=0.05, randomizer_config={"enabled": False})
        env_match = PandaPickPlaceDenseRewardEnv(distance_threshold=0.05, max_episode_steps=100, action_scale=0.05, randomizer_config=DR4_RANDOMIZER_CONFIG)
        eval_seeds = list(range(1001, 1101))
        cp = f"experiments/phase13_dense_dr/dr4/seed{s}/models/checkpoint_500000.zip"
        model = SAC.load(cp, env=env_held, device="cpu")
        ctrl = make_ctrl("E_ADAPTIVE_TRANSPORT", cfg)

        parts = []
        for rname, env_r in [("Heldout_DR4", env_held), ("Nominal_DR0", env_dr0), ("Matched_DR4", env_match)]:
            df_r = run_eval_episodes(env_r, model, ctrl, eval_seeds, s, rname)
            parts.append(df_r)
            print(
                f"[CONFIRM N=100] Seed {s} | {rname:11s} | E_ADAPTIVE_TRANSPORT | Succ={df_r['success'].mean()*100:5.1f}% | "
                f"Lift={df_r['lift_success'].mean()*100:5.1f}% | WrongDir={(df_r['failure_type']=='LIFT_AND_HELD_BUT_WRONG_DIRECTION').mean()*100:4.1f}% | "
                f"Dropped={(df_r['failure_type']=='LIFT_BUT_DROPPED').mean()*100:4.1f}% | FinalDist={df_r['final_dist'].mean():.4f}m",
                flush=True,
            )
        pd.concat(parts, ignore_index=True).to_csv(os.path.join(out_dir, f"confirm_e_seed{s}.csv"), index=False)
        env_held.close(); env_dr0.close(); env_match.close()
        elapsed = time.time() - t_start
        print(f"Seed {s} E_ADAPTIVE_TRANSPORT confirmation finished in {elapsed:.1f}s | RAM: {proc.memory_info().rss/(1024*1024):.1f} MB")

    elif args.mode == "merge-confirm":
        e42 = pd.read_csv(os.path.join(out_dir, "confirm_e_seed42.csv"))
        e43 = pd.read_csv(os.path.join(out_dir, "confirm_e_seed43.csv"))
        df_e = pd.concat([e42, e43], ignore_index=True)

        # Load cached Phase 14.3 confirmation data (A_BASELINE & D_DUAL_STAGE, N=100 each)
        df_prev = pd.read_csv("results/phase14_3/confirmation_episodes.csv")
        for col in ["bilateral_success", "entered_target_ever", "mean_transport_alignment", "dropped", "nan_detected"]:
            if col not in df_prev.columns:
                if col == "bilateral_success":
                    df_prev[col] = df_prev["bilateral_grasp"]
                elif col == "dropped":
                    df_prev[col] = df_prev["failure_type"] == "LIFT_BUT_DROPPED"
                elif col == "nan_detected":
                    df_prev[col] = False
                elif col == "entered_target_ever":
                    df_prev[col] = df_prev["success"]
                else:
                    df_prev[col] = 0.0

        common_cols = [c for c in df_e.columns if c in df_prev.columns]
        df_all = pd.concat([df_prev[common_cols], df_e[common_cols]], ignore_index=True)
        df_all.to_csv(os.path.join(out_dir, "confirmation_episodes.csv"), index=False)

        comb = df_all.copy()
        comb["training_seed"] = "COMBINED"
        summaries = summarize_df(df_all) + summarize_df(comb)
        with open(os.path.join(out_dir, "phase14_4_confirmation_results.json"), "w", encoding="utf-8") as f:
            json.dump({"mode": "confirmation_N100", "summaries": summaries}, f, indent=4)
        pd.DataFrame(summaries).to_csv(os.path.join(out_dir, "confirmation_summary.csv"), index=False)
        print("Merged Phase 14.4 confirmation summary:\n" + pd.DataFrame(summaries)[["training_seed", "regime", "controller", "episodes", "success_rate", "lift_rate", "wrong_direction_rate", "mean_final_dist", "mean_steps_to_success"]].to_string(index=False))


if __name__ == "__main__":
    main()

