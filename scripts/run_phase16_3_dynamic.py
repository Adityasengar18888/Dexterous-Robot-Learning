"""Phase 16.3: Dynamic Object Tracking, Predictive Grasp Positioning, & Replanning Benchmark."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

import numpy as np

from src.environments.dynamic_pick_place_env import PandaDynamicPickPlaceEnv
from src.perception.dynamic_tracker import DynamicObjectTracker, TrackerConfig


def _run_dynamic_condition(
    motion_mode: str,
    cube_speed: float,
    target_speed: float,
    predictive: bool,
    seeds: List[int],
) -> Dict[str, Any]:
    env = PandaDynamicPickPlaceEnv(
        render_mode=None,
        max_episode_steps=100,
        motion_mode=motion_mode,  # type: ignore[arg-type]
        cube_speed=cube_speed,
        target_speed=target_speed,
        disturbance_step=18 if motion_mode == "disturbance_step" else None,
    )
    tracker = DynamicObjectTracker(TrackerConfig(dt_step=0.0625, window_size=5, replan_err_thresh_m=0.022))

    succs, grasps = [], []
    vel_rmses, pred_errs = [], []
    total_disturbances = 0
    total_replans = 0
    t0 = time.perf_counter()

    try:
        for s in seeds:
            obs, _ = env.reset(seed=s)
            tracker.reset(obs["achieved_goal"][:3])
            grasped = False
            any_grasp = False
            hold_steps = 0
            last_info: Dict[str, Any] = {}

            for step_i in range(100):
                raw = obs["observation"]
                ee_pos = raw[:3]
                gw = float(raw[3])
                obs_cube_pos = obs["achieved_goal"][:3]
                gt_cube_vel = raw[7:10]
                dg = obs["desired_goal"][:3]

                t_info = tracker.update(obs_cube_pos, dt=0.0625)
                if not grasped and float(np.linalg.norm(gt_cube_vel)) > 1e-4:
                    vel_rmses.append(float(np.linalg.norm(t_info["vel_est"] - gt_cube_vel) * 1000.0))
                if t_info["innovation_err_m"] > 0.0:
                    pred_errs.append(float(t_info["innovation_err_m"] * 1000.0))

                # Predictive intercept lead time proportional to remaining XY flight steps
                xy_d = float(np.linalg.norm(ee_pos[:2] - obs_cube_pos[:2]))
                lead_s = min(0.28, xy_d / 0.45) if predictive else 0.0
                intercept_pos = tracker.predict_position(horizon_s=lead_s)

                cmd = np.zeros(4, dtype=np.float32)
                if grasped:
                    tgt = np.array([dg[0], dg[1], max(float(dg[2]), 0.055 if xy_d > 0.04 else float(dg[2]))], dtype=np.float32)
                    cmd[:3] = np.clip((tgt - obs_cube_pos) / env.action_scale, -0.85, 0.85)
                    dist_g = float(np.linalg.norm(obs_cube_pos - dg))
                    cmd[3] = 1.0 if (dg[2] <= 0.035 and dist_g <= 0.025) else -1.0
                else:
                    align_d = float(np.linalg.norm(ee_pos[:2] - intercept_pos[:2]))
                    if align_d > 0.022 or ee_pos[2] > 0.08:
                        tgt = np.array([intercept_pos[0], intercept_pos[1], 0.060 if align_d > 0.024 else 0.023], dtype=np.float32)
                        cmd[:3] = np.clip((tgt - ee_pos) / env.action_scale, -0.9, 0.9)
                        cmd[3] = 1.0
                    else:
                        tgt = np.array([intercept_pos[0], intercept_pos[1], 0.020], dtype=np.float32)
                        cmd[:3] = np.clip((tgt - ee_pos) / env.action_scale, -0.5, 0.5)
                        cmd[3] = -1.0
                        hold_steps += 1
                        if hold_steps >= 3 and gw < 0.058:
                            grasped = True

                obs, _, term, trunc, last_info = env.step(cmd)
                if last_info.get("cube_grasped") or obs["achieved_goal"][2] > 0.055:
                    any_grasp = True
                    grasped = True
                if grasped and np.linalg.norm(obs["observation"][:2] - obs["achieved_goal"][:2]) > 0.065:
                    grasped = False
                    hold_steps = 0
                if term or trunc:
                    break

            succs.append(float(last_info.get("is_success", False)))
            grasps.append(float(any_grasp))
            total_disturbances += tracker.disturbance_count
            total_replans += tracker.replan_count
    finally:
        env.close()

    return {
        "motion_mode": motion_mode,
        "cube_speed_mps": cube_speed,
        "target_speed_mps": target_speed,
        "predictive_tracking": predictive,
        "n_episodes": len(seeds),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "success_rate": round(float(np.mean(succs)), 4),
        "grasp_rate": round(float(np.mean(grasps)), 4),
        "mean_vel_err_mm_s": round(float(np.mean(vel_rmses)) if vel_rmses else 0.0, 2),
        "mean_pred_err_mm": round(float(np.mean(pred_errs)) if pred_errs else 0.0, 2),
        "disturbances_detected": int(total_disturbances),
        "replan_triggers": int(total_replans),
    }


def main() -> None:
    os.makedirs("results/phase16_17", exist_ok=True)
    seeds = list(range(8400, 8420))
    results = {
        "linear_no_prediction": _run_dynamic_condition("linear", 0.045, 0.0, False, seeds),
        "linear_predictive": _run_dynamic_condition("linear", 0.045, 0.0, True, seeds),
        "sinusoidal_predictive": _run_dynamic_condition("sinusoidal", 0.045, 0.0, True, seeds),
        "disturbance_step_predictive": _run_dynamic_condition("disturbance_step", 0.035, 0.0, True, seeds),
        "moving_target_and_cube": _run_dynamic_condition("linear", 0.035, 0.020, True, seeds),
    }
    with open("results/phase16_17/phase16_3_dynamic_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("[16.3] Dynamic manipulation results:")
    for k, v in results.items():
        print(" ", k, "succ:", v["success_rate"], "grasp:", v["grasp_rate"], "replans:", v["replan_triggers"])


if __name__ == "__main__":
    main()

