"""Phase 4 Orientation-Aware Assembly Evaluation Runner (Scripted & Learned Separated)."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

import numpy as np
from stable_baselines3 import SAC

from src.controllers.assembly_controller import ScriptedAssemblyController
from src.environments.assembly_dense_env import PandaAssemblyDenseEnv


def eval_scripted_assembly(n_episodes: int = 30, start_seed: int = 2001) -> Dict[str, Any]:
    env = PandaAssemblyDenseEnv(render_mode=None, max_episode_steps=120)
    ctrl = ScriptedAssemblyController(action_scale=env.action_scale, yaw_scale=env.yaw_scale)
    t0 = time.time()
    rows: List[Dict[str, Any]] = []

    try:
        for i in range(n_episodes):
            seed = start_seed + i
            obs, info = env.reset(seed=seed)
            ctrl.reset()

            grasped = False
            lifted = False
            aligned_above = False
            assembled = False
            jams = 0
            table_colls = 0
            steps_to_succ = -1
            ep_rew = 0.0
            max_depth = 0.0

            for step in range(1, env.max_episode_steps + 1):
                cmd, _ = ctrl.step(env, obs, info)
                obs, rew, terminated, truncated, info = env.step(cmd)
                ep_rew += float(rew)

                if info.get("bilateral_contact", False):
                    grasped = True
                if float(obs["achieved_goal"][2]) >= 0.085:
                    lifted = True
                if info.get("assembly_state") == "PART_B_ALIGNED":
                    aligned_above = True
                if info.get("jam_detected", False):
                    jams += 1
                if info.get("table_collision", False):
                    table_colls += 1
                max_depth = max(max_depth, float(info.get("insertion_depth_m", 0.0)))
                if info.get("is_success", False):
                    assembled = True
                    if steps_to_succ < 0:
                        steps_to_succ = step
                if terminated or truncated or ctrl.stage.value == "DONE":
                    break

            rows.append(
                {
                    "seed": int(seed),
                    "assembled": assembled,
                    "grasped": grasped,
                    "lifted": lifted,
                    "aligned_above": aligned_above or assembled,
                    "final_pos_err_m": round(float(info.get("distance", 0.0)), 4),
                    "final_ori_dc": round(float(info.get("orientation_error", 0.0)), 4),
                    "final_angle_deg": round(float(info.get("geodesic_angle_deg", 0.0)), 2),
                    "max_insertion_depth_m": round(max_depth, 4),
                    "jam_steps": jams,
                    "recovery_count": int(ctrl.recovery_count),
                    "table_coll_steps": table_colls,
                    "steps_executed": step,
                    "steps_to_success": steps_to_succ,
                    "ep_reward": round(ep_rew, 3),
                }
            )
    finally:
        env.close()

    n = len(rows)
    succ_steps = [r["steps_to_success"] for r in rows if r["steps_to_success"] > 0]
    return {
        "policy_type": "SCRIPTED",
        "n_episodes": n,
        "elapsed_seconds": round(time.time() - t0, 2),
        "grasp_rate": round(sum(r["grasped"] for r in rows) / n, 4),
        "lift_rate": round(sum(r["lifted"] for r in rows) / n, 4),
        "pre_insert_align_rate": round(sum(r["aligned_above"] for r in rows) / n, 4),
        "assembly_success_rate": round(sum(r["assembled"] for r in rows) / n, 4),
        "jam_episode_rate": round(sum(r["jam_steps"] > 0 for r in rows) / n, 4),
        "table_coll_episode_rate": round(sum(r["table_coll_steps"] > 0 for r in rows) / n, 4),
        "mean_final_pos_err_m": round(float(np.mean([r["final_pos_err_m"] for r in rows])), 4),
        "mean_final_angle_deg": round(float(np.mean([r["final_angle_deg"] for r in rows])), 2),
        "mean_max_insertion_depth_m": round(float(np.mean([r["max_insertion_depth_m"] for r in rows])), 4),
        "mean_steps_to_success": round(float(np.mean(succ_steps)), 2) if succ_steps else -1.0,
        "mean_episode_reward": round(float(np.mean([r["ep_reward"] for r in rows])), 3),
        "episodes": rows,
    }


def eval_learned_assembly(model_path: str, n_episodes: int = 30, start_seed: int = 2001) -> Dict[str, Any]:
    env = PandaAssemblyDenseEnv(render_mode=None, max_episode_steps=120)
    model = SAC.load(model_path, env=env, device="cpu")
    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    try:
        for i in range(n_episodes):
            seed = start_seed + i
            obs, info = env.reset(seed=seed)
            grasped = False
            lifted = False
            aligned_above = False
            assembled = False
            jams = 0
            ep_rew = 0.0
            for step in range(1, env.max_episode_steps + 1):
                act, _ = model.predict(obs, deterministic=True)
                obs, rew, terminated, truncated, info = env.step(act)
                ep_rew += float(rew)
                if info.get("bilateral_contact", False):
                    grasped = True
                if float(obs["achieved_goal"][2]) >= 0.085:
                    lifted = True
                if info.get("assembly_state") == "PART_B_ALIGNED":
                    aligned_above = True
                if info.get("jam_detected", False):
                    jams += 1
                if info.get("is_success", False):
                    assembled = True
                if terminated or truncated:
                    break
            rows.append(
                {
                    "seed": int(seed),
                    "assembled": assembled,
                    "grasped": grasped,
                    "lifted": lifted,
                    "aligned_above": aligned_above or assembled,
                    "final_pos_err_m": round(float(info.get("distance", 0.0)), 4),
                    "final_angle_deg": round(float(info.get("geodesic_angle_deg", 0.0)), 2),
                    "jam_steps": jams,
                    "ep_reward": round(ep_rew, 3),
                }
            )
    finally:
        env.close()
    n = len(rows)
    return {
        "policy_type": "LEARNED_SAC_HER",
        "model_path": model_path,
        "n_episodes": n,
        "elapsed_seconds": round(time.time() - t0, 2),
        "grasp_rate": round(sum(r["grasped"] for r in rows) / n, 4),
        "lift_rate": round(sum(r["lifted"] for r in rows) / n, 4),
        "pre_insert_align_rate": round(sum(r["aligned_above"] for r in rows) / n, 4),
        "assembly_success_rate": round(sum(r["assembled"] for r in rows) / n, 4),
        "jam_episode_rate": round(sum(r["jam_steps"] > 0 for r in rows) / n, 4),
        "mean_final_pos_err_m": round(float(np.mean([r["final_pos_err_m"] for r in rows])), 4),
        "mean_final_angle_deg": round(float(np.mean([r["final_angle_deg"] for r in rows])), 2),
        "mean_episode_reward": round(float(np.mean([r["ep_reward"] for r in rows])), 3),
        "episodes": rows,
    }


if __name__ == "__main__":
    os.makedirs("results/phase15_3_assembly", exist_ok=True)
    res = eval_scripted_assembly(n_episodes=30)
    with open("results/phase15_3_assembly/phase15_3_assembly_scripted_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print({k: v for k, v in res.items() if k != "episodes"})

