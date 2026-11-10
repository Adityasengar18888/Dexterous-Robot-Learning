"""Deterministic checkpoint evaluation script for Phase 15.2.5B.1."""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from typing import Any, Dict, List

import numpy as np
import pybullet as p
from stable_baselines3 import SAC
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.rl.evaluate import make_env
from src.rl.train import load_config


def _read_tb_scalars(tb_dir: str) -> Dict[str, List[List[float]]]:
    events = glob.glob(os.path.join(tb_dir, "SAC_*", "events.out.tfevents.*"))
    if not events:
        return {}
    out: Dict[str, List[List[float]]] = {}
    for ev_path in sorted(events):
        ea = EventAccumulator(ev_path)
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            pts = [[int(s.step), float(s.value)] for s in ea.Scalars(tag)]
            out.setdefault(tag, []).extend(pts)
    return out


def eval_checkpoint(model_path: str, label: str, num_timesteps: int, cfg: dict, seeds: List[int]) -> Dict[str, Any]:
    env = make_env(cfg["env"], render_mode=None)
    model = SAC.load(model_path, env=env, device="cpu")
    ep_rows: List[Dict[str, Any]] = []

    try:
        for seed in seeds:
            obs, _ = env.reset(seed=int(seed))
            start_z = float(obs["observation"][2])
            min_z = start_z
            min_d3 = float(np.linalg.norm(obs["observation"][:3] - obs["achieved_goal"][:3]))
            min_dxy = float(np.linalg.norm(obs["observation"][:2] - obs["achieved_goal"][:2]))
            min_dz_err = abs(start_z - max(env.pg_cfg.min_safe_ee_z, float(obs["achieved_goal"][2])))

            reached = False
            aligned = False
            safe_desc = False
            grasp_auth = False
            bilat_ever = False
            lifted = False
            held_goal = False
            premature_close = False
            hold_steps = 0
            table_coll_steps = 0
            down_actions = 0
            sat_count = 0
            abs_acts: List[np.ndarray] = []
            z_vals: List[float] = []
            xy_errs: List[float] = []
            z_errs: List[float] = []
            comp_sums: Dict[str, float] = {}
            ep_rew = 0.0

            for _step in range(env.max_episode_steps):
                act, _ = model.predict(obs, deterministic=True)
                act = np.asarray(act, dtype=np.float32)
                abs_acts.append(np.abs(act))
                sat_count += int(np.sum(np.abs(act) > 0.99))
                if float(act[2]) < -0.05:
                    down_actions += 1

                obs, rew, terminated, truncated, info = env.step(act)
                ep_rew += float(rew)
                for ck, cv in info.get("reward_components", {}).items():
                    comp_sums[ck] = comp_sums.get(ck, 0.0) + float(cv)

                ee = obs["observation"][:3]
                gw = float(obs["observation"][3])
                ag = obs["achieved_goal"][:3]
                d3 = float(np.linalg.norm(ee - ag))
                dxy = float(np.linalg.norm(ee[:2] - ag[:2]))
                z_tgt = max(env.pg_cfg.min_safe_ee_z, float(ag[2]))
                dz_err = abs(float(ee[2]) - z_tgt)

                z_vals.append(float(ee[2]))
                xy_errs.append(dxy)
                z_errs.append(dz_err)
                min_z = min(min_z, float(ee[2]))
                min_d3 = min(min_d3, d3)
                min_dxy = min(min_dxy, dxy)
                min_dz_err = min(min_dz_err, dz_err)

                if d3 <= 0.05:
                    reached = True
                if dxy <= 0.018 and gw > 0.065:
                    aligned = True
                tcoll = bool(info.get("table_collision", False))
                if tcoll:
                    table_coll_steps += 1
                if dz_err <= 0.015 and dxy <= 0.030 and gw > 0.065 and not tcoll:
                    safe_desc = True
                if dxy <= 0.018 and dz_err <= 0.015 and gw > 0.065 and info.get("orn_ok", True) and not tcoll:
                    grasp_auth = True
                bilat = bool(info.get("bilateral_contact", False))
                if not bilat and not grasp_auth and gw <= 0.065:
                    premature_close = True
                if bilat:
                    bilat_ever = True
                    if float(ag[2]) >= 0.075:
                        lifted = True
                if info.get("is_success", False):
                    held_goal = True
                    hold_steps += 1
                if terminated or truncated:
                    break

            ep_rows.append(
                {
                    "seed": int(seed),
                    "reach": reached,
                    "alignment": aligned,
                    "safe_descent": safe_desc,
                    "grasp_auth": grasp_auth,
                    "bilateral_grasp": bilat_ever,
                    "lift": lifted,
                    "hold_success": held_goal,
                    "hold_steps": hold_steps,
                    "dropped": bool(info.get("dropped_objects", 0) > 0),
                    "premature_closure": premature_close,
                    "table_coll_steps": table_coll_steps,
                    "start_ee_z_m": round(start_z, 4),
                    "min_ee_z_m": round(min_z, 4),
                    "final_ee_z_m": round(float(z_vals[-1]), 4),
                    "mean_ee_z_m": round(float(np.mean(z_vals)), 4),
                    "actual_downward_disp_m": round(float(start_z - min_z), 4),
                    "downward_action_freq": round(float(down_actions / max(1, len(z_vals))), 4),
                    "min_ee_obj_dist_m": round(min_d3, 4),
                    "min_xy_dist_m": round(min_dxy, 4),
                    "mean_xy_err_m": round(float(np.mean(xy_errs)), 4),
                    "min_vert_err_m": round(min_dz_err, 4),
                    "final_vert_err_m": round(float(z_errs[-1]), 4),
                    "mean_vert_err_m": round(float(np.mean(z_errs)), 4),
                    "ep_reward": round(ep_rew, 4),
                    "comp_sums": {k: round(v, 4) for k, v in comp_sums.items()},
                    "action_saturation_rate": round(float(sat_count / (4 * max(1, len(z_vals)))), 4),
                    "mean_abs_action": [round(float(x), 4) for x in np.mean(abs_acts, axis=0)],
                }
            )
    finally:
        env.close()

    n = len(ep_rows)
    grasped_n = sum(1 for r in ep_rows if r["bilateral_grasp"])
    comp_keys = list(ep_rows[0]["comp_sums"].keys()) if ep_rows else []
    return {
        "checkpoint": label,
        "model_path": model_path,
        "num_timesteps": num_timesteps,
        "n_episodes": n,
        "reach_rate": round(sum(r["reach"] for r in ep_rows) / n, 4),
        "alignment_rate": round(sum(r["alignment"] for r in ep_rows) / n, 4),
        "safe_descent_rate": round(sum(r["safe_descent"] for r in ep_rows) / n, 4),
        "grasp_auth_rate": round(sum(r["grasp_auth"] for r in ep_rows) / n, 4),
        "bilateral_grasp_rate": round(grasped_n / n, 4),
        "lift_rate": round(sum(r["lift"] for r in ep_rows) / n, 4),
        "lift_cond_on_grasp": round(sum(r["lift"] and r["bilateral_grasp"] for r in ep_rows) / grasped_n, 4) if grasped_n else 0.0,
        "hold_success_rate": round(sum(r["hold_success"] for r in ep_rows) / n, 4),
        "drop_rate_cond_grasp": round(sum(r["dropped"] and r["bilateral_grasp"] for r in ep_rows) / grasped_n, 4) if grasped_n else 0.0,
        "premature_closure_ep_rate": round(sum(r["premature_closure"] for r in ep_rows) / n, 4),
        "table_coll_ep_rate": round(sum(r["table_coll_steps"] > 0 for r in ep_rows) / n, 4),
        "mean_table_coll_steps": round(float(np.mean([r["table_coll_steps"] for r in ep_rows])), 4),
        "mean_hold_steps": round(float(np.mean([r["hold_steps"] for r in ep_rows])), 4),
        "mean_ee_height_m": round(float(np.mean([r["mean_ee_z_m"] for r in ep_rows])), 4),
        "mean_min_ee_z_m": round(float(np.mean([r["min_ee_z_m"] for r in ep_rows])), 4),
        "mean_final_ee_z_m": round(float(np.mean([r["final_ee_z_m"] for r in ep_rows])), 4),
        "mean_actual_downward_disp_m": round(float(np.mean([r["actual_downward_disp_m"] for r in ep_rows])), 4),
        "downward_action_freq": round(float(np.mean([r["downward_action_freq"] for r in ep_rows])), 4),
        "mean_xy_err_m": round(float(np.mean([r["mean_xy_err_m"] for r in ep_rows])), 4),
        "mean_min_xy_dist_m": round(float(np.mean([r["min_xy_dist_m"] for r in ep_rows])), 4),
        "mean_vert_err_m": round(float(np.mean([r["mean_vert_err_m"] for r in ep_rows])), 4),
        "mean_final_vert_err_m": round(float(np.mean([r["final_vert_err_m"] for r in ep_rows])), 4),
        "mean_min_vert_err_m": round(float(np.mean([r["min_vert_err_m"] for r in ep_rows])), 4),
        "mean_min_ee_dist_m": round(float(np.mean([r["min_ee_obj_dist_m"] for r in ep_rows])), 4),
        "mean_episode_reward": round(float(np.mean([r["ep_reward"] for r in ep_rows])), 4),
        "mean_reward_components": {ck: round(float(np.mean([r["comp_sums"].get(ck, 0.0) for r in ep_rows])), 4) for ck in comp_keys},
        "action_saturation_rate": round(float(np.mean([r["action_saturation_rate"] for r in ep_rows])), 4),
        "mean_abs_action": [round(float(x), 4) for x in np.mean([r["mean_abs_action"] for r in ep_rows], axis=0)],
        "episodes": ep_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multi_sort_phase15_5B1.yaml")
    parser.add_argument("--steps", nargs="+", type=int, default=[0, 2500, 5000, 7500, 10000])
    parser.add_argument("--n-episodes", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_dir = cfg["paths"]["model_save_path"]
    tb_dir = cfg["paths"]["tensorboard_log"]
    seeds = list(range(1001, 1001 + args.n_episodes))

    t0 = time.time()
    evals = []
    for st in args.steps:
        if st == 0:
            path = os.path.join(model_dir, "initial_model.zip")
            lbl = "0_initial"
        else:
            path = os.path.join(model_dir, f"checkpoint_{st}_steps.zip")
            lbl = str(st)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing checkpoint: {path}")
        row = eval_checkpoint(path, lbl, st, cfg, seeds)
        evals.append(row)
        print(
            f"[{lbl:>9s}] Reach={row['reach_rate']*100:5.1f}% | Align={row['alignment_rate']*100:5.1f}%"
            f" | SafeD={row['safe_descent_rate']*100:5.1f}% | Auth={row['grasp_auth_rate']*100:5.1f}%"
            f" | Grasp={row['bilateral_grasp_rate']*100:5.1f}% | Lift|G={row['lift_cond_on_grasp']*100:5.1f}%"
            f" | z_err={row['mean_vert_err_m']:.4f}m (minZ={row['mean_min_ee_z_m']:.4f}m, downF={row['downward_action_freq']*100:4.1f}%)"
            f" | dxy={row['mean_min_xy_dist_m']:.4f}m | Rew={row['mean_episode_reward']:.3f}"
        )

    out_payload = {
        "eval_elapsed_seconds": round(time.time() - t0, 2),
        "eval_seeds": seeds,
        "model_files": sorted(os.listdir(model_dir)),
        "tensorboard_scalars": _read_tb_scalars(tb_dir),
        "checkpoint_evaluations": evals,
    }
    for out_path in (
        "experiments/multi_sort_phase15_5B1/phase15_2_5B1_training_results.json",
        "results/phase15_2_5B1/phase15_2_5B1_training_results.json",
    ):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out_payload, f, indent=2)
    print(f"Saved results in {out_payload['eval_elapsed_seconds']}s.")


if __name__ == "__main__":
    main()

