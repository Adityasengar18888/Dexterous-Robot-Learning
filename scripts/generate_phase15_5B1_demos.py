"""Part 4: Generate validated 50-episode scripted demonstration dataset on PandaSinglePreGraspEnv."""

from __future__ import annotations

import json
import os
import pickle
from collections import Counter
from typing import Any, Dict, List

import numpy as np

from src.controllers.pregrasp_controller import (
    PreGraspConfig,
    PreGraspController,
    PreGraspStage,
    query_pybullet_telemetry,
)
from src.environments.single_pregrasp_env import (
    PandaSinglePreGraspEnv,
    SinglePreGraspRewardConfig,
)


def _hold_action(env: PandaSinglePreGraspEnv, goal: np.ndarray, cfg: PreGraspConfig) -> np.ndarray:
    cube_pos = env.cubes[0].get_position()
    delta = (goal - cube_pos) / env.action_scale
    cmd = np.zeros(4, dtype=np.float32)
    cmd[:3] = np.clip(delta, -0.35, 0.35)
    cmd[3] = cfg.close_gripper_cmd
    return cmd


def generate_demonstrations(
    out_dir: str = "experiments/multi_sort_phase15_5B1/seed42/demonstrations",
    target_episodes: int = 50,
    start_seed: int = 5000,
    max_attempts: int = 80,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    rcfg = SinglePreGraspRewardConfig(use_vertical_potential=True, w_vertical_potential=2.0)
    env = PandaSinglePreGraspEnv(render_mode=None, max_episode_steps=80, reward_config=rcfg)
    pg_cfg = PreGraspConfig(lift_verify_z_min=0.11)
    ctrl = PreGraspController(config=pg_cfg, action_scale=env.action_scale)

    episodes: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    stage_counts: Counter[str] = Counter()
    comp_totals: Counter[str] = Counter()

    try:
        for offset in range(max_attempts):
            if len(episodes) >= target_episodes:
                break
            seed = start_seed + offset
            obs, _ = env.reset(seed=seed)
            ctrl.reset(active_obj_idx=0, anchor_obj_pos=obs["achieved_goal"][:3])

            ep_obs: List[Dict[str, np.ndarray]] = []
            ep_next_obs: List[Dict[str, np.ndarray]] = []
            ep_acts: List[np.ndarray] = []
            ep_rews: List[float] = []
            ep_dones: List[bool] = []
            ep_infos: List[Dict[str, Any]] = []

            ep_return = 0.0
            hold_steps = 0
            valid = True
            reject_reason = ""
            stages_seen = set()

            for step_idx in range(1, env.max_episode_steps + 1):
                telem = query_pybullet_telemetry(env, active_obj_idx=0)
                if ctrl.stage != PreGraspStage.ACQUIRED:
                    cmd, _ = ctrl.step(step_idx, telem)
                    stage_name = ctrl.stage.value
                else:
                    cmd = _hold_action(env, obs["desired_goal"], pg_cfg)
                    stage_name = "HOLD_AT_GOAL"
                stages_seen.add(stage_name)
                stage_counts[stage_name] += 1

                if not np.all(np.isfinite(cmd)) or np.any(np.abs(cmd) > 1.0001):
                    valid = False
                    reject_reason = f"invalid_action_step_{step_idx}"
                    break

                next_obs, rew, terminated, truncated, info = env.step(cmd)
                if not np.isfinite(rew) or not all(np.all(np.isfinite(v)) for v in next_obs.values()):
                    valid = False
                    reject_reason = f"non_finite_transition_{step_idx}"
                    break

                if info.get("is_success", False):
                    hold_steps += 1
                ep_return += float(rew)
                for k, v in info.get("reward_components", {}).items():
                    comp_totals[k] += float(v)

                done = bool(terminated or truncated)
                info_rec = dict(info)
                info_rec["TimeLimit.truncated"] = bool(truncated and not terminated)
                info_rec["is_demo"] = True
                info_rec["demo_seed"] = int(seed)
                info_rec["demo_stage"] = stage_name

                ep_obs.append({k: v.copy() for k, v in obs.items()})
                ep_next_obs.append({k: v.copy() for k, v in next_obs.items()})
                ep_acts.append(cmd.copy().astype(np.float32))
                ep_rews.append(float(rew))
                ep_dones.append(done)
                ep_infos.append(info_rec)

                obs = next_obs
                if done:
                    break

            m = dict(ctrl.metrics)
            final_success = bool(ep_infos[-1].get("is_success", False)) if ep_infos else False
            if valid:
                if not final_success or hold_steps < 15:
                    valid = False
                    reject_reason = f"incomplete_hold(succ={final_success},hold={hold_steps})"
                elif m["premature_closure"] or m["table_collision_steps"] > 0 or info.get("collisions", 0) > 0:
                    valid = False
                    reject_reason = "safety_violation"
                elif m["max_pregrasp_disp_m"] > pg_cfg.max_pregrasp_disp_m or info.get("dropped_objects", 0) > 0:
                    valid = False
                    reject_reason = "disp_or_drop_violation"
                elif not (m["approach_reached"] and m["alignment_reached"] and m["safe_descent_reached"] and m["grasp_authorized"] and m["bilateral_grasped"] and m["lift_verified"]):
                    valid = False
                    reject_reason = "missing_pregrasp_stage"

            if valid:
                episodes.append(
                    {
                        "seed": int(seed),
                        "length": len(ep_acts),
                        "return": float(ep_return),
                        "hold_steps": int(hold_steps),
                        "stages_seen": sorted(stages_seen),
                        "metrics": m,
                        "observations": ep_obs,
                        "next_observations": ep_next_obs,
                        "actions": ep_acts,
                        "rewards": ep_rews,
                        "dones": ep_dones,
                        "infos": ep_infos,
                    }
                )
            else:
                rejected.append({"seed": int(seed), "reason": reject_reason, "metrics": m})
    finally:
        env.close()

    pkl_path = os.path.join(out_dir, "single_pregrasp_demos_50ep.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(episodes, f)

    n_eps = len(episodes)
    n_trans = sum(e["length"] for e in episodes)
    summary = {
        "target_episodes": target_episodes,
        "accepted_episodes": n_eps,
        "rejected_episodes": len(rejected),
        "rejected_details": rejected,
        "total_transitions": n_trans,
        "accepted_seeds": [e["seed"] for e in episodes],
        "mean_episode_length": float(np.mean([e["length"] for e in episodes])) if n_eps else 0.0,
        "mean_episode_return": float(np.mean([e["return"] for e in episodes])) if n_eps else 0.0,
        "mean_hold_steps": float(np.mean([e["hold_steps"] for e in episodes])) if n_eps else 0.0,
        "max_pregrasp_disp_m": float(max(e["metrics"]["max_pregrasp_disp_m"] for e in episodes)) if n_eps else 0.0,
        "total_table_collisions": int(sum(e["metrics"]["table_collision_steps"] for e in episodes)),
        "total_premature_closures": int(sum(1 for e in episodes if e["metrics"]["premature_closure"])),
        "stage_step_counts": dict(stage_counts),
        "mean_reward_components": {k: float(v / max(1, n_eps)) for k, v in comp_totals.items()},
        "dataset_path": pkl_path,
    }
    json_path = os.path.join(out_dir, "demo_dataset_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    s = generate_demonstrations()
    print(json.dumps({k: v for k, v in s.items() if k != "accepted_seeds"}, indent=2))

