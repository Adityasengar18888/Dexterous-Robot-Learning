"""Phase 1 (15.2.5B.2): Generate balanced scripted demonstration dataset for SinglePreGrasp.

Fixes Phase 15.2.5B.1's 78.4% closed-gripper HOLD_AT_GOAL bias by using gentle approach/descent
gains and terminating demo episodes cleanly after 2 verified holding steps so that:
  - Open approach / alignment / descent: ~65-75%
  - Grasp authorization & closure:       ~10-15%
  - Lifting:                             ~10-15%
  - Stable holding:                      ~5-10%
"""

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


def generate_balanced_demos(
    out_dir: str = "experiments/multi_sort_phase15_5B2/seed42/demonstrations",
    target_episodes: int = 50,
    start_seed: int = 6000,
    max_attempts: int = 80,
    target_hold_steps: int = 5,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    rcfg = SinglePreGraspRewardConfig(use_vertical_potential=True, w_vertical_potential=2.0)
    env = PandaSinglePreGraspEnv(render_mode=None, max_episode_steps=80, reward_config=rcfg)

    # Slightly gentler approach/descent speeds give richer open-gripper trajectories (~18-21 open steps)
    pg_cfg = PreGraspConfig(
        max_approach_cmd=0.55,
        max_align_cmd=0.45,
        max_descent_dz=0.22,
        lift_verify_z_min=0.11,
    )
    ctrl = PreGraspController(config=pg_cfg, action_scale=env.action_scale)

    episodes: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    stage_counts: Counter[str] = Counter()

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
            ep_stages: List[str] = []

            ep_return = 0.0
            hold_steps = 0
            valid = True
            reject_reason = ""

            for step_idx in range(1, env.max_episode_steps + 1):
                telem = query_pybullet_telemetry(env, active_obj_idx=0)
                if ctrl.stage != PreGraspStage.ACQUIRED:
                    cmd, _ = ctrl.step(step_idx, telem)
                    stage_name = ctrl.stage.value
                else:
                    cmd = _hold_action(env, obs["desired_goal"], pg_cfg)
                    stage_name = "HOLD_AT_GOAL"

                if not np.all(np.isfinite(cmd)) or np.any(np.abs(cmd) > 1.0001):
                    valid = False
                    reject_reason = f"invalid_action_{step_idx}"
                    break

                next_obs, rew, terminated, truncated, info = env.step(cmd)
                if not np.isfinite(rew) or not all(np.all(np.isfinite(v)) for v in next_obs.values()):
                    valid = False
                    reject_reason = f"non_finite_{step_idx}"
                    break

                if info.get("is_success", False):
                    hold_steps += 1
                ep_return += float(rew)
                done_now = bool(terminated or truncated or (hold_steps >= target_hold_steps))

                info_rec = dict(info)
                # HerReplayBuffer expects TimeLimit.truncated if episode stops smoothly at demo cap
                info_rec["TimeLimit.truncated"] = bool(done_now and not terminated)
                info_rec["is_demo"] = True
                info_rec["demo_seed"] = int(seed)
                info_rec["demo_stage"] = stage_name

                ep_obs.append({k: v.copy() for k, v in obs.items()})
                ep_next_obs.append({k: v.copy() for k, v in next_obs.items()})
                ep_acts.append(cmd.copy().astype(np.float32))
                ep_rews.append(float(rew))
                ep_dones.append(done_now)
                ep_infos.append(info_rec)
                ep_stages.append(stage_name)

                obs = next_obs
                if done_now:
                    break

            m = dict(ctrl.metrics)
            final_succ = bool(ep_infos[-1].get("is_success", False)) if ep_infos else False
            if valid:
                if not final_succ or hold_steps < target_hold_steps:
                    valid = False
                    reject_reason = f"incomplete_hold({final_succ},{hold_steps})"
                elif m["premature_closure"] or m["table_collision_steps"] > 0 or info.get("collisions", 0) > 0:
                    valid = False
                    reject_reason = "safety_violation"
                elif m["max_pregrasp_disp_m"] > pg_cfg.max_pregrasp_disp_m or info.get("dropped_objects", 0) > 0:
                    valid = False
                    reject_reason = "disp_or_drop"
                elif not (m["approach_reached"] and m["alignment_reached"] and m["safe_descent_reached"] and m["grasp_authorized"] and m["bilateral_grasped"] and m["lift_verified"]):
                    valid = False
                    reject_reason = "missing_stage"

            if valid:
                for st in ep_stages:
                    stage_counts[st] += 1
                episodes.append(
                    {
                        "seed": int(seed),
                        "length": len(ep_acts),
                        "return": float(ep_return),
                        "hold_steps": int(hold_steps),
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

    total_t = sum(e["length"] for e in episodes)
    open_steps = stage_counts["APPROACH"] + stage_counts["ALIGNMENT"] + stage_counts["CONTROLLED_DESCENT"]
    auth_grasp_steps = stage_counts["GRASP_AUTHORIZATION"] + stage_counts["GRASP"]
    lift_steps = stage_counts["LIFT_VERIFICATION"]
    hold_total = stage_counts["ACQUIRED"] + stage_counts["HOLD_AT_GOAL"]

    all_acts = np.concatenate([np.asarray(e["actions"]) for e in episodes], axis=0) if episodes else np.zeros((0, 4))
    open_act_pct = float(np.mean(all_acts[:, 3] > 0.0) * 100.0) if len(all_acts) else 0.0

    pkl_path = os.path.join(out_dir, "balanced_pregrasp_demos_50ep.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(episodes, f)

    summary = {
        "accepted_episodes": len(episodes),
        "rejected_episodes": len(rejected),
        "rejected_details": rejected,
        "total_transitions": total_t,
        "mean_episode_length": float(np.mean([e["length"] for e in episodes])) if episodes else 0.0,
        "mean_episode_return": float(np.mean([e["return"] for e in episodes])) if episodes else 0.0,
        "stage_step_counts": dict(stage_counts),
        "distribution_pct": {
            "open_approach_align_descent_pct": round(100.0 * open_steps / max(1, total_t), 2),
            "auth_and_close_pct": round(100.0 * auth_grasp_steps / max(1, total_t), 2),
            "lift_pct": round(100.0 * lift_steps / max(1, total_t), 2),
            "hold_pct": round(100.0 * hold_total / max(1, total_t), 2),
            "open_gripper_cmd_pct": round(open_act_pct, 2),
        },
        "total_table_collisions": int(sum(e["metrics"]["table_collision_steps"] for e in episodes)),
        "total_premature_closures": int(sum(1 for e in episodes if e["metrics"]["premature_closure"])),
        "dataset_path": pkl_path,
    }
    with open(os.path.join(out_dir, "balanced_demo_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    out = generate_balanced_demos()
    print(json.dumps(out, indent=2))

