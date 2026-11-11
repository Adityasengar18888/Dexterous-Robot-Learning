"""Phase 1 (15.2.5B.3): 100-Episode Single-Object Unseen Generalization & Stress Evaluation."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

import numpy as np
import pybullet as p
from stable_baselines3 import SAC

from src.environments.single_pregrasp_env import PandaSinglePreGraspEnv, SinglePreGraspRewardConfig


def run_b3_generalization(model_path: str, label: str = "5B4_best", n_episodes: int = 100, start_seed: int = 7001) -> Dict[str, Any]:
    rcfg = SinglePreGraspRewardConfig(use_vertical_potential=True, w_vertical_potential=2.0)
    env = PandaSinglePreGraspEnv(render_mode=None, max_episode_steps=80, reward_config=rcfg)
    model = SAC.load(model_path, env=env, device="cpu")
    t0 = time.time()
    rows: List[Dict[str, Any]] = []

    try:
        for i in range(n_episodes):
            seed = start_seed + i
            rng = np.random.default_rng(seed + 99991)
            obs, _ = env.reset(seed=seed)

            # Vary physics, orientation, and initial EE starting offset across unseen seeds
            cube_id = env.cubes[0].body_id
            cid = env.sim.client_id
            mass = float(rng.uniform(0.06, 0.22))
            lat_fric = float(rng.uniform(0.35, 1.40))
            p.changeDynamics(cube_id, -1, mass=mass, lateralFriction=lat_fric, physicsClientId=cid)

            yaw = float(rng.uniform(-np.pi, np.pi))
            c_pos = env.cubes[0].get_position()
            c_ori = p.getQuaternionFromEuler([0.0, 0.0, yaw])
            p.resetBasePositionAndOrientation(cube_id, c_pos, c_ori, physicsClientId=cid)

            # Vary initial EE starting position (different approach distances)
            ee0 = env.controller.get_ee_position() + rng.uniform([-0.06, -0.06, -0.08], [0.06, 0.06, 0.01]).astype(np.float32)
            ee0 = env.controller.robot.clip_to_workspace(ee0)
            env.controller.move_ee(position=ee0, steps=20, tolerance=0.005)
            obs = env._get_obs()

            # Vary disturbance probability (25% of episodes receive a small lateral tap during lift/grasp)
            apply_disturb = (i % 4 == 3)
            disturbed = False

            reached = False
            aligned = False
            grasped = False
            lifted = False
            held_goal = False
            hold_steps = 0
            tcoll_steps = 0
            ep_rew = 0.0
            steps_taken = 0

            for s in range(1, env.max_episode_steps + 1):
                steps_taken = s
                act, _ = model.predict(obs, deterministic=True)
                obs, rew, terminated, truncated, info = env.step(act)
                ep_rew += float(rew)

                ee = obs["observation"][:3]
                ag = obs["achieved_goal"][:3]
                gw = float(obs["observation"][3])
                if np.linalg.norm(ee - ag) <= 0.05:
                    reached = True
                if np.linalg.norm(ee[:2] - ag[:2]) <= 0.018 and gw > 0.065:
                    aligned = True
                if info.get("table_collision", False):
                    tcoll_steps += 1
                if info.get("bilateral_contact", False):
                    grasped = True
                    if apply_disturb and not disturbed and ag[2] >= 0.045:
                        p.applyExternalForce(cube_id, -1, [float(rng.uniform(-1.2, 1.2)), float(rng.uniform(-1.2, 1.2)), 0.0], ag.tolist(), p.WORLD_FRAME, physicsClientId=cid)
                        disturbed = True
                    if float(ag[2]) >= 0.075:
                        lifted = True
                if info.get("is_success", False):
                    held_goal = True
                    hold_steps += 1
                if terminated or truncated:
                    break

            rows.append(
                {
                    "seed": int(seed),
                    "mass": round(mass, 3),
                    "friction": round(lat_fric, 3),
                    "yaw_rad": round(yaw, 3),
                    "disturbed": disturbed,
                    "reach": reached,
                    "align": aligned,
                    "bilateral_grasp": grasped,
                    "lift": lifted,
                    "hold_success": held_goal,
                    "hold_steps": hold_steps,
                    "dropped": bool(info.get("dropped_objects", 0) > 0),
                    "table_coll_steps": tcoll_steps,
                    "steps": steps_taken,
                    "ep_reward": round(ep_rew, 3),
                }
            )
    finally:
        env.close()

    n = len(rows)
    gn = sum(r["bilateral_grasp"] for r in rows)
    return {
        "label": label,
        "model_path": model_path,
        "n_episodes": n,
        "elapsed_seconds": round(time.time() - t0, 2),
        "reach_rate": round(sum(r["reach"] for r in rows) / n, 4),
        "alignment_rate": round(sum(r["align"] for r in rows) / n, 4),
        "bilateral_grasp_rate": round(gn / n, 4),
        "lift_rate": round(sum(r["lift"] for r in rows) / n, 4),
        "lift_cond_on_grasp": round(sum(r["lift"] and r["bilateral_grasp"] for r in rows) / gn, 4) if gn else 0.0,
        "hold_success_rate": round(sum(r["hold_success"] for r in rows) / n, 4),
        "drop_rate_cond_grasp": round(sum(r["dropped"] and r["bilateral_grasp"] for r in rows) / gn, 4) if gn else 0.0,
        "table_coll_ep_rate": round(sum(r["table_coll_steps"] > 0 for r in rows) / n, 4),
        "mean_episode_steps": round(float(np.mean([r["steps"] for r in rows])), 2),
        "mean_episode_reward": round(float(np.mean([r["ep_reward"] for r in rows])), 3),
        "episodes": rows,
    }

