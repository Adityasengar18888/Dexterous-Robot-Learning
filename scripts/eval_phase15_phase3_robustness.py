"""Phase 3: Frozen-Policy Generalization & Robustness Evaluation Harness.

Evaluates frozen policies (zero retraining) across DR1, DR2, DR4, position/orientation
randomization, observation noise, moderate observation delay, and multiple seeds.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

import numpy as np
import pybullet as p
from stable_baselines3 import SAC

from src.environments.single_pregrasp_env import PandaSinglePreGraspEnv, SinglePreGraspRewardConfig
from src.environments.dense_pick_place_env import PandaPickPlaceDenseRewardEnv
from src.rl.adaptive_controllers import AdaptiveTransportController, ControllerConfig, get_step_telemetry
from src.rl.sim_to_real_latency_wrapper import LatencyWrapper
from src.rl.sim_to_real_observation_wrapper import ObservationNoiseWrapper


def eval_single_pregrasp_condition(
    model_path: str,
    cond_name: str,
    seeds: List[int],
    randomize_pos: bool = True,
    randomize_ori: bool = False,
    randomize_mass_fric: bool = False,
    obs_noise_m: float = 0.0,
    obs_delay_steps: int = 0,
    target_z_span: tuple[float, float] | None = None,
) -> Dict[str, Any]:
    rcfg = SinglePreGraspRewardConfig(use_vertical_potential=True, w_vertical_potential=2.0)
    env = PandaSinglePreGraspEnv(render_mode=None, max_episode_steps=80, reward_config=rcfg)
    noise_wrap = ObservationNoiseWrapper(std_m=obs_noise_m / 2.0, clip_bound_m=obs_noise_m, seed=2026)
    lat_wrap = LatencyWrapper(obs_delay_steps=obs_delay_steps, action_delay_steps=0)

    model = SAC.load(model_path, env=env, device="cpu")
    t0 = time.time()
    rows: List[Dict[str, Any]] = []

    try:
        for seed in seeds:
            rng = np.random.default_rng(int(seed) + 42424)
            obs, _ = env.reset(seed=int(seed))
            cid = env.sim.client_id
            cube_id = env.cubes[0].body_id

            if not randomize_pos:
                fixed_pos = np.array([0.45, 0.0, 0.025], dtype=np.float32)
                p.resetBasePositionAndOrientation(cube_id, fixed_pos.tolist(), [0, 0, 0, 1], physicsClientId=cid)
                env.goal = np.array([fixed_pos[0], fixed_pos[1], env.pg_cfg.target_lift_z], dtype=np.float32)

            if randomize_ori:
                yaw = float(rng.uniform(-np.pi, np.pi))
                pos = env.cubes[0].get_position()
                p.resetBasePositionAndOrientation(cube_id, pos.tolist(), p.getQuaternionFromEuler([0, 0, yaw]), physicsClientId=cid)

            if randomize_mass_fric:
                mass = float(rng.uniform(0.06, 0.25))
                fric = float(rng.uniform(0.30, 1.40))
                p.changeDynamics(cube_id, -1, mass=mass, lateralFriction=fric, physicsClientId=cid)

            if target_z_span is not None:
                cpos = env.cubes[0].get_position()
                env.goal = np.array([cpos[0], cpos[1], float(rng.uniform(target_z_span[0], target_z_span[1]))], dtype=np.float32)

            obs = env._get_obs()
            noise_wrap.reset(episode_seed=int(seed))
            t_stub = {"ee_pos": obs["observation"][:3].copy(), "cube_pos": obs["achieved_goal"][:3].copy(), "desired_goal": obs["desired_goal"].copy(), "rel_dist": 0.5}
            obs_p, t_p = noise_wrap.perturb(obs, t_stub) if obs_noise_m > 0.0 else (obs, t_stub)
            lat_wrap.reset(obs_p, t_p)

            grasped, lifted, held, coll = False, False, False, 0
            ep_rew = 0.0
            for _ in range(env.max_episode_steps):
                t_s = {"ee_pos": obs["observation"][:3].copy(), "cube_pos": obs["achieved_goal"][:3].copy(), "desired_goal": obs["desired_goal"].copy(), "rel_dist": 0.5}
                obs_n, t_n = noise_wrap.perturb(obs, t_s) if obs_noise_m > 0.0 else (obs, t_s)
                obs_in, _ = lat_wrap.push_and_get_obs(obs_n, t_n) if obs_delay_steps > 0 else (obs_n, t_n)
                act, _ = model.predict(obs_in, deterministic=True)
                obs, rew, term, trunc, info = env.step(act)
                ep_rew += float(rew)
                if info.get("table_collision", False):
                    coll += 1
                if info.get("bilateral_contact", False):
                    grasped = True
                    if float(env.cubes[0].get_position()[2]) >= 0.075:
                        lifted = True
                if info.get("is_success", False):
                    held = True
                if term or trunc:
                    break
            rows.append({"seed": int(seed), "grasped": grasped, "lifted": lifted, "held": held, "colls": coll, "rew": round(ep_rew, 3)})
    finally:
        env.close()

    n = len(rows)
    gn = sum(r["grasped"] for r in rows)
    return {
        "condition": cond_name,
        "n_episodes": n,
        "elapsed_s": round(time.time() - t0, 2),
        "grasp_rate": round(gn / n, 4),
        "lift_rate": round(sum(r["lifted"] for r in rows) / n, 4),
        "lift_cond_grasp": round(sum(r["lifted"] and r["grasped"] for r in rows) / gn, 4) if gn else 0.0,
        "hold_success_rate": round(sum(r["held"] for r in rows) / n, 4),
        "table_coll_ep_rate": round(sum(r["colls"] > 0 for r in rows) / n, 4),
        "mean_reward": round(float(np.mean([r["rew"] for r in rows])), 3),
    }


def eval_pickplace_dr_frozen(n_per_seed: int = 20) -> Dict[str, Any]:
    """Evaluate frozen Phase 14 E_ADAPTIVE_TRANSPORT across DR1, DR2, DR4, noise, delay."""
    seeds_cfg = {
        "DR4_Seed42": "experiments/phase13_dense_dr/dr4/seed42/models/checkpoint_500000.zip",
        "DR4_Seed43": "experiments/phase13_dense_dr/dr4/seed43/models/checkpoint_500000.zip",
    }
    dr_presets = {
        "DR1_POS": {"enabled": True, "object_position": {"enabled": True}},
        "DR2_POSE_TARGET": {"enabled": True, "object_position": {"enabled": True}, "target_position": {"enabled": True}, "object_orientation": {"enabled": True}},
        "DR4_FULL_PHYSICS": {"enabled": True, "object_position": {"enabled": True}, "target_position": {"enabled": True}, "object_orientation": {"enabled": True}, "mass": {"enabled": True, "min": 0.10, "max": 0.25}, "friction": {"enabled": True, "min": 0.30, "max": 1.20}},
        "DR4_OBS_NOISE_5MM": {"enabled": True, "object_position": {"enabled": True}, "target_position": {"enabled": True}, "object_orientation": {"enabled": True}, "mass": {"enabled": True, "min": 0.10, "max": 0.25}, "friction": {"enabled": True, "min": 0.30, "max": 1.20}},
        "DR4_OBS_DELAY_1": {"enabled": True, "object_position": {"enabled": True}, "target_position": {"enabled": True}, "object_orientation": {"enabled": True}, "mass": {"enabled": True, "min": 0.10, "max": 0.25}, "friction": {"enabled": True, "min": 0.30, "max": 1.20}},
    }
    eval_seeds = list(range(3001, 3001 + n_per_seed))
    results = []

    for model_name, model_path in seeds_cfg.items():
        for cond, rcfg in dr_presets.items():
            env = PandaPickPlaceDenseRewardEnv(render_mode=None, distance_threshold=0.05, max_episode_steps=100, action_scale=0.05, randomizer_config=rcfg)
            noise_bound = 0.005 if cond == "DR4_OBS_NOISE_5MM" else 0.0
            noise_wrap = ObservationNoiseWrapper(std_m=noise_bound / 2.0, clip_bound_m=noise_bound, seed=2026)
            lat_wrap = LatencyWrapper(obs_delay_steps=1 if cond == "DR4_OBS_DELAY_1" else 0, action_delay_steps=0)
            model = SAC.load(model_path, env=env, device="cpu")
            ctrl = AdaptiveTransportController(ControllerConfig())
            succs, grasps, lifts = 0, 0, 0
            for s in eval_seeds:
                obs, info = env.reset(seed=s)
                ctrl.reset()
                noise_wrap.reset(episode_seed=s)
                t0_raw = get_step_telemetry(env)
                obs_p0, t0_p = noise_wrap.perturb(obs, t0_raw) if noise_bound > 0 else (obs, t0_raw)
                lat_wrap.reset(obs_p0, t0_p)
                ep_g, ep_l = False, False
                for _ in range(env.max_episode_steps):
                    t_raw = get_step_telemetry(env)
                    obs_n, t_n = noise_wrap.perturb(obs, t_raw) if noise_bound > 0 else (obs, t_raw)
                    obs_d, t_d = lat_wrap.push_and_get_obs(obs_n, t_n)
                    raw_act, _ = model.predict(obs_d, deterministic=True)
                    act, _ = ctrl.step_action(raw_act, t_d)
                    obs, _, term, trunc, info = env.step(act)
                    if t_raw.get("has_contact", False) or t_raw.get("contact_symmetry", False):
                        ep_g = True
                    if float(obs["achieved_goal"][2]) >= 0.075:
                        ep_l = True
                    if term or trunc:
                        break
                if info.get("is_success", False):
                    succs += 1
                if ep_g:
                    grasps += 1
                if ep_l:
                    lifts += 1
            env.close()
            results.append({
                "model": model_name,
                "condition": cond,
                "n_episodes": n_per_seed,
                "success_rate": round(succs / n_per_seed, 4),
                "grasp_rate": round(grasps / n_per_seed, 4),
                "lift_rate": round(lifts / n_per_seed, 4),
            })
            print(f"[PickPlace {model_name} | {cond:>18s}] Succ={succs/n_per_seed*100:5.1f}% | Grasp={grasps/n_per_seed*100:5.1f}% | Lift={lifts/n_per_seed*100:5.1f}%")
    return {"pickplace_dr_results": results}
