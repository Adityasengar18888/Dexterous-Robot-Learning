"""Phase 17.1 (MuJoCo Cross-Sim Validation) & Phase 17.2 (Sim-to-Real Safety/Stress Suite)."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

import numpy as np
from stable_baselines3 import SAC

from src.cross_sim.mujoco_panda_env import MuJoCoPandaPickPlaceEnv
from src.deployment.runtime import (
    ActionPostprocessor,
    CheckpointLoader,
    DeploymentRuntime,
    EnvironmentAdapter,
    ObsPreprocessor,
    PolicyInferenceInterface,
    RuntimeConfig,
    SafetyMonitor,
)
from src.deployment.hardware_interface import (
    CartesianDeltaCommand,
    FrankaState,
    HardwareSafetyConfig,
    SafetySupervisor,
)
from src.environments.pick_place_env import PandaPickPlaceEnv
from src.hrl.hrl_agent import HierarchicalManipulatorAgent
from src.hrl.skills import SkillLibrary, extract_skill_features
from src.perception.rgbd_camera import CameraConfig, SimulatedRGBDCamera
from src.perception.vision_detector import VisionObjectLocalizer


def evaluate_in_mujoco(
    policy_name: str,
    seeds: List[int],
    sac_model: SAC | None = None,
    hrl_agent: HierarchicalManipulatorAgent | None = None,
    cube_mass: float = 0.1,
    cube_friction: float = 1.0,
    action_scale: float = 0.05,
    obs_ee_z_offset: float = 0.0,
    grip_force_scale: float = 1.0,
) -> Dict[str, Any]:
    """Run frozen PyBullet-trained policy inside MuJoCo (zero retraining)."""
    env = MuJoCoPandaPickPlaceEnv(
        max_episode_steps=100,
        action_scale=action_scale,
        cube_mass=cube_mass,
        cube_friction=cube_friction,
        gripper_force_scale=grip_force_scale,
    )
    succs, grasps, bilats, min_ee_obj_dists, min_goal_dists = [], [], [], [], []
    t0 = time.perf_counter()

    try:
        for s in seeds:
            obs, info = env.reset(seed=s)
            any_g, any_b = False, False
            min_d_ee = 1.0
            min_d_g = 1.0
            cur_skill = 0
            opt_steps = 0
            last_info = dict(info)

            for _ in range(100):
                p_obs = {
                    "observation": obs["observation"].copy(),
                    "achieved_goal": obs["achieved_goal"].copy(),
                    "desired_goal": obs["desired_goal"].copy(),
                }
                if obs_ee_z_offset != 0.0:
                    p_obs["observation"][2] += obs_ee_z_offset

                if sac_model is not None:
                    act, _ = sac_model.predict(p_obs, deterministic=True)
                else:
                    assert hrl_agent is not None
                    feat = extract_skill_features(p_obs)
                    if opt_steps == 0:
                        cur_skill, _, _ = hrl_agent.select_skill(p_obs, deterministic=True)
                    else:
                        _, tp = hrl_agent.skill_lib.act(cur_skill, feat)
                        if tp >= 0.5 or opt_steps >= hrl_agent.max_option_steps:
                            cur_skill, _, _ = hrl_agent.select_skill(p_obs, deterministic=True)
                            opt_steps = 0
                    act, _ = hrl_agent.skill_lib.act(cur_skill, feat)
                    opt_steps += 1

                obs, _, term, trunc, last_info = env.step(np.asarray(act, dtype=np.float32))
                min_d_ee = min(min_d_ee, float(np.linalg.norm(obs["observation"][:3] - obs["achieved_goal"][:3])))
                min_d_g = min(min_d_g, float(last_info["distance"]))
                any_g = any_g or bool(last_info.get("grasp_success", False))
                any_b = any_b or bool(last_info.get("bilateral_contact", False))
                if term or trunc:
                    break

            succs.append(float(last_info.get("is_success", False)))
            grasps.append(float(any_g))
            bilats.append(float(any_b))
            min_ee_obj_dists.append(min_d_ee)
            min_goal_dists.append(min_d_g)
    finally:
        env.close()

    return {
        "policy": policy_name,
        "n_episodes": len(seeds),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "success_rate": round(float(np.mean(succs)), 4),
        "grasp_rate": round(float(np.mean(grasps)), 4),
        "bilateral_contact_rate": round(float(np.mean(bilats)), 4),
        "mean_min_ee_obj_dist_m": round(float(np.mean(min_ee_obj_dists)), 4),
        "mean_min_goal_dist_m": round(float(np.mean(min_goal_dists)), 4),
    }


def run_sim_to_real_stress_suite(sac_model: SAC, seeds: List[int]) -> Dict[str, Any]:
    """Phase 17.2 simulation-side Sim-to-Real stress tests (NOT hardware proof)."""
    conditions = [
        {"name": "A0_Clean_Nominal", "ee_noise": 0.0, "obj_noise": 0.0, "delay": 0, "ctrl_jitter": False, "mass": 0.1, "fric": 1.0, "cam_calib_err": 0.0},
        {"name": "S1_Sensor_Noise_5mm", "ee_noise": 0.005, "obj_noise": 0.005, "delay": 0, "ctrl_jitter": False, "mass": 0.1, "fric": 1.0, "cam_calib_err": 0.0},
        {"name": "S2_Camera_Calib_Err_8mm", "ee_noise": 0.0, "obj_noise": 0.0, "delay": 0, "ctrl_jitter": False, "mass": 0.1, "fric": 1.0, "cam_calib_err": 0.008},
        {"name": "S3_Actuator_Latency_1step", "ee_noise": 0.0, "obj_noise": 0.0, "delay": 1, "ctrl_jitter": False, "mass": 0.1, "fric": 1.0, "cam_calib_err": 0.0},
        {"name": "S4_Actuator_Latency_2step", "ee_noise": 0.0, "obj_noise": 0.0, "delay": 2, "ctrl_jitter": False, "mass": 0.1, "fric": 1.0, "cam_calib_err": 0.0},
        {"name": "S5_Control_Freq_Jitter", "ee_noise": 0.0, "obj_noise": 0.0, "delay": 0, "ctrl_jitter": True, "mass": 0.1, "fric": 1.0, "cam_calib_err": 0.0},
        {"name": "S6_Low_Friction_Heavy_Mass", "ee_noise": 0.0, "obj_noise": 0.0, "delay": 0, "ctrl_jitter": False, "mass": 0.25, "fric": 0.45, "cam_calib_err": 0.0},
    ]
    results: Dict[str, Any] = {}
    import pybullet as p

    _pp_dr_cfg = {"enabled": True, "object_position": {"enabled": True}, "target_position": {"enabled": True}}
    for cfg in conditions:
        env = PandaPickPlaceEnv(render_mode=None, max_episode_steps=100, randomizer_config=_pp_dr_cfg)
        cam = None
        det = None
        safety = SafetyMonitor(RuntimeConfig(env_name="PickPlace", workspace_min=(0.22, -0.32, 0.012), workspace_max=(0.72, 0.32, 0.48)))

        succs, grasps, colls, clamps = [], [], [], []
        try:
            for s in seeds:
                rng = np.random.default_rng(s + 99)
                obs, _ = env.reset(seed=s)
                if cfg["cam_calib_err"] > 0 and cam is None:
                    cam = SimulatedRGBDCamera(env.sim.client_id, CameraConfig(width=128, height=128))
                    det = VisionObjectLocalizer(cam, cube_half_extent_m=0.025)
                safety.reset()
                if cfg["mass"] != 0.1 or cfg["fric"] != 1.0:
                    p.changeDynamics(env.cube.body_id, -1, mass=cfg["mass"], lateralFriction=cfg["fric"], physicsClientId=env.sim.client_id)
                obs_queue: List[Dict[str, np.ndarray]] = []
                any_g = False
                last_info: Dict[str, Any] = {}
                calib_bias = rng.uniform(-cfg["cam_calib_err"], cfg["cam_calib_err"], size=3).astype(np.float32)

                for _ in range(100):
                    if cfg["ctrl_jitter"]:
                        env.control_steps = int(rng.integers(10, 21))
                    else:
                        env.control_steps = 15

                    cur = {k: v.copy() for k, v in obs.items()}
                    if cam is not None and det is not None:
                        rgb, depth_m, raw_d, _ = cam.capture()
                        d_out = det.detect_red_cube(rgb, raw_d, depth_m)
                        if d_out.detected:
                            vis_p = d_out.position_m + calib_bias
                            cur["observation"][4:7] = vis_p
                            cur["achieved_goal"] = vis_p

                    if cfg["ee_noise"] > 0:
                        cur["observation"][:3] += rng.normal(0.0, cfg["ee_noise"], size=3).astype(np.float32)
                    if cfg["obj_noise"] > 0:
                        noise = rng.normal(0.0, cfg["obj_noise"], size=3).astype(np.float32)
                        cur["observation"][4:7] += noise
                        cur["achieved_goal"] += noise

                    obs_queue.append(cur)
                    pol_obs = obs_queue[max(0, len(obs_queue) - 1 - cfg["delay"])]

                    act, _ = sac_model.predict(pol_obs, deterministic=True)
                    safe_act, _ = safety.check_and_filter(cur["observation"][:3], np.asarray(act, dtype=np.float32))
                    obs, _, term, trunc, last_info = env.step(safe_act)
                    if obs["achieved_goal"][2] > 0.055:
                        any_g = True
                    if term or trunc:
                        break
                succs.append(float(last_info.get("is_success", False)))
                grasps.append(float(any_g or last_info.get("grasp_success", False)))
                colls.append(float(last_info.get("collisions", 0)))
                clamps.append(float(safety.ws_clamps))
        finally:
            env.close()

        results[cfg["name"]] = {
            "n_episodes": len(seeds),
            "success_rate": round(float(np.mean(succs)), 4),
            "grasp_rate": round(float(np.mean(grasps)), 4),
            "mean_collisions": round(float(np.mean(colls)), 4),
            "mean_safety_ws_clamps": round(float(np.mean(clamps)), 3),
        }

    # Explicit E-stop latency & SafetySupervisor unit stress benchmark
    sup = SafetySupervisor(HardwareSafetyConfig())
    t_e0 = time.perf_counter()
    for i in range(500):
        st = FrankaState(
            q=np.zeros(7, dtype=np.float32),
            dq=np.zeros(7, dtype=np.float32),
            ee_pos=np.array([0.5, 0.0, 0.2], dtype=np.float32),
            ee_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            gripper_width=0.05,
            ext_wrench=np.array([50.0 if i == 250 else 2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        sup.validate_command(st, CartesianDeltaCommand(np.array([0.02, 0.0, 0.0], dtype=np.float32)))
    estop_avg_us = (time.perf_counter() - t_e0) * 1e6 / 500.0
    results["SafetySupervisor_EStop_Check"] = {
        "estop_triggered_on_50N_spike": bool(sup.estop_latched),
        "blocked_post_trip_commands": int(sup.blocked_commands),
        "mean_check_latency_us": round(estop_avg_us, 2),
        "physical_safety_disclaimer": "Simulation-only check; NOT proof of physical hardware safety.",
    }
    return results


def main() -> None:
    os.makedirs("results/phase16_17", exist_ok=True)
    sac42 = CheckpointLoader.load_sb3_sac("experiments/phase13_dense_dr/dr4/seed42/models/checkpoint_500000.zip", device="cpu")
    sac43 = CheckpointLoader.load_sb3_sac("experiments/phase13_dense_dr/dr4/seed43/models/checkpoint_500000.zip", device="cpu")
    skill_lib = SkillLibrary(in_dim=16)
    skill_lib.load("experiments/phase16_17/hrl/skill_library.pt")
    hrl = HierarchicalManipulatorAgent(skill_lib=skill_lib, seed=42)
    hrl.load("experiments/phase16_17/hrl/best_hrl_meta.pt")

    seeds_20 = list(range(8600, 8620))

    # Zero-shot frozen PyBullet policies in MuJoCo (no retraining!)
    mj_zeroshot_42 = evaluate_in_mujoco("PyBullet_SAC_DR4_Seed42_ZeroShot", seeds_20, sac_model=sac42)
    mj_zeroshot_43 = evaluate_in_mujoco("PyBullet_SAC_DR4_Seed43_ZeroShot", seeds_20, sac_model=sac43)
    mj_zeroshot_hrl = evaluate_in_mujoco("PyBullet_HRL_ZeroShot", seeds_20, hrl_agent=hrl)

    # Calibrated physics / contact adaptation test in MuJoCo (higher gripper force & friction)
    mj_calib_42 = evaluate_in_mujoco(
        "PyBullet_SAC_DR4_Seed42_CalibratedMuJoCo",
        seeds_20,
        sac_model=sac42,
        cube_friction=1.5,
        grip_force_scale=2.0,
    )
    mj_calib_hrl = evaluate_in_mujoco(
        "PyBullet_HRL_CalibratedMuJoCo",
        seeds_20,
        hrl_agent=hrl,
        cube_friction=1.5,
        grip_force_scale=2.0,
    )

    cross_sim = {
        "zero_shot_eval": {
            "sac_dr4_seed42": mj_zeroshot_42,
            "sac_dr4_seed43": mj_zeroshot_43,
            "hrl_meta": mj_zeroshot_hrl,
        },
        "calibrated_physics_eval": {
            "sac_dr4_seed42": mj_calib_42,
            "hrl_meta": mj_calib_hrl,
        },
        "physics_comparison": {
            "pybullet_timestep_s": 1.0 / 240.0,
            "mujoco_timestep_s": 1.0 / 240.0,
            "cube_mass_kg": 0.1,
            "cube_size_m": 0.05,
            "pybullet_ee_init": [0.307, 0.0, 0.590],
        },
    }
    with open("results/phase16_17/phase17_cross_simulator_results.json", "w", encoding="utf-8") as f:
        json.dump(cross_sim, f, indent=2)
    print("[17.1] MuJoCo zero-shot:", {k: (v["success_rate"], v["grasp_rate"]) for k, v in cross_sim["zero_shot_eval"].items()})
    print("[17.1] MuJoCo calibrated:", {k: (v["success_rate"], v["grasp_rate"]) for k, v in cross_sim["calibrated_physics_eval"].items()})

    safety_res = run_sim_to_real_stress_suite(sac42, seeds=seeds_20)
    with open("results/phase16_17/phase17_safety_assessment.json", "w", encoding="utf-8") as f:
        json.dump(safety_res, f, indent=2)
    print("[17.2] Sim-to-Real stress:", {k: v.get("success_rate") for k, v in safety_res.items() if "success_rate" in v})


if __name__ == "__main__":
    main()

