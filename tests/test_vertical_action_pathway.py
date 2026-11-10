"""Part 2 Vertical Action-Pathway Diagnostic Tests for PandaSinglePreGraspEnv."""

import numpy as np
import pybullet as p

from src.environments.single_pregrasp_env import PandaSinglePreGraspEnv
from src.rl.train import make_env as train_make_env
from src.rl.evaluate import make_env as eval_make_env


def _run_vertical_probe(action_vec: np.ndarray, n_steps: int = 10, seed: int = 42):
    env = PandaSinglePreGraspEnv(render_mode=None, max_episode_steps=80)
    try:
        obs, _ = env.reset(seed=seed)
        start_ee = obs["observation"][:3].copy()
        robot = env.controller.robot
        cid = env.sim.client_id
        table_id = env.sim.table_id
        joint_limit_events = 0
        table_collisions = 0
        max_ik_err = 0.0
        step_logs = []

        for s in range(n_steps):
            prev_ee = env.controller.get_ee_position().copy()
            obs, _, _, _, info = env.step(np.asarray(action_vec, dtype=np.float32))
            curr_ee = obs["observation"][:3].copy()
            dz_actual = float(curr_ee[2] - prev_ee[2])
            joints = env.controller.get_joint_positions()
            at_limit = any(
                joints[i] <= robot.joint_lower_limits[i] + 1e-3
                or joints[i] >= robot.joint_upper_limits[i] - 1e-3
                for i in range(7)
            )
            if at_limit:
                joint_limit_events += 1
            if bool(info.get("table_collision", False)) or len(p.getContactPoints(bodyA=robot.body_id, bodyB=table_id, physicsClientId=cid)) > 0:
                table_collisions += 1

            raw_target = prev_ee + np.clip(action_vec[:3], -1.0, 1.0) * env.action_scale
            clipped_target = robot.clip_to_workspace(raw_target)
            ik_err = float(np.linalg.norm(curr_ee - clipped_target))
            max_ik_err = max(max_ik_err, ik_err)

            step_logs.append(
                {
                    "step": s + 1,
                    "cmd_z": float(action_vec[2]),
                    "prev_z": float(prev_ee[2]),
                    "curr_z": float(curr_ee[2]),
                    "dz_actual": dz_actual,
                    "gw": float(obs["observation"][3]),
                    "ik_err": ik_err,
                    "at_limit": at_limit,
                    "table_coll": bool(info.get("table_collision", False)),
                }
            )
        end_ee = obs["observation"][:3].copy()
        return {
            "start_ee": start_ee,
            "end_ee": end_ee,
            "total_dz": float(end_ee[2] - start_ee[2]),
            "joint_limit_events": joint_limit_events,
            "table_collisions": table_collisions,
            "max_ik_err": max_ik_err,
            "end_gw": float(obs["observation"][3]),
            "obs": obs,
            "steps": step_logs,
        }
    finally:
        env.close()


def test_observation_and_action_pathway_spec():
    cfg = {
        "name": "SinglePreGrasp",
        "distance_threshold": 0.05,
        "max_episode_steps": 80,
        "action_scale": 0.05,
    }
    env_tr = train_make_env(cfg, render_mode=None)
    env_ev = eval_make_env(cfg, render_mode=None)
    try:
        obs_tr, _ = env_tr.reset(seed=42)
        obs_ev, _ = env_ev.reset(seed=42)
        assert env_tr.action_space.shape == (4,) == env_ev.action_space.shape
        assert np.allclose(env_tr.action_space.low, -1.0)
        assert np.allclose(env_tr.action_space.high, 1.0)
        assert env_tr.action_scale == env_ev.action_scale == 0.05
        # Verify obs contains EE pose [0:3], gripper [3], cube pos [4:7]
        assert obs_tr["observation"].shape == (22,)
        assert np.allclose(obs_tr["observation"], obs_ev["observation"], atol=1e-5)
        ee_z = float(obs_tr["observation"][2])
        obj_z = float(obs_tr["observation"][6])
        assert np.isclose(ee_z, float(env_tr.controller.get_ee_position()[2]), atol=1e-5)
        assert np.isclose(obj_z, float(env_tr.cubes[0].get_position()[2]), atol=1e-5)
    finally:
        env_tr.close()
        env_ev.close()


def test_negative_vertical_command_produces_measurable_descent_and_gripper_independence():
    res_neg_open = _run_vertical_probe(np.array([0.0, 0.0, -1.0, +1.0]), n_steps=8, seed=42)
    res_neg_close = _run_vertical_probe(np.array([0.0, 0.0, -1.0, -1.0]), n_steps=8, seed=42)
    res_zero = _run_vertical_probe(np.array([0.0, 0.0, 0.0, +1.0]), n_steps=8, seed=42)
    res_small = _run_vertical_probe(np.array([0.0, 0.0, -0.1, +1.0]), n_steps=8, seed=42)
    res_pos = _run_vertical_probe(np.array([0.0, 0.0, +1.0, +1.0]), n_steps=8, seed=42)

    # Downward command (-1.0) must descend cleanly (> 0.30 m over 8 steps of 0.05 m)
    assert res_neg_open["total_dz"] <= -0.30, f"Expected strong descent, got {res_neg_open['total_dz']}"
    assert res_small["total_dz"] <= -0.025, f"Expected small descent, got {res_small['total_dz']}"
    assert abs(res_zero["total_dz"]) <= 0.015, f"Expected near-zero z drift, got {res_zero['total_dz']}"
    assert res_pos["total_dz"] >= -0.015, f"Positive command from home ceiling should not descend: {res_pos['total_dz']}"
    # No joint limits or table collision during mid-air descent
    assert res_neg_open["joint_limit_events"] == 0
    assert res_neg_open["table_collisions"] == 0
    # Gripper independence: open (+1) vs close (-1) achieve nearly identical EE z
    assert abs(res_neg_open["end_ee"][2] - res_neg_close["end_ee"][2]) <= 0.005
    assert res_neg_open["end_gw"] > 0.070
    assert res_neg_close["end_gw"] < 0.010

