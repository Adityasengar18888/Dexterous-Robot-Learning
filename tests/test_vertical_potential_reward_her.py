"""Part 3 Synthetic & Live HER Compatibility Tests for Continuous Vertical Potential Reward."""

from __future__ import annotations

import numpy as np
import pytest
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.monitor import Monitor

from src.environments.single_pregrasp_env import (
    PandaSinglePreGraspEnv,
    SinglePreGraspRewardConfig,
)


def _make_b1_env(w_z: float = 2.0) -> PandaSinglePreGraspEnv:
    cfg = SinglePreGraspRewardConfig(
        use_vertical_potential=True,
        w_vertical_potential=w_z,
    )
    return PandaSinglePreGraspEnv(render_mode=None, max_episode_steps=80, reward_config=cfg)


def _synth_info(
    prev_z: float,
    curr_z: float,
    obj_xy=(0.45, 0.0),
    prev_obj_z: float = 0.025,
    curr_obj_z: float = 0.025,
    ee_xy=(0.45, 0.0),
    gw: float = 0.08,
    tcoll: bool = False,
    orn_ok: bool = True,
    bilat: bool = False,
):
    return {
        "prev_ee_pos": np.array([ee_xy[0], ee_xy[1], prev_z], dtype=np.float32),
        "ee_pos": np.array([ee_xy[0], ee_xy[1], curr_z], dtype=np.float32),
        "prev_ag": np.array([obj_xy[0], obj_xy[1], prev_obj_z], dtype=np.float32),
        "prev_gw": float(gw),
        "gripper_width": float(gw),
        "prev_bilateral_contact": bool(bilat),
        "bilateral_contact": bool(bilat),
        "prev_orn_ok": bool(orn_ok),
        "orn_ok": bool(orn_ok),
        "prev_table_collision": bool(tcoll),
        "table_collision": bool(tcoll),
    }


def test_vertical_potential_all_part3_cases():
    env = _make_b1_env(w_z=2.0)
    try:
        ag = np.array([0.45, 0.0, 0.025], dtype=np.float32)
        dg = np.array([0.45, 0.0, 0.140], dtype=np.float32)

        # 1. Downward progress (open gripper, ungrasped, far from XY -> STILL gets vertical progress!)
        env.compute_reward(ag, dg, _synth_info(0.40, 0.35, ee_xy=(0.30, 0.15)))
        c_down = dict(env._last_reward_components)
        assert np.isclose(c_down["r_vert_progress"], 0.10, atol=1e-5)
        assert np.isclose(c_down["r_vert_potential"], -2.0 * (0.35 - 0.025), atol=1e-5)

        # 2. Upward movement away from target
        env.compute_reward(ag, dg, _synth_info(0.35, 0.40, ee_xy=(0.30, 0.15)))
        c_up = dict(env._last_reward_components)
        assert np.isclose(c_up["r_vert_progress"], -0.10, atol=1e-5)
        # Oscillation sum = 0.0
        assert np.isclose(c_down["r_vert_progress"] + c_up["r_vert_progress"], 0.0, atol=1e-6)

        # 3. Stationary hovering
        env.compute_reward(ag, dg, _synth_info(0.48, 0.48, ee_xy=(0.45, 0.0)))
        c_hover = dict(env._last_reward_components)
        assert np.isclose(c_hover["r_vert_progress"], 0.0, atol=1e-6)

        # 4. Reaching target height & stopping shaping once valid grasp height is reached
        env.compute_reward(ag, dg, _synth_info(0.060, 0.030))
        c_enter = dict(env._last_reward_components)
        assert c_enter["r_vert_progress"] > 0.05

        env.compute_reward(ag, dg, _synth_info(0.031, 0.027))
        c_inside = dict(env._last_reward_components)
        assert np.isclose(c_inside["r_vert_progress"], 0.0, atol=1e-6)

        # 5. Movement below safe grasp height / table collision
        env.compute_reward(ag, dg, _synth_info(0.025, 0.015, tcoll=False))
        c_below = dict(env._last_reward_components)
        assert c_below["r_vert_progress"] < 0.0

        env.compute_reward(ag, dg, _synth_info(0.030, 0.025, tcoll=True))
        c_tcoll = dict(env._last_reward_components)
        assert c_tcoll["r_vert_progress"] <= 0.0
        assert c_tcoll["r_collision"] < 0.0

        # Closed gripper during descent must NOT receive positive vertical reward
        env.compute_reward(ag, dg, _synth_info(0.40, 0.35, gw=0.01))
        c_closed = dict(env._last_reward_components)
        assert c_closed["r_vert_progress"] == 0.0
        assert c_closed["r_open_comp"] < 0.0
    finally:
        env.close()


def test_vertical_potential_her_relabeling_batched_parity_and_active_object_change():
    env = _make_b1_env(w_z=2.0)
    try:
        # 6 & 7. Relabeled goals + active-object position height changes + batched/single parity
        cases = [
            # Standard cube at z=0.025, relabeled goal A
            (
                np.array([0.45, 0.0, 0.025], dtype=np.float32),
                np.array([0.45, 0.0, 0.140], dtype=np.float32),
                _synth_info(0.35, 0.30, prev_obj_z=0.025, curr_obj_z=0.025),
            ),
            # Relabeled goal B (identical active object -> identical r_vert_progress)
            (
                np.array([0.45, 0.0, 0.025], dtype=np.float32),
                np.array([0.45, 0.0, 0.025], dtype=np.float32),
                _synth_info(0.35, 0.30, prev_obj_z=0.025, curr_obj_z=0.025),
            ),
            # Active object change (e.g. second object resting higher z=0.055 -> grasp height changes dynamically!)
            (
                np.array([0.52, -0.10, 0.055], dtype=np.float32),
                np.array([0.52, -0.10, 0.140], dtype=np.float32),
                _synth_info(0.055, 0.055, obj_xy=(0.52, -0.10), ee_xy=(0.52, -0.10), prev_obj_z=0.055, curr_obj_z=0.055),
            ),
        ]

        single_rews = []
        for ag_i, dg_i, info_i in cases:
            r_s = env.compute_reward(ag_i, dg_i, info_i)
            single_rews.append(r_s)

        # Verify Case 0 vs Case 1 relabeled goal has identical vertical potential terms
        env.compute_reward(cases[0][0], cases[0][1], cases[0][2])
        v0 = env._last_reward_components["r_vert_progress"]
        env.compute_reward(cases[1][0], cases[1][1], cases[1][2])
        v1 = env._last_reward_components["r_vert_progress"]
        assert np.isclose(v0, v1, atol=1e-6)

        # Verify Case 2 (active object at z=0.055) has target_z = 0.055 -> phi_z = 0.0 when ee_z = 0.055!
        env.compute_reward(cases[2][0], cases[2][1], cases[2][2])
        assert np.isclose(env._last_reward_components["r_vert_potential"], 0.0, atol=1e-6)

        # Batched vs single agreement (list-of-dicts as passed by SB3 HerReplayBuffer)
        ag_b = np.stack([c[0] for c in cases], axis=0)
        dg_b = np.stack([c[1] for c in cases], axis=0)
        info_list = [c[2] for c in cases]
        rew_b = env.compute_reward(ag_b, dg_b, info_list)
        assert rew_b.shape == (3,)
        assert np.allclose(rew_b, np.array(single_rews, dtype=np.float32), atol=1e-5)
    finally:
        env.close()


def test_b1_live_her_replay_buffer_recomputation():
    env = Monitor(_make_b1_env(w_z=2.0))
    model = SAC(
        "MultiInputPolicy",
        env,
        seed=42,
        buffer_size=2000,
        batch_size=32,
        learning_starts=100,
        device="cpu",
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={"n_sampled_goal": 4, "goal_selection_strategy": "future", "copy_info_dict": True},
        verbose=0,
    )
    obs = model.env.reset()
    for step in range(40):
        act = np.array([[0.0, 0.0, -0.8, 1.0]], dtype=np.float32)
        next_obs, rew, dones, infos = model.env.step(act)
        if step == 39:
            dones = np.array([True], dtype=bool)
            infos[0]["TimeLimit.truncated"] = True
        model.replay_buffer.add(obs, next_obs, act, rew, dones, infos)
        obs = next_obs
    batch = model.replay_buffer.sample(32)
    rewards = batch.rewards.detach().cpu().numpy()
    assert rewards.shape == (32, 1)
    assert np.all(np.isfinite(rewards))
    env.close()

