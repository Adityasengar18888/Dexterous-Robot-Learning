"""Tests for Phase 15.2.5B SinglePreGraspEnv, HER compatibility, and B->C Policy Transfer."""

from __future__ import annotations

import numpy as np
import pytest
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.monitor import Monitor

from src.controllers.pregrasp_controller import (
    PreGraspConfig,
    PreGraspController,
    PreGraspStage,
    query_pybullet_telemetry,
)
from src.rl.train import load_config, make_env, _validate_config
from src.rl.transfer_utils import transfer_single_to_multi_sac


def test_single_pregrasp_registration_spaces_and_her():
    cfg = load_config("configs/multi_sort_phase15_5B.yaml")
    _validate_config(cfg)
    env = Monitor(make_env(cfg["env"]))
    obs = env.reset(seed=42)[0]
    assert obs["observation"].shape == (22,)
    assert obs["achieved_goal"].shape == (3,)
    assert obs["desired_goal"].shape == (3,)
    assert env.action_space.shape == (4,)

    rl = cfg["rl"]
    model = SAC(
        rl["policy"], env, seed=rl["seed"], buffer_size=2000, batch_size=32,
        learning_rate=rl["learning_rate"], learning_starts=100, gamma=rl["gamma"], tau=rl["tau"],
        device="cpu", replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={"n_sampled_goal": 4, "goal_selection_strategy": "future", "copy_info_dict": True},
        verbose=0,
    )
    v_obs = model.env.reset()
    for s in range(30):
        act = np.array([env.action_space.sample()], dtype=np.float32)
        n_obs, r, d, infos = model.env.step(act)
        if s == 29:
            d = np.array([True], dtype=bool)
            infos[0]["TimeLimit.truncated"] = True
        model.replay_buffer.add(v_obs, n_obs, act, r, d, infos)
        v_obs = n_obs
    batch = model.replay_buffer.sample(16)
    rew = batch.rewards.detach().cpu().numpy()
    assert rew.shape == (16, 1) and np.all(np.isfinite(rew))
    env.close()


def test_single_pregrasp_scripted_reward_audit():
    env = make_env({"name": "SinglePreGrasp", "max_episode_steps": 80})
    obs, _ = env.reset(seed=1001)
    ctrl = PreGraspController(config=PreGraspConfig(lift_verify_z_min=0.11), action_scale=env.action_scale)
    ctrl.reset(active_obj_idx=0, anchor_obj_pos=obs["achieved_goal"][:3])
    total_rew = 0.0
    for step in range(1, 80):
        telem = query_pybullet_telemetry(env, active_obj_idx=0)
        cmd, _ = ctrl.step(step, telem)
        obs, rew, _, _, info = env.step(cmd)
        total_rew += float(rew)
        assert np.isfinite(rew)
        if info["is_success"]:
            break
    assert info["is_success"] is True
    assert total_rew > 5.0
    env.close()


def test_b_to_c_policy_transfer_exact_functional_equivalence():
    env_b = Monitor(make_env({"name": "SinglePreGrasp", "max_episode_steps": 80}))
    env_c = Monitor(make_env({"name": "MultiSortPreGrasp", "num_objects": 2, "max_episode_steps": 300}))
    src = SAC("MultiInputPolicy", env_b, seed=42, buffer_size=1000, verbose=0, device="cpu")
    tgt = SAC("MultiInputPolicy", env_c, seed=42, buffer_size=1000, verbose=0, device="cpu")

    report = transfer_single_to_multi_sac(src, tgt)
    assert report["transferred_params"] > 0
    assert report["mapped_input_cols"] == 28

    # Construct matching B and C observations (with arbitrary non-zero Object 2 / Goal 2 in C!)
    obs_b, _ = env_b.unwrapped.reset(seed=77)
    o_c_base, _ = env_c.unwrapped.reset(seed=77)
    obs_c = {
        "observation": np.concatenate([
            obs_b["observation"][0:4],          # ee + gw
            obs_b["observation"][4:13],         # obj0
            o_c_base["observation"][13:22],     # arbitrary non-zero obj1!
            obs_b["observation"][13:22],        # bins
        ]).astype(np.float32),
        "achieved_goal": np.concatenate([obs_b["achieved_goal"], o_c_base["achieved_goal"][3:6]]).astype(np.float32),
        "desired_goal": np.concatenate([obs_b["desired_goal"], o_c_base["desired_goal"][3:6]]).astype(np.float32),
    }

    act_b, _ = src.predict(obs_b, deterministic=True)
    act_c, _ = tgt.predict(obs_c, deterministic=True)
    assert np.allclose(act_b, act_c, atol=1e-5)
    env_b.close()
    env_c.close()

