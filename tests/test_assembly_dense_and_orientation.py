"""Unit & HER tests for Phase 4 Orientation-Aware Assembly (Quaternion math, contacts, HER parity)."""

from __future__ import annotations

import numpy as np
import pybullet as p
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.monitor import Monitor

from src.environments.assembly_dense_env import (
    PandaAssemblyDenseEnv,
    normalize_quaternion,
    quaternion_geodesic_error,
    wrap_angle,
)


def test_quaternion_normalization_sign_invariance_and_wraparound():
    q = normalize_quaternion(np.array([0.0, 0.0, 2.0, 2.0], dtype=np.float32))
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)
    dc, rad = quaternion_geodesic_error(q, -q)
    assert np.isclose(dc, 0.0, atol=1e-6)
    assert np.isclose(rad, 0.0, atol=1e-6)

    q90 = np.array(p.getQuaternionFromEuler([0.0, 0.0, np.pi / 2.0]), dtype=np.float32)
    q0 = np.array(p.getQuaternionFromEuler([0.0, 0.0, 0.0]), dtype=np.float32)
    dc90, rad90 = quaternion_geodesic_error(q0, q90)
    assert np.isclose(rad90, np.pi / 2.0, atol=1e-4)
    assert np.isclose(dc90, 0.5, atol=1e-4)

    assert np.isclose(wrap_angle(np.pi + 0.25), -np.pi + 0.25, atol=1e-6)
    assert np.isclose(wrap_angle(-np.pi - 0.25), np.pi - 0.25, atol=1e-6)


def test_assembly_dense_batched_parity_and_her_buffer():
    env = Monitor(PandaAssemblyDenseEnv(render_mode=None, max_episode_steps=120))
    try:
        model = SAC(
            "MultiInputPolicy",
            env,
            seed=42,
            buffer_size=2000,
            batch_size=16,
            learning_starts=50,
            device="cpu",
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs={"n_sampled_goal": 4, "goal_selection_strategy": "future", "copy_info_dict": True},
            verbose=0,
        )
        obs = model.env.reset()
        ags, dgs, infos_collected, single_rews = [], [], [], []
        for s in range(15):
            act = np.array([[0.0, 0.0, -0.2, 0.1, 1.0]], dtype=np.float32)
            n_obs, r, d, infos = model.env.step(act)
            if s == 14:
                d = np.array([True], dtype=bool)
                infos[0]["TimeLimit.truncated"] = True
            model.replay_buffer.add(obs, n_obs, act, r, d, infos)
            ags.append(n_obs["achieved_goal"][0])
            dgs.append(n_obs["desired_goal"][0])
            infos_collected.append(dict(infos[0]))
            single_rews.append(float(r[0]))
            obs = n_obs

        batched_rew = env.unwrapped.compute_reward(np.stack(ags), np.stack(dgs), infos_collected)
        assert batched_rew.shape == (15,)
        assert np.allclose(batched_rew, np.array(single_rews, dtype=np.float32), atol=1e-5)

        batch = model.replay_buffer.sample(16)
        assert np.all(np.isfinite(batch.rewards.detach().cpu().numpy()))
    finally:
        env.close()

