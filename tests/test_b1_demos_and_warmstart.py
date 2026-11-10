"""Part 4 & Part 5 Tests: Demonstration Dataset Validation & Safe HerReplayBuffer Initialization."""

from __future__ import annotations

import os
import pickle
import numpy as np
import pytest
import torch as th
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.monitor import Monitor

from src.environments.single_pregrasp_env import (
    PandaSinglePreGraspEnv,
    SinglePreGraspRewardConfig,
)


DEMO_PKL = "experiments/multi_sort_phase15_5B1/seed42/demonstrations/single_pregrasp_demos_50ep.pkl"


def insert_demos_into_her_buffer(replay_buffer: HerReplayBuffer, demos: list) -> dict:
    """Safely insert validated scripted episodes into HerReplayBuffer preserving boundaries."""
    inserted = 0
    for ep in demos:
        T = int(ep["length"])
        for t in range(T):
            obs_b = {k: np.expand_dims(v, axis=0) for k, v in ep["observations"][t].items()}
            next_obs_b = {k: np.expand_dims(v, axis=0) for k, v in ep["next_observations"][t].items()}
            act_b = np.expand_dims(ep["actions"][t], axis=0).astype(np.float32)
            rew_b = np.array([ep["rewards"][t]], dtype=np.float32)
            done_b = np.array([ep["dones"][t]], dtype=bool)
            info_rec = dict(ep["infos"][t])
            info_rec["is_demo"] = True
            replay_buffer.add(obs_b, next_obs_b, act_b, rew_b, done_b, [info_rec])
            inserted += 1
    return {
        "inserted_episodes": len(demos),
        "inserted_transitions": inserted,
        "buffer_size": replay_buffer.size(),
        "buffer_pos": int(replay_buffer.pos),
        "current_ep_start": int(replay_buffer._current_ep_start[0]),
    }


def test_demo_dataset_exists_and_all_50_valid():
    assert os.path.exists(DEMO_PKL)
    with open(DEMO_PKL, "rb") as f:
        demos = pickle.load(f)
    assert len(demos) == 50
    for ep in demos:
        assert ep["length"] == 80
        assert ep["hold_steps"] >= 15
        assert ep["metrics"]["premature_closure"] is False
        assert ep["metrics"]["table_collision_steps"] == 0
        assert ep["infos"][-1]["is_success"] is True
        assert ep["dones"][-1] is True and not any(ep["dones"][:-1])
        acts = np.asarray(ep["actions"], dtype=np.float32)
        assert acts.shape == (80, 4) and np.all(np.isfinite(acts)) and np.all(np.abs(acts) <= 1.0001)
        rews = np.asarray(ep["rewards"], dtype=np.float32)
        assert np.all(np.isfinite(rews))


def test_her_buffer_demo_warmstart_and_short_rl_step():
    with open(DEMO_PKL, "rb") as f:
        demos = pickle.load(f)

    rcfg = SinglePreGraspRewardConfig(use_vertical_potential=True, w_vertical_potential=2.0)
    env = Monitor(PandaSinglePreGraspEnv(render_mode=None, max_episode_steps=80, reward_config=rcfg))
    try:
        model = SAC(
            "MultiInputPolicy",
            env,
            seed=42,
            buffer_size=200000,
            batch_size=64,
            learning_rate=0.0003,
            learning_starts=100,
            gamma=0.98,
            tau=0.005,
            device="cpu",
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs={"n_sampled_goal": 4, "goal_selection_strategy": "future", "copy_info_dict": True},
            verbose=0,
        )
        stats = insert_demos_into_her_buffer(model.replay_buffer, demos[:5])
        assert stats["inserted_transitions"] == 400
        assert stats["buffer_pos"] == 400
        assert stats["current_ep_start"] == 400
        assert np.all(model.replay_buffer.ep_length[:400, 0] == 80)

        batch = model.replay_buffer.sample(64)
        assert th.all(th.isfinite(batch.rewards))
        # Verify online learn transitions append cleanly after pos=400
        model.learn(total_timesteps=160, progress_bar=False)
        assert model.replay_buffer.pos == 560
        assert np.all(model.replay_buffer.ep_length[400:560, 0] == 80)
        batch2 = model.replay_buffer.sample(64)
        assert th.all(th.isfinite(batch2.rewards))
    finally:
        env.close()

