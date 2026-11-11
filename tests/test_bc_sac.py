"""Tests for Phase 1 BCDecayingSAC + Balanced Demo Distribution."""

from __future__ import annotations

import json
import os
import tempfile
import numpy as np
import torch as th
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.monitor import Monitor

from src.environments.single_pregrasp_env import PandaSinglePreGraspEnv, SinglePreGraspRewardConfig
from src.rl.bc_sac import BCDecayingSAC
from tests.test_b1_demos_and_warmstart import insert_demos_into_her_buffer


DEMO_PKL = "experiments/multi_sort_phase15_5B2/seed42/demonstrations/balanced_pregrasp_demos_50ep.pkl"
DEMO_JSON = "experiments/multi_sort_phase15_5B2/seed42/demonstrations/balanced_demo_summary.json"


def test_balanced_demo_distribution():
    assert os.path.exists(DEMO_PKL) and os.path.exists(DEMO_JSON)
    summary = json.load(open(DEMO_JSON))
    assert summary["accepted_episodes"] == 50
    assert summary["total_table_collisions"] == 0
    assert summary["total_premature_closures"] == 0
    pct = summary["distribution_pct"]
    assert 65.0 <= pct["open_approach_align_descent_pct"] <= 78.0
    assert pct["open_gripper_cmd_pct"] >= 70.0


def test_bc_decaying_sac_objective_entropy_floor_and_standard_save():
    import pickle
    demos = pickle.load(open(DEMO_PKL, "rb"))[:4]
    rcfg = SinglePreGraspRewardConfig(use_vertical_potential=True, w_vertical_potential=2.0)
    env = Monitor(PandaSinglePreGraspEnv(render_mode=None, max_episode_steps=80, reward_config=rcfg))
    try:
        model = BCDecayingSAC(
            "MultiInputPolicy",
            env,
            seed=42,
            buffer_size=5000,
            batch_size=32,
            learning_starts=50,
            device="cpu",
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs={"n_sampled_goal": 4, "goal_selection_strategy": "future", "copy_info_dict": True},
            bc_demos_path=DEMO_PKL,
            bc_lambda_init=0.305,
            bc_lambda_final=0.02,
            bc_decay_steps=5000,
            bc_batch_size=32,
            min_ent_coef=0.05,
            verbose=0,
        )
        summary = json.load(open(DEMO_JSON))
        assert model._demo_n == summary["total_transitions"]
        assert np.isclose(model.current_bc_lambda(), 0.305, atol=1e-5)
        model.num_timesteps = 5000
        assert np.isclose(model.current_bc_lambda(), 0.02, atol=1e-5)
        model.num_timesteps = 0

        insert_demos_into_her_buffer(model.replay_buffer, demos)
        model.learn(total_timesteps=80, progress_bar=False)
        assert float(th.exp(model.log_ent_coef).item()) >= 0.05 - 1e-6

        with tempfile.TemporaryDirectory() as tmp:
            p_zip = os.path.join(tmp, "test_bcsac.zip")
            model.save(p_zip)
            loaded = SAC.load(p_zip, env=env, device="cpu")
            obs, _ = env.unwrapped.reset(seed=42)
            a1, _ = model.predict(obs, deterministic=True)
            a2, _ = loaded.predict(obs, deterministic=True)
            assert np.allclose(a1, a2, atol=1e-5)
    finally:
        env.close()

