"""Phase 5 Integration & Final Validation Tests (Envs, Checkpoints, Dims, Rewards, Transfer)."""

from __future__ import annotations

import os
import numpy as np
import pytest
from stable_baselines3 import SAC

from src.rl.train import make_env
from src.rl.transfer_utils import transfer_single_to_multi_sac


ENV_SPECS = [
    ({"name": "Reach", "max_episode_steps": 50}, (4,), (3,), (4,)),
    ({"name": "Pick", "max_episode_steps": 100}, (10,), (3,), (4,)),
    ({"name": "PickPlace", "max_episode_steps": 100}, (10,), (3,), (4,)),
    ({"name": "Sort", "max_episode_steps": 150}, (22,), (3,), (4,)),
    ({"name": "MultiSort", "num_objects": 2, "max_episode_steps": 300}, (31,), (6,), (4,)),
    ({"name": "MultiSortDense", "num_objects": 2, "max_episode_steps": 300}, (31,), (6,), (4,)),
    ({"name": "MultiSortDenseV2", "num_objects": 2, "max_episode_steps": 300}, (31,), (6,), (4,)),
    ({"name": "SinglePreGrasp", "max_episode_steps": 80, "reward_config": {"use_vertical_potential": True, "w_vertical_potential": 2.0, "gate_goal_lift_on_lifted_dg": True}}, (22,), (3,), (4,)),
    ({"name": "MultiSortPreGrasp", "num_objects": 2, "max_episode_steps": 300}, (31,), (6,), (4,)),
    ({"name": "Assembly", "max_episode_steps": 200}, (22,), (7,), (5,)),
    ({"name": "AssemblyDense", "max_episode_steps": 120}, (22,), (7,), (5,)),
]


@pytest.mark.parametrize("cfg,obs_s,goal_s,act_s", ENV_SPECS)
def test_all_environments_init_reset_step_and_reward_consistency(cfg, obs_s, goal_s, act_s):
    env = make_env(cfg, render_mode=None)
    try:
        obs, info = env.reset(seed=42)
        assert env.observation_space["observation"].shape == obs_s
        assert env.observation_space["achieved_goal"].shape == goal_s
        assert env.observation_space["desired_goal"].shape == goal_s
        assert env.action_space.shape == act_s
        assert obs["observation"].shape == obs_s

        act = np.zeros(act_s, dtype=np.float32)
        n_obs, rew, term, trunc, s_info = env.step(act)
        assert np.isfinite(rew)
        assert all(np.all(np.isfinite(v)) for v in n_obs.values())

        # Verify compute_reward single vs batch=1 consistency
        r_single = env.compute_reward(n_obs["achieved_goal"], n_obs["desired_goal"], s_info)
        r_batch = env.compute_reward(
            np.expand_dims(n_obs["achieved_goal"], 0),
            np.expand_dims(n_obs["desired_goal"], 0),
            [s_info],
        )
        assert r_batch.shape == (1,)
        assert np.isclose(float(r_single), float(r_batch[0]), atol=1e-5)
    finally:
        env.close()


CHECKPOINTS = [
    ({"name": "PickPlace", "max_episode_steps": 100}, "experiments/phase13_dense_dr/dr4/seed42/models/checkpoint_500000.zip"),
    ({"name": "PickPlace", "max_episode_steps": 100}, "experiments/phase13_dense_dr/dr4/seed43/models/checkpoint_500000.zip"),
    ({"name": "MultiSortDenseV2", "num_objects": 2, "max_episode_steps": 300}, "experiments/multi_sort_phase15_4/seed42/models/best_model.zip"),
    ({"name": "SinglePreGrasp", "max_episode_steps": 80}, "experiments/multi_sort_phase15_5B/seed42/models/final_model.zip"),
    ({"name": "SinglePreGrasp", "max_episode_steps": 80, "reward_config": {"use_vertical_potential": True}}, "experiments/multi_sort_phase15_5B1/seed42/models/checkpoint_7500_steps.zip"),
    ({"name": "SinglePreGrasp", "max_episode_steps": 80, "reward_config": {"use_vertical_potential": True}}, "experiments/multi_sort_phase15_5B2/seed42/models/checkpoint_17500_steps.zip"),
    ({"name": "SinglePreGrasp", "max_episode_steps": 80, "reward_config": {"use_vertical_potential": True, "gate_goal_lift_on_lifted_dg": True}}, "experiments/multi_sort_phase15_5B7/seed42/models/bc_warmstart_model.zip"),
    ({"name": "AssemblyDense", "max_episode_steps": 120}, "experiments/assembly_phase15_3/seed42/models/final_model.zip"),
]


@pytest.mark.parametrize("cfg,ckpt_path", CHECKPOINTS)
def test_checkpoint_loading_and_deterministic_predict(cfg, ckpt_path):
    assert os.path.exists(ckpt_path), f"Missing checkpoint {ckpt_path}"
    env = make_env(cfg, render_mode=None)
    try:
        model = SAC.load(ckpt_path, env=env, device="cpu")
        obs, _ = env.reset(seed=42)
        act, _ = model.predict(obs, deterministic=True)
        assert act.shape == env.action_space.shape
        assert np.all(np.isfinite(act))
    finally:
        env.close()


def test_phase2_transfer_adapter_with_5b2_checkpoint():
    env_b = make_env({"name": "SinglePreGrasp", "max_episode_steps": 80}, render_mode=None)
    env_c = make_env({"name": "MultiSortPreGrasp", "num_objects": 2, "max_episode_steps": 300}, render_mode=None)
    try:
        src = SAC.load("experiments/multi_sort_phase15_5B2/seed42/models/checkpoint_17500_steps.zip", env=env_b, device="cpu")
        tgt = SAC("MultiInputPolicy", env_c, seed=42, buffer_size=1000, verbose=0, device="cpu")
        report = transfer_single_to_multi_sac(src, tgt)
        assert report["transferred_params"] == 373260
        assert report["target_total_params"] == 392460
        obs_b, _ = env_b.reset(seed=77)
        obs_c_base, _ = env_c.reset(seed=77)
        obs_c = {
            "observation": np.concatenate([obs_b["observation"][0:4], obs_b["observation"][4:13], obs_c_base["observation"][13:22], obs_b["observation"][13:22]]).astype(np.float32),
            "achieved_goal": np.concatenate([obs_b["achieved_goal"], obs_c_base["achieved_goal"][3:6]]).astype(np.float32),
            "desired_goal": np.concatenate([obs_b["desired_goal"], obs_c_base["desired_goal"][3:6]]).astype(np.float32),
        }
        a_b, _ = src.predict(obs_b, deterministic=True)
        a_c, _ = tgt.predict(obs_c, deterministic=True)
        assert np.allclose(a_b, a_c, atol=1e-5)
    finally:
        env_b.close()
        env_c.close()
