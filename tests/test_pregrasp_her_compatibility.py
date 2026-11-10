"""HER relabeling compatibility & isolation tests for PandaMultiSortPreGraspEnv."""

from __future__ import annotations

import numpy as np
import yaml
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.monitor import Monitor

from src.environments.pregrasp_multi_sort_env import PandaMultiSortPreGraspEnv
from tests.test_pregrasp_reward import _pg_info


def test_her_relabeling_stateless_recomputation_and_active_object_switch():
    env = PandaMultiSortPreGraspEnv(render_mode=None, num_objects=2)
    env.reset(seed=42)

    p0 = np.array([0.45, 0.0, 0.025], dtype=np.float32)
    p1 = np.array([0.45, 0.15, 0.025], dtype=np.float32)
    ag = np.concatenate([p0, p1])

    # EE moves closer to Cube 1 ([0.40,0.15] -> [0.44,0.15])
    info = _pg_info([0.40, 0.15, 0.14], [0.44, 0.15, 0.14], ag, prev_gw=0.075, curr_gw=0.075)

    # Original goal: Cube 0 NOT yet at [0.70, -0.25] -> active object = 0 (approaching Cube 1 gives negative/zero alignment to Cube 0)
    dg_orig = np.array([0.70, -0.25, 0.025, 0.70, 0.0, 0.025], dtype=np.float32)
    r_orig = env.compute_reward(ag, dg_orig, info)
    assert env._last_reward_components["active_obj_idx"] == 0

    # HER relabeled goal: Cube 0's goal relabeled ONTO p0 ([0.45, 0.0, 0.025]) -> active object switches dynamically to 1!
    dg_relabeled = np.array([0.45, 0.0, 0.025, 0.70, 0.0, 0.025], dtype=np.float32)
    r_relabel = env.compute_reward(ag, dg_relabeled, info)
    assert env._last_reward_components["active_obj_idx"] == 1
    assert env._last_reward_components["r_align_prog"] > 0.0
    assert r_relabel > r_orig

    # Mutating live env simulator state does NOT corrupt stateless batched HER recomputation
    env.cubes[0].reset(position=[0.30, -0.20, 0.025])
    env.obj1_verified = True
    r_relabel_batched = env.compute_reward(ag[None, :], dg_relabeled[None, :], [info])
    assert np.isclose(float(r_relabel_batched[0]), r_relabel, atol=1e-5)
    env.close()


def test_her_replay_buffer_rollout_and_virtual_sampling_no_nan():
    env = Monitor(PandaMultiSortPreGraspEnv(render_mode=None, num_objects=2))
    with open("configs/multi_sort_phase15_5A.yaml", "r", encoding="utf-8") as f:
        rl = yaml.safe_load(f)["rl"]

    model = SAC(
        rl["policy"],
        env,
        seed=rl["seed"],
        buffer_size=2000,
        batch_size=32,
        learning_rate=rl["learning_rate"],
        learning_starts=100,
        gamma=rl["gamma"],
        tau=rl["tau"],
        device="cpu",
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": rl["her_n_sampled_goal"],
            "goal_selection_strategy": rl["her_strategy"],
            "copy_info_dict": rl["her_copy_info_dict"],
        },
        verbose=0,
    )
    obs = model.env.reset()
    for s in range(30):
        act = np.array([env.action_space.sample()], dtype=np.float32)
        next_obs, r, done, infos = model.env.step(act)
        if s == 29:
            done = np.array([True], dtype=bool)
            infos[0]["TimeLimit.truncated"] = True
        model.replay_buffer.add(obs, next_obs, act, r, done, infos)
        obs = next_obs

    batch = model.replay_buffer.sample(32)
    rew = batch.rewards.detach().cpu().numpy()
    assert rew.shape == (32, 1)
    assert np.all(np.isfinite(rew))

    # Empty-info fallback also safe & finite
    fb = env.unwrapped.compute_reward(
        obs["achieved_goal"],
        obs["desired_goal"],
        [{}, {}][: len(obs["achieved_goal"])],
    )
    assert np.all(np.isfinite(fb))
    env.close()

