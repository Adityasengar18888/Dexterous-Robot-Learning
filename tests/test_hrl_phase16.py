"""Unit tests for Phase 16.1 Hierarchical RL skills, learned termination, and HER SMDP buffer."""

from __future__ import annotations

import numpy as np

from src.environments.pick_place_env import PandaPickPlaceEnv
from src.hrl.hrl_agent import HierarchicalManipulatorAgent, SMDPReplayBuffer, build_meta_state
from src.hrl.skills import NUM_SKILLS, SkillLibrary, compute_teacher_skill_target, extract_skill_features


def test_skill_features_and_teacher_shapes():
    env = PandaPickPlaceEnv(render_mode=None, max_episode_steps=40)
    try:
        obs, _ = env.reset(seed=42)
        feat = extract_skill_features(obs)
        assert feat.shape == (16,)
        assert np.all(np.isfinite(feat))
        mstate = build_meta_state(obs)
        assert mstate.shape == (18,)
        for k in range(NUM_SKILLS):
            cmd, done = compute_teacher_skill_target(k, obs)
            assert cmd.shape == (4,)
            assert 0.0 <= done <= 1.0
    finally:
        env.close()


def test_hrl_meta_selection_and_smdp_her_buffer():
    lib = SkillLibrary(in_dim=16)
    agent = HierarchicalManipulatorAgent(skill_lib=lib, seed=42)
    env = PandaPickPlaceEnv(render_mode=None, max_episode_steps=40)
    try:
        obs, _ = env.reset(seed=10)
        k, ent, probs = agent.select_skill(obs, deterministic=False)
        assert 0 <= k < NUM_SKILLS
        assert probs.shape == (NUM_SKILLS,)
        # Untrained random policy has high entropy (> 1.4 nats for 6 skills)
        assert ent > 1.4

        obs2, oinfo = agent.execute_option(env, obs, k)
        assert oinfo["tau"] >= 1
        replay = SMDPReplayBuffer(capacity=100, her_k=1, seed=42)
        tr = {
            "s": build_meta_state(obs),
            "k": k,
            "r": -0.2,
            "s_next": build_meta_state(obs2),
            "tau": oinfo["tau"],
            "done": 0.0,
            "start_obs": obs,
            "end_obs": obs2,
            "dropped": False,
        }
        added = replay.add_episode([tr, dict(tr)])
        assert added >= 2
    finally:
        env.close()

