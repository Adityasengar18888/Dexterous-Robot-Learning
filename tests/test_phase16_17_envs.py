"""Unit tests for Phase 16.3 Dynamic, 16.4 Contact Assembly, and 17.1 MuJoCo Panda Envs."""

from __future__ import annotations

import numpy as np

from src.cross_sim.mujoco_panda_env import MuJoCoPandaPickPlaceEnv
from src.environments.contact_assembly_env import ContactState, PandaContactAssemblyEnv
from src.environments.dynamic_pick_place_env import PandaDynamicPickPlaceEnv


def test_dynamic_pick_place_env_step_and_disturbance():
    env = PandaDynamicPickPlaceEnv(render_mode=None, max_episode_steps=40, motion_mode="disturbance_step", cube_speed=0.04, disturbance_step=5)
    try:
        obs, _ = env.reset(seed=42)
        p0 = obs["achieved_goal"].copy()
        disturbed = False
        for _ in range(8):
            obs, rew, _, _, info = env.step(np.zeros(4, dtype=np.float32))
            assert np.isfinite(rew)
            disturbed = disturbed or bool(info["disturbance_injected"])
        assert np.linalg.norm(obs["achieved_goal"][:2] - p0[:2]) > 1e-4
        assert disturbed is True
    finally:
        env.close()


def test_contact_assembly_env_telemetry():
    env = PandaContactAssemblyEnv(render_mode=None, max_episode_steps=50)
    try:
        obs, _ = env.reset(seed=42)
        assert obs["observation"].shape == (22,)
        assert obs["desired_goal"].shape == (7,)
        obs2, rew, _, _, info = env.step(np.zeros(5, dtype=np.float32))
        assert np.isfinite(rew)
        assert info["contact_state"] in {s.value for s in ContactState}
        assert "insertion_depth_m" in info
    finally:
        env.close()


def test_mujoco_panda_twin_observation_and_kinematics():
    env = MuJoCoPandaPickPlaceEnv(max_episode_steps=50)
    try:
        obs, info = env.reset(seed=42)
        assert obs["observation"].shape == (10,)
        assert obs["achieved_goal"].shape == (3,)
        assert obs["desired_goal"].shape == (3,)
        assert np.all(np.isfinite(obs["observation"]))
        # Neutral EE position should be near [0.307, 0.0, ~0.59]
        ee_init = obs["observation"][:3]
        assert 0.25 < ee_init[0] < 0.40
        assert abs(ee_init[1]) < 0.05
        assert 0.45 < ee_init[2] < 0.70

        obs2, rew, _, _, info2 = env.step(np.array([0.5, 0.0, -0.5, 1.0], dtype=np.float32))
        assert np.isfinite(rew)
        assert obs2["observation"][2] < ee_init[2]
    finally:
        env.close()

