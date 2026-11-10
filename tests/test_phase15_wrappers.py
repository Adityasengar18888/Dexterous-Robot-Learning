"""Automated unit tests for Phase 15.1 Sim-to-Real wrappers."""

import numpy as np
import pytest

from src.rl.adaptive_controllers import ControllerConfig
from src.rl.sim_to_real_observation_wrapper import ObservationNoiseWrapper
from src.rl.sim_to_real_latency_wrapper import LatencyWrapper
from src.rl.unprivileged_contact_wrapper import (
    PRIVILEGED_TELEMETRY_KEYS,
    strip_privileged_telemetry,
    UnprivilegedAdaptiveController,
)


def _sample_obs_telem():
    obs = {
        "observation": np.array([0.4, -0.2, 0.1, 0.05, 0.42, -0.2, 0.0, 0.01, -0.02, 0.03], dtype=np.float32),
        "achieved_goal": np.array([0.42, -0.2, 0.0], dtype=np.float32),
        "desired_goal": np.array([0.65, 0.22, 0.18], dtype=np.float32),
    }
    telem = {
        "ee_pos": np.array([0.4, -0.2, 0.1], dtype=np.float32),
        "cube_pos": np.array([0.42, -0.2, 0.0], dtype=np.float32),
        "cube_orn": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "cube_vel": np.array([0.01, -0.02, 0.03], dtype=np.float32),
        "gripper_width": 0.05,
        "contact_count": 3,
        "f9_contact_count": 1,
        "f10_contact_count": 2,
        "f9_normal_force": 25.0,
        "f10_normal_force": 25.5,
        "total_normal_force": 50.5,
        "total_lateral_force": 1.2,
        "contact_symmetry": True,
        "has_contact": True,
        "rel_dist": 0.102,
        "xy_dist": 0.02,
        "cube_z": 0.0,
        "cube_vz": 0.03,
        "ee_z": 0.1,
        "desired_goal": np.array([0.65, 0.22, 0.18], dtype=np.float32),
    }
    return obs, telem


def test_observation_noise_clean_identity():
    obs, telem = _sample_obs_telem()
    wrapper = ObservationNoiseWrapper(std_m=0.0, clip_bound_m=0.0, seed=2026)
    wrapper.reset(1001)
    n_obs, n_telem = wrapper.perturb(obs, telem)
    np.testing.assert_allclose(n_obs["observation"], obs["observation"])
    np.testing.assert_allclose(n_obs["desired_goal"], obs["desired_goal"])
    np.testing.assert_allclose(n_telem["ee_pos"], telem["ee_pos"])


def test_observation_noise_bounds_and_vel_preservation():
    obs, telem = _sample_obs_telem()
    for bound, std in [(0.003, 0.0015), (0.005, 0.0025)]:
        wrapper = ObservationNoiseWrapper(std_m=std, clip_bound_m=bound, seed=2026)
        wrapper.reset(1001)
        for _ in range(100):
            n_obs, n_telem = wrapper.perturb(obs, telem)
            # Positions bounded strictly within +-bound
            diff_ee = n_obs["observation"][0:3] - obs["observation"][0:3]
            diff_obj = n_obs["observation"][4:7] - obs["observation"][4:7]
            diff_goal = n_obs["desired_goal"] - obs["desired_goal"]
            assert np.all(np.abs(diff_ee) <= bound + 1e-7)
            assert np.all(np.abs(diff_obj) <= bound + 1e-7)
            assert np.all(np.abs(diff_goal) <= bound + 1e-7)
            # Velocities & gripper width left untouched
            np.testing.assert_allclose(n_obs["observation"][7:10], obs["observation"][7:10])
            assert n_obs["observation"][3] == obs["observation"][3]
            # RL obs and controller telemetry receive identical noisy object position
            np.testing.assert_allclose(n_obs["achieved_goal"], n_telem["cube_pos"])
            assert np.all(np.isfinite(n_obs["observation"]))


def test_latency_wrapper_fifo_ordering():
    obs, telem = _sample_obs_telem()
    lw = LatencyWrapper(obs_delay_steps=2, action_delay_steps=1)
    lw.reset(obs, telem)

    obs1 = {k: v.copy() for k, v in obs.items()}; obs1["observation"][0] = 1.0
    obs2 = {k: v.copy() for k, v in obs.items()}; obs2["observation"][0] = 2.0
    obs3 = {k: v.copy() for k, v in obs.items()}; obs3["observation"][0] = 3.0

    # At t=1 (delay=2), queue yields t=0 (0.4), t=2 yields t=0 (0.4), t=3 yields t=1 (1.0)
    assert pytest.approx(lw.push_and_get_obs(obs1, telem)[0]["observation"][0], abs=1e-5) == 0.4
    assert pytest.approx(lw.push_and_get_obs(obs2, telem)[0]["observation"][0], abs=1e-5) == 0.4
    assert pytest.approx(lw.push_and_get_obs(obs3, telem)[0]["observation"][0], abs=1e-5) == 1.0

    # Action delay=1 returns initial neutral action first, then previous command
    a0 = lw.push_and_get_action(np.array([0.5, 0.0, 0.0, -1.0], dtype=np.float32))
    np.testing.assert_allclose(a0, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    a1 = lw.push_and_get_action(np.array([-0.5, 0.0, 0.0, -1.0], dtype=np.float32))
    np.testing.assert_allclose(a1, np.array([0.5, 0.0, 0.0, -1.0], dtype=np.float32))


def test_unprivileged_contact_stripping_and_enforcement():
    obs, telem = _sample_obs_telem()
    stripped = strip_privileged_telemetry(telem)
    for k in PRIVILEGED_TELEMETRY_KEYS:
        assert k not in stripped
    assert "ee_pos" in stripped and "cube_pos" in stripped and "gripper_width" in stripped

    ctrl = UnprivilegedAdaptiveController(ControllerConfig())
    ctrl.reset()
    raw_act = np.array([0.1, 0.2, 0.3, -1.0], dtype=np.float32)

    # Attempting to pass privileged telemetry raises explicit AssertionError
    with pytest.raises(AssertionError):
        ctrl.step_action(raw_act, telem)

    # Filtered telemetry executes cleanly without NaNs
    cmd, info = ctrl.step_action(raw_act, stripped)
    assert cmd.shape == (4,)
    assert np.all(np.isfinite(cmd))
    assert np.all(cmd >= -1.0) and np.all(cmd <= 1.0)

