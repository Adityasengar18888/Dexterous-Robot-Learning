"""Comprehensive unit and HER-compatibility tests for PandaMultiSortDenseRewardEnv."""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.monitor import Monitor

from src.environments.dense_multi_sort_env import (
    MultiSortProximityRewardConfig,
    MultiSortRewardConfig,
    PandaMultiSortDenseRewardEnv,
    PandaMultiSortDenseV2Env,
    SortingStage,
)


def _make_transition_info(
    prev_ee: np.ndarray,
    curr_ee: np.ndarray,
    prev_ag: np.ndarray,
    prev_gw: float = 0.08,
    curr_gw: float = 0.08,
    prev_bilat: tuple[bool, bool] = (False, False),
    curr_bilat: tuple[bool, bool] = (False, False),
    bins: np.ndarray | None = None,
    c_idx: tuple[int, int] = (0, 1),
) -> dict:
    return {
        "prev_ee_pos": np.asarray(prev_ee, dtype=np.float32),
        "ee_pos": np.asarray(curr_ee, dtype=np.float32),
        "prev_ag": np.asarray(prev_ag, dtype=np.float32).reshape(6),
        "prev_gw": float(prev_gw),
        "gripper_width": float(curr_gw),
        "prev_bilateral_contact": np.asarray(prev_bilat, dtype=bool),
        "bilateral_contact": np.asarray(curr_bilat, dtype=bool),
        "bin_positions": (
            np.asarray(bins, dtype=np.float32)
            if bins is not None
            else np.array([[0.70, -0.25, 0.001], [0.70, 0.0, 0.001], [0.70, 0.25, 0.001]], dtype=np.float32)
        ),
        "active_color_indices": np.asarray(c_idx, dtype=np.int32),
    }


def test_1_reset_and_observation_space_compatibility():
    env = PandaMultiSortDenseRewardEnv(render_mode=None, num_objects=2)
    obs, info = env.reset(seed=42)
    assert obs["observation"].shape == (31,)
    assert obs["achieved_goal"].shape == (6,)
    assert obs["desired_goal"].shape == (6,)
    assert env.action_space.shape == (4,)
    assert env.observation_space.contains(obs)
    assert info["sorting_stage"] == SortingStage.IDLE.value
    assert info["obj1_verified"] is False and info["obj2_verified"] is False
    env.close()


def test_2_reward_calculation_per_stage():
    env = PandaMultiSortDenseRewardEnv(render_mode=None, num_objects=2)
    env.reset(seed=42)
    cfg = env.reward_cfg
    bins = np.array([[0.70, -0.25, 0.001], [0.70, 0.0, 0.001], [0.70, 0.25, 0.001]], dtype=np.float32)
    dg = np.array([0.70, -0.25, 0.025, 0.70, 0.0, 0.025], dtype=np.float32)
    p0 = np.array([0.45, 0.0, 0.025], dtype=np.float32)
    p1 = np.array([0.45, 0.15, 0.025], dtype=np.float32)
    prev_ag = np.concatenate([p0, p1])

    # Stage A: Moving EE 0.04m closer to Cube 0 gives positive reach potential
    info_a = _make_transition_info([0.45, 0.0, 0.10], [0.45, 0.0, 0.06], prev_ag, bins=bins)
    r_a = env.compute_reward(prev_ag, dg, info_a)
    assert np.isclose(r_a, cfg.w_reach * 0.04 + cfg.step_penalty, atol=1e-4)

    # Stage B: Verified bilateral grasp transition (NOT empty gripper close)
    info_b_empty = _make_transition_info([0.45, 0.0, 0.10], [0.45, 0.0, 0.10], prev_ag, prev_gw=0.08, curr_gw=0.00, bins=bins)
    r_b_empty = env.compute_reward(prev_ag, dg, info_b_empty)
    assert np.isclose(r_b_empty, cfg.step_penalty, atol=1e-4)

    info_b_grasp = _make_transition_info(
        [0.45, 0.0, 0.025], [0.45, 0.0, 0.025], prev_ag, prev_gw=0.08, curr_gw=0.05,
        prev_bilat=(False, False), curr_bilat=(True, False), bins=bins,
    )
    r_b_grasp = env.compute_reward(prev_ag, dg, info_b_grasp)
    assert np.isclose(r_b_grasp, cfg.w_grasp + cfg.step_penalty, atol=1e-4)

    # Stage C: Bounded lift while holding Cube 0 from z=0.025 -> z=0.120 outside bin XY
    ag_lifted = np.concatenate([[0.45, 0.0, 0.120], p1])
    info_c = _make_transition_info(
        [0.45, 0.0, 0.025], [0.45, 0.0, 0.120], prev_ag, prev_gw=0.05, curr_gw=0.05,
        prev_bilat=(True, False), curr_bilat=(True, False), bins=bins,
    )
    r_c = env.compute_reward(ag_lifted, dg, info_c)
    assert r_c > cfg.w_lift * 0.9

    # Stage D: Transport Cube 0 0.05m closer to Bin 0
    ag_t0 = np.concatenate([[0.50, -0.05, 0.120], p1])
    ag_t1 = np.concatenate([[0.55, -0.05, 0.120], p1])
    info_d = _make_transition_info(
        [0.50, -0.05, 0.120], [0.55, -0.05, 0.120], ag_t0, prev_gw=0.05, curr_gw=0.05,
        prev_bilat=(True, False), curr_bilat=(True, False), bins=bins,
    )
    r_d = env.compute_reward(ag_t1, dg, info_d)
    assert r_d > 0.0
    env.close()


def test_3_reward_boundedness():
    env = PandaMultiSortDenseRewardEnv(render_mode=None, num_objects=2)
    env.reset(seed=42)
    rng = np.random.default_rng(7)
    for _ in range(50):
        prev_ag = rng.uniform(0.2, 0.8, size=(6,)).astype(np.float32)
        curr_ag = rng.uniform(0.2, 0.8, size=(6,)).astype(np.float32)
        dg = rng.uniform(0.2, 0.8, size=(6,)).astype(np.float32)
        info = _make_transition_info(
            rng.uniform(0.2, 0.8, size=(3,)),
            rng.uniform(0.2, 0.8, size=(3,)),
            prev_ag,
            prev_gw=float(rng.uniform(0.0, 0.08)),
            curr_gw=float(rng.uniform(0.0, 0.08)),
            prev_bilat=(bool(rng.integers(2)), bool(rng.integers(2))),
            curr_bilat=(bool(rng.integers(2)), bool(rng.integers(2))),
        )
        r = env.compute_reward(curr_ag, dg, info)
        assert np.isfinite(r)
        assert -20.0 <= r <= 20.0
    env.close()


def test_4_reward_reproducibility():
    env1 = PandaMultiSortDenseRewardEnv(render_mode=None, num_objects=2)
    env2 = PandaMultiSortDenseRewardEnv(render_mode=None, num_objects=2)
    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)
    assert np.allclose(obs1["observation"], obs2["observation"])
    act = np.array([0.4, -0.3, -0.5, 1.0], dtype=np.float32)
    o1, r1, _, _, i1 = env1.step(act)
    o2, r2, _, _, i2 = env2.step(act)
    assert np.isclose(r1, r2, atol=1e-6)
    assert i1["sorting_stage"] == i2["sorting_stage"]
    env1.close()
    env2.close()


def test_5_first_object_placement_reward_not_complete():
    env = PandaMultiSortDenseRewardEnv(render_mode=None, num_objects=2)
    env.reset(seed=42)
    dg = np.array([0.70, -0.25, 0.025, 0.70, 0.0, 0.025], dtype=np.float32)
    p1 = np.array([0.40, 0.10, 0.025], dtype=np.float32)
    prev_ag = np.concatenate([[0.70, -0.25, 0.04], p1])
    curr_ag = np.concatenate([[0.70, -0.25, 0.025], p1])
    # Release Cube 0 inside Bin 0 (gw: 0.05 -> 0.075)
    info = _make_transition_info(
        [0.70, -0.25, 0.04], [0.70, -0.25, 0.04], prev_ag, prev_gw=0.050, curr_gw=0.075,
        prev_bilat=(True, False), curr_bilat=(False, False),
    )
    r = env.compute_reward(curr_ag, dg, info)
    c = env._last_reward_components
    assert c["r_place"] == pytest.approx(env.reward_cfg.w_place)
    assert c["r_complete"] == 0.0
    # Staying placed on next idle step must NOT repeatedly grant placement bonus
    info_stay = _make_transition_info([0.70, -0.25, 0.04], [0.70, -0.25, 0.04], curr_ag, prev_gw=0.075, curr_gw=0.075)
    _ = env.compute_reward(curr_ag, dg, info_stay)
    assert env._last_reward_components["r_place"] == 0.0
    assert env._last_reward_components["active_obj_idx"] == 1
    env.close()


def test_6_and_9_second_object_placement_and_full_completion():
    env = PandaMultiSortDenseRewardEnv(render_mode=None, num_objects=2)
    env.reset(seed=42)
    dg = np.array([0.70, -0.25, 0.025, 0.70, 0.0, 0.025], dtype=np.float32)
    p0_placed = np.array([0.70, -0.25, 0.025], dtype=np.float32)
    prev_ag = np.concatenate([p0_placed, [0.70, 0.0, 0.04]])
    curr_ag = np.concatenate([p0_placed, [0.70, 0.0, 0.025]])
    info = _make_transition_info(
        [0.70, 0.0, 0.04], [0.70, 0.0, 0.04], prev_ag, prev_gw=0.050, curr_gw=0.075,
        prev_bilat=(False, True), curr_bilat=(False, False),
    )
    r = env.compute_reward(curr_ag, dg, info)
    c = env._last_reward_components
    assert c["r_place"] == pytest.approx(env.reward_cfg.w_place)
    assert c["r_complete"] == pytest.approx(env.reward_cfg.w_complete)
    assert r > env.reward_cfg.w_complete
    # Verify subsequent step while both remain placed awards 0.0 (no farming)
    info_hold = _make_transition_info([0.70, 0.0, 0.04], [0.70, 0.0, 0.04], curr_ag, prev_gw=0.075, curr_gw=0.075)
    r_hold = env.compute_reward(curr_ag, dg, info_hold)
    assert r_hold == pytest.approx(0.0)
    env.close()


def test_7_incorrect_bin_placement_penalized():
    env = PandaMultiSortDenseRewardEnv(render_mode=None, num_objects=2)
    env.reset(seed=42)
    bins = np.array([[0.70, -0.25, 0.001], [0.70, 0.0, 0.001], [0.70, 0.25, 0.001]], dtype=np.float32)
    dg = np.array([0.70, -0.25, 0.025, 0.70, 0.0, 0.025], dtype=np.float32)
    p1 = np.array([0.45, 0.15, 0.025], dtype=np.float32)
    # Cube 0 (color idx 0) lowered & released into Bin 2 (wrong bin!)
    prev_ag = np.concatenate([[0.70, 0.25, 0.04], p1])
    curr_ag = np.concatenate([[0.70, 0.25, 0.025], p1])
    info = _make_transition_info(
        [0.70, 0.25, 0.04], [0.70, 0.25, 0.04], prev_ag, prev_gw=0.050, curr_gw=0.075,
        bins=bins, c_idx=(0, 1),
    )
    r = env.compute_reward(curr_ag, dg, info)
    c = env._last_reward_components
    assert c["r_wrong_bin"] == pytest.approx(-env.reward_cfg.w_wrong_bin)
    assert c["r_place"] == 0.0
    assert r < 0.0
    env.close()


def test_8_false_release_prevention_and_hover_rejection():
    env = PandaMultiSortDenseRewardEnv(render_mode=None, num_objects=2)
    env.reset(seed=42)
    g0 = env.goal[0:3].copy()
    g1 = env.goal[3:6].copy()

    # Move cleanly overhead above g0, open, place Cube 0 centered, descend vertically and pinch CLOSED
    env.controller.open_gripper(steps=15)
    env.controller.move_ee([g0[0], g0[1], 0.10], steps=40)
    env.cubes[1].reset(position=g1)
    env.cubes[0].reset(position=g0)
    env.controller.move_ee([g0[0], g0[1], 0.025], steps=40)
    obs, _, _, _, info = env.step(np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32))
    assert float(obs["observation"][3]) <= env.reward_cfg.release_width_threshold
    assert info["is_released"] is False
    # Cube 1 at g1 is unattended & placed (True); Cube 0 is still pinched at g0 (False)
    assert info["is_success_per_obj"] == [False, True]
    assert info["objects_placed"] == 1
    assert info["is_success"] is False

    # Open gripper completely (>0.065m) -> now verified placed!
    for _ in range(3):
        obs, _, _, _, info = env.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    assert float(obs["observation"][3]) > env.reward_cfg.release_width_threshold
    assert info["is_released"] is True
    assert info["objects_placed"] == 2
    assert info["is_success"] is True

    # Hover height rejection in both single & batched _get_info
    ag_hover = np.array([g0[0], g0[1], 0.06, g1[0], g1[1], 0.06], dtype=np.float32)
    assert env._get_info(ag_hover, env.goal)["is_success"] is False
    assert bool(env._get_info(ag_hover[None, :], env.goal[None, :])["is_success"][0]) is False
    env.close()


def test_10_and_11_and_12_her_relabeling_batched_consistency_and_nan():
    env = Monitor(PandaMultiSortDenseRewardEnv(render_mode=None, num_objects=2))
    with open("configs/multi_sort_phase15_dense.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["rl"]

    model = SAC(
        cfg["policy"],
        env,
        seed=cfg["seed"],
        buffer_size=2000,
        batch_size=32,
        learning_rate=cfg["learning_rate"],
        gamma=cfg["gamma"],
        tau=cfg["tau"],
        learning_starts=100,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": cfg["her_n_sampled_goal"],
            "goal_selection_strategy": cfg["her_strategy"],
            "copy_info_dict": cfg["her_copy_info_dict"],
        },
        verbose=0,
    )

    # Rollout 2 short trajectories into HerReplayBuffer and sample virtual HER batch
    for ep in range(2):
        obs = model.env.reset()
        for step in range(40):
            act = np.array([env.action_space.sample()], dtype=np.float32)
            # Trigger done cleanly on last step so HER records episode boundary
            next_obs, rew, done, infos = model.env.step(act)
            if step == 39:
                done = np.array([True], dtype=bool)
                infos[0]["TimeLimit.truncated"] = True
            model.replay_buffer.add(obs, next_obs, act, rew, done, infos)
            obs = next_obs

    batch = model.replay_buffer.sample(batch_size=16)
    r_np = batch.rewards.detach().cpu().numpy()
    assert r_np.shape == (16, 1)
    assert np.all(np.isfinite(r_np))

    # Empty-info fallback consistency check on batched ag/dg
    ag_b = np.tile(np.array([0.4, 0.0, 0.025, 0.5, 0.1, 0.025], dtype=np.float32), (4, 1))
    dg_b = np.tile(np.array([0.7, -0.25, 0.025, 0.7, 0.0, 0.025], dtype=np.float32), (4, 1))
    dg_b[0] = ag_b[0]  # Future relabeled match
    r_fb = env.unwrapped.compute_reward(ag_b, dg_b, [{} for _ in range(4)])
    assert r_fb.shape == (4,)
    assert np.all(np.isfinite(r_fb))
    assert r_fb[0] > r_fb[1]
    env.close()


def test_13_v2_proximity_multiple_distances_and_grasp_gating():
    env = PandaMultiSortDenseV2Env(render_mode=None, num_objects=2)
    env.reset(seed=42)
    dg = np.array([0.70, -0.25, 0.025, 0.70, 0.0, 0.025], dtype=np.float32)
    p0 = np.array([0.45, 0.0, 0.025], dtype=np.float32)
    p1 = np.array([0.45, 0.15, 0.025], dtype=np.float32)
    prev_ag = np.concatenate([p0, p1])

    # Stationary at multiple distances: r_proximity = -0.25 * clip(d, 0, 1)
    prox_vals = []
    for d in [0.50, 0.25, 0.05, 1.50]:
        ee = p0 + np.array([0.0, 0.0, d], dtype=np.float32)
        info_s = _make_transition_info(ee, ee, prev_ag)
        r = env.compute_reward(prev_ag, dg, info_s)
        c = env._last_reward_components
        expected_prox = -0.25 * min(d, 1.0)
        assert c["r_proximity"] == pytest.approx(expected_prox, abs=1e-5)
        assert c["r_reach_progress"] == pytest.approx(0.0, abs=1e-5)
        assert c["r_reach"] == pytest.approx(c["r_reach_progress"] + c["r_proximity"], abs=1e-5)
        prox_vals.append((d, r, c["r_proximity"]))

    # Monotonic: closer stationary distance gives strictly higher reward (0.05 > 0.25 > 0.50)
    assert prox_vals[2][1] > prox_vals[1][1] > prox_vals[0][1]

    # When Cube 0 is grasped (bilateral=True), r_proximity MUST vanish (0.0)
    info_grasp = _make_transition_info(
        p0, p0, prev_ag, prev_gw=0.05, curr_gw=0.05, prev_bilat=(True, False), curr_bilat=(True, False)
    )
    _ = env.compute_reward(prev_ag, dg, info_grasp)
    assert env._last_reward_components["r_proximity"] == 0.0
    assert env._last_reward_components["r_reach_progress"] == 0.0
    env.close()


def test_14_v2_active_object_switching_and_her_sampling():
    env = Monitor(PandaMultiSortDenseV2Env(render_mode=None, num_objects=2))
    raw_env = env.unwrapped
    dg = np.array([0.70, -0.25, 0.025, 0.70, 0.0, 0.025], dtype=np.float32)
    p0_placed = np.array([0.70, -0.25, 0.025], dtype=np.float32)
    p1 = np.array([0.40, 0.15, 0.025], dtype=np.float32)
    ag_p1 = np.concatenate([p0_placed, p1])

    # Cube 0 placed -> active object MUST switch to Cube 1 (ee at [0.40, 0.15, 0.225] => d_cube1 = 0.20m)
    ee_above_1 = np.array([0.40, 0.15, 0.225], dtype=np.float32)
    info_sw = _make_transition_info(ee_above_1, ee_above_1, ag_p1, prev_gw=0.075, curr_gw=0.075)
    _ = raw_env.compute_reward(ag_p1, dg, info_sw)
    assert raw_env._last_reward_components["active_obj_idx"] == 1
    assert raw_env._last_reward_components["r_proximity"] == pytest.approx(-0.25 * 0.20, abs=1e-5)

    # First-object placement transition vs Second-object completion transition
    ag_both_prev = np.concatenate([p0_placed, [0.70, 0.0, 0.04]])
    ag_both_curr = np.concatenate([p0_placed, [0.70, 0.0, 0.025]])
    info_done = _make_transition_info([0.70, 0.0, 0.04], [0.70, 0.0, 0.04], ag_both_prev, prev_gw=0.05, curr_gw=0.075)
    _ = raw_env.compute_reward(ag_both_curr, dg, info_done)
    assert raw_env._last_reward_components["r_place"] == pytest.approx(4.0)
    assert raw_env._last_reward_components["r_complete"] == pytest.approx(8.0)
    assert raw_env._last_reward_components["r_proximity"] == 0.0

    # Live HER replay buffer sampling with V2 config
    with open("configs/multi_sort_phase15_4.yaml", "r", encoding="utf-8") as f:
        rl = yaml.safe_load(f)["rl"]
    model = SAC(
        rl["policy"], env, seed=rl["seed"], buffer_size=2000, batch_size=32,
        learning_rate=rl["learning_rate"], learning_starts=100, gamma=rl["gamma"], tau=rl["tau"],
        device="cpu", replay_buffer_class=HerReplayBuffer,
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
    batch = model.replay_buffer.sample(16)
    rew = batch.rewards.detach().cpu().numpy()
    assert rew.shape == (16, 1) and np.all(np.isfinite(rew))
    env.close()

