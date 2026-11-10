"""Reward component & anti-exploitation tests for PandaMultiSortPreGraspEnv."""

from __future__ import annotations

import numpy as np
import pytest

from src.environments.pregrasp_multi_sort_env import (
    PandaMultiSortPreGraspEnv,
    PreGraspRewardConfig,
)


def _pg_info(
    prev_ee, curr_ee, prev_ag, prev_gw=0.08, curr_gw=0.08,
    prev_bilat=(False, False), curr_bilat=(False, False),
    prev_orn=True, curr_orn=True, prev_tcoll=False, curr_tcoll=False,
    any_contact=(False, False), c_idx=(0, 1),
) -> dict:
    return {
        "prev_ee_pos": np.asarray(prev_ee, dtype=np.float32),
        "ee_pos": np.asarray(curr_ee, dtype=np.float32),
        "prev_ag": np.asarray(prev_ag, dtype=np.float32).reshape(6),
        "prev_gw": float(prev_gw),
        "gripper_width": float(curr_gw),
        "prev_bilateral_contact": np.asarray(prev_bilat, dtype=bool),
        "bilateral_contact": np.asarray(curr_bilat, dtype=bool),
        "prev_orn_ok": bool(prev_orn),
        "orn_ok": bool(curr_orn),
        "prev_table_collision": bool(prev_tcoll),
        "table_collision": bool(curr_tcoll),
        "any_cube_contact": np.asarray(any_contact, dtype=bool),
        "bin_positions": np.array([[0.70, -0.25, 0.001], [0.70, 0.0, 0.001], [0.70, 0.25, 0.001]], dtype=np.float32),
        "active_color_indices": np.asarray(c_idx, dtype=np.int32),
    }


def test_pregrasp_reward_components_and_gates():
    env = PandaMultiSortPreGraspEnv(render_mode=None, num_objects=2)
    env.reset(seed=42)
    cfg = env.pg_cfg
    dg = np.array([0.70, -0.25, 0.025, 0.70, 0.0, 0.025], dtype=np.float32)
    p0 = np.array([0.45, 0.0, 0.025], dtype=np.float32)
    p1 = np.array([0.45, 0.15, 0.025], dtype=np.float32)
    ag = np.concatenate([p0, p1])

    # A. Horizontal alignment progress rewarded
    _ = env.compute_reward(ag, dg, _pg_info([0.35, 0.0, 0.14], [0.40, 0.0, 0.14], ag))
    assert env._last_reward_components["r_align_prog"] == pytest.approx(cfg.w_align_prog * 0.05)

    # B. Safe descent rewarded ONLY when aligned XY, open, no table collision
    _ = env.compute_reward(ag, dg, _pg_info([0.45, 0.0, 0.08], [0.45, 0.0, 0.04], ag, curr_gw=0.08))
    assert env._last_reward_components["r_vert_approach"] == pytest.approx(cfg.w_vert_approach * 0.04)

    # Unsafe descent (misaligned XY=0.10m OR closed gripper) gets ZERO vertical approach reward
    _ = env.compute_reward(ag, dg, _pg_info([0.55, 0.0, 0.08], [0.55, 0.0, 0.04], ag, curr_gw=0.08))
    assert env._last_reward_components["r_vert_approach"] == 0.0
    _ = env.compute_reward(ag, dg, _pg_info([0.45, 0.0, 0.08], [0.45, 0.0, 0.04], ag, curr_gw=0.00))
    assert env._last_reward_components["r_vert_approach"] == 0.0
    # Closing gripper BEFORE entering grasp authorization height (e.g. z=0.06m, |z-0.025|=0.035 > 0.015) triggers compliance penalty
    _ = env.compute_reward(ag, dg, _pg_info([0.45, 0.0, 0.08], [0.45, 0.0, 0.06], ag, curr_gw=0.00))
    assert env._last_reward_components["r_open_comp"] == pytest.approx(-cfg.w_premature_close)

    # D. Collision penalty (table + other cube contact)
    _ = env.compute_reward(ag, dg, _pg_info([0.45, 0.0, 0.02], [0.45, 0.0, 0.02], ag, curr_tcoll=True, any_contact=(False, True)))
    assert env._last_reward_components["r_collision"] == pytest.approx(-(cfg.w_table_coll + cfg.w_other_obj_coll))

    # E. Ungrasped object displacement penalty
    ag_knocked = np.concatenate([[0.48, 0.0, 0.025], p1])
    _ = env.compute_reward(ag_knocked, dg, _pg_info([0.48, 0.0, 0.025], [0.48, 0.0, 0.025], ag))
    assert env._last_reward_components["r_disp"] == pytest.approx(-cfg.w_disp * (0.03 - cfg.disp_deadzone_m))

    # F -> G. Grasp authorization (+w_auth), then closing onto bilateral grasp retains auth + adds w_grasp
    _ = env.compute_reward(ag, dg, _pg_info([0.45, 0.0, 0.06], [0.45, 0.0, 0.025], ag, prev_gw=0.08, curr_gw=0.08))
    assert env._last_reward_components["r_auth"] == pytest.approx(cfg.w_auth)
    _ = env.compute_reward(ag, dg, _pg_info([0.45, 0.0, 0.025], [0.45, 0.0, 0.025], ag, prev_gw=0.08, curr_gw=0.05, prev_bilat=(False, False), curr_bilat=(True, False)))
    assert env._last_reward_components["r_auth"] == 0.0
    assert env._last_reward_components["r_grasp"] == pytest.approx(cfg.w_grasp)
    env.close()


def test_pregrasp_reward_anti_exploitation_and_two_object_switching():
    env = PandaMultiSortPreGraspEnv(render_mode=None, num_objects=2)
    env.reset(seed=42)
    dg = np.array([0.70, -0.25, 0.025, 0.70, 0.0, 0.025], dtype=np.float32)
    p0 = np.array([0.45, 0.0, 0.025], dtype=np.float32)
    p1 = np.array([0.45, 0.15, 0.025], dtype=np.float32)
    ag = np.concatenate([p0, p1])

    # Entering then exiting auth without grasping cancels r_auth (+0.5 - 0.5 = 0)
    r_enter = env.compute_reward(ag, dg, _pg_info([0.45, 0.0, 0.06], [0.45, 0.0, 0.025], ag))
    r_exit = env.compute_reward(ag, dg, _pg_info([0.45, 0.0, 0.025], [0.45, 0.0, 0.06], ag))
    assert env._last_reward_components["r_auth"] == pytest.approx(-env.pg_cfg.w_auth)
    assert r_enter + r_exit < 0.0

    # Empty gripper close gives 0 grasp reward & negative compliance penalty
    _ = env.compute_reward(ag, dg, _pg_info([0.30, 0.0, 0.10], [0.30, 0.0, 0.10], ag, prev_gw=0.08, curr_gw=0.00))
    assert env._last_reward_components["r_grasp"] == 0.0
    assert env._last_reward_components["r_open_comp"] < 0.0

    # Once Cube 0 placed, active_obj_idx switches to 1
    p0_placed = np.array([0.70, -0.25, 0.025], dtype=np.float32)
    ag2 = np.concatenate([p0_placed, p1])
    _ = env.compute_reward(ag2, dg, _pg_info([0.40, 0.15, 0.14], [0.45, 0.15, 0.14], ag2, prev_gw=0.075, curr_gw=0.075))
    assert env._last_reward_components["active_obj_idx"] == 1
    assert env._last_reward_components["r_align_prog"] > 0.0
    env.close()
