"""Unit & integration tests for PreGraspController state machine (Phase 15.2.5A)."""

from __future__ import annotations

import numpy as np
import pytest

from src.controllers.pregrasp_controller import (
    PreGraspConfig,
    PreGraspController,
    PreGraspStage,
    query_pybullet_telemetry,
)
from src.environments.pregrasp_multi_sort_env import PandaMultiSortPreGraspEnv


def test_pregrasp_authorization_rejects_unsafe_states():
    ctrl = PreGraspController(config=PreGraspConfig(), action_scale=0.05)
    ctrl.reset(active_obj_idx=0, anchor_obj_pos=np.array([0.45, 0.0, 0.025], dtype=np.float32))

    # 1. Valid pose -> authorized
    valid_telem = {
        "ee_pos": np.array([0.45, 0.0, 0.025], dtype=np.float32),
        "obj_pos": np.array([0.45, 0.0, 0.025], dtype=np.float32),
        "orn_dot": 0.999,
        "table_contact_count": 0,
        "gripper_width": 0.080,
    }
    ok, reasons = ctrl.check_grasp_authorization(valid_telem)
    assert ok is True and reasons == []

    # 2. Misaligned XY -> rejected
    bad_xy = dict(valid_telem, ee_pos=np.array([0.48, 0.0, 0.025], dtype=np.float32))
    ok, _ = ctrl.check_grasp_authorization(bad_xy)
    assert ok is False

    # 3. Table collision -> rejected
    bad_table = dict(valid_telem, table_contact_count=2)
    ok, _ = ctrl.check_grasp_authorization(bad_table)
    assert ok is False

    # 4. Unsafe orientation -> rejected
    bad_orn = dict(valid_telem, orn_dot=0.90)
    ok, _ = ctrl.check_grasp_authorization(bad_orn)
    assert ok is False

    # 5. Prematurely closed gripper -> rejected
    bad_gw = dict(valid_telem, gripper_width=0.045)
    ok, _ = ctrl.check_grasp_authorization(bad_gw)
    assert ok is False

    # 6. Displaced active object (> 20 mm) -> rejected
    bad_disp = dict(valid_telem, obj_pos=np.array([0.48, 0.0, 0.025], dtype=np.float32), ee_pos=np.array([0.48, 0.0, 0.025], dtype=np.float32))
    ok, _ = ctrl.check_grasp_authorization(bad_disp)
    assert ok is False


def test_pregrasp_full_acquisition_and_transition_logging():
    env = PandaMultiSortPreGraspEnv(render_mode=None, num_objects=2, max_episode_steps=300)
    obs, _ = env.reset(seed=1001)
    ctrl = PreGraspController(config=PreGraspConfig(), action_scale=env.action_scale)
    ctrl.reset(active_obj_idx=0, anchor_obj_pos=obs["achieved_goal"][:3])

    for step in range(1, 60):
        telem = query_pybullet_telemetry(env, active_obj_idx=0)
        cmd, info = ctrl.step(step, telem)
        env.step(cmd)
        if ctrl.stage == PreGraspStage.ACQUIRED:
            break

    assert ctrl.stage == PreGraspStage.ACQUIRED
    assert ctrl.metrics["approach_reached"] is True
    assert ctrl.metrics["alignment_reached"] is True
    assert ctrl.metrics["safe_descent_reached"] is True
    assert ctrl.metrics["grasp_authorized"] is True
    assert ctrl.metrics["bilateral_grasped"] is True
    assert ctrl.metrics["lift_verified"] is True
    assert ctrl.metrics["premature_closure"] is False
    assert ctrl.metrics["table_collision_steps"] == 0
    stages_visited = [t["to"] for t in ctrl.transition_log]
    assert stages_visited == [
        PreGraspStage.ALIGNMENT.value,
        PreGraspStage.CONTROLLED_DESCENT.value,
        PreGraspStage.GRASP_AUTHORIZATION.value,
        PreGraspStage.GRASP.value,
        PreGraspStage.LIFT_VERIFICATION.value,
        PreGraspStage.ACQUIRED.value,
    ]
    env.close()

