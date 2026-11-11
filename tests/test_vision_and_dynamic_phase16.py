"""Unit tests for Phase 16.2 RGB-D Vision Localization & Phase 16.3 Dynamic Tracker."""

from __future__ import annotations

import numpy as np

from src.environments.pick_place_env import PandaPickPlaceEnv
from src.perception.dynamic_tracker import DynamicObjectTracker, TrackerConfig
from src.perception.rgbd_camera import CameraConfig, SimulatedRGBDCamera
from src.perception.vision_detector import VisionObjectLocalizer


def test_rgbd_camera_and_vision_localization_accuracy():
    env = PandaPickPlaceEnv(render_mode=None, max_episode_steps=50)
    try:
        obs, _ = env.reset(seed=42)
        gt_pos = obs["achieved_goal"][:3].copy()

        cam = SimulatedRGBDCamera(env.sim.client_id, CameraConfig(width=128, height=128))
        detector = VisionObjectLocalizer(cam, cube_half_extent_m=0.025)

        rgb, depth_m, raw_d, seg = cam.capture()
        assert rgb.shape == (128, 128, 3)
        assert depth_m.shape == (128, 128)

        det = detector.detect_red_cube(rgb, raw_d, depth_m)
        assert det.detected is True
        err_mm = float(np.linalg.norm(det.position_m[:2] - gt_pos[:2]) * 1000.0)
        assert err_mm < 15.0, f"XY visual localization error too large: {err_mm:.2f} mm"
        iou = detector.compute_segmentation_iou(det.pred_mask, seg, env.cube.body_id)
        assert iou > 0.70, f"Segmentation IoU unexpectedly low: {iou:.3f}"
    finally:
        env.close()


def test_dynamic_tracker_velocity_prediction_and_replan():
    tracker = DynamicObjectTracker(TrackerConfig(dt_step=0.05, window_size=4, replan_err_thresh_m=0.02))
    pos = np.array([0.45, -0.10, 0.025], dtype=np.float32)
    vel = np.array([0.04, 0.08, 0.0], dtype=np.float32)
    tracker.reset(pos)
    for step in range(1, 6):
        pos = pos + vel * 0.05
        out = tracker.update(pos, dt=0.05)
        assert not out["disturbance_detected"]

    pred = tracker.predict_position(horizon_s=0.10)
    expected = pos + vel * 0.10
    assert np.linalg.norm(pred[:2] - expected[:2]) < 0.01

    # Sudden jump simulates external disturbance -> triggers replanning
    jumped = pos + np.array([0.05, -0.05, 0.0], dtype=np.float32)
    out_jump = tracker.update(jumped, dt=0.05)
    assert out_jump["disturbance_detected"] is True
    assert out_jump["replan_triggered"] is True

