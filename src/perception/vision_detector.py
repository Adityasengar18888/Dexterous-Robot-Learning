"""Pure Vision Object Segmentation, 3D Localization, and 6D Pose Estimation (Phase 16.2).

Operates strictly on simulated RGB-D image buffers (`rgb`, `depth_m`, `raw_depth`).
Never reads hidden simulator object positions or orientations during inference!
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

from src.perception.rgbd_camera import SimulatedRGBDCamera


@dataclass
class VisualDetectionResult:
    """Estimated object & scene pose strictly from camera RGB-D observations."""

    detected: bool
    position_m: np.ndarray        # (3,) estimated 3D object center in world frame
    yaw_rad: float                # estimated dominant yaw orientation (modulo 90 deg for square prism)
    pose_6d: np.ndarray           # (6,) [x, y, z, roll, pitch, yaw]
    quat_xyzw: np.ndarray         # (4,) estimated quaternion
    pixel_count: int
    confidence: float
    pred_mask: np.ndarray         # (H, W) bool segmentation mask


class VisionObjectLocalizer:
    """RGB color + depth table-plane segmentation and PCA/OBB 6D pose estimation."""

    def __init__(
        self,
        camera: SimulatedRGBDCamera,
        cube_half_extent_m: float = 0.025,
        table_z_thresh_m: float = 0.008,
    ) -> None:
        self.cam = camera
        self.half_ext = float(cube_half_extent_m)
        self.table_z_thresh = float(table_z_thresh_m)

    @staticmethod
    def _yaw_to_quat(yaw: float) -> np.ndarray:
        half = 0.5 * float(yaw)
        return np.array([0.0, 0.0, float(np.sin(half)), float(np.cos(half))], dtype=np.float32)

    def detect_red_cube(
        self,
        rgb: np.ndarray,
        raw_depth: np.ndarray,
        depth_m: Optional[np.ndarray] = None,
    ) -> VisualDetectionResult:
        """Segment red/warm foreground cube from RGB + metric unprojected height and estimate 6D pose."""
        h, w, _ = rgb.shape
        r = rgb[:, :, 0].astype(np.float32)
        g = rgb[:, :, 1].astype(np.float32)
        b = rgb[:, :, 2].astype(np.float32)

        # Pure image-space chromaticity + dominance filter (no simulator IDs)
        red_mask = (r > 95.0) & ((r - g) > 35.0) & ((r - b) > 35.0)
        if not np.any(red_mask):
            return VisualDetectionResult(
                detected=False,
                position_m=np.zeros(3, dtype=np.float32),
                yaw_rad=0.0,
                pose_6d=np.zeros(6, dtype=np.float32),
                quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                pixel_count=0,
                confidence=0.0,
                pred_mask=np.zeros((h, w), dtype=bool),
            )

        # Keep largest connected component to reject stray shadows or background artifacts
        labeled, n_comp = ndimage.label(red_mask)
        if n_comp > 1:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0
            best_lbl = int(np.argmax(sizes))
            comp_mask = labeled == best_lbl
        else:
            comp_mask = red_mask

        pts_world = self.cam.unproject_points(raw_depth, comp_mask)
        # Keep points above tabletop and inside plausible manipulation workspace
        above_table = (
            (pts_world[:, 2] >= self.table_z_thresh)
            & (pts_world[:, 0] >= 0.15)
            & (pts_world[:, 0] <= 0.85)
            & (np.abs(pts_world[:, 1]) <= 0.45)
        )
        fg_pts = pts_world[above_table]
        if fg_pts.shape[0] < 4:
            return VisualDetectionResult(
                detected=False,
                position_m=np.zeros(3, dtype=np.float32),
                yaw_rad=0.0,
                pose_6d=np.zeros(6, dtype=np.float32),
                quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                pixel_count=int(fg_pts.shape[0]),
                confidence=0.0,
                pred_mask=comp_mask,
            )

        # Shift visible surface centroid along incoming camera viewing ray by ~half_ext to recover cube 3D center
        surf_center = np.median(fg_pts, axis=0)
        ray = surf_center - np.asarray(self.cam.cfg.eye_position, dtype=np.float32)
        ray_norm = float(np.linalg.norm(ray))
        ray_unit = ray / max(ray_norm, 1e-6)
        pos_est = (surf_center + (0.75 * self.half_ext) * ray_unit).astype(np.float32)
        pos_est[2] = float(max(0.015, pos_est[2]))

        # 6D orientation estimation via 2D PCA on top-face XY point cloud
        yaw_est = 0.0
        if fg_pts.shape[0] >= 8:
            centered_xy = fg_pts[:, :2] - np.mean(fg_pts[:, :2], axis=0, keepdims=True)
            cov = centered_xy.T @ centered_xy
            eigvals, eigvecs = np.linalg.eigh(cov)
            principal = eigvecs[:, int(np.argmax(eigvals))]
            yaw_est = float(np.arctan2(principal[1], principal[0]))

        pose_6d = np.array([pos_est[0], pos_est[1], pos_est[2], 0.0, 0.0, yaw_est], dtype=np.float32)
        conf = float(min(1.0, fg_pts.shape[0] / 35.0))
        return VisualDetectionResult(
            detected=True,
            position_m=pos_est,
            yaw_rad=yaw_est,
            pose_6d=pose_6d,
            quat_xyzw=self._yaw_to_quat(yaw_est),
            pixel_count=int(fg_pts.shape[0]),
            confidence=conf,
            pred_mask=comp_mask,
        )

    @staticmethod
    def compute_segmentation_iou(pred_mask: np.ndarray, seg_mask: np.ndarray, target_body_id: int) -> float:
        """Evaluation-only metric comparing visual mask against PyBullet renderer body ID mask."""
        gt_mask = seg_mask == int(target_body_id)
        inter = float(np.sum(pred_mask & gt_mask))
        union = float(np.sum(pred_mask | gt_mask))
        return float(inter / max(union, 1.0))

