"""Simulated PyBullet RGB-D Camera with exact clip-to-world 3D unprojection (Phase 16.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pybullet as p


@dataclass
class CameraConfig:
    """Front-oblique RGB-D camera configuration avoiding arm self-occlusion across grasp & lift."""

    width: int = 128
    height: int = 128
    fov_deg: float = 55.0
    near: float = 0.05
    far: float = 2.0
    eye_position: Tuple[float, float, float] = (0.92, 0.28, 0.45)
    target_position: Tuple[float, float, float] = (0.48, 0.0, 0.05)
    up_vector: Tuple[float, float, float] = (0.0, 0.0, 1.0)


class SimulatedRGBDCamera:
    """Captures synchronized RGB, metric depth (m), segmentation mask, and unprojected 3D world points."""

    def __init__(self, client_id: int, config: Optional[CameraConfig] = None) -> None:
        self.cid = int(client_id)
        self.cfg = config or CameraConfig()

        aspect = float(self.cfg.width) / float(self.cfg.height)
        view_list = p.computeViewMatrix(
            cameraEyePosition=list(self.cfg.eye_position),
            cameraTargetPosition=list(self.cfg.target_position),
            cameraUpVector=list(self.cfg.up_vector),
        )
        proj_list = p.computeProjectionMatrixFOV(
            fov=float(self.cfg.fov_deg),
            aspect=aspect,
            nearVal=float(self.cfg.near),
            farVal=float(self.cfg.far),
        )
        # PyBullet returns OpenGL column-major 4x4 matrices
        self.view_matrix = np.asarray(view_list, dtype=np.float64).reshape(4, 4).T
        self.proj_matrix = np.asarray(proj_list, dtype=np.float64).reshape(4, 4).T
        self.clip_to_world = np.linalg.inv(self.proj_matrix @ self.view_matrix)
        self.world_to_cam = self.view_matrix.copy()

        # Precompute normalized device X/Y pixel grids for vectorized unprojection
        h, w = self.cfg.height, self.cfg.width
        u = np.arange(w, dtype=np.float64)
        v = np.arange(h, dtype=np.float64)
        uu, vv = np.meshgrid(u, v)
        self._x_ndc = (2.0 * (uu + 0.5) / float(w)) - 1.0
        self._y_ndc = 1.0 - (2.0 * (vv + 0.5) / float(h))

        fy = (0.5 * h) / np.tan(np.deg2rad(0.5 * self.cfg.fov_deg))
        self.intrinsics_K = np.array(
            [[fy, 0.0, 0.5 * w], [0.0, fy, 0.5 * h], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    def capture(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (rgb_uint8[H,W,3], depth_m[H,W], raw_depth[H,W], seg_mask[H,W])."""
        h, w = self.cfg.height, self.cfg.width
        view_flat = self.view_matrix.T.reshape(-1).tolist()
        proj_flat = self.proj_matrix.T.reshape(-1).tolist()
        _, _, rgba, depth_buf, seg = p.getCameraImage(
            width=w,
            height=h,
            viewMatrix=view_flat,
            projectionMatrix=proj_flat,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.cid,
        )
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        raw_d = np.asarray(depth_buf, dtype=np.float64).reshape(h, w)
        near, far = self.cfg.near, self.cfg.far
        depth_m = (far * near / (far - (far - near) * np.clip(raw_d, 0.0, 0.999999))).astype(np.float32)
        seg_mask = np.asarray(seg, dtype=np.int32).reshape(h, w)
        return rgb, depth_m, raw_d, seg_mask

    def unproject_points(self, raw_depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Exact clip-space to world-frame 3D point cloud unprojection for masked pixels (N, 3)."""
        ys, xs = np.where(mask)
        if ys.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        x_ndc = self._x_ndc[ys, xs]
        y_ndc = self._y_ndc[ys, xs]
        z_ndc = 2.0 * raw_depth[ys, xs] - 1.0
        ones = np.ones_like(x_ndc)
        clip_pts = np.stack([x_ndc, y_ndc, z_ndc, ones], axis=0)  # (4, N)
        world_h = self.clip_to_world @ clip_pts
        world_xyz = (world_h[:3, :] / np.where(np.abs(world_h[3:4, :]) > 1e-8, world_h[3:4, :], 1e-8)).T
        return world_xyz.astype(np.float32)

    @staticmethod
    def transform_points(pts_world: np.ndarray, rigid_T: np.ndarray) -> np.ndarray:
        """Apply homogeneous 4x4 transformation matrix to Nx3 points."""
        if pts_world.shape[0] == 0:
            return pts_world
        hom = np.concatenate([pts_world, np.ones((pts_world.shape[0], 1), dtype=pts_world.dtype)], axis=1)
        out = (rigid_T @ hom.T).T[:, :3]
        return out.astype(np.float32)

