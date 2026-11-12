"""Franka Emika Panda URDF asset paths, joint limits, and live FK link transform helper wrapping src.robots.panda."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pybullet as p

from src.robots.panda import FrankaPanda

PACKAGE_DIR = Path(__file__).resolve().parent
URDF_PATH = PACKAGE_DIR / "panda.urdf"
MESHES_DIR = PACKAGE_DIR / "meshes"

# Exact link indices & visual mesh mapping matching panda.urdf
PANDA_LINK_MESH_SPECS: List[Dict[str, Any]] = [
    {"name": "panda_link0", "link_idx": -1, "visual": "collision/link0.obj", "collision": "collision/link0.obj", "rpy": [0.0, 0.0, 0.0]},
    {"name": "panda_link1", "link_idx": 0, "visual": "visual/link1.obj", "collision": "collision/link1.obj", "rpy": [0.0, 0.0, 0.0]},
    {"name": "panda_link2", "link_idx": 1, "visual": "visual/link2.obj", "collision": "collision/link2.obj", "rpy": [0.0, 0.0, 0.0]},
    {"name": "panda_link3", "link_idx": 2, "visual": "visual/link3.obj", "collision": "collision/link3.obj", "rpy": [0.0, 0.0, 0.0]},
    {"name": "panda_link4", "link_idx": 3, "visual": "visual/link4.obj", "collision": "collision/link4.obj", "rpy": [0.0, 0.0, 0.0]},
    {"name": "panda_link5", "link_idx": 4, "visual": "visual/link5.obj", "collision": "collision/link5.obj", "rpy": [0.0, 0.0, 0.0]},
    {"name": "panda_link6", "link_idx": 5, "visual": "visual/link6.obj", "collision": "collision/link6.obj", "rpy": [0.0, 0.0, 0.0]},
    {"name": "panda_link7", "link_idx": 6, "visual": "collision/link7.obj", "collision": "collision/link7.obj", "rpy": [0.0, 0.0, 0.0]},
    {"name": "panda_hand", "link_idx": 8, "visual": "visual/hand.obj", "collision": "collision/hand.obj", "rpy": [0.0, 0.0, 0.0]},
    {"name": "panda_leftfinger", "link_idx": 9, "visual": "visual/finger.obj", "collision": "collision/finger.obj", "rpy": [0.0, 0.0, 0.0]},
    {"name": "panda_rightfinger", "link_idx": 10, "visual": "visual/finger.obj", "collision": "collision/finger.obj", "rpy": [0.0, 0.0, 3.14159265359]},
]

PANDA_JOINT_NAMES: List[str] = [f"panda_joint{i}" for i in range(1, 8)]


def get_link_world_transforms(robot: FrankaPanda) -> List[Dict[str, Any]]:
    """Extract authoritative URDF frame transforms [x,y,z] + [qx,qy,qz,w] directly from PyBullet."""
    cid = robot.cid
    body_id = robot.body_id
    out: List[Dict[str, Any]] = []
    for spec in PANDA_LINK_MESH_SPECS:
        idx = int(spec["link_idx"])
        if idx < 0:
            pos, orn = robot.base_position, robot.base_orientation
        else:
            ls = p.getLinkState(body_id, idx, computeForwardKinematics=True, physicsClientId=cid)
            pos, orn = ls[4], ls[5]  # worldLinkFramePosition, worldLinkFrameOrientation
        out.append({
            "name": spec["name"],
            "link_idx": idx,
            "visual_mesh": spec["visual"],
            "collision_mesh": spec["collision"],
            "mesh_rpy": spec["rpy"],
            "position": [round(float(v), 5) for v in pos],
            "quaternion": [round(float(v), 5) for v in orn],
        })
    return out


__all__ = ["FrankaPanda", "URDF_PATH", "MESHES_DIR", "PANDA_LINK_MESH_SPECS", "PANDA_JOINT_NAMES", "get_link_world_transforms"]

