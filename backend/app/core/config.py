"""Application configuration and workspace paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
ROBOTICS_DIR = ROOT_DIR / "robotics"
FRANKA_ASSETS_DIR = ROBOTICS_DIR / "robots" / "franka_panda"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
RESULTS_DIR = ROOT_DIR / "results"
PUBLICATION_DIR = ROOT_DIR / "publication"

HOST = os.getenv("DEXTEROUS_HOST", "127.0.0.1")
PORT = int(os.getenv("DEXTEROUS_PORT", "8000"))
SIM_RATE_HZ = float(os.getenv("DEXTEROUS_SIM_RATE_HZ", "20.0"))
CORS_ORIGINS: List[str] = [
    o.strip()
    for o in os.getenv("DEXTEROUS_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if o.strip()
]
