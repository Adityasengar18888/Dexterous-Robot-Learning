"""Strict Pydantic v2 schemas for Robot Control, Simulation Telemetry, Policies, Tasks, and WebSockets."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


PANDA_LOWER_LIMITS = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
PANDA_UPPER_LIMITS = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
WS_BOUNDS = {"x": (0.20, 0.80), "y": (-0.45, 0.45), "z": (0.01, 0.65)}


class JointCommandRequest(BaseModel):
    positions: List[float] = Field(..., min_length=7, max_length=7, description="7 Panda arm joint angles in radians")
    steps: int = Field(default=60, ge=1, le=300)

    @field_validator("positions")
    @classmethod
    def check_limits(cls, v: List[float]) -> List[float]:
        for i, val in enumerate(v):
            lo, hi = PANDA_LOWER_LIMITS[i], PANDA_UPPER_LIMITS[i]
            if not (lo <= val <= hi):
                raise ValueError(f"Joint {i+1} target {val:.4f} rad outside Panda limits [{lo}, {hi}]")
        return [float(x) for x in v]


class CartesianTargetRequest(BaseModel):
    x: float = Field(..., ge=WS_BOUNDS["x"][0], le=WS_BOUNDS["x"][1])
    y: float = Field(..., ge=WS_BOUNDS["y"][0], le=WS_BOUNDS["y"][1])
    z: float = Field(..., ge=WS_BOUNDS["z"][0], le=WS_BOUNDS["z"][1])
    roll: Optional[float] = Field(default=None, ge=-3.1416, le=3.1416)
    pitch: Optional[float] = Field(default=None, ge=-3.1416, le=3.1416)
    yaw: Optional[float] = Field(default=None, ge=-3.1416, le=3.1416)
    steps: int = Field(default=90, ge=5, le=300)


class GripperCommandRequest(BaseModel):
    action: Literal["open", "close", "set_width"]
    width: Optional[float] = Field(default=None, ge=0.0, le=0.08, description="Aperture in meters [0.0, 0.08]")
    steps: int = Field(default=40, ge=5, le=200)


class EnvConfigRequest(BaseModel):
    environment: str = Field(default="PandaPickPlaceEnv")
    seed: int = Field(default=42, ge=0, le=1000000)
    dr_tier: str = Field(default="DR0", pattern="^(DR0|DR1|DR2|DR3|DR4)$")
    dr_enabled: bool = Field(default=False)


class TaskExecuteRequest(BaseModel):
    task_id: str
    controller_mode: Literal["SCRIPTED", "POLICY"] = "SCRIPTED"
    policy_id: Optional[str] = None
    seed: int = Field(default=42, ge=0, le=1000000)
    max_steps: int = Field(default=150, ge=10, le=500)


class PolicyLoadRequest(BaseModel):
    policy_id: str


class ControlResponse(BaseModel):
    ok: bool
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)

