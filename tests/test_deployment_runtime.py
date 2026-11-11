"""Unit tests for Phase 16.5 Simulation Deployment Architecture & Phase 17.3 Hardware Interfaces."""

from __future__ import annotations

import subprocess
import sys
import time
import numpy as np
import pytest

from src.deployment.runtime import (
    ActionPostprocessor,
    DeploymentRuntime,
    EnvironmentAdapter,
    ObsPreprocessor,
    PolicyInferenceInterface,
    RuntimeConfig,
    SafetyMonitor,
)
from src.deployment.hardware_interface import (
    CartesianDeltaCommand,
    FrankaState,
    HARDWARE_AVAILABLE,
    HardwareUnavailableError,
    SafetySupervisor,
    StubLiveFrankaDriver,
)


def test_runtime_imports_without_training_pipeline():
    """Runtime module must load without importing src.rl.train or HerReplayBuffer."""
    code = (
        "import sys; import src.deployment.runtime; "
        "assert 'src.rl.train' not in sys.modules; "
        "print('CLEAN_RUNTIME_OK')"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert "CLEAN_RUNTIME_OK" in out


def test_safety_monitor_workspace_clamp_and_estop():
    cfg = RuntimeConfig(workspace_min=(0.3, -0.2, 0.02), workspace_max=(0.6, 0.2, 0.4), max_contact_force_n=100.0, action_scale=0.05)
    mon = SafetyMonitor(cfg)
    act, info = mon.check_and_filter(np.array([0.59, 0.0, 0.1], dtype=np.float32), np.array([1.0, 0.0, 0.0, -1.0], dtype=np.float32), contact_force_n=10.0)
    assert info["estop"] is False
    assert info["clamped"] is True
    assert float(0.59 + act[0] * 0.05) <= 0.6001

    act2, info2 = mon.check_and_filter(np.array([0.5, 0.0, 0.1], dtype=np.float32), np.array([0.5, 0.5, 0.0, -1.0], dtype=np.float32), contact_force_n=150.0)
    assert info2["estop"] is True
    assert np.allclose(act2[:3], 0.0)


def test_hardware_interface_safety_and_refusal():
    assert HARDWARE_AVAILABLE is False
    sup = SafetySupervisor()
    now = time.time()
    good_state = FrankaState(
        q=np.zeros(7, dtype=np.float32),
        dq=np.zeros(7, dtype=np.float32),
        ee_pos=np.array([0.5, 0.0, 0.2], dtype=np.float32),
        ee_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        gripper_width=0.06,
        ext_wrench=np.zeros(6, dtype=np.float32),
        timestamp_s=now,
        robot_mode_ok=True,
    )
    ok, reason, safe_cmd = sup.validate_command(good_state, CartesianDeltaCommand(np.array([0.1, 0.0, 0.0], dtype=np.float32)), now_s=now)
    assert ok is True and reason == "ok"
    assert np.linalg.norm(safe_cmd.dpos_m) <= 0.0301

    # Stale state triggers E-stop
    ok_stale, reason_stale, _ = sup.validate_command(good_state, CartesianDeltaCommand(np.zeros(3, dtype=np.float32)), now_s=now + 1.0)
    assert ok_stale is False and "stale" in reason_stale

    drv = StubLiveFrankaDriver()
    with pytest.raises(HardwareUnavailableError):
        drv.send_cartesian_delta(safe_cmd)

