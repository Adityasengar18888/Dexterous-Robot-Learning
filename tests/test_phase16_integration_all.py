"""End-to-End Integration Validation for Phase 16 & Phase 17 Modules and Checkpoints."""

from __future__ import annotations

import os
import pytest

from src.deployment.runtime import CheckpointLoader, DeploymentRuntime, EnvironmentAdapter, PolicyInferenceInterface, RuntimeConfig
from src.environments.pick_place_env import PandaPickPlaceEnv


def test_frozen_phase15_and_phase16_deployment_smoke():
    """Verify DeploymentRuntime executes frozen checkpoint cleanly on PyBullet PickPlace."""
    ckpt = "experiments/phase13_dense_dr/dr4/seed42/models/checkpoint_500000.zip"
    assert os.path.exists(ckpt)
    cfg = RuntimeConfig(env_name="PickPlace", checkpoint_path=ckpt, max_episode_steps=100, enable_safety_monitor=True)
    env = PandaPickPlaceEnv(render_mode=None, max_episode_steps=100, randomizer_config={"enabled": True, "object_position": {"enabled": True}, "target_position": {"enabled": True}})
    try:
        policy = PolicyInferenceInterface.from_sb3_checkpoint(ckpt, policy_type="LEARNED_SAC_HER", device="cpu")
        rt = DeploymentRuntime(EnvironmentAdapter(env), policy, cfg)
        rec = rt.run_episode(seed=9901)
        assert "success" in rec
        assert rec["estop_triggered"] is False
        assert rec["mean_inference_ms"] > 0.0
    finally:
        env.close()

