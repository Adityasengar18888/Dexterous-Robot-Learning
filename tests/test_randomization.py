"""Tests for DomainRandomizer."""

import pytest
import numpy as np
from src.utils.randomization import DomainRandomizer

def test_randomization_disabled():
    rng = np.random.default_rng(42)
    config = {"enabled": False}
    dr = DomainRandomizer(config, rng)
    
    assert not dr.is_enabled("object_position")
    
    default_pos = np.array([0.5, 0.0, 0.1])
    pos = dr.sample_position("object_position", default_pos, np.zeros(3), np.ones(3))
    assert np.allclose(pos, default_pos)
    
def test_randomization_enabled():
    rng = np.random.default_rng(42)
    config = {
        "enabled": True,
        "object_position": {"enabled": True},
        "mass": {"enabled": True, "min": 0.1, "max": 0.5},
        "friction": {"enabled": True, "min": 0.5, "max": 1.5}
    }
    dr = DomainRandomizer(config, rng)
    
    assert dr.is_enabled("object_position")
    
    default_pos = np.array([0.5, 0.0, 0.1])
    bounds_min = np.array([0.0, 0.0, 0.0])
    bounds_max = np.array([1.0, 1.0, 1.0])
    
    pos1 = dr.sample_position("object_position", default_pos, bounds_min, bounds_max)
    assert not np.allclose(pos1, default_pos)
    assert np.all(pos1 >= bounds_min) and np.all(pos1 <= bounds_max)
    
    mass = dr.sample_mass(1.0)
    assert 0.1 <= mass <= 0.5
    
    friction = dr.sample_friction(2.0)
    assert 0.5 <= friction <= 1.5

def test_randomization_reproducibility():
    config = {
        "enabled": True,
        "object_position": {"enabled": True}
    }
    
    rng1 = np.random.default_rng(42)
    dr1 = DomainRandomizer(config, rng1)
    pos1 = dr1.sample_position("object_position", np.zeros(3), np.zeros(3), np.ones(3))
    
    rng2 = np.random.default_rng(42)
    dr2 = DomainRandomizer(config, rng2)
    pos2 = dr2.sample_position("object_position", np.zeros(3), np.zeros(3), np.ones(3))
    
    assert np.allclose(pos1, pos2)
    
def test_orientation_normalization():
    rng = np.random.default_rng(42)
    config = {"enabled": True, "object_orientation": {"enabled": True}}
    dr = DomainRandomizer(config, rng)
    
    ori = dr.sample_orientation("object_orientation", np.array([0,0,0,1]), yaw_only=False)
    assert np.isclose(np.linalg.norm(ori), 1.0)
    
    ori_yaw = dr.sample_orientation("object_orientation", np.array([0,0,0,1]), yaw_only=True)
    assert np.isclose(np.linalg.norm(ori_yaw), 1.0)
