"""Robotics module exposing robot models and interfaces."""

from src.robots.base_robot import BaseRobot
from src.robots.panda import FrankaPanda

__all__ = ["BaseRobot", "FrankaPanda"]
