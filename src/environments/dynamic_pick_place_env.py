"""Dynamic Moving-Object Pick-and-Place Environment (Phase 16.3).

Extends ``PandaPickPlaceEnv`` by advancing a smooth deterministic or disturbed
planar motion profile on ungrasped cubes and optional moving target waypoints.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple
import numpy as np
import pybullet as p

from src.environments.pick_place_env import PandaPickPlaceEnv


MotionMode = Literal["linear", "sinusoidal", "disturbance_step"]


class PandaDynamicPickPlaceEnv(PandaPickPlaceEnv):
    """Dynamic manipulation environment supporting moving cubes, target drift, and external impulses."""

    def __init__(
        self,
        render_mode: str | None = None,
        distance_threshold: float = 0.05,
        max_episode_steps: int = 100,
        control_steps: int = 15,
        action_scale: float = 0.05,
        motion_mode: MotionMode = "linear",
        cube_speed: float = 0.035,
        target_speed: float = 0.0,
        disturbance_step: Optional[int] = 18,
    ) -> None:
        self.motion_mode: MotionMode = motion_mode
        self.cube_speed = float(cube_speed)
        self.target_speed = float(target_speed)
        self.disturbance_step = disturbance_step
        self._cube_vel_cmd = np.zeros(3, dtype=np.float32)
        self._target_vel_cmd = np.zeros(3, dtype=np.float32)
        self._disturbance_injected = False
        super().__init__(
            render_mode=render_mode,
            distance_threshold=distance_threshold,
            max_episode_steps=max_episode_steps,
            control_steps=control_steps,
            action_scale=action_scale,
        )

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        # Keep initial dynamic cube inside central region so linear sweep stays reachable
        cx = float(self.np_random.uniform(0.42, 0.56))
        cy = float(self.np_random.uniform(-0.10, 0.10))
        self.cube.reset(position=[cx, cy, 0.025])

        angle = float(self.np_random.uniform(-np.pi, np.pi))
        self._cube_vel_cmd = np.array(
            [self.cube_speed * np.cos(angle), self.cube_speed * np.sin(angle), 0.0],
            dtype=np.float32,
        )
        t_angle = float(self.np_random.uniform(-np.pi, np.pi))
        self._target_vel_cmd = np.array(
            [self.target_speed * np.cos(t_angle), self.target_speed * np.sin(t_angle), 0.0],
            dtype=np.float32,
        )
        self._disturbance_injected = False
        self.sim.step(5)
        obs = self._get_obs()
        info = self._get_info(obs["achieved_goal"], self.goal)
        info["motion_mode"] = self.motion_mode
        info["disturbance_injected"] = False
        return obs, info

    def _is_cube_grasped(self) -> bool:
        cid = self.sim.client_id
        contacts = p.getContactPoints(bodyA=self.controller.robot.body_id, bodyB=self.cube.body_id, physicsClientId=cid)
        f9 = any(c[3] == 9 for c in contacts)
        f10 = any(c[3] == 10 for c in contacts)
        return bool(f9 and f10 and sum(c[9] for c in contacts) >= 3.0)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        dt = float(self.control_steps) * (1.0 / 240.0)
        grasped_before = self._is_cube_grasped()

        # Advance ungrasped dynamic cube smoothly inside workspace bounce boundaries
        if not grasped_before and self.cube_speed > 0.0:
            cid = self.sim.client_id
            pos, orn = p.getBasePositionAndOrientation(self.cube.body_id, physicsClientId=cid)
            p_arr = np.asarray(pos, dtype=np.float32)

            if self.motion_mode == "sinusoidal":
                phase = 0.25 * float(self.current_step)
                v = np.array(
                    [self._cube_vel_cmd[0] * np.cos(phase), self.cube_speed * np.sin(phase), 0.0],
                    dtype=np.float32,
                )
            else:
                v = self._cube_vel_cmd.copy()

            disturbed_now = False
            if (
                self.motion_mode == "disturbance_step"
                and self.disturbance_step is not None
                and self.current_step == int(self.disturbance_step)
                and not self._disturbance_injected
            ):
                # Sudden lateral impulse / velocity direction flip testing disturbance detection & replanning
                p_arr[0] = float(np.clip(p_arr[0] + 0.035, 0.36, 0.62))
                p_arr[1] = float(np.clip(p_arr[1] - 0.040, -0.22, 0.22))
                self._cube_vel_cmd = np.array([-self._cube_vel_cmd[1], self._cube_vel_cmd[0], 0.0], dtype=np.float32)
                v = self._cube_vel_cmd.copy()
                self._disturbance_injected = True
                disturbed_now = True

            next_p = p_arr + v * dt
            if next_p[0] < 0.36 or next_p[0] > 0.62:
                self._cube_vel_cmd[0] *= -1.0
                next_p[0] = float(np.clip(next_p[0], 0.36, 0.62))
            if next_p[1] < -0.22 or next_p[1] > 0.22:
                self._cube_vel_cmd[1] *= -1.0
                next_p[1] = float(np.clip(next_p[1], -0.22, 0.22))
            next_p[2] = float(max(0.024, p_arr[2]))
            p.resetBasePositionAndOrientation(self.cube.body_id, next_p.tolist(), list(orn), physicsClientId=cid)
            p.resetBaseVelocity(self.cube.body_id, v.tolist(), [0.0, 0.0, 0.0], physicsClientId=cid)
        else:
            disturbed_now = False

        if self.target_speed > 0.0:
            self.goal = np.clip(
                self.goal + self._target_vel_cmd * dt,
                self.target_workspace_min,
                self.target_workspace_max,
            ).astype(np.float32)

        obs, rew, term, trunc, info = super().step(action)
        info["motion_mode"] = self.motion_mode
        info["cube_grasped"] = bool(self._is_cube_grasped())
        info["disturbance_injected"] = bool(disturbed_now or self._disturbance_injected)
        return obs, rew, term, trunc, info

