"""MuJoCo Franka Emika Panda Pick-and-Place Environment for Cross-Simulator Validation (Phase 17.1).

Mirrors PyBullet `PandaPickPlaceEnv` exact coordinate frame, cube geometry (`0.05m`, `0.1kg`),
action semantics (`[dx, dy, dz, gripper]`, `action_scale=0.05`), and Dict observation layout
(`observation=(10,)`, `achieved_goal=(3,)`, `desired_goal=(3,)`).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


_PANDA_MJCF = """
<mujoco model="franka_panda_pickplace">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.0041667" integrator="implicitfast" gravity="0 0 -9.81" cone="elliptic" noslip_iterations="3"/>
  <default>
    <joint damping="2.0" armature="0.1"/>
    <geom friction="1.0 0.005 0.001" margin="0.0005"/>
  </default>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <geom name="table" type="plane" pos="0 0 0" size="2 2 0.1" rgba="0.85 0.85 0.85 1" friction="1.0 0.005 0.001"/>
    <site name="target_site" pos="0.5 0.0 0.1" size="0.025" rgba="0.1 0.9 0.2 0.5"/>

    <body name="link0" pos="0 0 0">
      <body name="link1" pos="0 0 0.333">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
        <geom type="capsule" size="0.05" fromto="0 0 -0.15 0 0 0" contype="0" conaffinity="0"/>
        <body name="link2" pos="0 0 0" quat="0.707107 -0.707107 0 0">
          <joint name="joint2" type="hinge" axis="0 0 1" range="-1.7628 1.7628"/>
          <geom type="capsule" size="0.05" fromto="0 0 0 0 -0.15 0" contype="0" conaffinity="0"/>
          <body name="link3" pos="0 -0.316 0" quat="0.707107 0.707107 0 0">
            <joint name="joint3" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
            <geom type="capsule" size="0.045" fromto="0 0 -0.12 0 0 0" contype="0" conaffinity="0"/>
            <body name="link4" pos="0.0825 0 0" quat="0.707107 0.707107 0 0">
              <joint name="joint4" type="hinge" axis="0 0 1" range="-3.0718 -0.0698"/>
              <geom type="capsule" size="0.045" fromto="0 0 0 0 0.12 0" contype="0" conaffinity="0"/>
              <body name="link5" pos="-0.0825 0.384 0" quat="0.707107 -0.707107 0 0">
                <inertial pos="0 0 0" mass="1.2" diaginertia="0.01 0.01 0.01"/>
                <joint name="joint5" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                <geom type="capsule" size="0.04" fromto="0 0 -0.12 0 0 0" contype="0" conaffinity="0"/>
                <body name="link6" pos="0 0 0" quat="0.707107 0.707107 0 0">
                  <inertial pos="0 0 0" mass="1.0" diaginertia="0.008 0.008 0.008"/>
                  <joint name="joint6" type="hinge" axis="0 0 1" range="-0.0175 3.7525"/>
                  <geom type="sphere" size="0.04" pos="0 0 0" contype="0" conaffinity="0"/>
                  <body name="link7" pos="0.088 0 0" quat="0.707107 0.707107 0 0">
                    <inertial pos="0 0 0" mass="0.8" diaginertia="0.005 0.005 0.005"/>
                    <joint name="joint7" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                    <geom type="capsule" size="0.035" fromto="0 0 0 0 0 0.08" contype="0" conaffinity="0"/>
                    <body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">
                      <inertial pos="0 0 0.03" mass="0.7" diaginertia="0.005 0.005 0.005"/>
                      <geom name="palm" type="box" size="0.03 0.08 0.025" pos="0 0 0.025" contype="0" conaffinity="0" rgba="0.9 0.9 0.9 1"/>
                      <site name="ee_site" pos="0 0 0.1034" size="0.008" rgba="0.2 0.2 0.9 0.8"/>
                      <body name="left_finger" pos="0 0 0.0584">
                        <joint name="finger_joint1" type="slide" axis="0 1 0" range="0 0.04" damping="1.5"/>
                        <geom name="left_pad" type="box" size="0.012 0.008 0.024" pos="0 0.008 0.042" friction="1.2 0.01 0.001" rgba="0.2 0.2 0.2 1"/>
                      </body>
                      <body name="right_finger" pos="0 0 0.0584">
                        <joint name="finger_joint2" type="slide" axis="0 -1 0" range="0 0.04" damping="1.5"/>
                        <geom name="right_pad" type="box" size="0.012 0.008 0.024" pos="0 -0.008 0.042" friction="1.2 0.01 0.001" rgba="0.2 0.2 0.2 1"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <body name="cube" pos="0.5 0.0 0.025">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="box" size="0.025 0.025 0.025" mass="0.1" friction="1.0 0.005 0.001" rgba="0.8 0.2 0.2 1"/>
    </body>
  </worldbody>

  <actuator>
    <position name="act1" joint="joint1" kp="500" kv="50" forcerange="-87 87"/>
    <position name="act2" joint="joint2" kp="500" kv="50" forcerange="-87 87"/>
    <position name="act3" joint="joint3" kp="500" kv="50" forcerange="-87 87"/>
    <position name="act4" joint="joint4" kp="500" kv="50" forcerange="-87 87"/>
    <position name="act5" joint="joint5" kp="300" kv="30" forcerange="-12 12"/>
    <position name="act6" joint="joint6" kp="300" kv="30" forcerange="-12 12"/>
    <position name="act7" joint="joint7" kp="300" kv="30" forcerange="-12 12"/>
    <position name="act_f1" joint="finger_joint1" kp="400" kv="20" forcerange="-25 25"/>
    <position name="act_f2" joint="finger_joint2" kp="400" kv="20" forcerange="-25 25"/>
  </actuator>
</mujoco>
"""


class MuJoCoPandaPickPlaceEnv(gym.Env):
    """MuJoCo twin of `PandaPickPlaceEnv` with matching observation/goal/action spaces."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        distance_threshold: float = 0.05,
        max_episode_steps: int = 100,
        control_steps: int = 15,
        action_scale: float = 0.05,
        cube_mass: float = 0.1,
        cube_friction: float = 1.0,
        gripper_force_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = mujoco.MjModel.from_xml_string(_PANDA_MJCF)
        self.data = mujoco.MjData(self.model)

        self.distance_threshold = float(distance_threshold)
        self.max_episode_steps = int(max_episode_steps)
        self.control_steps = int(control_steps)
        self.action_scale = float(action_scale)
        self.current_step = 0
        self.collision_count = 0

        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.table_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        self.left_pad_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_pad")
        self.right_pad_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_pad")

        self.model.body_mass[self.cube_body_id] = float(cube_mass)
        self.model.geom_friction[self.cube_geom_id, 0] = float(cube_friction)
        for act_name in ("act_f1", "act_f2"):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            self.model.actuator_forcerange[aid] = np.array([-25.0 * gripper_force_scale, 25.0 * gripper_force_scale])

        self.neutral_q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)
        self.workspace_min = np.array([0.4, -0.2, 0.025], dtype=np.float32)
        self.workspace_max = np.array([0.6, 0.2, 0.025], dtype=np.float32)
        self.target_workspace_min = np.array([0.3, -0.25, 0.025], dtype=np.float32)
        self.target_workspace_max = np.array([0.7, 0.25, 0.2], dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
                "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            }
        )
        self.goal = np.zeros(3, dtype=np.float32)
        self._finger_target = 0.04

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.collision_count = 0
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:7] = self.neutral_q
        self.data.qpos[7:9] = 0.04
        self._finger_target = 0.04

        cube_pos = self.np_random.uniform(low=self.workspace_min, high=self.workspace_max, size=(3,)).astype(np.float32)
        cube_pos[2] = 0.025
        # qpos[9:16] is cube freejoint [x, y, z, qw, qx, qy, qz]
        self.data.qpos[9:12] = cube_pos
        self.data.qpos[12:16] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        self.goal = self.np_random.uniform(
            low=self.target_workspace_min, high=self.target_workspace_max, size=(3,)
        ).astype(np.float32)
        self.model.site_pos[self.target_site_id] = self.goal

        self.data.ctrl[:7] = self.neutral_q
        self.data.ctrl[7:9] = 0.04
        mujoco.mj_forward(self.model, self.data)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        return obs, self._get_info(obs["achieved_goal"], self.goal)

    def _solve_ik_step(self, target_ee_pos: np.ndarray, n_iters: int = 12) -> np.ndarray:
        """Damped-least-squares Jacobian IK keeping EE Z-axis downward [-Z_world]."""
        q_work = self.data.qpos[:7].copy()
        d_save_q = self.data.qpos.copy()
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        desired_z_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        lam = 0.04

        for _ in range(n_iters):
            self.data.qpos[:7] = q_work
            mujoco.mj_kinematics(self.model, self.data)
            ee_pos = self.data.site_xpos[self.ee_site_id].copy()
            ee_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
            cur_z = ee_mat[:, 2]

            err_pos = target_ee_pos - ee_pos
            err_rot = np.cross(cur_z, desired_z_world) * 0.4
            err = np.concatenate([err_pos, err_rot])

            if np.linalg.norm(err_pos) < 1e-4 and np.linalg.norm(err_rot) < 1e-3:
                break

            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
            J = np.vstack([jacp[:, :7], jacr[:, :7]])
            dq = J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(6), err)
            q_work = np.clip(q_work + 0.7 * dq, self.model.jnt_range[:7, 0], self.model.jnt_range[:7, 1])

        self.data.qpos[:] = d_save_q
        mujoco.mj_forward(self.model, self.data)
        return q_work

    def _get_obs(self) -> Dict[str, np.ndarray]:
        ee_pos = self.data.site_xpos[self.ee_site_id].astype(np.float32).copy()
        gw = float(self.data.qpos[7] + self.data.qpos[8])
        cube_pos = self.data.qpos[9:12].astype(np.float32).copy()
        cube_vel = self.data.qvel[6:9].astype(np.float32).copy()
        obs = np.concatenate([ee_pos, [gw], cube_pos, cube_vel]).astype(np.float32)
        return {
            "observation": obs,
            "achieved_goal": cube_pos.copy(),
            "desired_goal": self.goal.copy(),
        }

    def _query_bilateral_contact(self) -> bool:
        left, right = False, False
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            geoms = (int(con.geom1), int(con.geom2))
            if self.cube_geom_id in geoms:
                if self.left_pad_id in geoms:
                    left = True
                if self.right_pad_id in geoms:
                    right = True
        return bool(left and right)

    def _get_info(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Dict[str, Any]:
        dist = float(np.linalg.norm(achieved_goal - desired_goal))
        return {
            "is_success": bool(dist <= self.distance_threshold),
            "distance": dist,
            "episode_length": self.current_step,
            "grasp_success": bool(achieved_goal[2] > 0.05),
            "bilateral_contact": self._query_bilateral_contact(),
            "collisions": self.collision_count,
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        act = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        dx, dy, dz = act[:3] * self.action_scale
        cur_ee = self.data.site_xpos[self.ee_site_id].copy()
        tgt_ee = np.clip(
            cur_ee + np.array([dx, dy, dz], dtype=np.float64),
            np.array([0.2, -0.3, 0.025], dtype=np.float64),
            np.array([0.7, 0.3, 0.50], dtype=np.float64),
        )
        if tgt_ee[2] <= 0.026 and cur_ee[2] <= 0.032 and dz < -0.01:
            self.collision_count += 1

        q_tgt = self._solve_ik_step(tgt_ee)
        self._finger_target = 0.04 if act[3] > 0.0 else 0.0
        self.data.ctrl[:7] = q_tgt
        self.data.ctrl[7:9] = self._finger_target

        for _ in range(self.control_steps):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info(obs["achieved_goal"], self.goal)
        reward = -1.0 if info["distance"] > self.distance_threshold else 0.0
        truncated = self.current_step >= self.max_episode_steps
        return obs, float(reward), False, truncated, info

