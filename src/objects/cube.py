"""Cube object for manipulation tasks."""

import pybullet as p
import numpy as np

class Cube:
    """A generic cube object for manipulation tasks."""
    
    def __init__(self, client_id: int, size: float = 0.05, mass: float = 0.1, color: list = [0.8, 0.2, 0.2, 1.0]):
        self.client_id = client_id
        self.size = size
        self.mass = mass
        self.color = color
        self.body_id = -1
        
    def load(self, position=(0.5, 0.0, 0.025), orientation=(0, 0, 0, 1)):
        half_extents = [self.size / 2.0] * 3
        
        col_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self.client_id
        )
        vis_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=self.color,
            physicsClientId=self.client_id
        )
        
        self.body_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=col_shape_id,
            baseVisualShapeIndex=vis_shape_id,
            basePosition=list(position),
            baseOrientation=list(orientation),
            physicsClientId=self.client_id
        )
        
        # Increase friction for robust grasping
        p.changeDynamics(
            self.body_id,
            -1,
            lateralFriction=1.0,
            spinningFriction=0.005,
            rollingFriction=0.005,
            physicsClientId=self.client_id
        )
        
    def reset(self, position, orientation=(0, 0, 0, 1)):
        if self.body_id < 0:
            return
            
        p.resetBasePositionAndOrientation(
            self.body_id,
            list(position),
            list(orientation),
            physicsClientId=self.client_id
        )
        p.resetBaseVelocity(
            self.body_id,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0],
            physicsClientId=self.client_id
        )

    def get_position(self) -> np.ndarray:
        if self.body_id < 0:
            return np.zeros(3, dtype=np.float32)
        pos, _ = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.client_id)
        return np.array(pos, dtype=np.float32)

    def get_velocity(self) -> np.ndarray:
        if self.body_id < 0:
            return np.zeros(3, dtype=np.float32)
        lin_vel, _ = p.getBaseVelocity(self.body_id, physicsClientId=self.client_id)
        return np.array(lin_vel, dtype=np.float32)
