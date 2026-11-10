"""Assembly parts (Peg and Receptacle) for Phase 9."""

import pybullet as p
import numpy as np

class Receptacle:
    """A U-shaped or Hollow box base for insertion assembly."""
    def __init__(self, client_id: int):
        self.client_id = client_id
        self.body_id = -1
        
    def load(self):
        # Create a hollow box using 5 collision shapes
        # Dimensions for the inner hole: 0.08 x 0.08, depth: 0.05
        # Total outer dimensions: 0.1 x 0.1 x 0.055
        
        half_extents_bottom = [0.05, 0.05, 0.005]
        half_extents_wall_x = [0.005, 0.05, 0.025]
        half_extents_wall_y = [0.04, 0.005, 0.025]
        
        # Center of mass offsets
        pos_bottom = [0, 0, 0.005]
        pos_left = [-0.045, 0, 0.035]
        pos_right = [0.045, 0, 0.035]
        pos_front = [0, -0.045, 0.035]
        pos_back = [0, 0.045, 0.035]
        
        col_bottom = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents_bottom, physicsClientId=self.client_id)
        col_left = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents_wall_x, physicsClientId=self.client_id)
        col_right = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents_wall_x, physicsClientId=self.client_id)
        col_front = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents_wall_y, physicsClientId=self.client_id)
        col_back = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents_wall_y, physicsClientId=self.client_id)
        
        vis_bottom = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents_bottom, rgbaColor=[0.3, 0.3, 0.8, 1], physicsClientId=self.client_id)
        vis_left = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents_wall_x, rgbaColor=[0.3, 0.3, 0.8, 1], physicsClientId=self.client_id)
        vis_right = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents_wall_x, rgbaColor=[0.3, 0.3, 0.8, 1], physicsClientId=self.client_id)
        vis_front = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents_wall_y, rgbaColor=[0.3, 0.3, 0.8, 1], physicsClientId=self.client_id)
        vis_back = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents_wall_y, rgbaColor=[0.3, 0.3, 0.8, 1], physicsClientId=self.client_id)
        
        self.body_id = p.createMultiBody(
            baseMass=0.0, # Static object
            baseCollisionShapeIndex=col_bottom,
            baseVisualShapeIndex=vis_bottom,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            linkMasses=[0.1, 0.1, 0.1, 0.1],
            linkCollisionShapeIndices=[col_left, col_right, col_front, col_back],
            linkVisualShapeIndices=[vis_left, vis_right, vis_front, vis_back],
            linkPositions=[pos_left, pos_right, pos_front, pos_back],
            linkOrientations=[[0,0,0,1]]*4,
            linkInertialFramePositions=[[0,0,0]]*4,
            linkInertialFrameOrientations=[[0,0,0,1]]*4,
            linkParentIndices=[0, 0, 0, 0],
            linkJointTypes=[p.JOINT_FIXED]*4,
            linkJointAxis=[[0,0,1]]*4,
            physicsClientId=self.client_id
        )

    def reset(self, position: np.ndarray, orientation: np.ndarray):
        p.resetBasePositionAndOrientation(self.body_id, position, orientation, physicsClientId=self.client_id)

    def get_position(self) -> np.ndarray:
        pos, _ = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.client_id)
        return np.array(pos, dtype=np.float32)

    def get_orientation(self) -> np.ndarray:
        _, ori = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.client_id)
        return np.array(ori, dtype=np.float32)


class Peg:
    """A rectangular peg that fits tightly into the receptacle."""
    def __init__(self, client_id: int):
        self.client_id = client_id
        self.body_id = -1
        
    def load(self):
        # The hole is 0.08 x 0.08. We make the peg 0.06 x 0.06 x 0.06
        half_extents = [0.03, 0.03, 0.03]
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client_id)
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.8, 0.8, 0.2, 1], physicsClientId=self.client_id)
        
        self.body_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=[0, 0, 0.03],
            physicsClientId=self.client_id
        )
        
        # Increase friction for better grasping
        p.changeDynamics(self.body_id, -1, lateralFriction=1.0, spinningFriction=0.01, physicsClientId=self.client_id)

    def reset(self, position: np.ndarray, orientation: np.ndarray):
        p.resetBasePositionAndOrientation(self.body_id, position, orientation, physicsClientId=self.client_id)
        p.resetBaseVelocity(self.body_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client_id)

    def get_position(self) -> np.ndarray:
        pos, _ = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.client_id)
        return np.array(pos, dtype=np.float32)

    def get_orientation(self) -> np.ndarray:
        _, ori = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.client_id)
        return np.array(ori, dtype=np.float32)
        
    def get_velocity(self) -> np.ndarray:
        lin_vel, _ = p.getBaseVelocity(self.body_id, physicsClientId=self.client_id)
        return np.array(lin_vel, dtype=np.float32)
