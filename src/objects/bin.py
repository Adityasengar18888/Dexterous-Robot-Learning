"""Target bin object for sorting tasks."""
import pybullet as p
import numpy as np

class SortingBin:
    """A visual zone representing a sorting bin."""
    def __init__(self, client_id: int, size: float = 0.15, color: list = [0.2, 0.8, 0.2, 0.5]):
        self.client_id = client_id
        self.size = size
        self.color = color
        self.body_id = -1
        
    def load(self, position=(0,0,0)):
        half_extents = [self.size / 2.0, self.size / 2.0, 0.001]
        vis_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=self.color,
            physicsClientId=self.client_id
        )
        self.body_id = p.createMultiBody(
            baseMass=0, # static
            baseCollisionShapeIndex=-1, # no collision, just visual zone
            baseVisualShapeIndex=vis_shape_id,
            basePosition=list(position),
            physicsClientId=self.client_id
        )
        
    def reset(self, position):
        if self.body_id < 0:
            return
        p.resetBasePositionAndOrientation(
            self.body_id,
            list(position),
            [0, 0, 0, 1],
            physicsClientId=self.client_id
        )
