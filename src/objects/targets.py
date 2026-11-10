"""Visual target markers for goal-conditioned tasks."""
import pybullet as p

class VisualTarget:
    """A ghost object that visualizes the desired goal without collision."""
    
    def __init__(self, client_id: int, size: float = 0.05, color=(0.2, 0.8, 0.2, 0.5)):
        self.client_id = client_id
        self.size = size
        self.color = color
        self.body_id = -1
        
    def load(self, position):
        half_extents = [self.size / 2.0] * 3
        # No collision shape for ghost target
        vis_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=list(self.color),
            physicsClientId=self.client_id
        )
        self.body_id = p.createMultiBody(
            baseMass=0, # static
            baseCollisionShapeIndex=-1,
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
