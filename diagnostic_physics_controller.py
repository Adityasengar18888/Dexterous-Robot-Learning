import numpy as np
import pybullet as p
from src.utils.simulation import PyBulletSimulation
from src.robots.controller import PandaController
import time

def test_physics_and_controller():
    sim = PyBulletSimulation(gui=False)
    sim.connect()
    
    # Check physics state
    
    controller = PandaController(simulation=sim)
    controller.load()
    controller.reset()
    
    # Test valid Cartesian targets
    targets = [
        [0.4, 0.0, 0.4],
        [0.5, 0.2, 0.2],
        [0.3, -0.2, 0.1],
        [0.6, 0.0, 0.5]
    ]
    
    max_err = 0.0
    errs = []
    
    for pos in targets:
        try:
            actual = controller.move_ee(position=pos, steps=200, tolerance=0.005)
            err = np.linalg.norm(np.array(pos) - actual)
            errs.append(err)
            max_err = max(max_err, err)
            print(f"Target: {pos} | Actual: {actual.round(4)} | Error: {err:.4f}")
            
            joints = controller.get_joint_positions()
            if not np.all(np.isfinite(joints)):
                print("WARNING: Non-finite joints detected!")
                
        except Exception as e:
            print(f"Error moving to {pos}: {e}")

    print(f"Mean position error: {np.mean(errs):.4f}m")
    print(f"Max position error: {max_err:.4f}m")
    
    # Test Gripper
    print("Testing Gripper...")
    w_open = controller.open_gripper()
    print(f"Opened width: {w_open:.4f}")
    w_close = controller.close_gripper()
    print(f"Closed width: {w_close:.4f}")
    
    sim.disconnect()

if __name__ == "__main__":
    test_physics_and_controller()
