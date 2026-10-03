"""Manual demonstration script for Franka Panda robot in PyBullet."""

import argparse
import sys
import time
import numpy as np

from src.utils.simulation import PyBulletSimulation
from src.robots.controller import PandaController


def run_demo(gui: bool = False, sleep_time: float = 0.0) -> bool:
    """
    Execute the Franka Panda kinematic control demonstration.

    Args:
        gui: If True, launches PyBullet with OpenGL GUI visualizer.
        sleep_time: Pause duration between waypoint motions for visual tracking.
    """
    print("=" * 60)
    print("  Franka Panda Robotic Simulation Demo (Phase 1)")
    print("=" * 60)
    print(f"Mode: {'GUI (Interactive)' if gui else 'DIRECT (Headless)'}")

    # 1. Launch PyBullet Simulation
    print("\n[1/5] Launching PyBullet simulation environment...")
    sim = PyBulletSimulation(
        gui=gui,
        timestep=1.0 / 240.0,
        gravity=(0.0, 0.0, -9.81),
        load_ground=True,
        load_table=True,
    )
    sim.connect()

    # 2. Load Franka Panda Robot Controller
    print("[2/5] Loading Franka Emika Panda robot...")
    controller = PandaController(simulation=sim)
    controller.load()

    initial_ee_pos = controller.get_ee_position()
    print(f"      Initial EE Position: [x={initial_ee_pos[0]:.3f}, y={initial_ee_pos[1]:.3f}, z={initial_ee_pos[2]:.3f}]")
    print(f"      Initial Gripper Width: {controller.get_gripper_state()['width'] * 100:.1f} cm")

    # 3. Move End-Effector to several safe Cartesian positions
    print("\n[3/5] Moving End-Effector through safe Cartesian waypoints...")
    waypoints = [
        ("Waypoint 1 - Center Hover", np.array([0.50, 0.00, 0.30])),
        ("Waypoint 2 - Forward Reach", np.array([0.62, 0.00, 0.20])),
        ("Waypoint 3 - Left Offset", np.array([0.55, -0.18, 0.22])),
        ("Waypoint 4 - Right Offset", np.array([0.55, 0.18, 0.22])),
        ("Waypoint 5 - Table Pre-Grasp", np.array([0.50, 0.00, 0.12])),
    ]

    for name, target in waypoints:
        print(f"      -> Commanding {name}: {target}")
        reached = controller.move_ee(target, steps=120)
        err = np.linalg.norm(reached - target)
        print(f"         Reached: [{reached[0]:.3f}, {reached[1]:.3f}, {reached[2]:.3f}] (error: {err * 1000:.1f} mm)")
        if sleep_time > 0:
            time.sleep(sleep_time)

    # 4. Actuate Gripper (Close & Open)
    print("\n[4/5] Actuating Gripper (Close & Open)...")
    print("      -> Closing gripper...")
    closed_width = controller.close_gripper(steps=120)
    print(f"         Closed aperture: {closed_width * 100:.2f} cm (State: {controller.get_gripper_state()})")
    if sleep_time > 0:
        time.sleep(sleep_time)

    print("      -> Opening gripper...")
    open_width = controller.open_gripper(steps=120)
    print(f"         Open aperture: {open_width * 100:.2f} cm (State: {controller.get_gripper_state()})")
    if sleep_time > 0:
        time.sleep(sleep_time)

    # 5. Return to home / Reset and Clean Exit
    print("\n[5/5] Resetting robot to neutral state and exiting cleanly...")
    controller.reset()
    final_ee_pos = controller.get_ee_position()
    print(f"      Neutral EE Position: [{final_ee_pos[0]:.3f}, {final_ee_pos[1]:.3f}, {final_ee_pos[2]:.3f}]")

    sim.disconnect()
    print("\nSimulation disconnected cleanly. Demo completed successfully.")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description="Franka Panda PyBullet Simulation Demo")
    parser.add_argument("--gui", action="store_true", help="Launch in interactive PyBullet GUI mode")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between steps in seconds (useful with --gui)")
    args = parser.parse_args()

    run_demo(gui=args.gui, sleep_time=args.delay)


if __name__ == "__main__":
    main()
