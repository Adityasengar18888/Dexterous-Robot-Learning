"""Reusable PyBullet simulation manager."""

from __future__ import annotations

from typing import Sequence, Tuple
import pybullet as p
import pybullet_data


class PyBulletSimulation:
    """
    Simulation manager for PyBullet environments.

    Handles connection lifecycle (GUI / DIRECT), timestep configuration, gravity,
    ground/table geometry setup, physics stepping, and camera parameters.
    """

    def __init__(
        self,
        gui: bool = False,
        timestep: float = 1.0 / 240.0,
        gravity: Sequence[float] = (0.0, 0.0, -9.81),
        load_ground: bool = True,
        load_table: bool = True,
        table_position: Sequence[float] = (0.5, 0.0, -0.65),
    ) -> None:
        """
        Initialize the simulation manager.

        Args:
            gui: True for OpenGL interactive visualizer, False for headless DIRECT mode.
            timestep: Physics simulation delta time in seconds (default: 240 Hz).
            gravity: 3D gravity vector (default: [0, 0, -9.81]).
            load_ground: Whether to load plane.urdf ground plane.
            load_table: Whether to load a table in front of the robot.
            table_position: Base position for table.urdf.
        """
        self.gui = gui
        self.timestep = float(timestep)
        self.gravity = list(gravity)
        self.load_ground = load_ground
        self.load_table = load_table
        self.table_position = list(table_position)

        self.client_id: int = -1
        self.ground_id: int = -1
        self.table_id: int = -1

    def connect(self) -> int:
        """Connect to the PyBullet physics server and configure physics."""
        if self.client_id >= 0:
            return self.client_id

        connection_mode = p.GUI if self.gui else p.DIRECT
        self.client_id = p.connect(connection_mode)
        if self.client_id < 0:
            raise RuntimeError("Failed to connect to PyBullet physics server.")

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)

        if self.gui:
            # Disable noisy visualizer panels for a clean rendering viewport
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.client_id)
            self.set_camera()

        self.reset()
        return self.client_id

    def disconnect(self) -> None:
        """Disconnect from PyBullet server if connected."""
        if self.client_id >= 0:
            try:
                p.disconnect(physicsClientId=self.client_id)
            except Exception:
                pass
            self.client_id = -1
            self.ground_id = -1
            self.table_id = -1

    def reset(self) -> None:
        """Reset the simulation environment, reapply physics parameters, and reload scene props."""
        if self.client_id < 0:
            self.connect()

        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(self.gravity[0], self.gravity[1], self.gravity[2], physicsClientId=self.client_id)
        p.setTimeStep(self.timestep, physicsClientId=self.client_id)
        p.setPhysicsEngineParameter(numSolverIterations=150, physicsClientId=self.client_id)

        # Load environment scene assets
        if self.load_ground:
            self.ground_id = p.loadURDF("plane.urdf", [0, 0, -0.65], useFixedBase=True, physicsClientId=self.client_id)

        if self.load_table:
            # Table surface sits around z = 0.0 with table_position z = -0.625
            self.table_id = p.loadURDF(
                "table/table.urdf",
                self.table_position,
                useFixedBase=True,
                physicsClientId=self.client_id,
            )

    def step(self, steps: int = 1) -> None:
        """Advance physics simulation by the specified number of steps."""
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.client_id)

    def set_camera(
        self,
        distance: float = 1.35,
        yaw: float = 50.0,
        pitch: float = -32.0,
        target_position: Sequence[float] = (0.5, 0.0, 0.15),
    ) -> None:
        """Configure the debug visualizer camera pose."""
        if self.client_id >= 0 and self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=distance,
                cameraYaw=yaw,
                cameraPitch=pitch,
                cameraTargetPosition=list(target_position),
                physicsClientId=self.client_id,
            )

    def is_connected(self) -> bool:
        """Check if physics client is connected."""
        return self.client_id >= 0

    def __enter__(self) -> PyBulletSimulation:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
