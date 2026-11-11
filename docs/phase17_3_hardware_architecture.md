# Phase 17.3 — Optional Hardware Integration Architecture *(Simulation Interface Design Only)*

**Status:** Physical execution **`NOT_ATTEMPTED`** (simulation-only repository; no physical Franka Panda robot connected).  
**Simulation Interface Contract Module:** [`src/deployment/hardware_interface.py`](../src/deployment/hardware_interface.py)  

## 1. Safety Disclaimer

No real robot commands may be issued unless:
1. Physical hardware availability flag and hardware safety interlock are explicitly verified.
2. Workspace bounds, velocity caps, force torque limits (`40.0 N`), and operator E-stop hardware latch are validated.
3. All `StubLiveFrankaDriver` calls refuse live actuation by throwing `HardwareUnavailableError`.

## 2. ROS 2 Interface Specification (`ROS2TopicContract`)

| Logical Stream | ROS 2 Topic / Service | Message / Action Type | Nominal Rate |
|---|---|---|---|
| Robot State Telemetry | `/franka_robot_state_broadcaster/robot_state` | `franka_msgs/msg/FrankaRobotState` | `1000 Hz` internal / `20 Hz` policy |
| Cartesian Delta / Vel Cmd | `/cartesian_velocity_controller/command` | `geometry_msgs/msg/TwistStamped` | `20 Hz` |
| Gripper Grasp Action | `/panda_gripper/grasp` | `franka_msgs/action/Grasp` | Event / skill-triggered |
| RGB Color Image | `/camera/color/image_raw` | `sensor_msgs/msg/Image` | `30 Hz` |
| Aligned Depth Image | `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/msg/Image` | `30 Hz` |
| Camera Calibration | `/camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | Static / `30 Hz` |
| Safety E-Stop Service | `/safety_supervisor/emergency_stop` | Custom trigger / `std_srvs/srv/Trigger` | Synchronous `< 1 ms` |

## 3. Policy Inference <-> SafetySupervisor Dataflow

1. **`CameraStreamInterface.read_rgbd()`** supplies synchronized color, metric depth, and camera intrinsics `K` into `VisionObjectLocalizer`.
2. **`FrankaDriverInterface.read_state()`** snapshots `FrankaState(q, dq, ee_pos, ee_quat, gripper_width, ext_wrench, timestamp_s, robot_mode_ok)`.
3. **`DeploymentRuntime` (`src/deployment/runtime.py`)** runs frozen inference (`PolicyInferenceInterface`) without importing training pipelines.
4. **`SafetySupervisor.validate_command()`** enforces:
   - Non-latched E-stop check (`operator_ack` required to clear)
   - State freshness check (`age <= 0.15 s`)
   - Finite telemetry & command verification
   - External wrench threshold (`<= 40.0 N`)
   - Per-step translation cap (`<= 0.03 m`) and workspace clipping (`[0.25..0.72, -0.30..0.30, 0.015..0.48]`).

