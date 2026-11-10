# Reach Task Benchmark

## Environment Definition
The **Reach** task serves as the foundational benchmark for the goal-conditioned robotic manipulation platform. The agent controls a 7-DOF Franka Panda arm using end-effector Cartesian delta actions to reach a spatially generated target.

## Coordinate System & Workspace Limits
- **Origin (0,0,0)**: Robot base link.
- **X-axis**: Forward direction (towards the table).
- **Y-axis**: Left/right direction.
- **Z-axis**: Up/down direction (table surface is at `Z = 0.0`).

### Reachable Workspace
Target generation and kinematics clamping share the following bounds to guarantee safety:
- **Min Bounds**: `[0.2, -0.3, 0.05]` (x, y, z)
- **Max Bounds**: `[0.7, 0.3, 0.4]` (x, y, z)
- *Note:* The minimum `Z` is `0.05m`, guaranteeing the target never collides with the tabletop.

## Reward Definition
A **Sparse Binary Reward** is enforced:
- **Reward**: `0.0` if `distance(achieved_goal, desired_goal) <= distance_threshold`.
- **Reward**: `-1.0` otherwise.
- **No reward shaping** is introduced to retain the strict HER formulation.

## Success Definition
An episode is considered successful as soon as the achieved goal (end-effector Cartesian position) enters the Euclidean distance threshold around the desired goal.
- **Threshold (`distance_threshold`)**: Configurable, defaults to `0.05` meters.
- **Diagnostic Metrics**: `distance` (L2 norm), `is_success` (Boolean evaluation), and `episode_length` metrics are returned inside the environment `info` dictionary.

## Configuration
- `max_episode_steps`: Default 50. Ensures agents do not drift indefinitely.
- Deterministic seeding supported via standard Gymnasium `env.reset(seed=X)` semantics.
