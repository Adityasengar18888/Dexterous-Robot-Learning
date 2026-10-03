# Experiment Registry

This document records all structured experiment configurations, execution dates, and validation runs.

| Experiment ID | Phase | Task | Algorithm | Seeds | Status | Notes |
|---|---|---|---|---|---|---|
| `EXP-P0-001` | Phase 1 | Kinematic Validation | Direct IK / Motor | N/A | **COMPLETED** | Verified Franka Panda kinematics, IK convergence (<10mm), Cartesian delta control, gripper limits, and determinism. |
| `EXP-P1-REACH` | Phase 2 & 3 | Reach | SAC + HER | 42, 101, 202 | *Pending Phase 2 Approval* | Goal-conditioned reach baseline under sparse reward. |
