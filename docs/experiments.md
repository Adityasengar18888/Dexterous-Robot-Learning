# Experiment Registry

This document records all structured experiment configurations, execution dates, and validation runs.

| Experiment ID | Phase | Task | Algorithm | Seeds | Status | Notes |
|---|---|---|---|---|---|---|
| `EXP-P0-001` | Phase 1 | Kinematic Validation | Direct IK / Motor | N/A | **COMPLETED** | Verified Franka Panda kinematics, IK convergence (<10mm), Cartesian delta control, gripper limits, and determinism. |
| `EXP-P1-REACH` | Phase 2 & 3 | Reach | SAC + HER | 42, 101, 202 | *Pending Phase 2 Approval* | Goal-conditioned reach baseline under sparse reward. |
| `EXP-P6-PICKPLACE` | Phase 6 | Pick & Place | SAC vs SAC+HER | 10 | **SMOKE COMPLETED** | Compared E1 (no HER) and E2 (HER). Verified HER replay logic via actor loss convergence. Full runs pending. |
| `EXP-P7-SORT` | Phase 7 | Sorting | SAC + HER | 10 | **SMOKE COMPLETED** | Single-object dynamic bin sorting with goal-conditioned target generation. Model compiles and trains successfully. |
| `EXP-P8-MULTI_SORT` | Phase 8 | Multi-Sorting | SAC + HER | 20 | **SMOKE COMPLETED** | Multi-object tracking (configurably scaled, tested with N=2). Custom target generation mapped precisely per object without environment hard-coding. |
| `EXP-P9-ASSEMBLY` | Phase 9 | Assembly | SAC + HER | 30 | **SMOKE COMPLETED** | Orientation-aware 5D action space peg insertion. Tracked mathematically robust quaternion error and explicit state telemetry (NOT_ASSEMBLED vs ASSEMBLED). |
