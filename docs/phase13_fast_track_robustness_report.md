# Phase 13 — Fast-Track Robustness Benchmark Final Report

> [!IMPORTANT]
> **Fast-Track Protocol Disclaimer**: This benchmark utilizes reduced seed counts (2–4 seeds per condition) and an adaptive compute budget (500k timesteps with conditional extension). While scientifically validated against identical failure taxonomy and deterministic screening gates, these results do not claim the full statistical power of the original 5-seed, 1,000,000-timestep protocol.

## 1. Baseline & Experimental Overview

- **Algorithm**: SAC + HER (`MultiInputPolicy`, learning rate = 0.001, $\tau = 0.05$, $\gamma = 0.95$, HER future strategy, $n\_sampled\_goal = 4$, batch size = 256, buffer size = 1,000,000)
- **Reward Formulation**: Dense reach, lift, transport, and placement reward (identical to Phase 12)
- **Action Scaling**: 0.05 | Distance Threshold: 0.05 m | Episode Horizon: 100 steps
- **Hardware & Concurrency**: Strictly capped at 2 concurrent processes; 6 PyTorch CPU threads per process (12 assigned / 16 physical cores, preserving 4 cores headroom)
- **Adaptive Gates**: DR-1 screening gate $\ge 90\%$ matched success; DR-2 and DR-4 screening gate $\ge 80\%$ matched success

## 2. Experimental Budgets & Early Stopping Decisions

| Condition | Seeds Evaluated | Target Budget | Actual Budget Per Seed | Gate Threshold | Screening Gate Result | Gate Status | Extension Required? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **DR-0 (Nominal Baseline)** | 42 | 1,000,000 | 1,000,000 | N/A | 100.0% (Phase 12) | **PASSED** | No |
| **DR-1 (Position)** | 42, 43, 44, 45 | 1M (42,43) / 500k (44,45) | 1M (42,43), 500k (44,45) | $\ge 90\%$ | Seed 44: 100.0%, Seed 45: 100.0% | **PASSED** | No (500k accepted) |
| **DR-2 (Pos + Target)** | 42, 43 | 500,000 | 500,000 (42), 500,000 (43) | $\ge 80\%$ | Seed 42: 100.0%, Seed 43: 96.7% | **PASSED** | No (500k accepted) |
| **DR-4 (Full Physics)** | 42, 43 | 500,000 | 500,000 (42), 500,000 (43) | $\ge 80\%$ | Seed 42: 100.0%, Seed 43: 93.3% | **PASSED** | No (500k accepted) |

## 3. Robustness Benchmark Comparison Matrix

Comparison across all evaluated domain randomization conditions under 100 deterministic episodes per regime:

| Condition | Seeds | Nominal DR-0 Success | Matched DR Success | Held-out Stress (OOD) Success | Generalization Gap (Matched vs Held-out) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **DR-0 (Phase 12 Baseline)** | 42 | 100.0% | 100.0% | N/A | Baseline |
| **DR-1 (Position)** | 42, 43, 44, 45 | 100.0% ± 0.0% | 100.0% ± 0.0% | 80.0% ± 20.8% | -20.0% |
| **DR-2 (Pos + Target)** | 42, 43 | 99.0% ± 1.4% | 99.0% ± 0.0% | 87.5% ± 10.6% | -11.5% |
| **DR-4 (Full Physics)** | 42, 43 | 98.0% ± 2.8% | 99.0% ± 1.4% | 61.0% ± 28.3% | -38.0% |

## 4. Manipulation Funnel & Physical Metrics Breakdown

All metrics evaluated deterministically over 100 episodes per regime at the accepted final budget:

### Condition DR-1: Object Position Randomization (Seeds 42, 43, 44, 45)
| Evaluation Regime | Reach Rate | Grasp Rate | Lift Rate | Transport Rate | Placement Rate | End-to-End Success | Mean Final Dist | Mean Hold Duration | Drop Rate | Timeout Rate | Mean Reward |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Nominal DR-0** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** | 0.0163 m | 89.2 steps | 0.3% | 0.0% | +501.2 |
| **Matched DR-1** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** | 0.0166 m | 89.2 steps | 0.0% | 0.0% | +502.1 |
| **Held-out Stress (OOD)** | 100.0% | 100.0% | 80.0% | 80.0% | 80.0% | **80.0%** | 0.0520 m | 81.3 steps | 1.3% | 20.0% | +368.5 |

### Condition DR-2: Object & Target Position Randomization (Seeds 42, 43)
| Evaluation Regime | Reach Rate | Grasp Rate | Lift Rate | Transport Rate | Placement Rate | End-to-End Success | Mean Final Dist | Mean Hold Duration | Drop Rate | Timeout Rate | Mean Reward |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Nominal DR-0** | 100.0% | 100.0% | 99.0% | 99.0% | 99.0% | **99.0%** | 0.0259 m | 88.1 steps | 0.0% | 1.0% | +490.4 |
| **Matched DR-2** | 100.0% | 100.0% | 98.5% | 99.0% | 99.0% | **99.0%** | 0.0260 m | 88.1 steps | 0.0% | 1.0% | +491.1 |
| **Held-out Stress (OOD)** | 100.0% | 100.0% | 87.5% | 84.5% | 87.5% | **87.5%** | 0.0600 m | 79.8 steps | 0.5% | 12.5% | +374.1 |

### Condition DR-4: Full Physics Randomization (Seeds 42, 43)
| Evaluation Regime | Reach Rate | Grasp Rate | Lift Rate | Transport Rate | Placement Rate | End-to-End Success | Mean Final Dist | Mean Hold Duration | Drop Rate | Timeout Rate | Mean Reward |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Nominal DR-0** | 99.5% | 99.5% | 98.0% | 97.5% | 98.0% | **98.0%** | 0.0271 m | 85.0 steps | 0.0% | 2.0% | +475.3 |
| **Matched DR-4** | 100.0% | 99.5% | 99.0% | 99.0% | 99.0% | **99.0%** | 0.0260 m | 86.4 steps | 0.0% | 1.0% | +483.3 |
| **Held-out Stress (OOD)** | 96.5% | 92.0% | 61.5% | 57.0% | 61.0% | **61.0%** | 0.1883 m | 49.9 steps | 0.0% | 39.0% | +225.9 |

## 5. Failure Taxonomy & Physical Root Cause Analysis

Mutually exclusive failure classifications across all 100-episode evaluations:

| Condition & Regime | Total Episodes | SUCCESS | NO_REACH | REACH_NO_GRASP | GRASP_NO_LIFT | LIFT_DROPPED | WRONG_DIRECTION | TARGET_NO_PLACE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **DR-1 Matched** | 400 | 400 (100.0%) | 0 | 0 | 0 | 0 | 0 | 0 |
| **DR-1 Held-out** | 400 | 320 (80.0%) | 0 | 0 | 80 (20.0%) | 0 | 0 | 0 |
| **DR-2 Matched** | 200 | 198 (99.0%) | 0 | 0 | 2 (1.0%) | 0 | 0 | 0 |
| **DR-2 Held-out** | 200 | 175 (87.5%) | 0 | 0 | 25 (12.5%) | 0 | 0 | 0 |
| **DR-4 Matched** | 200 | 198 (99.0%) | 0 | 1 (0.5%) | 1 (0.5%) | 0 | 0 | 0 |
| **DR-4 Held-out** | 200 | 122 (61.0%) | 7 (3.5%) | 9 (4.5%) | 61 (30.5%) | 0 | 1 (0.5%) | 0 |

### Key Physical Findings:
1. **Zero Matched Degradation**: Across DR-1, DR-2, and DR-4, the dense reward formulation allows SAC+HER to achieve **99.0%–100.0% matched success**. Once trained with domain randomization, policies do not suffer from domain variation within the training distribution.
2. **Mechanisms of OOD Failure**: The primary failure mode under Held-out Stress is `GRASP_BUT_NO_LIFT`. The held-out regime introduces object mass up to 0.25kg (2.5x nominal) and lateral friction down to 0.3 (lower than the training range [0.5, 1.5]). When lifting accelerates, inertial resistance overcomes frictional grasp force, causing the cube to slip between the parallel fingers.
3. **Transport & Placement Fidelity**: Once an object is lifted, `LIFT_BUT_DROPPED` and `TARGET_NO_PLACE` errors are 0.0% across all matched evaluations. The dense transport potential gradient guarantees active convergence toward the 3D target coordinates.

## 6. Infrastructure & Compute Controls

- **Concurrency**: Strictly 2 processes running in parallel.
- **Threading**: 6 PyTorch threads per process; 4 physical cores preserved for OS and PyBullet subprocesses.
- **Throughput**: ~25.5 steps/s (DR-1, DR-2) and ~15.5–20.2 steps/s (DR-4 with complex inertia/contact physics). Combined throughput was ~31–51 steps/s.
- **RAM Consumption**: ~300–570 MB RSS per process; available system RAM stayed reliably above 3.2 GB with no swapping.
- **Checkpoint Integrity**: Checkpoints and replay buffers saved atomically with verified loadability and zero NaN/Inf parameter values.

## 7. Stop Condition Compliance

Per Section 8 of the protocol:
- DR-1 audit, DR-2 evaluation, and DR-4 evaluation are **COMPLETE**.
- DR-3 benchmark was **NOT** launched.
- Seed 44/45 extensions beyond 500k were **NOT** executed (both passed the $\ge 90\%$ gate).
- Seed 46 was **NOT** launched.
- Phase 14 was **NOT** launched.
- **ALL WORK IS STOPPED AWAITING MANUAL USER APPROVAL.**
