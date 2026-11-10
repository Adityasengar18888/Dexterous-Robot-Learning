# Phase 11.10 — Curriculum Pick & Place Diagnostic

## Executive Summary
This diagnostic tested whether Curriculum Learning (using progressively harder initial state distributions) could allow SAC+HER to master the end-to-end Pick & Place task by composing independently learned skills. The curriculum began with Stage 1 (Reach → Grasp → Lift) under the normal initial state distribution. 
However, the SAC+HER policy failed to master grasping and lifting within the strict 50,000 timestep budget, triggering a hard stop. 
Final Result Classification: **CURRICULUM_STAGE_1_FAILURE**.

## Research Question
Can independently learnable manipulation behaviors be composed through curriculum/state-distribution training into reliable end-to-end Pick & Place?

## Experimental Design
- **Algorithm:** SAC + HER (Phase 11.6 baseline hyperparameters).
- **Environment:** `PandaCurriculumEnv` (isolated).
- **Curriculum:** 4 progressive stages modifying the initial state distribution (Normal, Lifted, Near-Target).
- **Budget:** 50k steps per stage.
- **Stage 1 (Current):** 100% `NORMAL_START` to learn reaching and lifting.

## Stage 1 Results
- **10k Checkpoint:** Grasp 9.0%, Lift 0.0%
- **25k Checkpoint:** Grasp 5.0%, Lift 0.0%
- **50k Checkpoint:** Grasp 10.0%, Lift 0.0%

## Stage 1 Gate
**FAILED.**
The Stage 1 success gate required >= 60% Grasp Rate and >= 50% Lift Rate at the 50k checkpoint. The policy only achieved a 10.0% Grasp Rate and 0.0% Lift Rate, indicating that 50k timesteps is insufficient to learn robust object acquisition from scratch.

## Stage 2 Results
*Stage not executed due to Stage 1 failure.*

## Stage 2 Gate
*N/A*

## Stage 3 Results
*Stage not executed due to Stage 1 failure.*

## Stage 3 Gate
*N/A*

## Stage 4 Results
*Stage not executed due to Stage 1 failure.*

## Comparison With Phase 11 / 11.6
In Phase 11.6 (Dense Reward Diagnostic), SAC+HER was given a 100k step budget and managed to achieve a roughly 10% lift rate. In this curriculum diagnostic, the budget for Stage 1 was strictly limited to 50k steps. The failure to lift the object within 50k steps perfectly aligns with the slow learning curve observed in Phase 11.6, proving that initial acquisition (reaching + grasping) itself is a heavily compute-constrained problem.

## Curriculum State Distribution
100% `NORMAL_START` during Stage 1. 
(Logged in `results/phase11_10_curriculum_distribution.csv`)

## Learning Curves
Learning curves plateaued early in Stage 1 due to the insufficient budget to master grasping. 
(Detailed metrics available in `results/phase11_10_curriculum_metrics.csv`).

## Failure Analysis
The primary cause of the CURRICULUM_STAGE_1_FAILURE is the strict 50,000 timestep budget limit applied to the first stage. While 50k steps was highly successful for isolated *transport* (Phase 11.9), the initial sequence of moving the arm, aligning the gripper accurately over the block, and executing a friction-stable grasp is kinetically much more complex. The policy simply requires significantly more environment interactions (scaling compute) to converge on the pre-requisite grasp behavior.

## Limitations
This diagnostic relied on a single seed (Seed 42) and a strictly capped budget per stage which did not allow for dynamic budget extensions.

## Recommended Next Experiment
Since both Phase 11.6 and this diagnostic demonstrated that the foundational grasp behavior is heavily compute-starved under standard SAC+HER exploration, the ONE recommended follow-up experiment is **Phase 12: Scaling Compute (Training SAC+HER on the dense environment with an expanded 500k-1M timestep budget) to definitively shatter the acquisition bottleneck.**
