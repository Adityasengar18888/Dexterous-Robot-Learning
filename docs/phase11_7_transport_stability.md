# Phase 11.7 — Transport & Grasp Stability Diagnostic

## 1. Objective
To precisely identify the failure mechanism between successful lifting and final placement in the dense-reward agent, using deterministic evaluation instrumentation.

## 2. Failure Taxonomy (100 Episodes)
- **NO_REACH**: 87
- **REACH_BUT_NO_GRASP**: 2
- **GRASP_BUT_NO_LIFT**: 10
- **LIFT_BUT_DROPPED**: 0
- **LIFT_AND_HELD_BUT_WRONG_DIRECTION**: 1
- **REACHED_TARGET_BUT_NO_PLACE**: 0
- **SUCCESS**: 0

## 3. Grasp Stability Analysis
- Lifts occurred in 1 episodes.
- Median hold duration: 72.0 steps
- Max hold duration: 72 steps
- % surviving >5 steps: 100.0%
- % surviving >25 steps: 100.0%

## 4. Reward Decomposition (Averages)
- **All Episodes**: Reach=-22.0, Transport=-24.6
- **Lifted Episodes**: Reach=-6.9, Lift=72.0, Transport=-41.5

## 5. Diagnostic Conclusion
The dense-reward policy's primary failure point is precisely identified in the taxonomy above.
