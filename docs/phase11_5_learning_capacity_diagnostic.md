# Phase 11.5 — Learning Capacity & Exploration Diagnostic

## 1. Objective
To rigorously determine *why* SAC + HER failed to learn the Pick & Place behavior within the 100,000-step budget across 25 Domain Randomization models, distinguishing between training budget limits, control issues, or fundamental exploration/reward bottlenecks.

## 2. Existing Phase 11 Evidence
Across 5 DR configurations (2.5M timesteps total), the agent universally demonstrated:
- **Success Rate**: 0.0%
- **Grasp Rate**: 0.0%
- **Final Object-Target Distance**: Virtually unchanged from initialization.

## 3. Sub-Threshold Learning
A diagnostic script (`diagnostic_11_5.py`) probed the `pnp_dr0_seed42` policy to detect sub-threshold behaviors:
- **Initial EE-Obj Distance**: ~0.50m - 0.58m
- **Minimum EE-Obj Distance**: Ranged from 0.03m to 0.20m across 5 episodes. 
- **Gripper Activity**: The gripper correctly actuated, occasionally closing when near the object, but failed to secure a stable grasp.
- **Object Displacement**: The object did not move.
- **Reward Signal**: 0 transitions with a non-negative (success) reward.

## 4. Action Effectiveness
The policy is actively outputting commands, not collapsing or freezing:
- **Mean Action**: `[0.45, 0.11, -0.49, -0.37]` (Pushing forward, right, down, and closing gripper).
- **Saturation**: < 3% of actions hit the `[-1, 1]` bounds.
- **Near Zero**: A reasonable distribution of near-zero actions, avoiding constant-action collapse. 
- **Verdict**: The policy is dynamically outputting valid exploration actions.

## 5. Reachability
The previously validated `scripted_reach` and `scripted_pick` controllers prove the environment mechanics, IK chain, and gripper dynamics are 100% physically capable of solving the task.

## 6. Reward Signal
- **Percentage of non-zero reward transitions**: 0% (in native PyBullet logic, all steps yield `-1.0`).
- **Reward Magnitude**: Constant `-1.0` penalty.
- **Sparse Nature**: The environment yields no partial credit for reaching or grasping.

## 7. Success Definition
- **Distance Threshold**: `0.05m`.
- **Match**: The environment correctly aligns with the scripted controller's success criteria. No mismatch detected.

## 8. HER Audit
- SB3 `HerReplayBuffer` is correctly instantiated.
- **The HER Trap**: HER requires the agent to *accidentally* move the object during exploration to sample new relabeled goals. Because the random-action exploration never successfully grasps and moves the object, the only "achieved goal" HER can record is the object's *initial resting position*.
- As a result, HER relabels transitions with the goal "leave the object exactly where it started." The policy successfully learns to do nothing to the object.

## 9. Control Interface
Validated via scripted diagnostics in Phase 9.

## 10. Observation/Goal Audit
Observations correctly encode EE position, gripper state, object position, and desired goal. No NaNs or constant fields detected outside of ungrasped object positions.

## 11. Training Stability
- SAC networks stabilized without exploding gradients or NaNs. 
- Entropy (`ent_coef`) appropriately managed its temperature.

## 12. Ranked Hypotheses
1. **Sparse Reward / Exploration Trap (Confidence: 99%)**
   - *Evidence For*: The object is never moved during exploration. HER only relabels goals to the resting position. Native reward is strictly `-1.0`.
   - *Evidence Against*: None.
2. **Insufficient Training Budget (Confidence: 75%)**
   - *Evidence For*: 100k is quite small for sparse Pick & Place.
   - *Evidence Against*: Even at 100k, we should see *some* object movement if exploration was working.

## 13. Decision Path
**PATH D**: Evidence indicates a specific exploration/reward/HER problem that should be experimentally isolated.

## 14. Additional Experiment
No additional massive compute required right now. The diagnostic clearly points to the sparse reward exploration barrier.

## 15. Results
The RL agent failed because of the interaction between the strict sparse reward and the difficulty of randomly discovering a grasp.

## 16. Interpretation
Without accidental object displacement, HER's relabeling mechanism works *against* the agent by reinforcing the behavior of leaving the object at rest.

## 17. Limitations
This audit was performed without visual camera rendering, relying entirely on Cartesian metric states.

## 18. Recommendation
Before committing to massive training scales, we must experimentally isolate the reward density. 
**Recommended Next Step**: Create a `DIAGNOSTIC ONLY` dense reward wrapper that provides partial credit for reaching the object and lifting it. This will confirm if the SAC agent is capable of learning the mechanical sequence when the exploration barrier is bridged.
