# Phase 16 & 17 Failure Analysis & Root-Cause Engineering Log

## 1. Phase 16.1 Hierarchical RL (`PickPlace`, `12k` Steps)
- **What Failed Initially:** Repeatable mid-air hover reward (`+0.35` when `obj_z >= 0.055`) induced an option loop hovering above table (`93.3%` grasp, `0%` final placement at `25k`). Replacing it with potential-difference shaping `(h1 - h0)` yielded clean placement (`30.0%` over 30 unseen episodes vs `0.0%` Flat SAC+HER and `0.0%` untrained step-0).
- **Remaining Gap:** `12k` total CPU steps (`~120` episodes) is insufficient for Discrete SAC over `6` options to match the `500k` Phase 14 policy (`100.0%`); disturbed drop-recovery reached `10.0%` success on 10 episodes (`drop_recovery_triggers=0` because mid-lift drop reset occurred cleanly within option boundaries, re-triggering `GRASP` to `30.32%`).

## 2. Phase 16.2 Vision-Based Manipulation (RGB-D)
- **Initial Failure:** Straight-overhead camera `(0.50, 0.0, 0.72)` suffered severe robot-arm occlusion (`detection_rate = 18.65%`, `loc_err = 29.88 mm`, `0%` PickPlace).
- **Fix & Measured Gain:** Repositioning to an oblique `45°` view `(0.92, 0.28, 0.45)` + viewing-ray half-extent offset jumped detection to `99.9%–100.0%` (`14.81–15.55 mm` 3D error, `0.9983–0.9999` IoU), lifting `SAC_RGBD_Vision` to `60.0%` grasp / `15.0%` success and `HRL_RGBD_Vision` to `25.0%` grasp / `20.0%` success without hidden GT state.
- **Remaining Gap:** Closed-loop `~15 mm` depth centroid jitter plus gripper-finger occlusions during fine placement reduce vision transport from `100%` (GT) to `15–20%`; policies trained *only* on GT state lack visual-noise augmentation during training.

## 3. Phase 16.3 Dynamic Tracking
- **Behavior:** Predictive intercept raised linear moving-cube PickPlace from `95.0% -> 100.0%` (`100%` grasp across all 5 conditions, `95%` sinusoidal, `95%` step-disturbance, `100%` moving target+cube).
- **Telemetry Artifact:** Post-grasp acceleration during robot lift exceeded the `0.022 m` innovation threshold (`1068–1185` replan triggers across 2,000 steps) because the tracker stayed active after grasp pickup rather than masking post-grasp transport.

## 4. Phase 16.4 Contact-Rich Assembly
- **What Improved:** Adding lateral wall-force admittance and active jam retract-wiggle (`CompliantImpedanceAssemblyController`) improved assembly from `50.0% -> 53.33%` and lowered final position error (`0.0532 m -> 0.0468 m`) over 30 seeds.
- **Blocked Subset:** Learned compliant RL remains `BLOCKED_PREREQUISITES_LOW_GRASP_RL` (`Phase 15 5k SAC: 3.33% grasp, 0% assembly`). PyBullet `wall_force` impulses are simulator contact constraints, NOT hardware F/T sensor signals.

## 5. Phase 16.6 Autonomous Multi-Object Execution
- **Outcome:** RGB-D + learned HRL + progress monitor completed Object 1 in `25.0%` of `200`-step episodes (`229` replan triggers, `1049` SafetyMonitor clamps), with `0.0%` full 2-object completion because `obj1` grasping/transport consumed most of the `200`-step budget and `GRASP -> LIFT -> TRANSPORT` transitions had no `500k` fine-tuning.

## 6. Phase 17.1 Cross-Simulator Validation (PyBullet -> MuJoCo 4.0.1)
- **Zero-Shot Transfer Failure Classification:**
  1. **Observation / Frame Height Mismatch (`~9.4 mm` tabletop settling & `0.485 m vs 0.59 m` EE init Z):** PyBullet `PandaPickPlaceEnv` settles cube center to `0.0156 m` while MuJoCo elliptic plane settles cleanly at `0.0250 m`.
  2. **Actuator & Contact Dynamics Mismatch:** Even reaching `3.94 cm` from the cube with `40.0%` bilateral contact (`sac_dr4_seed42`), MuJoCo `kp=400 / 25N` finger PD actuators slipped during rapid lift (`5%` grasp, `0%` place zero-shot). Calibrating MuJoCo friction (`1.5`) and gripper force (`2.0x`) raised bilateral contact to `50.0%`, grasp to `15.0%`, and achieved `5.0%` full PickPlace transfer without retraining.

