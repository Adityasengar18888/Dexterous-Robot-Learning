# Phase 3 Robustness & Domain Randomization Report

**Script:** `scripts/run_phase3_complete.py` (`393.36s` total wall time)  
**Artifact:** `results/phase15_phase3/phase15_phase3_robustness_results.json`  
**Chart:** `results/phase15_master/phase15_robustness_generalization.png`  

## 1. Frozen `PickPlace` Policy (`E_ADAPTIVE_TRANSPORT`, `N=20` per seed per condition)

Evaluated two independent frozen DR4 checkpoints (`DR4_Seed42`, `DR4_Seed43`) across pose randomization (`DR1_POS`), full target+pose randomization (`DR2_POSE_TARGET`), physics randomization (`DR4_FULL_PHYSICS`), `±5 mm` observation noise (`DR4_OBS_NOISE_5MM`), and 1-step observation delay (`DR4_OBS_DELAY_1`).

| Condition | DR4 Seed 42 Succ / Grasp / Lift | DR4 Seed 43 Succ / Grasp / Lift | Combined Mean Success |
|---|---|---|---|
| `DR1_POS` | `100.0% / 100.0% / 70.0%` | `100.0% / 100.0% / 70.0%` | `100.0%` |
| `DR2_POSE_TARGET` | `100.0% / 100.0% / 80.0%` | `90.0% / 95.0% / 75.0%` | `95.0%` |
| `DR4_FULL_PHYSICS` | `95.0% / 100.0% / 50.0%` | `95.0% / 100.0% / 60.0%` | `95.0%` |
| `DR4_OBS_NOISE_5MM` | `100.0% / 100.0% / 60.0%` | `95.0% / 100.0% / 65.0%` | `97.5%` |
| `DR4_OBS_DELAY_1` | `95.0% / 100.0% / 80.0%` | `65.0% / 100.0% / 50.0%` | `80.0%` |

Across all `DR1/DR2/DR4/Noise5mm` evaluations without observation delay (`160` episodes total), `E_ADAPTIVE_TRANSPORT` (`LEARNED_SAC_HER`) achieved `96.875%` success and `99.375%` grasp rate. One-step latency (`DR4_OBS_DELAY_1`) exposes seed variance (`95%` vs `65%`).

## 2. Frozen `SinglePreGrasp` Multi-Seed Generalization (`2 x 20 = 40` eps per condition)

Two disjoint evaluation batches (`batch_A`: seeds `4001–4020`, `batch_B`: seeds `5001–5020`) compared `5B7 bc_warmstart_model.zip` (`BEHAVIOR_CLONING`) and `5B2 checkpoint_17500_steps.zip` (`LEARNED_SAC_HER`):

| Model (Policy Source) | Condition | Mean Grasp ± Std | Mean Lift ± Std | Mean Hold | Table Collision Ep Rate |
|---|---|---|---|---|---|
| `5B7_BC_WARMSTART` (`BEHAVIOR_CLONING`) | `Nominal` | `85.0% ± 5.0%` | `67.5% ± 2.5%` | `67.5%` | `0.0%` |
| `5B7_BC_WARMSTART` (`BEHAVIOR_CLONING`) | `DR1_Pos_Ori` | `75.0% ± 5.0%` | `50.0% ± 5.0%` | `50.0%` | `0.0%` |
| `5B7_BC_WARMSTART` (`BEHAVIOR_CLONING`) | `DR2_TargetZ_Ori` | `75.0% ± 5.0%` | `50.0% ± 0.0%` | `50.0%` | `0.0%` |
| `5B7_BC_WARMSTART` (`BEHAVIOR_CLONING`) | `DR4_Phys_Noise_Delay` | `95.0% ± 0.0%` | `82.5% ± 2.5%` | `80.0%` | `0.0%` |
| `5B2_17500_SAC_HER` (`LEARNED_SAC_HER`) | `Nominal` | `67.5% ± 2.5%` | `7.5% ± 2.5%` | `7.5%` | `32.5%` |
| `5B2_17500_SAC_HER` (`LEARNED_SAC_HER`) | `DR1_Pos_Ori` | `60.0% ± 5.0%` | `2.5% ± 2.5%` | `0.0%` | `30.0%` |
| `5B2_17500_SAC_HER` (`LEARNED_SAC_HER`) | `DR2_TargetZ_Ori` | `52.5% ± 2.5%` | `2.5% ± 2.5%` | `2.5%` | `22.5%` |
| `5B2_17500_SAC_HER` (`LEARNED_SAC_HER`) | `DR4_Phys_Noise_Delay` | `42.5% ± 12.5%` | `7.5% ± 7.5%` | `7.5%` | `40.0%` |

## 3. Conclusion

Phase 3 generalization is **COMPLETE**: frozen `PickPlace` DR4 models maintain `95–100%` success across pose, target, physics, and `±5 mm` sensor noise, while single-object `5B7` BC maintains `75–95%` grasp / `50–82.5%` lift with `0%` table collisions across all perturbations.

