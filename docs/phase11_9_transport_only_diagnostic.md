# Phase 11.9 — Transport-Only SAC+HER Diagnostic

## 1. Executive Summary
This diagnostic isolated the continuous-control transport stage of the Pick & Place task. The environment strictly initialized the object already physically grasped and lifted in the air, removing the reaching and grasping bottlenecks entirely. 
After exactly 50,000 steps of isolated training, the SAC+HER policy resulted in a classification of **TRANSPORT_LEARNED**.

## 2. Experimental Configuration
- Model: Fresh SAC instance (`experiments/pnp_transport_seed42/`)
- Environment: Isolated `PandaTransportDiagnosticEnv`
- Timesteps: 50,000
- Seed: 42

## 3. Initial-State Validation
- Initialization was physically validated via pybullet stepping.
- Success rate of initialization > 95%.

## 4. Reward Validation
- Strict `-distance(object, target)` reward was used.
- Tested deterministic geometries proved moving toward target increases reward, and moving away decreases reward.

## 5. Learning Curve

| Timesteps | Success | Mean Final Distance | Mean Min Distance | >25% Progress | Alignment |
|---:|---:|---:|---:|---:|---:|
| 5000 | 1.0% | 0.188 | 0.113 | 79.0% | 50.6% |
| 10000 | 6.0% | 0.122 | 0.052 | 97.0% | 5.2% |
| 20000 | 98.0% | 0.021 | 0.010 | 100.0% | 48.0% |
| 30000 | 97.0% | 0.013 | 0.005 | 100.0% | 74.0% |
| 40000 | 100.0% | 0.012 | 0.004 | 100.0% | 90.0% |
| 50000 | 100.0% | 0.009 | 0.003 | 100.0% | 80.0% |

## 6. Decision Classification
**TRANSPORT_LEARNED**

## 7. Recommended Next Experiment
Based on the transport classification, the recommended next step is to analyze how the learning capacity changes when scaling compute or when chaining the behaviors together via curriculum.
