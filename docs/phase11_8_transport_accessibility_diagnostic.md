# Phase 11.8 — Transport Accessibility & Exploration Diagnostic

## 1. Objective
To determine whether the existing `pnp_dr0_dense_seed42` policy can perform transport when a valid lifted/grasped state is made accessible, decoupling reaching failures from transport failures.

## 2. Experimental Setup
- Model: Phase 11.6 SAC dense-reward (`experiments/pnp_dr0_dense_seed42/models/final_model.zip`)
- Environment: Isolated `PandaPickPlaceDiagnosticEnv`
- Evaluation: Deterministic inference, 100 episodes per state
- Budget: 0 additional training timesteps

## 3. Physical State Initialization
The diagnostic physically manipulated the simulated robot to acquire the object securely before yielding control to the policy. 

## 4. Initialization Metrics Table

| Initial State | Reach | Grasp | Lift | Success | Drop Rate |
|---|---:|---:|---:|---:|---:|
| A_REACH_START | 11.0% | 6.0% | 0.0% | 0.0% | 0.0% |
| B_NEAR_OBJECT | N/A | N/A | N/A | N/A | N/A |
| C_GRASPED_ON_GROUND | 23.9% | 19.7% | 5.6% | 0.0% | 2.8% |
| D_LIFTED_AND_HELD | 57.1% | 57.1% | 57.1% | 0.0% | 28.6% |
| E_NEAR_TARGET_AND_HELD | 100.0% | 100.0% | 100.0% | 0.0% | 25.0% |

### Deep Metrics
**State D (LIFTED AND HELD)**:
- Target-Direction Cosine Similarity: -0.209
- % Positive Alignment: 28.6%
- % Negative Alignment: 71.4%
- Mean Transport Progress: 0.082m

**State E (NEAR TARGET AND HELD)**:
- Placement Success Rate: 0.0%

## 5. Primary Bottleneck Classification
**TRANSPORT_CONTROL_BOTTLENECK**
LIFTED_AND_HELD states are valid, but the policy does not reliably reduce object-target distance, and target-direction alignment is negative. The policy has not learned transport control.

## 6. Reproducibility
- Model path: `experiments/pnp_dr0_dense_seed42/models/final_model.zip`
- Training Seed: 42
- Evaluation Seed offset: 8000
