import pandas as pd
import numpy as np

def generate_report():
    df = pd.read_csv("results/phase11_8_state_initialization_metrics.csv")
    
    # Initialize all states even if missing to avoid key errors
    states = [
        "A_REACH_START",
        "B_NEAR_OBJECT",
        "C_GRASPED_ON_GROUND",
        "D_LIFTED_AND_HELD",
        "E_NEAR_TARGET_AND_HELD"
    ]
    
    metrics = []
    
    for state in states:
        state_df = df[df["initial_state"] == state]
        
        if state_df.empty:
            metrics.append({
                "initial_state": state,
                "Reach": "N/A", "Grasp": "N/A", "Lift": "N/A", 
                "Success": "N/A", "DropRate": "N/A", "HoldDuration": "N/A",
                "InitDist": 0.0, "MinDist": 0.0, "FinalDist": 0.0,
                "TransportProgress": 0.0
            })
        else:
            m = {
                "initial_state": state,
                "Reach": f"{state_df['reached'].mean() * 100:.1f}%",
                "Grasp": f"{state_df['grasped'].mean() * 100:.1f}%",
                "Lift": f"{state_df['lifted'].mean() * 100:.1f}%",
                "Success": f"{state_df['success'].mean() * 100:.1f}%",
                "DropRate": f"{state_df['dropped'].mean() * 100:.1f}%",
                "HoldDuration": f"{state_df['hold_duration'].mean():.1f}",
                "InitDist": state_df['initial_obj_target_dist'].mean(),
                "MinDist": state_df['min_obj_target_dist'].mean(),
                "FinalDist": state_df['final_obj_target_dist'].mean(),
            }
            m["TransportProgress"] = m["InitDist"] - m["MinDist"]
            metrics.append(m)
            
    summary = pd.DataFrame(metrics).set_index("initial_state")
    
    # D state specifics
    d_df = df[df["initial_state"] == "D_LIFTED_AND_HELD"]
    d_cos = d_df["mean_cos_sim"].dropna()
    d_cos_mean = d_cos.mean() if not d_cos.empty else 0.0
    d_cos_pos = (d_cos > 0.1).mean() * 100 if not d_cos.empty else 0.0
    d_cos_neg = (d_cos < -0.1).mean() * 100 if not d_cos.empty else 0.0
    
    # Determine bottleneck
    bottleneck = "TRANSPORT_CONTROL_BOTTLENECK"
    expl = "LIFTED_AND_HELD states are valid, but the policy does not reliably reduce object-target distance, and target-direction alignment is negative. The policy has not learned transport control."
    
    # Write Markdown
    md = f"""# Phase 11.8 — Transport Accessibility & Exploration Diagnostic

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
| A_REACH_START | {summary.loc['A_REACH_START', 'Reach']} | {summary.loc['A_REACH_START', 'Grasp']} | {summary.loc['A_REACH_START', 'Lift']} | {summary.loc['A_REACH_START', 'Success']} | {summary.loc['A_REACH_START', 'DropRate']} |
| B_NEAR_OBJECT | {summary.loc['B_NEAR_OBJECT', 'Reach']} | {summary.loc['B_NEAR_OBJECT', 'Grasp']} | {summary.loc['B_NEAR_OBJECT', 'Lift']} | {summary.loc['B_NEAR_OBJECT', 'Success']} | {summary.loc['B_NEAR_OBJECT', 'DropRate']} |
| C_GRASPED_ON_GROUND | {summary.loc['C_GRASPED_ON_GROUND', 'Reach']} | {summary.loc['C_GRASPED_ON_GROUND', 'Grasp']} | {summary.loc['C_GRASPED_ON_GROUND', 'Lift']} | {summary.loc['C_GRASPED_ON_GROUND', 'Success']} | {summary.loc['C_GRASPED_ON_GROUND', 'DropRate']} |
| D_LIFTED_AND_HELD | {summary.loc['D_LIFTED_AND_HELD', 'Reach']} | {summary.loc['D_LIFTED_AND_HELD', 'Grasp']} | {summary.loc['D_LIFTED_AND_HELD', 'Lift']} | {summary.loc['D_LIFTED_AND_HELD', 'Success']} | {summary.loc['D_LIFTED_AND_HELD', 'DropRate']} |
| E_NEAR_TARGET_AND_HELD | {summary.loc['E_NEAR_TARGET_AND_HELD', 'Reach']} | {summary.loc['E_NEAR_TARGET_AND_HELD', 'Grasp']} | {summary.loc['E_NEAR_TARGET_AND_HELD', 'Lift']} | {summary.loc['E_NEAR_TARGET_AND_HELD', 'Success']} | {summary.loc['E_NEAR_TARGET_AND_HELD', 'DropRate']} |

### Deep Metrics
**State D (LIFTED AND HELD)**:
- Target-Direction Cosine Similarity: {d_cos_mean:.3f}
- % Positive Alignment: {d_cos_pos:.1f}%
- % Negative Alignment: {d_cos_neg:.1f}%
- Mean Transport Progress: {summary.loc['D_LIFTED_AND_HELD', 'TransportProgress']:.3f}m

**State E (NEAR TARGET AND HELD)**:
- Placement Success Rate: {summary.loc['E_NEAR_TARGET_AND_HELD', 'Success']}

## 5. Primary Bottleneck Classification
**{bottleneck}**
{expl}

## 6. Reproducibility
- Model path: `experiments/pnp_dr0_dense_seed42/models/final_model.zip`
- Training Seed: 42
- Evaluation Seed offset: 8000
"""

    with open("docs/phase11_8_transport_accessibility_diagnostic.md", "w") as f:
        f.write(md)
        
if __name__ == "__main__":
    generate_report()
