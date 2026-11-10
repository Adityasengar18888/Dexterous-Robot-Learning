# Phase 15.2.4 — Reward Comparison (`PandaMultiSortDenseRewardEnv` vs `PandaMultiSortDenseV2Env`)

**Versioned Environment:** [`PandaMultiSortDenseV2Env`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/environments/dense_multi_sort_env.py) (`MultiSortProximityRewardConfig`)  
**Config:** [`configs/multi_sort_phase15_4.yaml`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/configs/multi_sort_phase15_4.yaml)

---

## 1. Reward Formulation Comparison

| Component | Phase 15.2.3 (`MultiSortRewardConfig`, `V1`) | Phase 15.2.4 (`MultiSortProximityRewardConfig`, `V2`) |
| :--- | :---: | :---: |
| **Stage A — Reach Progress** | `4.0 * clip(d_prev - d_curr, -0.05, 0.05) * (~bilat) * active_mask` | Retained unchanged (`w_reach = 4.0`) |
| **Stage A — Continuous Proximity** | `0.0` (`w_proximity = 0.0`) | **`-0.25 * clip(d_curr, 0.0, 1.0) * (~bilat) * (~both_curr) * active_mask`** |
| **Stage A — Reach Total** | `R_progress` | `R_progress + R_proximity` |
| **Stage B — Verified Grasp** | `1.0 * Δ(bilat_f9_f10 >= 5.0N) * active_mask` | Retained unchanged (`w_grasp = 1.0`) |
| **Stage C — Bounded Lift** | `1.5 * Δh * outside_bin_xy * held_any * active_mask` | Retained unchanged (`w_lift = 1.5`) |
| **Stage D — Transport** | `6.0 * clip(g_prev - g_curr, -0.05, 0.05) * active_mask` | Retained unchanged (`w_transport = 6.0`) |
| **Wrong-Bin Deterrence** | `-2.0 * I[entered wrong bin k]` | Retained unchanged (`w_wrong_bin = 2.0`) |
| **Stage E — Placement / Disturb** | `+4.0 * newly_placed - 4.5 * newly_disturbed` | Retained unchanged |
| **Stage F — Completion** | `+8.0 * (newly_both - lost_both)` | Retained unchanged (`w_complete = 8.0`) |
| **Step Penalty** | `-0.02 * (1 - both_curr)` | Retained unchanged (`step_penalty = -0.02`) |

---

## 2. Empirical Magnitude Verification (Pre-Training Audit on Seed 1001)

| Regime (Seed 1001) | `V1` (`w_proximity = 0.0`) | `V2` (`w_proximity = 0.25`) | Magnitude Assessment |
| :--- | :---: | :---: | :--- |
| **Idle (`300` zero-action steps)** | `total = -6.000` (`r_step = -6.0`) | `total = -45.076` (`r_prox = -39.009`, `r_step = -6.0`) | Strong negative baseline vs lingering far from cube |
| **Approaching near cube (`d ≈ 0.02 m`, ungrasped)** | `~ -5.50` (`+0.50` vs idle) | `~ -6.18` (`+38.89` vs idle) | Continuous smooth gradient even before grasp contact |
| **Scripted Full Two-Object Sort (`76` steps, `20/20`)** | `total = +25.448` (`r_reach = +3.167`, `r_lift = +3.0`, `r_transport = +4.861`, `r_place = +8.0`, `r_complete = +8.0`, `r_step = -1.58`) | **`total = +23.710`** (`r_reach_prog = +3.167`, **`r_prox = -1.738`**, `r_lift = +3.0`, `r_transport = +4.861`, `r_place = +8.0`, `r_complete = +8.0`, `r_step = -1.58`) | **Balanced**: `+23.71` vs `-45.08` (`+68.79` span); `-1.738` proximity cost is minor compared to `+22.0` transport/lift/place/complete rewards |

**Pre-Training Dominance Gate:** Passed (`r_proximity` per step $\in [-0.25, 0.0]$, vanishes identically once `bilateral_contact` is acquired, switches cleanly to Object 2 after verified Object 1 release, and yields `+0` after full completion).

---

## 3. Pre-Training Verification Summary

- `pytest tests/test_dense_multi_sort_env.py tests/test_multi_sort_env.py tests/test_phase15_wrappers.py -v`: **`18 passed in 25.80s`** (includes V2 multi-distance monotonicity, grasp gating, Object 1 -> Object 2 switching, and `HerReplayBuffer(future, n=4, copy_info_dict=True)` virtual reward sampling).
- Scripted two-object benchmark on `PandaMultiSortDenseV2Env`: **`20 / 20 (100.0%)`**, `mean_steps = 76.0`.

