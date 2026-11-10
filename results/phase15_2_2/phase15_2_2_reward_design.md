# Phase 15.2.2 — Dense Multi-Object Sorting Reward Design & HER Specification

**Implementation:** [`src/environments/dense_multi_sort_env.py`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/environments/dense_multi_sort_env.py)  
**Class:** `PandaMultiSortDenseRewardEnv(PandaMultiSortEnv)`  
**Preservation:** Original [`PandaMultiSortEnv`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/environments/multi_sort_env.py) sparse reward, physics, 31D obs, 6D goals, 4D actions, and Phase 14 controllers remain completely unmodified.

---

## 1. Release-Verification Correction

Empirically measured Franka Panda gripper widths on the `0.050 m` cube:
- Empty closed (`action[3] = -1.0`): `0.00006–0.0044 m`
- Pinching `0.050 m` cube (`action[3] = -1.0`): **`0.04988–0.0506 m`**
- Opening step 1 (`action[3] = +1.0`): `0.05211 m`
- Opening step 2+: **`0.0766–0.0800 m`**

Therefore, `PandaMultiSortEnv`'s `width > 0.04 m` check triggers prematurely while holding the cube closed. `PandaMultiSortDenseRewardEnv` adopts:
1. **`release_width_threshold = 0.065 m`** (midpoint between pinched `0.0506 m` and open `0.080 m`).
2. **Per-Object Release Verification**: While picking Object 2 later, the robot closes its gripper (`gw <= 0.065 m`), which must not falsely un-verify an unattended Object 1 resting inside Bin 1. Hence object `k ∈ {0, 1}` is verified free/released iff:
   $$\text{obj\_released}_k = (\text{gw} > 0.065) \lor \Big(\neg \text{bilateral}_k \land \|\mathbf{p}_{\text{ee}} - \mathbf{p}_k\|_2 > 0.065\Big)$$
3. **Single & Batched `_get_info` Parity**: Placement of object `k` requires `dist_k <= 0.05 m`, `pos_k[2] < 0.050 m`, and `obj_released_k`. Batched `_get_info(ag, dg)` enforces `(distances <= 0.05) & (ag[..., 2] < 0.05)` (plus `gripper_widths > 0.065` when supplied), rejecting mid-air hovers (`z = 0.06 m`).

---

## 2. Sequential Sorting State Machine (`SortingStage`)

Transitions follow the 16-state progression:

```text
IDLE -> SELECT_OBJECT_1 -> REACH_OBJECT_1 -> GRASP_OBJECT_1 -> LIFT_OBJECT_1 -> TRANSPORT_OBJECT_1 -> PLACE_OBJECT_1 -> VERIFY_OBJECT_1
     -> SELECT_OBJECT_2 -> REACH_OBJECT_2 -> GRASP_OBJECT_2 -> LIFT_OBJECT_2 -> TRANSPORT_OBJECT_2 -> PLACE_OBJECT_2 -> VERIFY_OBJECT_2 -> COMPLETE
```

- `obj1_verified` becomes `True` iff Object 0 is placed (`dist_0 <= 0.05`, `z_0 < 0.05`, `obj_released_0`), and automatically **revokes** to `False` if Object 0 is disturbed outside its bin while sorting Object 2.
- `COMPLETE` triggers strictly when `obj1_verified and obj2_verified` hold simultaneously.

---

## 3. Staged Bounded Reward Equations & Weights (`MultiSortRewardConfig`)

Let active object $k_{\text{act}} \in \{0, 1\}$ be `0` if $\neg \text{placed}_{\text{prev},0}$, `1` if $\text{placed}_{\text{prev},0} \land \neg \text{placed}_{\text{prev},1}$, with $\text{active\_mask} = 1 - \mathbb{I}[\text{placed}_{\text{prev},0} \land \text{placed}_{\text{prev},1}]$. Every component is bounded and computed solely from `(ag, dg, info)`:

| Stage | Weight / Value | Mathematical Form | Boundedness & Anti-Hacking |
| :--- | :---: | :--- | :--- |
| **A — Reach** | `w_reach = 4.0` | `w_reach * clip(d_ee_prev - d_ee_curr, -0.05, 0.05) * (~bilat_curr) * active_mask` | Potential difference; per-step $\in [-0.20, +0.20]$, telescoping across trajectory |
| **B — Grasp** | `w_grasp = 1.0` | `w_grasp * (I[bilat_curr] - I[bilat_prev]) * active_mask` | Requires real PyBullet `f9 & f10 >= 5.0 N`; empty gripper close gives `0.0`; drop gives `-1.0` |
| **C — Lift** | `w_lift = 1.5` | `w_lift * (h_curr - h_prev) * outside_bin_xy * held_any * active_mask` | `h(z) = clip((z - 0.025) / 0.095, 0, 1)`; cumulative lift `<= +1.5` per object; descent inside bin (`xy <= 0.06m`) is unpenalized |
| **D — Transport** | `w_transport = 6.0` | `w_transport * clip(d_goal_prev - d_goal_curr, -0.05, 0.05) * active_mask` | Strictly toward assigned bin `dg[k_act]`; per-step $\in [-0.30, +0.30]$ |
| **Wrong Bin** | `w_wrong_bin = 2.0` | `-w_wrong_bin * I[entered wrong bin k] * not_her_target` | Penalizes releasing into wrong-colored bin; automatically disabled if HER relabeled onto that bin |
| **E — Placement** | `w_place = 4.0`, `w_disturb = 4.5` | `w_place * newly_placed - w_disturb * newly_disturbed` | Fires once per object on valid entry+release; `-4.5` makes disturb+re-place `-0.5` net loss |
| **F — Complete** | `w_complete = 8.0` | `w_complete * (newly_both - lost_both)` | Terminal milestone bonus only when BOTH objects are simultaneously valid & placed |
| **Step Penalty** | `step_penalty = -0.02` | `-0.02 * (1 - both_curr)` | Encourages fast sorting (`0.0` once complete; bounded `-6.0` max idle penalty) |

### HER Compatibility Dual-Mode Design
1. **Transition Context Mode (`copy_info_dict=True`)**: `step()` populates goal-agnostic physical transition arrays (`prev_ee_pos`, `ee_pos`, `prev_ag`, `prev_gw`, `gripper_width`, `prev_bilateral_contact`, `bilateral_contact`, `bin_positions`, `active_color_indices`). When `HerReplayBuffer` relabels `desired_goal -> dg_virtual`, `compute_reward(next_ag, dg_virtual, infos)` deterministically evaluates Stages A–F relative to `dg_virtual` with zero hidden `self.` state dependencies.
2. **Pure `(ag, dg)` Fallback (`info == {}`)**: Computes smooth distance shaping `-0.5 * sum(clip(d_k, 0, 1))`, lift indicator, `+2.0` per object placed (`z < 0.05`), and `+5.0` full completion bonus cleanly without NaN/Inf.

