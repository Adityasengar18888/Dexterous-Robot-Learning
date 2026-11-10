# Phase 15.2.5B.1 — Continuous Vertical Potential Reward & HER Compatibility Audit

**Project:** Robotic Manipulation using SAC + HER  
**Environment:** [`PandaSinglePreGraspEnv`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/src/environments/single_pregrasp_env.py) (`SinglePreGraspRewardConfig(use_vertical_potential=True, w_vertical_potential=2.0)`)  
**HER Verification Suite:** [`tests/test_vertical_potential_reward_her.py`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/tests/test_vertical_potential_reward_her.py) (`3 passed`; combined `8 passed in 30.55s`)

---

## 1. Continuous Vertical Potential Formulation

In Phase 15.2.5B, `r_vert_approach` was gated on `xy_d_curr <= 0.030`, giving zero descent signal across `99%` of exploration space and trapping SAC at `z ≈ 0.50 m`. Phase 15.2.5B.1 introduces an un-gated potential-based vertical term:

$$z_{\text{grasp}}(o) = \max\left(z_{\min,\text{safe}},\; o_z + z_{\text{offset}}\right), \quad z_{\min,\text{safe}} = 0.0205\text{ m},\; z_{\text{offset}} = 0.000\text{ m}$$

$$\Phi_z(s) = -w_z \operatorname{clip}\left(|z_{EE} - z_{\text{grasp}}|, 0, 1\right), \quad w_z = 2.0$$

$$R_{z,\text{progress}} = \left(\Phi_z(s_{t+1}) - \Phi_z(s_t)\right) \cdot \mathbb{I}[\neg b_{t} \wedge \neg b_{t+1}]$$

Safeguards applied to $R_{z,\text{progress}}$:
1. **Valid grasp height deadband**: Once $|z_{EE,t} - z_{\text{grasp}}| \le 0.008\text{ m}$ and $|z_{EE,t+1} - z_{\text{grasp}}| \le 0.008\text{ m}$ with $z_{EE,t+1} \ge z_{\text{grasp}} - 0.002\text{ m}$, $R_{z,\text{progress}} = 0.0$.
2. **Safety gating on positive gains**: Positive $R_{z,\text{progress}} > 0$ requires open gripper (`width > 0.065 m`), zero table collision (`~curr_tcoll`), downward gripper orientation (`orn_ok`), and $z_{EE,t+1} \ge z_{\text{grasp}} - 0.002\text{ m}$. Negative movement away from $z_{\text{grasp}}$ or into the table remains negative, guaranteeing **zero oscillation reward** (`+0.10 + (-0.10) = 0.0`) and **strictly negative premature-closure / table-scrape loops**.
3. **Logging Separation**: `r_vert_potential` ($\Phi_z(s_{t+1})$ ungrasped) is logged strictly as a diagnostic telemetry component and is **never** added cumulatively to `total_reward`.

---

## 2. HER Compatibility Verification

Every quantity required to recompute $\Phi_z(s_t)$ and $\Phi_z(s_{t+1})$ under arbitrary relabeled `desired_goal` — `prev_ee_pos`, `ee_pos`, `prev_ag`, `prev_gw`, `gripper_width`, `prev_bilateral_contact`, `bilateral_contact`, `prev_orn_ok`, `orn_ok`, `prev_table_collision`, `table_collision` — is stored in transition `info` (`her_copy_info_dict: true`), with zero reliance on hidden simulator state.

Verified scenarios in [`tests/test_vertical_potential_reward_her.py`](file:///c:/Users/Welcome/OneDrive/Desktop/newshit/tests/test_vertical_potential_reward_her.py):
- Downward progress far from XY (`ee_xy=(0.30, 0.15)`): `r_vert_progress = +0.1000`
- Upward withdrawal: `r_vert_progress = -0.1000` (`down + up = 0.0000`)
- Stationary hovering (`0.48 -> 0.48 m`): `r_vert_progress = 0.0000`
- Entering & remaining inside valid grasp height (`0.031 -> 0.027 m`): `0.0000`
- Movement below safe grasp height (`0.025 -> 0.015 m`) or table collision: `< 0.0` (never rewarded)
- Closed-gripper descent (`gw = 0.01 m`): `r_vert_progress = 0.0` and `r_open_comp = -0.06`
- Relabeled goals (`desired_goal` switched) & elevated active object (`obj_z = 0.055 m`): dynamic `z_grasp` update verified (`batched == single` within `atol = 1e-5`)
- Live `HerReplayBuffer(future, n_sampled_goal=4, copy_info_dict=True).sample(32)`: all recomputed rewards finite (`0` NaN/Inf).

---

## 3. Part 6 Pre-Training Reward Magnitude Audit (`80`-Step Trajectories, `seed = 1001`)

| Reward Component | `idle` | `premature_close` | `xy_align_only` | `vertical_descent` | `scripted_full` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `r_align_prog` (XY progress) | `+0.0006` | `+0.5236` | `+0.6246` | `+0.6246` | `+0.6222` |
| `r_align_prox` (XY prox) | `-3.1212` | `-0.5456` | `-0.0661` | `-0.0699` | `-0.0607` |
| **`r_align`** (total horizontal) | **`-3.1206`** | **`-0.0221`** | **`+0.5585`** | **`+0.5547`** | **`+0.5615`** |
| `r_vert_potential` (sum, logged only) | `-74.3834` | `-11.6925` | `-74.3964` | `-9.7547` | `-5.4744` |
| `final_r_vert_potential` ($\Phi_z(s_T)$) | `-0.9301` | `-0.0434` | `-0.9302` | `-0.0091` | `0.0000` |
| **`r_vert_progress`** ($R_{z,\text{progress}}$) | **`-0.0005`** | **`-0.0011`** | **`-0.0006`** | **`+0.9139`** | **`+0.9212`** |
| `r_open_comp` | `-4.8000` | `-4.6200` | `0.0000` | `0.0000` | `0.0000` |
| `r_collision` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `r_disp` (object knock) | `0.0000` | `-0.0703` | `0.0000` | `0.0000` | `0.0000` |
| `r_auth` (grasp authorization) | `0.0000` | `0.0000` | `0.0000` | `+0.5000` | `+0.5000` |
| `r_grasp` (bilateral contact) | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `+1.5000` |
| `r_lift` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `+1.5000` |
| `r_goal_lift` | `0.0000` | `+0.0734` | `0.0000` | `0.0000` | `+0.7072` |
| `r_hold` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `+4.0000` |
| `r_step` | `-1.6000` | `-1.6000` | `-1.6000` | `-1.6000` | `-0.4800` |
| **`total_reward` (Episode Return)** | **`-9.5211`** | **`-6.2401`** | **`-1.0421`** | **`+0.3686`** | **`+9.2098`** |

**Conclusion:** The reward landscape is strictly ordered:
`idle (-9.52) < premature_close (-6.24) < xy_align_only (-1.04) < vertical_descent (+0.37) < scripted_full (+9.21)`, with `r_vert_progress (+0.92)` properly scaled beneath grasp + lift + hold (`+7.00`).

