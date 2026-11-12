"""Phase 15.1 — Sim-to-Real Transfer Readiness Robustness Benchmark.

Evaluates the frozen Phase 14 E_ADAPTIVE_TRANSPORT controller on unmodified
DR4 Seed 42 and Seed 43 checkpoints under:
  - A0/B0/C0 Clean Baseline
  - A1/A2 Observation Position Noise (+-3 mm, +-5 mm)
  - B1/B2/B3/B4 Observation/Action FIFO Delays (1, 2, 4 obs; 1+1 combined)
  - C1 Unprivileged-Contact Conservative Kinematic Mode
"""

from __future__ import annotations

import os
import json
import time
import yaml
import psutil
import hashlib
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from stable_baselines3 import SAC

from src.environments.dense_pick_place_env import PandaPickPlaceDenseRewardEnv
from src.rl.adaptive_controllers import (
    ControllerConfig,
    AdaptiveTransportController,
    get_step_telemetry,
)
from src.rl.sim_to_real_observation_wrapper import ObservationNoiseWrapper
from src.rl.sim_to_real_latency_wrapper import LatencyWrapper
from src.rl.unprivileged_contact_wrapper import (
    UnprivilegedAdaptiveController,
    strip_privileged_telemetry,
)


CONDITIONS: List[Dict[str, Any]] = [
    {"id": "A0_CLEAN_BASELINE",     "group": "Baseline",     "noise_bound": 0.0,   "noise_std": 0.0,    "obs_delay": 0, "act_delay": 0, "unprivileged": False},
    {"id": "A1_NOISE_3MM",          "group": "Obs_Noise",    "noise_bound": 0.003, "noise_std": 0.0015, "obs_delay": 0, "act_delay": 0, "unprivileged": False},
    {"id": "A2_NOISE_5MM",          "group": "Obs_Noise",    "noise_bound": 0.005, "noise_std": 0.0025, "obs_delay": 0, "act_delay": 0, "unprivileged": False},
    {"id": "B1_OBS_DELAY_1",        "group": "Latency",      "noise_bound": 0.0,   "noise_std": 0.0,    "obs_delay": 1, "act_delay": 0, "unprivileged": False},
    {"id": "B2_OBS_DELAY_2",        "group": "Latency",      "noise_bound": 0.0,   "noise_std": 0.0,    "obs_delay": 2, "act_delay": 0, "unprivileged": False},
    {"id": "B3_OBS_DELAY_4",        "group": "Latency",      "noise_bound": 0.0,   "noise_std": 0.0,    "obs_delay": 4, "act_delay": 0, "unprivileged": False},
    {"id": "B4_COMBINED_DELAY_1_1", "group": "Latency",      "noise_bound": 0.0,   "noise_std": 0.0,    "obs_delay": 1, "act_delay": 1, "unprivileged": False},
    {"id": "C1_UNPRIVILEGED",       "group": "Contact_Mode", "noise_bound": 0.0,   "noise_std": 0.0,    "obs_delay": 0, "act_delay": 0, "unprivileged": True},
]


def readiness_class(success_rate: float) -> str:
    if success_rate >= 0.90:
        return "A - ROBUST"
    if success_rate >= 0.75:
        return "B - DEGRADED BUT USABLE"
    if success_rate >= 0.50:
        return "C - FRAGILE"
    return "D - FAILED"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest().upper()


def build_heldout_env() -> PandaPickPlaceDenseRewardEnv:
    with open("configs/pick_place_heldout.yaml", "r", encoding="utf-8") as f:
        h_cfg = yaml.safe_load(f)["env"]
    h_kw = {k: v for k, v in h_cfg.items() if k != "name"}
    if "randomizer" in h_kw:
        h_kw["randomizer_config"] = h_kw.pop("randomizer")
    return PandaPickPlaceDenseRewardEnv(**h_kw)


def run_condition(
    env: PandaPickPlaceDenseRewardEnv,
    model: SAC,
    cond: Dict[str, Any],
    training_seed: int,
    eval_seeds: List[int],
    cfg: ControllerConfig,
    noise_seed_base: int = 2026,
) -> pd.DataFrame:
    noise_wrap = ObservationNoiseWrapper(
        std_m=cond["noise_std"],
        clip_bound_m=cond["noise_bound"],
        seed=noise_seed_base,
    )
    lat_wrap = LatencyWrapper(
        obs_delay_steps=cond["obs_delay"],
        action_delay_steps=cond["act_delay"],
    )

    if cond["unprivileged"]:
        ctrl = UnprivilegedAdaptiveController(
            cfg,
            settle_closure_steps=2,
            jaw_cube_width_min=0.042,
            jaw_cube_width_max=0.066,
        )
    else:
        ctrl = AdaptiveTransportController(cfg)

    rows: List[Dict[str, Any]] = []
    for es in eval_seeds:
        obs, _ = env.reset(seed=int(es))
        ctrl.reset()
        noise_wrap.reset(episode_seed=int(es))

        t0_true = get_step_telemetry(env)
        obs_p0, t0_p = noise_wrap.perturb(obs, t0_true)
        if cond["unprivileged"]:
            t0_p = strip_privileged_telemetry(t0_p)
        lat_wrap.reset(obs_p0, t0_p)

        desired_goal_true = obs["desired_goal"].copy()
        min_ee_obj_dist = float(t0_true["rel_dist"])
        max_cube_z = float(t0_true["cube_z"])
        grasped = False
        had_bilateral = False
        lifted = False
        dropped = False
        entered_target_ever = False
        overshoot_events = 0
        was_in_target_prev = False
        current_hold = 0
        max_hold = 0
        time_to_first_success: Optional[int] = None
        nan_detected = False
        total_reward = 0.0
        step = 0
        done = False
        info_step: Dict[str, Any] = {}

        # Active step loop (using noisy/delayed interface strictly for policy + controller)
        obs_cur = obs
        while not done:
            t_true_before = get_step_telemetry(env)
            obs_noisy, t_noisy = noise_wrap.perturb(obs_cur, t_true_before)
            if cond["unprivileged"]:
                t_noisy = strip_privileged_telemetry(t_noisy)

            obs_delayed, t_delayed = lat_wrap.push_and_get_obs(obs_noisy, t_noisy)
            raw_action, _ = model.predict(obs_delayed, deterministic=True)
            cmd_action, _ = ctrl.step_action(raw_action, t_delayed)
            exec_action = lat_wrap.push_and_get_action(cmd_action)

            if not (np.all(np.isfinite(exec_action)) and np.all(np.isfinite(obs_delayed["observation"]))):
                nan_detected = True

            obs_cur, reward, terminated, truncated, info_step = env.step(exec_action)
            done = terminated or truncated
            step += 1
            total_reward += float(reward)

            # Evaluate physical truth in simulator (never leaked back to unprivileged/noisy agent)
            ta = get_step_telemetry(env)
            rel_dist = float(ta["rel_dist"])
            cube_z = float(ta["cube_z"])
            norm_force = float(ta["total_normal_force"])

            min_ee_obj_dist = min(min_ee_obj_dist, rel_dist)
            max_cube_z = max(max_cube_z, cube_z)
            if not cond["unprivileged"] and ta["contact_symmetry"] and norm_force >= 10.0:
                had_bilateral = True

            is_held = (
                (rel_dist < 0.05)
                and (float(exec_action[3]) < 0.0)
                and (ta["has_contact"] or ta["gripper_width"] < 0.05)
            )
            if is_held:
                grasped = True
                current_hold += 1
                max_hold = max(max_hold, current_hold)
            else:
                if max_hold > 5:
                    dropped = True
                current_hold = 0

            if cube_z > 0.05 and (is_held or ta["has_contact"]):
                lifted = True

            g_dist = float(np.linalg.norm(obs_cur["achieved_goal"] - desired_goal_true))
            in_target_now = g_dist <= 0.05
            if in_target_now:
                entered_target_ever = True
            if was_in_target_prev and (not in_target_now):
                overshoot_events += 1
            was_in_target_prev = in_target_now

            if info_step.get("is_success", False) and time_to_first_success is None:
                time_to_first_success = step

        final_dist = float(info_step.get("transport_dist", np.linalg.norm(obs_cur["achieved_goal"] - desired_goal_true)))
        reached = bool(min_ee_obj_dist < 0.05)
        transported = bool(final_dist < 0.05)
        final_success = bool(info_step.get("is_success", False))
        timeout = bool((not final_success) and (step >= env.unwrapped.max_episode_steps))

        if final_success:
            failure_type = "SUCCESS"
        elif not reached:
            failure_type = "NO_REACH"
        elif not grasped:
            failure_type = "REACH_BUT_NO_GRASP"
        elif not lifted:
            failure_type = "GRASP_BUT_NO_LIFT"
        elif dropped or (max_hold < 5):
            failure_type = "LIFT_BUT_DROPPED"
        elif final_dist > 0.05:
            failure_type = "LIFT_AND_HELD_BUT_WRONG_DIRECTION"
        else:
            failure_type = "REACHED_TARGET_BUT_NO_PLACE"

        rows.append({
            "condition": cond["id"],
            "group": cond["group"],
            "training_seed": training_seed,
            "eval_seed": int(es),
            "reach_success": reached,
            "grasp_success": grasped,
            "bilateral_observable": (not cond["unprivileged"]),
            "bilateral_success": had_bilateral if not cond["unprivileged"] else None,
            "lift_success": lifted,
            "transport_success": transported,
            "placement_success": final_success,
            "success": final_success,
            "final_dist": final_dist,
            "dropped": bool(failure_type == "LIFT_BUT_DROPPED"),
            "wrong_direction": bool(failure_type == "LIFT_AND_HELD_BUT_WRONG_DIRECTION"),
            "overshoot_events": overshoot_events,
            "timeout": timeout,
            "ep_length": step,
            "time_to_first_success": time_to_first_success if time_to_first_success is not None else -1,
            "episode_reward": total_reward,
            "failure_type": failure_type,
            "nan_detected": nan_detected,
        })

    return pd.DataFrame(rows)


def render_readiness_chart(summary_df: pd.DataFrame, out_path: str) -> None:
    w, h_px = 1320, 780
    img = Image.new("RGB", (w, h_px), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((30, 18), "Phase 15.1: Sim-to-Real Transfer Readiness Benchmark (E_ADAPTIVE_TRANSPORT, N=20/seed)", fill=(0, 0, 0))
    draw.text((30, 40), "Seed 42 & Seed 43 Success Rate (%) & Final Distance Across Observation Noise, Latency, and Unprivileged Contact", fill=(85, 85, 85))

    cond_order = [c["id"] for c in CONDITIONS]
    short_lbl = {
        "A0_CLEAN_BASELINE": "A0 Clean",
        "A1_NOISE_3MM": "A1 +-3mm",
        "A2_NOISE_5MM": "A2 +-5mm",
        "B1_OBS_DELAY_1": "B1 d=1",
        "B2_OBS_DELAY_2": "B2 d=2",
        "B3_OBS_DELAY_4": "B3 d=4",
        "B4_COMBINED_DELAY_1_1": "B4 1+1",
        "C1_UNPRIVILEGED": "C1 Unpriv",
    }

    # Panel 1 (Top): Success rate per condition for Seed 42, Seed 43, Combined
    x0, y0, pw, ph = 30, 80, 1260, 320
    draw.rectangle([x0, y0, x0 + pw, y0 + ph], outline=(200, 200, 200), fill=(250, 250, 250))
    draw.text((x0 + 12, y0 + 8), "End-to-End Held-out Success Rate (%) with Readiness Thresholds (A>=90%, B>=75%, C>=50%)", fill=(0, 0, 0))
    pad_l, pad_r, pad_t, pad_b = 55, 20, 35, 42
    gx0, gy0, gw, gh = x0 + pad_l, y0 + pad_t, pw - pad_l - pad_r, ph - pad_t - pad_b

    for pct, col in [(0, (230, 230, 230)), (50, (230, 170, 50)), (75, (40, 145, 210)), (90, (25, 170, 90)), (100, (230, 230, 230))]:
        yy = gy0 + gh - int((pct / 100.0) * gh)
        draw.line([gx0, yy, gx0 + gw, yy], fill=col, width=2 if pct in (50, 75, 90) else 1)
        draw.text((x0 + 8, yy - 6), f"{pct}%", fill=(90, 90, 90))

    grp_w = gw / len(cond_order)
    bar_w = int((grp_w * 0.72) / 3)
    c_cols = {"42": (45, 130, 205), "43": (25, 170, 95), "COMBINED": (85, 95, 115)}

    for gi, cid in enumerate(cond_order):
        bx = gx0 + int(gi * grp_w + grp_w * 0.14)
        draw.text((bx + 6, gy0 + gh + 10), short_lbl.get(cid, cid), fill=(20, 20, 20))
        for si, sk in enumerate(["42", "43", "COMBINED"]):
            sub = summary_df[(summary_df["condition"] == cid) & (summary_df["training_seed"].astype(str) == sk)]
            v = float(sub["success_rate"].iloc[0]) * 100.0 if len(sub) > 0 else 0.0
            bh = int((v / 100.0) * gh)
            xb = bx + si * (bar_w + 3)
            yb = gy0 + gh - bh
            draw.rectangle([xb, yb, xb + bar_w, gy0 + gh], fill=c_cols[sk])
            draw.text((xb, max(gy0 + 2, yb - 13)), f"{v:.0f}", fill=(0, 0, 0))

    # Panel 2 (Bottom): Combined Final Distance (cm)
    y0b, phb = 425, 290
    draw.rectangle([x0, y0b, x0 + pw, y0b + phb], outline=(200, 200, 200), fill=(250, 250, 250))
    draw.text((x0 + 12, y0b + 8), "Mean Final Object-Target Distance (cm) — Combined Seed 42 & 43 (Red line = 5.0 cm success threshold)", fill=(0, 0, 0))
    gx0b, gy0b, gwb, ghb = x0 + pad_l, y0b + pad_t, pw - pad_l - pad_r, phb - pad_t - pad_b
    y_max_cm = 30.0
    for cm in [0, 5, 10, 20, 30]:
        yy = gy0b + ghb - int((cm / y_max_cm) * ghb)
        draw.line([gx0b, yy, gx0b + gwb, yy], fill=(220, 60, 60) if cm == 5 else (230, 230, 230), width=2 if cm == 5 else 1)
        draw.text((x0 + 8, yy - 6), f"{cm}cm", fill=(90, 90, 90))

    bar_wb = int(grp_w * 0.5)
    for gi, cid in enumerate(cond_order):
        bx = gx0b + int(gi * grp_w + grp_w * 0.25)
        draw.text((bx, gy0b + ghb + 10), short_lbl.get(cid, cid), fill=(20, 20, 20))
        sub = summary_df[(summary_df["condition"] == cid) & (summary_df["training_seed"].astype(str) == "COMBINED")]
        cm_val = float(sub["mean_final_dist"].iloc[0]) * 100.0 if len(sub) > 0 else 0.0
        bh = int(min(1.0, cm_val / y_max_cm) * ghb)
        draw.rectangle([bx, gy0b + ghb - bh, bx + bar_wb, gy0b + ghb], fill=(110, 125, 155))
        draw.text((bx + 6, max(gy0b + 2, gy0b + ghb - bh - 14)), f"{cm_val:.1f}cm", fill=(0, 0, 0))

    lx = 30
    for lbl, col in [("Seed 42 (N=20)", c_cols["42"]), ("Seed 43 (N=20)", c_cols["43"]), ("Combined (N=40)", c_cols["COMBINED"])]:
        draw.rectangle([lx, 738, lx + 18, 752], fill=col)
        draw.text((lx + 24, 738), lbl, fill=(0, 0, 0))
        lx += 200
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="Run worker for seed 42 or 43")
    parser.add_argument("--merge", action="store_true", help="Merge Seed 42 & 43 results")
    args = parser.parse_args()

    out_dir = "results/phase15_1"
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()
    proc = psutil.Process(os.getpid())

    with open("configs/phase15_1_config.yaml", "r", encoding="utf-8") as f:
        p15_cfg = yaml.safe_load(f)

    # Checkpoint Hash Verification Gate (Rule Section 1)
    for sk in ["42", "43"]:
        cp_info = p15_cfg["checkpoints"][f"seed{sk}"]
        actual_hash = sha256_file(cp_info["path"])
        if actual_hash != cp_info["sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch for seed {sk}: {actual_hash} != {cp_info['sha256']}")

    if args.seed is not None:
        s = int(args.seed)
        env = build_heldout_env()
        cp_path = p15_cfg["checkpoints"][f"seed{s}"]["path"]
        model = SAC.load(cp_path, env=env, device="cpu")
        cfg = ControllerConfig()
        eval_seeds = list(range(1001, 1021))

        all_parts = []
        for cond in CONDITIONS:
            # Hard budget safety stop at 40 minutes
            if (time.time() - t_start) > 40 * 60:
                print(f"[BUDGET STOP] Stopping Seed {s} prior to {cond['id']} to preserve 45m limit.")
                break
            df_c = run_condition(env, model, cond, s, eval_seeds, cfg, noise_seed_base=2026)
            all_parts.append(df_c)
            print(
                f"[Seed {s}] {cond['id']:22s} | Succ={df_c['success'].mean()*100:5.1f}% | "
                f"Reach={df_c['reach_success'].mean()*100:5.1f}% | Grasp={df_c['grasp_success'].mean()*100:5.1f}% | "
                f"Lift={df_c['lift_success'].mean()*100:5.1f}% | Drop={df_c['dropped'].mean()*100:4.1f}% | "
                f"WrongDir={df_c['wrong_direction'].mean()*100:4.1f}% | Dist={df_c['final_dist'].mean():.4f}m",
                flush=True,
            )

        env.close()
        df_s = pd.concat(all_parts, ignore_index=True)
        df_s.to_csv(os.path.join(out_dir, f"phase15_1_seed{s}.csv"), index=False)
        elapsed = time.time() - t_start
        print(f"Seed {s} finished in {elapsed:.1f}s | RAM: {proc.memory_info().rss/(1024*1024):.1f} MB")
        return

    if args.merge:
        d42 = pd.read_csv(os.path.join(out_dir, "phase15_1_seed42.csv"))
        d43 = pd.read_csv(os.path.join(out_dir, "phase15_1_seed43.csv"))
        df_all = pd.concat([d42, d43], ignore_index=True)
        df_all.to_csv(os.path.join(out_dir, "phase15_1_episodes.csv"), index=False)

        # Compute baselines per seed & combined
        base_rates: Dict[str, float] = {}
        for sk, sdf in [("42", d42), ("43", d43), ("COMBINED", df_all)]:
            b = sdf[sdf["condition"] == "A0_CLEAN_BASELINE"]
            base_rates[sk] = float(b["success"].mean())

        summary_rows = []
        for cid in [c["id"] for c in CONDITIONS]:
            for sk, sdf in [("42", d42), ("43", d43), ("COMBINED", df_all)]:
                sub = sdf[sdf["condition"] == cid]
                if len(sub) == 0:
                    continue
                succ = float(sub["success"].mean())
                b_succ = base_rates[sk]
                bilat_obs = bool(sub["bilateral_observable"].iloc[0])
                bilat_rate = float(sub["bilateral_success"].dropna().mean()) if (bilat_obs and sub["bilateral_success"].notna().any()) else None
                summary_rows.append({
                    "condition": cid,
                    "group": str(sub["group"].iloc[0]),
                    "training_seed": sk,
                    "episodes": len(sub),
                    "success_rate": round(succ, 4),
                    "delta_pp_vs_baseline": round((succ - b_succ) * 100.0, 2),
                    "readiness_class": readiness_class(succ),
                    "reach_rate": round(float(sub["reach_success"].mean()), 4),
                    "grasp_rate": round(float(sub["grasp_success"].mean()), 4),
                    "bilateral_observable": bilat_obs,
                    "bilateral_rate": round(bilat_rate, 4) if bilat_rate is not None else None,
                    "lift_rate": round(float(sub["lift_success"].mean()), 4),
                    "transport_rate": round(float(sub["transport_success"].mean()), 4),
                    "placement_rate": round(float(sub["placement_success"].mean()), 4),
                    "drop_rate": round(float(sub["dropped"].mean()), 4),
                    "wrong_direction_rate": round(float(sub["wrong_direction"].mean()), 4),
                    "timeout_rate": round(float(sub["timeout"].mean()), 4),
                    "mean_overshoot_events": round(float(sub["overshoot_events"].mean()), 2),
                    "mean_ep_length": round(float(sub["ep_length"].mean()), 2),
                    "mean_steps_to_success": round(float(sub[sub["time_to_first_success"] >= 0]["time_to_first_success"].mean()), 2) if (sub["time_to_first_success"] >= 0).any() else -1.0,
                    "mean_final_dist": round(float(sub["final_dist"].mean()), 5),
                    "mean_reward": round(float(sub["episode_reward"].mean()), 2),
                    "any_nan": bool(sub["nan_detected"].any()),
                })

        df_sum = pd.DataFrame(summary_rows)
        df_sum.to_csv(os.path.join(out_dir, "phase15_1_summary.csv"), index=False)
        with open(os.path.join(out_dir, "phase15_1_results.json"), "w", encoding="utf-8") as f:
            json.dump({
                "status": "PHASE 15.1 COMPLETE",
                "checkpoint_hashes_verified": True,
                "summaries": summary_rows,
            }, f, indent=4)

        chart_path = os.path.join(out_dir, "phase15_1_readiness_chart.png")
        render_readiness_chart(df_sum, chart_path)
        print("Merged Phase 15.1 summary & saved results JSON and readiness chart.")


if __name__ == "__main__":
    main()

