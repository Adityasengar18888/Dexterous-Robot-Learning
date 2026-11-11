"""Phase 16.4: Contact-Rich Peg-in-Hole Manipulation Benchmark & Compliant Recovery Evaluation."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

import numpy as np

from src.controllers.assembly_controller import ScriptedAssemblyController
from src.controllers.compliant_assembly_controller import CompliantImpedanceAssemblyController
from src.environments.contact_assembly_env import PandaContactAssemblyEnv


def evaluate_controller(controller_type: str, seeds: List[int]) -> Dict[str, Any]:
    env = PandaContactAssemblyEnv(render_mode=None, max_episode_steps=120)
    if controller_type == "SCRIPTED_COMPLIANT":
        ctrl = CompliantImpedanceAssemblyController(action_scale=env.action_scale, yaw_scale=env.yaw_scale)
    else:
        ctrl = ScriptedAssemblyController(action_scale=env.action_scale, yaw_scale=env.yaw_scale)

    grasps, lifts, aligns, assems, jams, colls = [], [], [], [], [], []
    depths, wall_forces, pos_errs, ang_errs, steps_succ = [], [], [], [], []
    recoveries = 0
    t0 = time.perf_counter()

    try:
        for s in seeds:
            obs, info = env.reset(seed=s)
            ctrl.reset()
            any_g, any_l, any_a, any_j, any_c, any_succ = False, False, False, False, False, False
            max_depth = 0.0
            max_wall_f = 0.0
            ep_steps = 0
            last_info = dict(info)

            for _ in range(120):
                if isinstance(ctrl, CompliantImpedanceAssemblyController):
                    act, c_info = ctrl.step_compliant(env, obs, last_info)
                else:
                    act, c_info = ctrl.step(env, obs, last_info)

                obs, _, term, trunc, last_info = env.step(act)
                ep_steps += 1
                any_g = any_g or bool(last_info.get("bilateral_contact", False))
                any_l = any_l or bool(obs["achieved_goal"][2] >= 0.095)
                any_a = any_a or (
                    np.linalg.norm(obs["achieved_goal"][:2] - obs["desired_goal"][:2]) <= 0.025
                    and obs["achieved_goal"][2] >= 0.090
                )
                any_j = any_j or bool(last_info.get("jam_detected", False))
                any_c = any_c or bool(last_info.get("table_collision", False))
                if last_info.get("is_success", False):
                    any_succ = True
                max_depth = max(max_depth, float(last_info.get("insertion_depth_m", 0.0)))
                max_wall_f = max(max_wall_f, float(last_info.get("wall_force", 0.0)))
                if any_succ and c_info.get("stage") == "DONE":
                    break
                if term or trunc:
                    break

            succ = bool(any_succ or last_info.get("is_success", False))
            grasps.append(float(any_g))
            lifts.append(float(any_l))
            aligns.append(float(any_a))
            assems.append(float(succ))
            jams.append(float(any_j))
            colls.append(float(any_c))
            depths.append(max_depth)
            wall_forces.append(max_wall_f)
            pos_errs.append(float(last_info.get("distance", 0.0)))
            ang_errs.append(float(last_info.get("geodesic_angle_deg", 0.0)))
            if succ:
                steps_succ.append(ep_steps)
            recoveries += int(getattr(ctrl, "recovery_count", 0))
    finally:
        env.close()

    return {
        "policy_type": controller_type,
        "n_episodes": len(seeds),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "grasp_rate": round(float(np.mean(grasps)), 4),
        "lift_rate": round(float(np.mean(lifts)), 4),
        "pre_insert_align_rate": round(float(np.mean(aligns)), 4),
        "assembly_success_rate": round(float(np.mean(assems)), 4),
        "jam_episode_rate": round(float(np.mean(jams)), 4),
        "table_coll_ep_rate": round(float(np.mean(colls)), 4),
        "mean_max_insertion_depth_m": round(float(np.mean(depths)), 4),
        "mean_peak_wall_force_n": round(float(np.mean(wall_forces)), 2),
        "mean_final_pos_err_m": round(float(np.mean(pos_errs)), 4),
        "mean_final_angle_deg": round(float(np.mean(ang_errs)), 2),
        "mean_steps_to_success": round(float(np.mean(steps_succ)), 2) if steps_succ else -1.0,
        "total_jam_recoveries": int(recoveries),
        "sim_force_note": "PyBullet contact impulses are simulator constraints, not hardware F/T equivalent.",
    }


def main() -> None:
    os.makedirs("results/phase16_17", exist_ok=True)
    seeds_30 = list(range(8500, 8530))

    p15_baseline = evaluate_controller("SCRIPTED_PHASE15_BASELINE", seeds_30)
    p16_compliant = evaluate_controller("SCRIPTED_COMPLIANT", seeds_30)

    # Also summarize Phase 15 learned assembly baseline from existing verified JSON without re-running failing 5k
    learned_ref = {}
    p15_lrn_path = "results/phase15_3_assembly/phase15_3_assembly_learned_results.json"
    if os.path.exists(p15_lrn_path):
        with open(p15_lrn_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        learned_ref = {
            "bc_warmstart_only_BEHAVIOR_CLONING": {
                "grasp_rate": raw["bc_warmstart_only"]["grasp_rate"],
                "assembly_success_rate": raw["bc_warmstart_only"]["assembly_success_rate"],
            },
            "sac_her_5000_LEARNED_SAC_HER": {
                "grasp_rate": raw["sac_her_5000"]["grasp_rate"],
                "assembly_success_rate": raw["sac_her_5000"]["assembly_success_rate"],
            },
        }

    out = {
        "phase15_scripted_baseline_30ep": p15_baseline,
        "phase16_compliant_impedance_30ep": p16_compliant,
        "phase15_learned_reference": learned_ref,
        "learned_compliant_assembly_status": "BLOCKED_PREREQUISITES_LOW_GRASP_RL",
    }
    with open("results/phase16_17/phase16_4_contact_assembly_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("[16.4] Phase 15 Scripted:", p15_baseline["assembly_success_rate"], "Phase 16 Compliant:", p16_compliant["assembly_success_rate"])


if __name__ == "__main__":
    main()

