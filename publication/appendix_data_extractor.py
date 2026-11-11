"""Extracts genuine archival tables from repository artifacts for Deep Chapters and Appendices B-H."""

from __future__ import annotations

import ast
import csv
import glob
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
PUB = ROOT / "publication"


def extract_tb_table() -> List[List[str]]:
    tb = json.loads((PUB / "benchmark_data" / "tb_scalars_all.json").read_text(encoding="utf-8"))
    rows = []
    for run_key in sorted(tb.keys()):
        tags = tb[run_key]
        if not tags:
            rows.append([run_key, "0", "-", "-", "-", "Aborted / Empty"])
            continue
        def last_val(t):
            d = tags.get(t)
            if not d or not d.get("values"):
                return "-"
            return f"{d['values'][-1]:.3f}"
        max_step = max((d["steps"][-1] for d in tags.values() if d.get("steps")), default=0)
        rows.append([
            run_key,
            f"{max_step:,}",
            last_val("train/actor_loss"),
            last_val("train/critic_loss"),
            last_val("train/ent_coef"),
            last_val("time/fps"),
        ])
    return rows


def extract_npz_table() -> List[List[str]]:
    rows = []
    for p in sorted(glob.glob(str(ROOT / "**" / "evaluations.npz"), recursive=True)):
        rel = str(Path(p).relative_to(ROOT)).replace("\\", "/")
        try:
            d = np.load(p)
            ts = d["timesteps"]
            rew = d["results"].mean(axis=1) if "results" in d else np.array([0.0])
            succ = d["successes"].mean(axis=1) if "successes" in d else np.array([0.0])
            rows.append([
                rel.replace("/models/evaluations.npz", "").replace("/evaluations.npz", ""),
                str(len(ts)),
                f"{int(ts[-1]):,}" if len(ts) else "0",
                f"{rew[0]:.1f} -> {rew[-1]:.1f}",
                f"{succ.max()*100:.1f}%",
            ])
        except Exception as exc:
            rows.append([rel, "ERR", "-", str(exc)[:30], "-"])
    return rows


def extract_d5_telemetry_sample() -> List[List[str]]:
    s_rows = list(csv.DictReader((ROOT / "results" / "phase13_5" / "d5_lift_trajectories" / "success_sample_1_seed42_ep5001.csv").open()))
    f_rows = list(csv.DictReader((ROOT / "results" / "phase13_5" / "d5_lift_trajectories" / "grasp_but_no_lift_sample_1_seed42_ep5002.csv").open()))
    out = []
    f_map = {int(r["step"]): r for r in f_rows}
    for r in s_rows:
        st = int(r["step"])
        if st % 3 == 0 and st <= 99:
            fr = f_map.get(st, {})
            out.append([
                str(st),
                f"{float(r['total_normal_force']):.1f} N",
                f"{float(r['cube_z']):.4f} m",
                f"{float(r['gripper_width'])*100:.2f} cm",
                f"{float(fr.get('total_normal_force', 0.0)):.1f} N",
                f"{float(fr.get('cube_z', 0.0)):.4f} m",
                f"{float(fr.get('gripper_width', 0.0))*100:.2f} cm",
            ])
    return out


def extract_d4_contact_geometry() -> List[List[str]]:
    rows = list(csv.DictReader((ROOT / "results" / "phase13_5" / "d4_contact_geometry" / "contact_geometry_summary.csv").open()))
    return [
        [r["yaw_deg"] + " deg", f"{float(r['success_rate'])*100:.1f}%", f"{float(r['grasp_rate'])*100:.1f}%",
         f"{float(r['lift_rate'])*100:.1f}%", f"{float(r['mean_normal_force']):.1f} N",
         f"{float(r['mean_slip_events']):.2f}"]
        for r in rows
    ]


def extract_phase11_diagnostic_tables() -> Dict[str, List[List[str]]]:
    r11_8 = list(csv.DictReader((ROOT / "results" / "phase11_8_state_initialization_metrics.csv").open()))
    by_state: Dict[str, List[Dict[str, str]]] = {}
    for r in r11_8:
        by_state.setdefault(r["initial_state"], []).append(r)
    s11_8 = []
    for k, grp in by_state.items():
        n = len(grp)
        succ = 100.0 * sum(1 for x in grp if x["success"] in ("True", "1", "true")) / n
        gr = 100.0 * sum(1 for x in grp if x["grasped"] in ("True", "1", "true")) / n
        li = 100.0 * sum(1 for x in grp if x["lifted"] in ("True", "1", "true")) / n
        dr = 100.0 * sum(1 for x in grp if x["dropped"] in ("True", "1", "true")) / n
        s11_8.append([k, str(n), f"{gr:.1f}%", f"{li:.1f}%", f"{dr:.1f}%", f"{succ:.1f}%"])

    r11_10 = list(csv.DictReader((ROOT / "results" / "phase11_10_curriculum_metrics.csv").open()))
    s11_10 = [
        [r["stage"], r["timesteps"], r["initial_state_type"], f"{float(r['success_rate'])*100:.1f}%",
         f"{float(r['grasp_rate'])*100:.1f}%", f"{float(r['lift_rate'])*100:.1f}%"]
        for r in r11_10
    ]

    r11_10b = list(csv.DictReader((ROOT / "results" / "phase11_10b_acquisition_metrics.csv").open()))
    s11_10b = [
        [r["Total Steps"], f"{float(r['Reach'])*100:.1f}%", f"{float(r['Grasp'])*100:.1f}%",
         f"{float(r['Lift'])*100:.1f}%", f"{float(r['Object Movement'])*100:.1f}%",
         r["Hold Duration"], f"{float(r['Timeout'])*100:.1f}%"]
        for r in r11_10b
    ]
    return {"11_8": s11_8, "11_10": s11_10, "11_10b": s11_10b}


def extract_phase13_checkpoints_and_taxonomy() -> tuple[List[List[str]], List[List[str]]]:
    chk_rows = []
    for f in sorted(glob.glob(str(ROOT / "experiments" / "phase13_dense_dr" / "*" / "*" / "metrics" / "checkpoint_metrics.csv"))):
        parts = Path(f).parts
        dr_tag, seed_tag = parts[-4].upper(), parts[-3]
        for r in csv.DictReader(open(f, encoding="utf-8")):
            chk_rows.append([
                f"{dr_tag}/{seed_tag}",
                r["checkpoint"],
                r["regime"],
                f"{float(r['reach_rate'])*100:.0f}%",
                f"{float(r['grasp_rate'])*100:.0f}%",
                f"{float(r['lift_rate'])*100:.0f}%",
                f"{float(r['success_rate'])*100:.0f}%",
                f"{float(r['mean_final_dist']):.4f} m",
            ])

    tax_rows = []
    for f in sorted(glob.glob(str(ROOT / "experiments" / "phase13_dense_dr" / "*" / "*" / "metrics" / "failure_taxonomy.csv"))):
        parts = Path(f).parts
        dr_tag, seed_tag = parts[-4].upper(), parts[-3]
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if float(r["count"]) > 0:
                tax_rows.append([
                    f"{dr_tag}/{seed_tag}",
                    r["checkpoint"],
                    r["regime"],
                    r["failure_type"],
                    r["count"],
                    f"{float(r['fraction'])*100:.1f}%",
                ])
    return chk_rows, tax_rows


def extract_phase14_lite_comparison() -> List[List[str]]:
    rows = list(csv.DictReader((ROOT / "results" / "phase14_lite" / "controller_comparison.csv").open()))
    return [
        [
            r["training_seed"],
            r["controller"][:18],
            r["regime"],
            f"{float(r['success_rate'])*100:.1f}%",
            f"{float(r['grasp_to_lift_conversion'])*100:.1f}%",
            f"{float(r['mean_slip_events']):.2f}",
            f"{float(r['mean_peak_vert_jerk']):.2f}",
            f"{float(r['grasp_but_no_lift_rate'])*100:.1f}%",
        ]
        for r in rows
    ]


def extract_phase14_3_confirmation() -> List[List[str]]:
    rows = list(csv.DictReader((ROOT / "results" / "phase14_3" / "confirmation_summary.csv").open()))
    return [
        [
            r["training_seed"],
            r["controller"][:18],
            r["regime"],
            f"{float(r['success_rate'])*100:.1f}%",
            f"{float(r['bilateral_grasp_rate'])*100:.1f}%",
            f"{float(r['grasp_but_no_lift_rate'])*100:.1f}%",
            f"{float(r['mean_slip_events']):.2f}",
            f"{float(r['mean_steps_to_success']):.1f}",
        ]
        for r in rows
    ]


def extract_phase15_phase3_and_assembly() -> tuple[List[List[str]], List[List[str]]]:
    p15_3 = json.loads((ROOT / "results" / "phase15_phase3" / "phase15_phase3_robustness_results.json").read_text())
    pp_rows = [
        [r["model"], r["condition"], str(r["n_episodes"]),
         f"{r['success_rate']*100:.1f}%", f"{r['grasp_rate']*100:.1f}%", f"{r.get('lift_rate',0)*100:.1f}%"]
        for r in p15_3.get("pickplace_dr_results", [])
    ]

    scr_a = json.loads((ROOT / "results" / "phase15_3_assembly" / "phase15_3_assembly_scripted_results.json").read_text())
    lrn_a = json.loads((ROOT / "results" / "phase15_3_assembly" / "phase15_3_assembly_learned_results.json").read_text())
    asm_rows = [
        ["Scripted Baseline", str(scr_a["n_episodes"]), f"{scr_a['grasp_rate']*100:.1f}%",
         f"{scr_a['pre_insert_align_rate']*100:.1f}%", f"{scr_a['assembly_success_rate']*100:.2f}%",
         f"{scr_a.get('mean_final_angle_deg', 0):.2f} deg"]
    ]
    for k, v in lrn_a.items():
        if isinstance(v, dict) and "n_episodes" in v:
            asm_rows.append([
                k, str(v["n_episodes"]), f"{v['grasp_rate']*100:.1f}%",
                f"{v['pre_insert_align_rate']*100:.2f}%", f"{v['assembly_success_rate']*100:.2f}%",
                f"{v.get('mean_final_angle_deg', 0):.2f} deg",
            ])
    return pp_rows, asm_rows


def extract_phase15_ablation_ledger() -> List[List[str]]:
    entries = []
    for sub in ["5B", "5B1", "5B2", "5B3", "5B4", "5B5", "5B6", "5B7", "5B8"]:
        for jf in sorted(glob.glob(str(ROOT / "results" / f"phase15_2_{sub}" / "*.json"))):
            try:
                d = json.loads(Path(jf).read_text(encoding="utf-8"))
                summary = {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool))}
                desc = ", ".join(f"{k}={v}" for k, v in list(summary.items())[:6])
                entries.append([sub, Path(jf).name, desc[:95]])
            except Exception:
                pass
    return entries


def extract_phase17_physics_diff() -> List[List[str]]:
    d = json.loads((ROOT / "results" / "phase16_17" / "phase17_cross_simulator_results.json").read_text())
    pc = d.get("physics_comparison", {})
    return [[str(k), str(v)] for k, v in pc.items()]


def extract_all_30_yamls() -> List[List[str]]:
    rows = []
    for p in sorted(glob.glob(str(ROOT / "configs" / "**" / "*.yaml"), recursive=True)):
        rel = str(Path(p).relative_to(ROOT)).replace("\\", "/")
        y = yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}
        if not isinstance(y, dict):
            continue
        env = y.get("env", {}) if isinstance(y.get("env"), dict) else {}
        rl = y.get("rl", {}) if isinstance(y.get("rl"), dict) else {}
        rows.append([
            rel.replace("configs/", ""),
            str(list(y.keys())),
            str(env.get("max_episode_steps", env.get("task", "-"))),
            str(rl.get("learning_rate", "-")),
            str(rl.get("gamma", "-")),
            str(rl.get("total_timesteps", "-")),
        ])
    return rows


def extract_src_api_table() -> List[List[str]]:
    rows = []
    for p in sorted((ROOT / "src").rglob("*.py")):
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
            classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
            funcs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
            n_lines = len(p.read_text(encoding="utf-8", errors="ignore").splitlines())
            rows.append([
                rel,
                str(n_lines),
                ", ".join(classes[:4]) if classes else "-",
                ", ".join(funcs[:4]) if funcs else "-",
            ])
        except Exception:
            pass
    return rows


def extract_phase16_skill_table() -> List[List[str]]:
    bm = json.loads((ROOT / "results" / "phase16_17" / "phase16_benchmark_results.json").read_text(encoding="utf-8"))
    skills = bm.get("phase16_1_hrl", {}).get("skill_library_training_metrics", {})
    rows = []
    for name, d in skills.items():
        rows.append([
            name,
            str(d.get("n_samples", "-")),
            f"{d.get('actor_mse', 0.0):.6f}",
            f"{d.get('term_bce', 0.0):.6f}",
            f"{d.get('term_accuracy', 0.0)*100:.2f}%",
        ])
    return rows


def extract_full_file_inventory() -> List[List[str]]:
    inv = json.loads((PUB / "project_file_inventory.json").read_text(encoding="utf-8"))
    rows = []
    for i, f in enumerate(inv["files"], 1):
        rows.append([
            str(i),
            f["path"],
            f["category"],
            str(f["lines"]),
            f"{f['size_bytes']:,} B",
        ])
    return rows


def extract_episode_archives() -> Dict[str, List[List[str]]]:
    p11_6 = [
        [
            r["evaluation_condition"],
            r["evaluation_seed"],
            r["grasp_success"],
            r["success"],
            f"{float(r['final_distance']):.4f} m",
            f"{float(r['reward']):.1f}",
            r["collision_count"],
        ]
        for r in csv.DictReader((ROOT / "results" / "phase11_6_dense_reward_metrics.csv").open(encoding="utf-8"))
    ]
    p11_9 = [
        [
            str(i + 1),
            r["success"],
            f"{float(r['initial_dist']):.4f} m",
            f"{float(r['final_dist']):.4f} m",
            r["hold_duration"],
            r["dropped"],
        ]
        for i, r in enumerate(csv.DictReader((ROOT / "results" / "phase11_9_transport_metrics.csv").open(encoding="utf-8")))
    ]
    p15_1 = [
        [
            r["condition"][:22],
            r["training_seed"],
            r["eval_seed"],
            r["reach_success"],
            r["grasp_success"],
            r["lift_success"],
            r["transport_success"],
        ]
        for r in csv.DictReader((ROOT / "results" / "phase15_1" / "phase15_1_episodes.csv").open(encoding="utf-8"))
    ]
    p14_4 = [
        [
            r["training_seed"],
            r["regime"],
            r["controller"][:14],
            r["eval_seed"],
            r["bilateral_success"],
            r["lift_success"],
            f"{float(r['final_dist']):.4f}",
            r["slip_events"],
            r["failure_type"][:18],
        ]
        for r in csv.DictReader((ROOT / "results" / "phase14_4" / "confirmation_episodes.csv").open(encoding="utf-8"))
    ]
    return {"p11_6": p11_6, "p11_9": p11_9, "p15_1": p15_1, "p14_4": p14_4}
