"""Service reading verified repository experiments, TensorBoard logs, evaluations, and policy checkpoints."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from backend.app.core.config import (
    CHECKPOINTS_DIR,
    EXPERIMENTS_DIR,
    PUBLICATION_DIR,
    RESULTS_DIR,
    ROOT_DIR,
)


POLICY_CATALOG: List[Dict[str, Any]] = [
    {
        "policy_id": "scripted_pregrasp_multisort",
        "name": "Phase 15.2.5A Scripted PreGrasp Two-Object Controller",
        "provenance": "SCRIPTED",
        "algorithm": "5-Check Bilateral Gate FSM",
        "compatible_envs": ["PandaMultiSortPreGraspEnv", "PandaMultiSortEnv", "PandaPickPlaceEnv"],
        "obs_dim": 31,
        "goal_dim": 6,
        "action_dim": 4,
        "checkpoint_path": "src/controllers/pregrasp_controller.py",
        "verified_benchmark": "50/50 (100.0%) Two-Object Sorting, 76.4 steps, 0 colls",
    },
    {
        "policy_id": "scripted_compliant_assembly",
        "name": "Phase 16.4 Compliant Force-Impedance Assembly Controller",
        "provenance": "SCRIPTED",
        "algorithm": "Compliant Wall-Force Insertion + Jam Recovery",
        "compatible_envs": ["PandaContactAssemblyEnv", "PandaAssemblyDenseEnv", "PandaAssemblyEnv"],
        "obs_dim": 22,
        "goal_dim": 7,
        "action_dim": 5,
        "checkpoint_path": "src/controllers/compliant_assembly_controller.py",
        "verified_benchmark": "53.33% Full Insertion (pos_err 0.0468 m), 22 recoveries",
    },
    {
        "policy_id": "sac_dr4_seed42_adaptive_e",
        "name": "Phase 13/14 SAC+HER DR4 Seed 42 + Adaptive Transport Shield (Ctrl E)",
        "provenance": "LEARNED_SAC_HER",
        "algorithm": "SAC + HER (k=4, future) + E_ADAPTIVE_TRANSPORT",
        "compatible_envs": ["PandaPickPlaceDenseRewardEnv", "PandaPickPlaceEnv", "PandaPickEnv"],
        "obs_dim": 10,
        "goal_dim": 3,
        "action_dim": 4,
        "checkpoint_path": "experiments/phase13_dense_dr/dr4/seed42/models/checkpoint_500000_steps.zip",
        "verified_benchmark": "97.0% Held-out DR4, 100.0% Nominal DR0 (Combined E: 98.0%)",
    },
    {
        "policy_id": "sac_dr4_seed43_adaptive_e",
        "name": "Phase 13/14 SAC+HER DR4 Seed 43 + Adaptive Transport Shield (Ctrl E)",
        "provenance": "LEARNED_SAC_HER",
        "algorithm": "SAC + HER (k=4, future) + E_ADAPTIVE_TRANSPORT",
        "compatible_envs": ["PandaPickPlaceDenseRewardEnv", "PandaPickPlaceEnv"],
        "obs_dim": 10,
        "goal_dim": 3,
        "action_dim": 4,
        "checkpoint_path": "experiments/phase13_dense_dr/dr4/seed43/models/checkpoint_500000_steps.zip",
        "verified_benchmark": "99.0% Held-out DR4 (33% -> 99% under Ctrl E), 100% DR0",
    },
    {
        "policy_id": "bc_5b7_pregrasp_masked",
        "name": "Phase 15.2.5B7 Progress-Masked Behavioral Cloning",
        "provenance": "BEHAVIOR_CLONING",
        "algorithm": "Filtered L2 BC (Open-Approach Balanced Demos)",
        "compatible_envs": ["PandaSinglePreGraspEnv"],
        "obs_dim": 22,
        "goal_dim": 3,
        "action_dim": 4,
        "checkpoint_path": "experiments/multi_sort_phase15_5B7/seed42/models/bc_warmstart_model.zip",
        "verified_benchmark": "75.0% Grasp / 45.0% Lift / 0.0% Table Collisions (100 eps)",
    },
    {
        "policy_id": "hybrid_5b8_bc_sac",
        "name": "Phase 15.2.5B8 Decaying Hybrid BC-SAC + HER Guard",
        "provenance": "HYBRID_BC_SAC",
        "algorithm": "BCDecayingSAC + gate_goal_lift_on_lifted_dg",
        "compatible_envs": ["PandaSinglePreGraspEnv"],
        "obs_dim": 22,
        "goal_dim": 3,
        "action_dim": 4,
        "checkpoint_path": "experiments/multi_sort_phase15_5B8/seed42/models/final_model.zip",
        "verified_benchmark": "66.0% Grasp / 26.0% Lift (100 eps)",
    },
]


def _resolve_best_checkpoint(rel_path: str) -> Path:
    p = ROOT_DIR / rel_path
    if p.exists():
        return p
    # Check fallback candidate names inside same models directory
    if p.parent.exists():
        for cand in ["best_model.zip", "final_model.zip", "checkpoint_500000.zip"]:
            cp = p.parent / cand
            if cp.exists():
                return cp
    return p


def list_policies(active_env: str = "PandaPickPlaceEnv") -> List[Dict[str, Any]]:
    out = []
    for entry in POLICY_CATALOG:
        resolved = _resolve_best_checkpoint(entry["checkpoint_path"])
        exists = resolved.exists()
        compat = active_env in entry["compatible_envs"]
        out.append({
            **entry,
            "resolved_path": str(resolved.relative_to(ROOT_DIR)).replace("\\", "/") if exists else entry["checkpoint_path"],
            "exists": exists,
            "file_size_kb": round(resolved.stat().st_size / 1024.0, 1) if exists else 0.0,
            "compatible_with_active_env": compat,
            "dependencies_ok": True,
        })
    return out


def get_experiments() -> List[Dict[str, Any]]:
    csv_path = PUBLICATION_DIR / "experiment_inventory.csv"
    if not csv_path.exists():
        return []
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            # Distinguish scripted vs learned provenance cleanly
            method_up = r["method"].upper()
            prov = "SCRIPTED" if "SCRIPTED" in method_up else (
                "HYBRID_BC_SAC" if "BC" in method_up and "SAC" in method_up else (
                    "BEHAVIOR_CLONING" if "BC" in method_up else "LEARNED_SAC_HER"
                )
            )
            rows.append({**r, "provenance": prov})
    return rows


def get_phase_matrix() -> List[Dict[str, Any]]:
    p = PUBLICATION_DIR / "phase_completion_matrix.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("completed_or_partial_phases", [])


def get_debugging_history() -> List[Dict[str, Any]]:
    p = PUBLICATION_DIR / "debugging_history.csv"
    if not p.exists():
        return []
    return list(csv.DictReader(p.open("r", encoding="utf-8")))


def get_evaluation_detail(experiment_id: str) -> Optional[Dict[str, Any]]:
    exps = {e["exp_id"]: e for e in get_experiments()}
    if experiment_id not in exps:
        return None
    meta = exps[experiment_id]

    detail: Dict[str, Any] = {"metadata": meta, "series": [], "summary": {}}

    if experiment_id == "EXP-P12-1M" and (RESULTS_DIR / "phase12_1m" / "checkpoint_metrics.csv").exists():
        detail["series"] = list(csv.DictReader((RESULTS_DIR / "phase12_1m" / "checkpoint_metrics.csv").open()))
    elif experiment_id == "EXP-P14-4" and (RESULTS_DIR / "phase14_4" / "phase14_4_confirmation_results.json").exists():
        detail["summary"] = json.loads((RESULTS_DIR / "phase14_4" / "phase14_4_confirmation_results.json").read_text())
    elif experiment_id == "EXP-P15-1" and (RESULTS_DIR / "phase15_1" / "phase15_1_results.json").exists():
        detail["summary"] = json.loads((RESULTS_DIR / "phase15_1" / "phase15_1_results.json").read_text())
    elif experiment_id == "EXP-P11-SPARSE" and (RESULTS_DIR / "phase11_final_summary.csv").exists():
        detail["series"] = list(csv.DictReader((RESULTS_DIR / "phase11_final_summary.csv").open()))[:20]

    # Check matching TensorBoard scalars archive if available
    tb_p = PUBLICATION_DIR / "benchmark_data" / "tb_scalars_all.json"
    if tb_p.exists() and not detail["series"]:
        tb = json.loads(tb_p.read_text(encoding="utf-8"))
        # Attach closest matching run curve
        for k, tags in tb.items():
            if experiment_id.split("-")[-1].lower() in k.lower() and tags:
                act = tags.get("train/actor_loss", {})
                if act:
                    detail["tb_run"] = k
                    detail["series"] = [
                        {"step": s, "actor_loss": v}
                        for s, v in zip(act.get("steps", [])[:60], act.get("values", [])[:60])
                    ]
                    break
    return detail

