"""Comprehensive backend REST, PyBullet physics, IK, Policy, Task, E-Stop & WebSocket tests."""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from fastapi.testclient import TestClient
from backend.app.main import app


def test_health_status_and_assets():
    with TestClient(app) as client:
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert r.json()["physical_robot"] == "NOT_ATTEMPTED"

        r_sys = client.get("/api/system/status")
        assert r_sys.status_code == 200
        assert r_sys.json()["assets_mounted"] is True

        r_urdf = client.get("/assets/franka_panda/panda.urdf")
        assert r_urdf.status_code == 200
        assert "<robot name=\"panda\"" in r_urdf.text

        r_mesh = client.get("/assets/franka_panda/meshes/visual/hand.obj")
        assert r_mesh.status_code == 200


def test_robot_info_limits_and_joint_control():
    with TestClient(app) as client:
        info = client.get("/api/robot/info").json()
        assert len(info["joint_names"]) == 7
        assert len(info["link_specs"]) == 11

        # Out-of-limit rejection (Pydantic 422)
        bad = client.post("/api/robot/joints", json={"positions": [9.0, 0, 0, -2.0, 0, 1.5, 0.7]})
        assert bad.status_code == 422

        # Valid joint command updates PyBullet state
        target = [0.25, -0.6, 0.1, -2.1, 0.0, 1.6, 0.8]
        ok = client.post("/api/robot/joints", json={"positions": target, "steps": 60})
        assert ok.status_code == 200
        ach = ok.json()["achieved_joints"]
        assert abs(ach[0] - 0.25) < 0.05


def test_cartesian_ik_gripper_and_estop_safety():
    with TestClient(app) as client:
        client.post("/api/simulation/reset")

        cart = client.post("/api/robot/cartesian-target", json={"x": 0.45, "y": 0.10, "z": 0.25, "steps": 120})
        assert cart.status_code == 200
        data = cart.json()
        assert data["ik_converged"] is True
        assert data["position_error_m"] < 0.02

        g_close = client.post("/api/robot/gripper", json={"action": "close", "steps": 40})
        assert g_close.status_code == 200
        assert g_close.json()["actual_width"] < 0.02

        g_open = client.post("/api/robot/gripper", json={"action": "open", "steps": 40})
        assert g_open.status_code == 200
        assert g_open.json()["actual_width"] > 0.06

        # Emergency stop must block subsequent commands until reset
        estop = client.post("/api/robot/emergency-stop")
        assert estop.status_code == 200
        assert estop.json()["emergency_stopped"] is True

        blocked = client.post("/api/robot/cartesian-target", json={"x": 0.4, "y": 0.0, "z": 0.3})
        assert blocked.status_code == 409

        reset = client.post("/api/simulation/reset")
        assert reset.status_code == 200
        assert reset.json()["emergency_stopped"] is False


def test_pause_resume_tasks_policies_experiments_and_ws():
    with TestClient(app) as client:
        assert client.post("/api/simulation/pause").json()["status"] == "PAUSED"
        assert client.post("/api/simulation/resume").json()["status"] == "RUNNING"

        tasks = client.get("/api/tasks").json()["tasks"]
        assert any(t["task_id"] == "pick_place" and t["status"] == "IMPLEMENTED" for t in tasks)
        assert any(t["task_id"] == "physical_fci_execution" and t["status"] == "NOT IMPLEMENTED" for t in tasks)

        not_impl = client.post("/api/tasks/execute", json={"task_id": "physical_fci_execution"})
        assert not_impl.status_code == 501

        run_task = client.post("/api/tasks/execute", json={"task_id": "reach_target", "controller_mode": "SCRIPTED", "max_steps": 40})
        assert run_task.status_code == 200
        assert client.post("/api/tasks/stop").status_code == 200

        pols = client.get("/api/policies").json()["policies"]
        assert len(pols) >= 5
        load_sac = client.post("/api/policies/load", json={"policy_id": "sac_dr4_seed42_adaptive_e"})
        assert load_sac.status_code == 200
        assert load_sac.json()["policy"]["provenance"] == "LEARNED_SAC_HER"

        exps = client.get("/api/experiments").json()["experiments"]
        assert len(exps) >= 20
        ev = client.get("/api/evaluation/EXP-P12-1M")
        assert ev.status_code == 200
        assert len(ev.json()["series"]) >= 5
        assert client.get("/api/evaluation/DOES_NOT_EXIST").status_code == 404

        with client.websocket_connect("/ws/simulation") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "simulation_state"
            assert len(msg["payload"]["link_transforms"]) == 11
            assert len(msg["payload"]["joint_positions"]) == 7

        with client.websocket_connect("/ws/events") as ws_ev:
            ev_msg = ws_ev.receive_json()
            assert ev_msg["type"] == "events_snapshot"

