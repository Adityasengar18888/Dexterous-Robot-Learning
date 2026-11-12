"""WebSocket endpoints streaming authoritative simulation telemetry, historical training traces, and events."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.app.core.config import PUBLICATION_DIR
from backend.app.managers.simulation_manager import sim_manager

ws_router = APIRouter()


@ws_router.websocket("/ws/simulation")
async def ws_simulation(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            snap = sim_manager.get_snapshot()
            await websocket.send_json({"type": "simulation_state", "payload": snap})
            await asyncio.sleep(0.05)  # 20 Hz UI telemetry
    except WebSocketDisconnect:
        return
    except Exception:
        return


@ws_router.websocket("/ws/training")
async def ws_training(websocket: WebSocket) -> None:
    """Streams real archived TensorBoard / experiment curves (does not auto-launch GPU/CPU training)."""
    await websocket.accept()
    try:
        tb_file = PUBLICATION_DIR / "benchmark_data" / "tb_scalars_all.json"
        if tb_file.exists():
            tb = json.loads(tb_file.read_text(encoding="utf-8"))
            summary = []
            for rk, tags in sorted(tb.items()):
                if tags:
                    act = tags.get("train/actor_loss", {})
                    ent = tags.get("train/ent_coef", {})
                    fps = tags.get("time/fps", {})
                    if act.get("steps"):
                        summary.append({
                            "run": rk,
                            "steps": act["steps"][-1],
                            "actor_loss": round(act["values"][-1], 4),
                            "ent_coef": round(ent["values"][-1], 4) if ent.get("values") else None,
                            "fps": round(fps["values"][-1], 1) if fps.get("values") else None,
                        })
            await websocket.send_json({"type": "training_archive_snapshot", "runs": summary})
        while True:
            await asyncio.sleep(2.0)
            await websocket.send_json({"type": "training_heartbeat", "active_training_jobs": 0})
    except WebSocketDisconnect:
        return
    except Exception:
        return


@ws_router.websocket("/ws/events")
async def ws_events(websocket: WebSocket) -> None:
    await websocket.accept()
    last_count = 0
    try:
        while True:
            events = sim_manager.get_events()
            if len(events) != last_count or last_count == 0:
                await websocket.send_json({"type": "events_snapshot", "events": events[-50:]})
                last_count = len(events)
            await asyncio.sleep(0.35)
    except WebSocketDisconnect:
        return
    except Exception:
        return

