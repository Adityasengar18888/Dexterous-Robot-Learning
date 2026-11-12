"""FastAPI Workstation Entry Point for Dexterous-Robot-Learning."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import router
from backend.app.core.config import CORS_ORIGINS, FRANKA_ASSETS_DIR
from backend.app.managers.simulation_manager import sim_manager
from backend.app.websocket.streams import ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize simulation cleanly on startup so Franka Panda is immediately observable
    sim_manager.start()
    yield
    sim_manager.shutdown()


app = FastAPI(
    title="Dexterous-Robot-Learning Robotics Workstation API",
    version="17.0.0",
    description="Authoritative PyBullet Franka Panda Simulation, Robot Control, Task Execution, and Experiment Platform.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRANKA_ASSETS_DIR.exists():
    app.mount("/assets/franka_panda", StaticFiles(directory=str(FRANKA_ASSETS_DIR)), name="franka_assets")

app.include_router(router)
app.include_router(ws_router)

