"""Contact-Rich Peg-in-Hole Manipulation & Force/Compliance Telemetry (Phase 16.4).

Extends ``PandaAssemblyDenseEnv`` while preserving Phase 15 ``PandaAssemblyEnv`` and
``PandaAssemblyDenseEnv`` untouched. NOTE: Contact forces reported by PyBullet are
simulation constraint impulses and are NOT hardware-equivalent F/T sensor readings.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Tuple

import numpy as np
import pybullet as p

from src.environments.assembly_dense_env import PandaAssemblyDenseEnv


class ContactState(str, Enum):
    FREE_AIR = "FREE_AIR"
    RIM_CONTACT = "RIM_CONTACT"
    WALL_ALIGNED = "WALL_ALIGNED"
    SEATED = "SEATED"
    JAMMED = "JAMMED"


class PandaContactAssemblyEnv(PandaAssemblyDenseEnv):
    """Contact-aware peg-in-hole environment adding directional wall/axial force telemetry
    and a simulation contact-state classifier (`FREE_AIR`, `RIM_CONTACT`, `WALL_ALIGNED`,
    `SEATED`, `JAMMED`) for compliant impedance insertion and failed-insertion recovery.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.recovery_events: int = 0

    def query_rich_contacts(self) -> Dict[str, Any]:
        base_ctx = self._query_assembly_contacts()
        cid = self.sim.client_id
        peg_id = self.peg.body_id
        base_id = self.base.body_id

        peg_base = p.getContactPoints(bodyA=peg_id, bodyB=base_id, physicsClientId=cid)
        # Directional normal resultant pushing peg away from receptacle contacts (N)
        force_vec = np.zeros(3, dtype=np.float32)
        for c in peg_base:
            fn = float(c[9])
            n_on_b = np.asarray(c[7], dtype=np.float32)  # unit normal from B toward A (peg)
            force_vec += fn * n_on_b

        peg_pos = self.peg.get_position()
        base_pos = self.base.get_position()
        xy_err = float(np.linalg.norm(peg_pos[:2] - base_pos[:2]))
        insertion_depth_m = float(max(0.0, 0.115 - peg_pos[2]))

        wall_f = float(base_ctx["wall_force"])
        base_f = float(base_ctx["peg_base_force"])

        if wall_f >= 22.0 and insertion_depth_m < 0.060:
            c_state = ContactState.JAMMED
        elif peg_pos[2] <= 0.058 and xy_err <= self.distance_threshold and wall_f < 22.0:
            c_state = ContactState.SEATED
        elif len(peg_base) > 0 and peg_pos[2] > 0.068:
            c_state = ContactState.RIM_CONTACT
        elif len(peg_base) > 0:
            c_state = ContactState.WALL_ALIGNED
        else:
            c_state = ContactState.FREE_AIR

        base_ctx.update(
            {
                "contact_force_vec": force_vec,
                "xy_err_m": xy_err,
                "insertion_depth_m": insertion_depth_m,
                "contact_state": c_state.value,
                "sim_force_disclaimer": "pybullet_impulse_not_hardware_ft",
            }
        )
        return base_ctx

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        obs, rew, term, trunc, info = super().step(action)
        rich = self.query_rich_contacts()
        # Small extra compliance bonus for deeper low-force insertion without jamming
        if rich["contact_state"] == ContactState.WALL_ALIGNED.value and rich["wall_force"] < 10.0:
            rew += 0.15
        info["contact_state"] = rich["contact_state"]
        info["contact_force_vec"] = rich["contact_force_vec"].tolist()
        info["insertion_depth_m"] = round(float(rich["insertion_depth_m"]), 5)
        info["wall_force"] = round(float(rich["wall_force"]), 3)
        return obs, float(rew), term, trunc, info

