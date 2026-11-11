"""Generate validated orientation-aware assembly demonstrations for Phase 4 SAC+HER."""

from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, List

import numpy as np

from src.controllers.assembly_controller import ScriptedAssemblyController
from src.environments.assembly_dense_env import PandaAssemblyDenseEnv


def generate_assembly_demos(
    out_dir: str = "experiments/assembly_phase15_3/seed42/demonstrations",
    target_episodes: int = 35,
    start_seed: int = 8000,
    max_attempts: int = 75,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    env = PandaAssemblyDenseEnv(render_mode=None, max_episode_steps=120)
    ctrl = ScriptedAssemblyController(action_scale=env.action_scale, yaw_scale=env.yaw_scale)
    episodes: List[Dict[str, Any]] = []
    rejected = 0

    try:
        for offset in range(max_attempts):
            if len(episodes) >= target_episodes:
                break
            seed = start_seed + offset
            obs, info = env.reset(seed=seed)
            ctrl.reset()

            ep_obs, ep_nobs, ep_acts, ep_rews, ep_dones, ep_infos = [], [], [], [], [], []
            succ_steps = 0
            ep_ret = 0.0

            for _step in range(1, env.max_episode_steps + 1):
                cmd, _ = ctrl.step(env, obs, info)
                n_obs, rew, term, trunc, info = env.step(cmd)
                ep_ret += float(rew)
                if info.get("is_success", False):
                    succ_steps += 1
                done_now = bool(term or trunc or (succ_steps >= 3 and ctrl.stage.value == "DONE"))
                inf_rec = dict(info)
                inf_rec["TimeLimit.truncated"] = bool(done_now and not term)
                inf_rec["is_demo"] = True

                ep_obs.append({k: v.copy() for k, v in obs.items()})
                ep_nobs.append({k: v.copy() for k, v in n_obs.items()})
                ep_acts.append(cmd.copy().astype(np.float32))
                ep_rews.append(float(rew))
                ep_dones.append(done_now)
                ep_infos.append(inf_rec)
                obs = n_obs
                if done_now:
                    break

            if ep_infos and bool(ep_infos[-1].get("is_success", False)):
                episodes.append(
                    {
                        "seed": int(seed),
                        "length": len(ep_acts),
                        "return": float(ep_ret),
                        "observations": ep_obs,
                        "next_observations": ep_nobs,
                        "actions": ep_acts,
                        "rewards": ep_rews,
                        "dones": ep_dones,
                        "infos": ep_infos,
                    }
                )
            else:
                rejected += 1
    finally:
        env.close()

    pkl_path = os.path.join(out_dir, "assembly_demos.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(episodes, f)
    summary = {
        "accepted_episodes": len(episodes),
        "rejected_episodes": rejected,
        "total_transitions": sum(e["length"] for e in episodes),
        "mean_length": float(np.mean([e["length"] for e in episodes])) if episodes else 0.0,
        "mean_return": float(np.mean([e["return"] for e in episodes])) if episodes else 0.0,
        "dataset_path": pkl_path,
    }
    with open(os.path.join(out_dir, "assembly_demo_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    print(json.dumps(generate_assembly_demos(), indent=2))

