import os
import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from src.rl.evaluate import make_env
from src.environments.dense_pick_place_env import PandaPickPlaceDenseRewardEnv

def main():
    print("Starting Phase 11.7 Diagnostic...")
    
    with open("configs/pick_place_baseline.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    env_kwargs = config["env"].copy()
    if "name" in env_kwargs:
        del env_kwargs["name"]
    if "randomizer" in env_kwargs:
        env_kwargs["randomizer_config"] = env_kwargs.pop("randomizer")
        
    env = PandaPickPlaceDenseRewardEnv(**env_kwargs)
    
    model_path = "experiments/pnp_dr0_dense_seed42/models/final_model.zip"
    model = SAC.load(model_path, env=env)
    
    num_episodes = 100
    
    episodes_data = []
    
    for ep in range(num_episodes):
        obs, info = env.reset(seed=4200 + ep)
        done = False
        
        ep_history = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            
            ee_pos = next_info["ee_pos"]
            obj_pos = next_obs["achieved_goal"]
            target_pos = next_obs["desired_goal"]
            gripper_width = next_obs["observation"][3]
            
            ee_obj_dist = next_info["reach_dist"]
            obj_target_dist = next_info["transport_dist"]
            
            # Simple grasp heuristic: EE is close to object, gripper is somewhat closed, and Z > 0.025
            is_lifted = obj_pos[2] > 0.05
            is_grasped = ee_obj_dist < 0.05 and action[3] < 0.0
            
            ep_history.append({
                "ee_pos": ee_pos.copy(),
                "obj_pos": obj_pos.copy(),
                "target_pos": target_pos.copy(),
                "gripper_action": action[3],
                "gripper_width": gripper_width,
                "ee_obj_dist": ee_obj_dist,
                "obj_target_dist": obj_target_dist,
                "is_lifted": is_lifted,
                "is_grasped": is_grasped,
                "is_success": next_info["is_success"],
                "reach_reward": next_info["reach_reward"],
                "lift_reward": next_info["lift_reward"],
                "transport_reward": next_info["transport_reward"],
                "success_bonus": next_info["success_bonus"],
                "total_reward": next_info["total_reward"]
            })
            
            obs = next_obs
            done = terminated or truncated
            
        episodes_data.append(ep_history)
        
    env.close()
    
    print(f"Collected data for {len(episodes_data)} episodes.")
    
    # Analyze Taxonomies and Metrics
    results = {
        "NO_REACH": 0,
        "REACH_BUT_NO_GRASP": 0,
        "GRASP_BUT_NO_LIFT": 0,
        "LIFT_BUT_DROPPED": 0,
        "LIFT_AND_HELD_BUT_WRONG_DIRECTION": 0,
        "REACHED_TARGET_BUT_NO_PLACE": 0,
        "SUCCESS": 0
    }
    
    grasp_durations = []
    
    # Reward tracking
    rewards_summary = {
        "all": {"reach": 0, "lift": 0, "transport": 0, "success": 0, "total": 0, "count": 0},
        "lifted": {"reach": 0, "lift": 0, "transport": 0, "success": 0, "total": 0, "count": 0},
        "failed": {"reach": 0, "lift": 0, "transport": 0, "success": 0, "total": 0, "count": 0}
    }
    
    for ep_hist in episodes_data:
        # Determine episode taxonomy
        reached = any(step["ee_obj_dist"] < 0.05 for step in ep_hist)
        grasped = any(step["is_grasped"] for step in ep_hist)
        lifted = any(step["is_lifted"] for step in ep_hist)
        success = ep_hist[-1]["is_success"]
        
        # Calculate Grasp Durations
        current_grasp_len = 0
        max_grasp_len_ep = 0
        for step in ep_hist:
            if step["is_lifted"]: # True grasp/lift
                current_grasp_len += 1
                max_grasp_len_ep = max(max_grasp_len_ep, current_grasp_len)
            else:
                current_grasp_len = 0
        if lifted:
            grasp_durations.append(max_grasp_len_ep)
            
        # Taxonomy classification
        if success:
            results["SUCCESS"] += 1
        elif lifted:
            if max_grasp_len_ep < 5 or not ep_hist[-1]["is_lifted"]:
                results["LIFT_BUT_DROPPED"] += 1
            else:
                # Did it move toward target?
                min_obj_target = min(step["obj_target_dist"] for step in ep_hist if step["is_lifted"])
                if min_obj_target < 0.1:
                    results["REACHED_TARGET_BUT_NO_PLACE"] += 1
                else:
                    results["LIFT_AND_HELD_BUT_WRONG_DIRECTION"] += 1
        elif grasped:
            results["GRASP_BUT_NO_LIFT"] += 1
        elif reached:
            results["REACH_BUT_NO_GRASP"] += 1
        else:
            results["NO_REACH"] += 1
            
        # Rewards
        r_reach = sum(s["reach_reward"] for s in ep_hist)
        r_lift = sum(s["lift_reward"] for s in ep_hist)
        r_transport = sum(s["transport_reward"] for s in ep_hist)
        r_success = sum(s["success_bonus"] for s in ep_hist)
        r_total = sum(s["total_reward"] for s in ep_hist)
        
        cat = "all"
        rewards_summary[cat]["reach"] += r_reach
        rewards_summary[cat]["lift"] += r_lift
        rewards_summary[cat]["transport"] += r_transport
        rewards_summary[cat]["success"] += r_success
        rewards_summary[cat]["total"] += r_total
        rewards_summary[cat]["count"] += 1
        
        if lifted:
            cat = "lifted"
        else:
            cat = "failed"
            
        rewards_summary[cat]["reach"] += r_reach
        rewards_summary[cat]["lift"] += r_lift
        rewards_summary[cat]["transport"] += r_transport
        rewards_summary[cat]["success"] += r_success
        rewards_summary[cat]["total"] += r_total
        rewards_summary[cat]["count"] += 1
        
    print("\n--- Failure Taxonomy ---")
    for k, v in results.items():
        print(f"{k}: {v}")
        
    print("\n--- Grasp Stability ---")
    if grasp_durations:
        print(f"Median Hold Duration: {np.median(grasp_durations)}")
        print(f"Max Hold Duration: {np.max(grasp_durations)}")
        print(f"Surviving >5 steps: {np.mean(np.array(grasp_durations) > 5)*100:.1f}%")
        print(f"Surviving >10 steps: {np.mean(np.array(grasp_durations) > 10)*100:.1f}%")
        print(f"Surviving >25 steps: {np.mean(np.array(grasp_durations) > 25)*100:.1f}%")
    else:
        print("No lifts recorded.")
        
    print("\n--- Reward Decomposition ---")
    for cat, data in rewards_summary.items():
        if data["count"] > 0:
            c = data["count"]
            print(f"[{cat.upper()}] Reach: {data['reach']/c:.1f}, Lift: {data['lift']/c:.1f}, Transport: {data['transport']/c:.1f}, Total: {data['total']/c:.1f}")

    # Write report
    report = f"""# Phase 11.7 — Transport & Grasp Stability Diagnostic

## 1. Objective
To precisely identify the failure mechanism between successful lifting and final placement in the dense-reward agent, using deterministic evaluation instrumentation.

## 2. Failure Taxonomy (100 Episodes)
- **NO_REACH**: {results['NO_REACH']}
- **REACH_BUT_NO_GRASP**: {results['REACH_BUT_NO_GRASP']}
- **GRASP_BUT_NO_LIFT**: {results['GRASP_BUT_NO_LIFT']}
- **LIFT_BUT_DROPPED**: {results['LIFT_BUT_DROPPED']}
- **LIFT_AND_HELD_BUT_WRONG_DIRECTION**: {results['LIFT_AND_HELD_BUT_WRONG_DIRECTION']}
- **REACHED_TARGET_BUT_NO_PLACE**: {results['REACHED_TARGET_BUT_NO_PLACE']}
- **SUCCESS**: {results['SUCCESS']}

## 3. Grasp Stability Analysis
- Lifts occurred in {len(grasp_durations)} episodes.
- Median hold duration: {np.median(grasp_durations) if grasp_durations else 0} steps
- Max hold duration: {np.max(grasp_durations) if grasp_durations else 0} steps
- % surviving >5 steps: {np.mean(np.array(grasp_durations) > 5)*100 if grasp_durations else 0:.1f}%
- % surviving >25 steps: {np.mean(np.array(grasp_durations) > 25)*100 if grasp_durations else 0:.1f}%

## 4. Reward Decomposition (Averages)
- **All Episodes**: Reach={rewards_summary['all']['reach']/num_episodes:.1f}, Transport={rewards_summary['all']['transport']/num_episodes:.1f}
- **Lifted Episodes**: Reach={rewards_summary['lifted']['reach']/max(1, rewards_summary['lifted']['count']):.1f}, Lift={rewards_summary['lifted']['lift']/max(1, rewards_summary['lifted']['count']):.1f}, Transport={rewards_summary['lifted']['transport']/max(1, rewards_summary['lifted']['count']):.1f}

## 5. Diagnostic Conclusion
The dense-reward policy's primary failure point is precisely identified in the taxonomy above.
"""
    with open("docs/phase11_7_transport_stability.md", "w") as f:
        f.write(report)
        
if __name__ == "__main__":
    main()
