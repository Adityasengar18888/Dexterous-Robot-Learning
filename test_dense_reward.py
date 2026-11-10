import numpy as np
from src.environments.dense_pick_place_env import PandaPickPlaceDenseRewardEnv

def main():
    env = PandaPickPlaceDenseRewardEnv(render_mode=None)
    
    # 1. EE far from object
    achieved = np.array([0.5, 0.0, 0.025])
    desired = np.array([0.5, 0.2, 0.025])
    info = {"ee_pos": np.array([0.5, -0.2, 0.2])}
    r1 = env.compute_reward(achieved, desired, info)
    print(f"1. EE far: {r1}")
    
    # 2. EE moving toward object
    info["ee_pos"] = np.array([0.5, -0.1, 0.1])
    r2 = env.compute_reward(achieved, desired, info)
    print(f"2. EE closer: {r2} (Difference: {r2 - r1} -> should be positive)")
    assert r2 > r1
    
    # 3. EE moving away
    info["ee_pos"] = np.array([0.5, -0.3, 0.3])
    r3 = env.compute_reward(achieved, desired, info)
    print(f"3. EE away: {r3} (Difference: {r3 - r2} -> should be negative)")
    assert r3 < r2
    
    # 4. Valid grasp (but not lifted) -> EE is on object
    info["ee_pos"] = achieved.copy()
    r4 = env.compute_reward(achieved, desired, info)
    print(f"4. Grasp (on object): {r4}")
    
    # 5. Object lifted
    achieved_lifted = np.array([0.5, 0.0, 0.1])
    info["ee_pos"] = achieved_lifted.copy()
    r5 = env.compute_reward(achieved_lifted, desired, info)
    print(f"5. Lifted: {r5} (Difference: {r5 - r4} -> should be positive, around +1.0)")
    assert r5 > r4
    
    # 6. Object transported toward target
    achieved_closer = np.array([0.5, 0.1, 0.1])
    info["ee_pos"] = achieved_closer.copy()
    r6 = env.compute_reward(achieved_closer, desired, info)
    print(f"6. Transport closer: {r6} (Difference: {r6 - r5} -> should be positive)")
    assert r6 > r5
    
    # 7. Object transported away
    achieved_away = np.array([0.5, -0.1, 0.1])
    info["ee_pos"] = achieved_away.copy()
    r7 = env.compute_reward(achieved_away, desired, info)
    print(f"7. Transport away: {r7} (Difference: {r7 - r6} -> should be negative)")
    assert r7 < r6
    
    # 8. Successful placement
    achieved_success = desired.copy()
    info["ee_pos"] = achieved_success.copy()
    r8 = env.compute_reward(achieved_success, desired, info)
    print(f"8. Success: {r8} (Difference: {r8 - r6} -> should be highly positive)")
    assert r8 > r6
    
    # HER compatibility test (passing list of dicts)
    print("\nTesting HER compatibility (list of dicts):")
    achieved_batch = np.array([achieved, achieved_lifted])
    desired_batch = np.array([desired, desired])
    info_batch = [{"ee_pos": achieved}, {"ee_pos": achieved_lifted}]
    r_batch = env.compute_reward(achieved_batch, desired_batch, info_batch)
    print(f"Batch reward: {r_batch}")
    assert np.allclose(r_batch, [r4, r5])

    print("ALL TESTS PASSED")

if __name__ == "__main__":
    main()
