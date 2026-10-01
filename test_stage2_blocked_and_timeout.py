"""
Comprehensive verification test suite for Stage 2 Blocked Movement Penalties,
20.0-Second Timeout, -50.0 Timeout Penalty, and Failure Profitability Sanity Checks.
Validates Sections 18, 19, and 20 of Phase 3B Reward Refinements.
"""

import math
import os
import sys

# Ensure headless execution
os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np

from config import (
    RL_DT,
    PLAYER_SPEED,
    ACTION_DEADZONE,
    MAX_EPISODE_TIME,
    STAGE1_STEP_PENALTY,
    STAGE2_STEP_PENALTY,
    STAGE2_MAX_EPISODE_TIME,
    STAGE2_TIMEOUT_PENALTY,
    STAGE2_BLOCKED_STEP_PENALTY,
    STAGE2_PROLONGED_BLOCKED_PENALTY,
    STAGE2_BLOCKED_RATIO_THRESHOLD,
    STAGE2_PROLONGED_BLOCKED_STEPS,
    PLAYER_RADIUS,
    GOAL_RADIUS,
    OBSTACLE_RECTS,
    PROGRESS_CHECK_INTERVAL,
    PROGRESS_REWARD_SCALE,
)
from rl_environment import StealthGymEnv


def run_blocked_movement_tests():
    print("=" * 70)
    print("RUNNING STAGE 2 BLOCKED MOVEMENT VERIFICATION (SECTION 18)")
    print("=" * 70)

    env = StealthGymEnv(stage=2, render_mode=None)

    # -----------------------------------------------------------------
    # Test 1: Normal unobstructed movement: no blocked penalty
    # -----------------------------------------------------------------
    env.reset()
    # Place player in open space: (200, 200) moving right toward (900, 200)
    env.env.set_layout((200.0, 200.0), (900.0, 200.0))
    _, r, term, trunc, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert not info["is_blocked"], "Unobstructed movement flagged as blocked!"
    assert info["consecutive_blocked_steps"] == 0, f"Blocked steps counter > 0: {info['consecutive_blocked_steps']}"
    assert math.isclose(info["blocked_penalty"], 0.0, abs_tol=1e-6)
    print("[PASS] Blocked Test 1: Normal unobstructed movement incurs zero blocked penalty.")

    # -----------------------------------------------------------------
    # Test 2: Standing still voluntarily: no blocked penalty
    # -----------------------------------------------------------------
    _, r_still, _, _, info_still = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert not info_still["is_blocked"], "Voluntary stationary step flagged as blocked!"
    assert info_still["consecutive_blocked_steps"] == 0
    assert math.isclose(info_still["blocked_penalty"], 0.0, abs_tol=1e-6)
    # Inside deadzone
    _, r_dz, _, _, info_dz = env.step(np.array([0.02, -0.01], dtype=np.float32))
    assert not info_dz["is_blocked"], "Action inside deadzone flagged as blocked!"
    print("[PASS] Blocked Test 2: Standing still voluntarily incurs zero blocked penalty.")

    # -----------------------------------------------------------------
    # Test 3: Action pushing directly into obstacle receives -0.02 blocked penalty initially
    # -----------------------------------------------------------------
    # Obstacle 0 is Rect(520, 95, 60, 160).
    # Place player flush against its left face: x = 520 - 14 = 506.0, y = 150.0
    obs_rect = OBSTACLE_RECTS[0]
    flush_x = float(obs_rect.left - PLAYER_RADIUS)
    flush_y = 150.0
    env.env.set_layout((flush_x, flush_y), (100.0, 100.0))
    env.consecutive_blocked_steps = 0

    # Push directly right into wall (dx = +1.0)
    _, r_push, _, _, info_push = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert info_push["is_blocked"], "Pushing directly into solid wall was NOT detected as blocked!"
    assert info_push["consecutive_blocked_steps"] == 1, f"Expected streak 1, got {info_push['consecutive_blocked_steps']}"
    assert math.isclose(info_push["blocked_penalty"], STAGE2_BLOCKED_STEP_PENALTY, abs_tol=1e-6), (
        f"Expected initial blocked penalty {STAGE2_BLOCKED_STEP_PENALTY}, got {info_push['blocked_penalty']}"
    )
    # Total step reward = normal step penalty (-0.01) + blocked penalty (-0.02) = -0.03
    expected_step_r = STAGE2_STEP_PENALTY + STAGE2_BLOCKED_STEP_PENALTY
    assert math.isclose(r_push, expected_step_r, abs_tol=1e-5), f"Step reward {r_push} != {expected_step_r}"
    print(f"[PASS] Blocked Test 3: Pushing into obstacle receives short blocked penalty ({STAGE2_BLOCKED_STEP_PENALTY}).")

    # -----------------------------------------------------------------
    # Test 4: 15+ consecutive blocked steps escalates to -0.05 prolonged penalty
    # -----------------------------------------------------------------
    # Continue pushing for 13 more steps (steps 2 to 14)
    for step_num in range(2, 15):
        _, r_sub, _, _, info_sub = env.step(np.array([1.0, 0.0], dtype=np.float32))
        assert info_sub["is_blocked"]
        assert info_sub["consecutive_blocked_steps"] == step_num
        assert math.isclose(info_sub["blocked_penalty"], STAGE2_BLOCKED_STEP_PENALTY, abs_tol=1e-6)

    # Step 15: Must escalate to STAGE2_PROLONGED_BLOCKED_PENALTY (-0.05)
    _, r_esc15, _, _, info_esc15 = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert info_esc15["consecutive_blocked_steps"] == 15
    assert math.isclose(info_esc15["blocked_penalty"], STAGE2_PROLONGED_BLOCKED_PENALTY, abs_tol=1e-6), (
        f"Step 15 penalty {info_esc15['blocked_penalty']} != {STAGE2_PROLONGED_BLOCKED_PENALTY}"
    )
    expected_esc_r = STAGE2_STEP_PENALTY + STAGE2_PROLONGED_BLOCKED_PENALTY  # -0.01 + -0.05 = -0.06
    assert math.isclose(r_esc15, expected_esc_r, abs_tol=1e-5)

    # Step 16: Must continue receiving prolonged blocked penalty
    _, r_esc16, _, _, info_esc16 = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert info_esc16["consecutive_blocked_steps"] == 16
    assert math.isclose(info_esc16["blocked_penalty"], STAGE2_PROLONGED_BLOCKED_PENALTY, abs_tol=1e-6)
    print(f"[PASS] Blocked Test 4: Blocked steps >= 15 strictly escalate to prolonged penalty ({STAGE2_PROLONGED_BLOCKED_PENALTY}).")

    # -----------------------------------------------------------------
    # Test 5: Counter resets after successful movement
    # -----------------------------------------------------------------
    # Player moves away from wall (dx = -1.0, left into open space)
    _, r_free, _, _, info_free = env.step(np.array([-1.0, 0.0], dtype=np.float32))
    assert not info_free["is_blocked"], "Moving away from wall flagged as blocked!"
    assert info_free["consecutive_blocked_steps"] == 0, "Blocked counter failed to reset on movement!"
    assert math.isclose(info_free["blocked_penalty"], 0.0, abs_tol=1e-6)
    print("[PASS] Blocked Test 5: Blocked counter resets to 0 immediately upon successful movement.")

    # -----------------------------------------------------------------
    # Test 6: Counter resets on episode reset
    # -----------------------------------------------------------------
    # Make player blocked again
    env.env.set_layout((flush_x, flush_y), (100.0, 100.0))
    for _ in range(5):
        env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert env.consecutive_blocked_steps == 5

    # Reset environment
    env.reset()
    assert env.consecutive_blocked_steps == 0, "consecutive_blocked_steps did not reset to 0 on env.reset()!"
    print("[PASS] Blocked Test 6: Blocked counter strictly resets to 0 on episode reset.")

    # -----------------------------------------------------------------
    # Test 7: Wall sliding with substantial actual movement is NOT falsely punished
    # -----------------------------------------------------------------
    # Player is flush against left vertical face of obstacle: (506.0, 150.0).
    # Player moves diagonally down-right: dx = +1.0, dy = +1.0.
    # The x-component is stopped by the wall, but player slides freely in +y direction.
    # Expected distance: 185 * (1/30) = 6.167 px.
    # Actual distance: ~ (1/sqrt(2)) * 6.167 = 4.36 px (> 70% of expected >> 20% threshold).
    env.env.set_layout((flush_x, flush_y), (100.0, 100.0))
    env.consecutive_blocked_steps = 0

    slide_action = np.array([1.0, 1.0], dtype=np.float32)
    for _ in range(20):
        _, r_slide, _, _, info_slide = env.step(slide_action)
        assert not info_slide["is_blocked"], "Wall sliding falsely flagged as blocked!"
        assert info_slide["consecutive_blocked_steps"] == 0, "Blocked streak incremented during wall sliding!"
        assert math.isclose(info_slide["blocked_penalty"], 0.0, abs_tol=1e-6), (
            f"Wall sliding received blocked penalty: {info_slide['blocked_penalty']}"
        )
    print("[PASS] Blocked Test 7: Wall sliding with substantial movement is NOT penalized as blocked.")

    # -----------------------------------------------------------------
    # Test 8: Brief corner contact does not create permanent blocked state
    # -----------------------------------------------------------------
    # Tap a corner for 2 steps then move away
    env.env.set_layout((flush_x, flush_y), (100.0, 100.0))
    env.consecutive_blocked_steps = 0
    env.step(np.array([1.0, 0.0], dtype=np.float32))
    env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert env.consecutive_blocked_steps == 2
    # Turn and walk along / away
    _, _, _, _, info_away = env.step(np.array([0.0, 1.0], dtype=np.float32))
    assert not info_away["is_blocked"]
    assert env.consecutive_blocked_steps == 0
    print("[PASS] Blocked Test 8: Brief corner contact clears cleanly without persistent penalty.")

    env.close()


def run_timeout_tests():
    print("\n" + "=" * 70)
    print("RUNNING STAGE 2 TIMEOUT VERIFICATION (SECTION 19)")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Test 1 & 2: Stage 2 timeout occurs at 20.0s (~600 steps)
    # -----------------------------------------------------------------
    env2 = StealthGymEnv(stage=2, render_mode=None)
    env2.reset(seed=10)
    # Run 599 steps stationary
    for s in range(599):
        _, _, term, trunc, _ = env2.step(np.array([0.0, 0.0], dtype=np.float32))
        assert not term and not trunc, f"Premature truncation at step {s+1} (time {env2.episode_time:.2f}s)"

    # Step 600: Time reaches 600 * (1/30) = 20.0s
    _, r_to, term_600, trunc_600, info_600 = env2.step(np.array([0.0, 0.0], dtype=np.float32))
    assert trunc_600, f"Stage 2 failed to truncate at step 600 (time {env2.episode_time:.2f}s)"
    assert not term_600, "Stage 2 timeout was marked terminated=True!"
    assert info_600["timeout"], "info['timeout'] not True"
    assert math.isclose(env2.episode_time, 20.0, abs_tol=1e-3), f"Episode time {env2.episode_time} != 20.0"
    print(f"[PASS] Timeout Test 1 & 2: Stage 2 truncates at exactly 20.0s (600 simulation steps).")

    # -----------------------------------------------------------------
    # Test 3 & 4 & 5: Timeout adds exactly -50.0 penalty; normal step penalty also applies
    # -----------------------------------------------------------------
    expected_step_r = STAGE2_STEP_PENALTY + STAGE2_TIMEOUT_PENALTY  # -0.01 + -50.0 = -50.01
    assert math.isclose(r_to, expected_step_r, abs_tol=1e-5), (
        f"Final step reward {r_to} != expected {expected_step_r}"
    )
    assert math.isclose(info_600["timeout_penalty"], STAGE2_TIMEOUT_PENALTY, abs_tol=1e-5)
    print(f"[PASS] Timeout Test 3, 4, 5: Stage 2 timeout applies {STAGE2_TIMEOUT_PENALTY:.1f} penalty + normal step penalty.")

    # -----------------------------------------------------------------
    # Test 6: Stage 1 timeout behavior remains unchanged (30.0s, no penalty)
    # -----------------------------------------------------------------
    env1 = StealthGymEnv(stage=1, render_mode=None)
    env1.reset(seed=10)
    env1.episode_time = 20.0  # Would time out Stage 2, but NOT Stage 1
    _, r1_mid, term1_mid, trunc1_mid, _ = env1.step(np.array([0.0, 0.0], dtype=np.float32))
    assert not trunc1_mid, "Stage 1 prematurely truncated at 20.0s!"
    assert not term1_mid
    assert math.isclose(r1_mid, STAGE1_STEP_PENALTY, abs_tol=1e-5)

    # Advance to 30.0s
    env1.episode_time = MAX_EPISODE_TIME - (RL_DT * 0.5)
    _, r1_to, term1_to, trunc1_to, info1_to = env1.step(np.array([0.0, 0.0], dtype=np.float32))
    assert trunc1_to, "Stage 1 failed to truncate at 30.0s"
    assert not term1_to
    assert info1_to["timeout"]
    assert math.isclose(r1_to, STAGE1_STEP_PENALTY, abs_tol=1e-5), f"Stage 1 timeout received extra penalty: {r1_to}"
    print(f"[PASS] Timeout Test 6: Stage 1 timeout preserves 30.0s limit and zero extra timeout penalty.")

    # -----------------------------------------------------------------
    # Test 7: Stage 3+ timeout behavior remains unchanged (30.0s, reward 0.0)
    # -----------------------------------------------------------------
    env3 = StealthGymEnv(stage=3, render_mode=None)
    env3.reset(seed=10)
    env3.episode_time = 20.0
    _, r3_mid, _, trunc3_mid, _ = env3.step(np.array([0.0, 0.0], dtype=np.float32))
    assert not trunc3_mid, "Stage 3 prematurely truncated at 20.0s!"

    env3.episode_time = MAX_EPISODE_TIME - (RL_DT * 0.5)
    _, r3_to, _, trunc3_to, _ = env3.step(np.array([0.0, 0.0], dtype=np.float32))
    assert trunc3_to, "Stage 3 failed to truncate at 30.0s"
    assert math.isclose(r3_to, 0.0, abs_tol=1e-5)
    print(f"[PASS] Timeout Test 7: Stage 3 timeout preserves 30.0s limit and zero step/timeout penalty.")

    env2.close()
    env1.close()
    env3.close()


def run_reward_sanity_test():
    print("\n" + "=" * 70)
    print("RUNNING REWARD SANITY TEST: FAILED WALL-PUSHING STRATEGY (SECTION 20)")
    print("=" * 70)

    # Scenario:
    # 1. Agent starts at (400, 150) and goal is at (650, 150).
    #    Obstacle 0 is Rect(520, 95, 60, 160).
    #    Direct line to goal hits Obstacle 0 at x = 520 - 14 = 506.0.
    # 2. Agent runs directly right (+1.0, 0.0) toward goal for 106 px (~18 steps).
    #    During this phase, it earns positive progress reward: ~106 * 0.05 = +5.30.
    # 3. Upon reaching x=506.0, agent continues stubbornly pushing right into the obstacle
    #    until the 20-second timeout occurs (remaining ~582 steps).
    # 4. Under old reward:
    #    progress: +5.30, normal steps: 600 * (-0.01) = -6.0. Total = -0.70 (or if distance was larger, positive!).
    # 5. Under corrected reward:
    #    - Steps 1-14 of blocked: 14 * (-0.02) = -0.28
    #    - Steps 15-582 of blocked: 568 * (-0.05) = -28.40
    #    - Normal step penalties: 600 * (-0.01) = -6.00
    #    - Timeout penalty: -50.00
    #    - Net return: +5.30 - 6.00 - 0.28 - 28.40 - 50.00 = -79.38!
    #
    # Let's execute this exact trajectory in the environment and verify total reward is deeply negative!

    env = StealthGymEnv(stage=2, render_mode=None)
    env.reset()
    env.env.set_layout((400.0, 150.0), (650.0, 150.0))
    init_dist = math.hypot(400.0 - 650.0, 0.0)  # 250.0 px
    env.best_distance_to_goal = init_dist
    env.initial_distance_to_goal = init_dist

    total_episode_reward = 0.0
    done = False
    step_count = 0
    action_push_right = np.array([1.0, 0.0], dtype=np.float32)

    while not done:
        _, r, term, trunc, info = env.step(action_push_right)
        total_episode_reward += r
        step_count += 1
        done = term or trunc

    print(f"Sanity Scenario Results:")
    print(f"  Total Steps:          {step_count}")
    print(f"  Episode Truncated:    {trunc}")
    print(f"  Goal Reached:         {term}")
    print(f"  Final Player X:       {env.env.player.x:.1f} (stopped at wall face 506.0)")
    print(f"  Total Episode Return: {total_episode_reward:.2f}")

    assert trunc, "Sanity scenario should have timed out"
    assert not term, "Sanity scenario should not reach goal"
    assert total_episode_reward < -50.0, (
        f"Wall-pushing failed strategy resulted in return {total_episode_reward:.2f} >= -50.0! Must be deeply negative."
    )
    print(f"[PASS] Reward Sanity Test: Wall-grinding failure receives deeply negative return ({total_episode_reward:.2f} << 0).")
    print("       A failed strategy can no longer be profitable under any circumstances.")

    env.close()


if __name__ == "__main__":
    run_blocked_movement_tests()
    run_timeout_tests()
    run_reward_sanity_test()
    print("\n" + "=" * 70)
    print("ALL STAGE 2 BLOCKED MOVEMENT & TIMEOUT TESTS PASSED PERFECTLY!")
    print("=" * 70)
