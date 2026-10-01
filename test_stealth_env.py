"""
Comprehensive automated verification suite for the 14 Acceptance Tests of Phase 1.
Runs headless simulation steps to assert physics, collision, LOS, and determinism.
"""

import math
import os
import sys

# Ensure headless pygame mode for CI / test execution
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
from config import (
    PLAYER_SPEED,
    PLAYER_START_POS,
    OBSTACLE_RECTS,
    PLAY_AREA,
    OBSERVER_CONFIGS,
    VISION_DISTANCE,
    VISION_ANGLE_RAD,
)
from environment import StealthEnvironment
from geometry import check_line_of_sight, normalize_vector


def run_acceptance_tests():
    print("=" * 60)
    print("RUNNING ACCEPTANCE TEST SUITE - PHASE 1 STEALTH SANDBOX")
    print("=" * 60)

    # -------------------------------------------------------------
    # Test 1: Program launches and initializes without errors
    # -------------------------------------------------------------
    env = StealthEnvironment()
    assert env.player is not None, "Player not initialized"
    assert len(env.observers) == 3, f"Expected 3 observers, got {len(env.observers)}"
    print("[PASS] Test 1: Program initializes and loads all components cleanly.")

    # -------------------------------------------------------------
    # Test 2: Player responds to directional input
    # -------------------------------------------------------------
    env.reset()
    init_x, init_y = env.player.pos
    dt = 0.1
    # Move right (D / Right Arrow)
    env.step((1.0, 0.0), dt)
    assert env.player.x > init_x, "Player did not move right"
    assert math.isclose(env.player.y, init_y, abs_tol=1e-3), "Player y changed during horizontal move"
    print("[PASS] Test 2: Player moves correctly with directional input.")

    # -------------------------------------------------------------
    # Test 3: Diagonal movement is normalized (not faster)
    # -------------------------------------------------------------
    env.reset()
    # Move pure horizontal for 1 second
    env.player.x, env.player.y = 300.0, 200.0
    p1_start = env.player.pos
    env.player.move((1.0, 0.0), 1.0, [], env.play_area)
    dist_cardinal = math.hypot(env.player.x - p1_start[0], env.player.y - p1_start[1])

    # Move diagonal (1.0, 1.0) for 1 second
    env.player.x, env.player.y = 300.0, 200.0
    p2_start = env.player.pos
    env.player.move((1.0, 1.0), 1.0, [], env.play_area)
    dist_diagonal = math.hypot(env.player.x - p2_start[0], env.player.y - p2_start[1])

    assert math.isclose(dist_cardinal, PLAYER_SPEED, rel_tol=1e-3), f"Cardinal speed mismatch: {dist_cardinal}"
    assert math.isclose(dist_diagonal, PLAYER_SPEED, rel_tol=1e-3), f"Diagonal speed mismatch: {dist_diagonal}"
    assert math.isclose(dist_cardinal, dist_diagonal, rel_tol=1e-3), "Diagonal speed is faster than cardinal!"
    print(f"[PASS] Test 3: Diagonal movement normalization verified (dist={dist_diagonal:.2f}px/s).")

    # -------------------------------------------------------------
    # Test 4: Player cannot cross obstacles (solid collision + wall sliding)
    # -------------------------------------------------------------
    env.reset()
    # Position player directly left of Left Horizontal Wall: Rect(175, 320, 250, 60)
    obs = OBSTACLE_RECTS[2]
    env.player.x = obs.left - env.player.radius - 5.0
    env.player.y = obs.centery
    # Push into the obstacle aggressively for 2 seconds
    for _ in range(100):
        env.player.move((1.0, 0.0), 0.02, env.obstacles, env.play_area)
    # Verify player remains outside the obstacle
    assert env.player.x <= obs.left - env.player.radius + 1e-4, "Player penetrated obstacle on X!"
    assert not env.player.get_bounding_rect().colliderect(obs), "Player bounding box intersects obstacle!"

    # Test wall sliding: moving diagonally (1.0, 1.0) against the wall moves down along Y without penetrating X
    y_before = env.player.y
    env.player.move((1.0, 1.0), 0.1, env.obstacles, env.play_area)
    assert env.player.x <= obs.left - env.player.radius + 1e-4, "Player penetrated X during slide!"
    assert env.player.y > y_before, "Player failed to slide along Y axis!"
    print("[PASS] Test 4: Obstacle collision and wall sliding verified without penetration.")

    # -------------------------------------------------------------
    # Test 5: Player cannot leave the room (outer boundaries)
    # -------------------------------------------------------------
    env.reset()
    # Move violently past left boundary
    env.player.x = env.play_area.left + env.player.radius
    for _ in range(50):
        env.player.move((-1.0, 0.0), 0.05, env.obstacles, env.play_area)
    assert env.player.x >= env.play_area.left + env.player.radius - 1e-4, "Player breached left boundary!"

    # Move violently past bottom boundary
    env.player.y = env.play_area.bottom - env.player.radius
    for _ in range(50):
        env.player.move((0.0, 1.0), 0.05, env.obstacles, env.play_area)
    assert env.player.y <= env.play_area.bottom - env.player.radius + 1e-4, "Player breached bottom boundary!"
    print("[PASS] Test 5: Room outer boundary constraints verified on all axes.")

    # -------------------------------------------------------------
    # Test 6: All three observers move continuously on fixed patrol paths
    # -------------------------------------------------------------
    env.reset()
    obs_initial_positions = [obs.pos for obs in env.observers]
    for _ in range(60):
        env.step((0.0, 0.0), 0.05)
    for i, obs in enumerate(env.observers):
        curr_pos = obs.pos
        init_pos = obs_initial_positions[i]
        moved_dist = math.hypot(curr_pos[0] - init_pos[0], curr_pos[1] - init_pos[1])
        assert moved_dist > 5.0, f"Observer {obs.name} did not move (dist={moved_dist})"
    print("[PASS] Test 6: All three observers move continuously on their patrol loops.")

    # -------------------------------------------------------------
    # Test 7: Vision cones rotate according to facing direction
    # -------------------------------------------------------------
    obs = env.observers[0]
    # Check that facing angle matches movement direction
    next_wp = obs.patrol_points[obs.target_wp_index]
    expected_angle = math.atan2(next_wp[1] - obs.y, next_wp[0] - obs.x)
    assert math.isclose(obs.facing_angle, expected_angle, abs_tol=1e-3), "Observer vision angle does not match heading!"
    print("[PASS] Test 7: Observer vision cones rotate accurately with facing direction.")

    # -------------------------------------------------------------
    # Test 8: Entering an unobstructed vision cone triggers detection
    # -------------------------------------------------------------
    env.reset()
    obs = env.observers[0]
    # Place player directly in front of observer 1 within vision cone and clear LOS
    front_dist = 80.0
    env.player.x = obs.x + math.cos(obs.facing_angle) * front_dist
    env.player.y = obs.y + math.sin(obs.facing_angle) * front_dist
    env.step((0.0, 0.0), 0.01)
    assert env.state == StealthEnvironment.STATE_DETECTED, "Unobstructed player was NOT detected!"
    assert env.detected_by.id == obs.id, f"Wrong detector: {env.detected_by.name}"
    print(f"[PASS] Test 8: Direct unobstructed detection triggered correctly by {obs.name}.")

    # -------------------------------------------------------------
    # Test 9: Standing behind an obstacle prevents detection
    # -------------------------------------------------------------
    env.reset()
    # Observer positioned directly above top vertical wall (wall: x in [520, 580], y in [95, 255])
    # Observer heads downwards across the wall
    obs = env.observers[0]
    obs.patrol_points = [(550.0, 65.0), (550.0, 280.0)]
    obs.x, obs.y = 550.0, 65.0
    obs.target_wp_index = 1
    # Place player on opposite side of top wall at (550, 270)
    # Total distance is 205px (<= 225px vision distance)
    env.player.x, env.player.y = 550.0, 270.0
    env.step((0.0, 0.0), 0.01)
    assert env.state == StealthEnvironment.STATE_PLAYING, "Player behind obstacle was falsely detected!"
    assert obs.last_los_info["blocked_by_obstacle"] == True, "Wall did not register as blocking LOS!"
    assert obs.last_los_info["is_visible"] == False, "Player should not be visible behind wall!"
    print("[PASS] Test 9: Obstacle occlusion strictly blocks line-of-sight and prevents detection.")

    # -------------------------------------------------------------
    # Test 10: Moving from behind obstacle into visible space triggers detection
    # -------------------------------------------------------------
    # Position observer at (610, 65) patrolling downwards past the right side of the wall
    # Top wall ends at x=580, so corridor at x=610 is completely clear
    obs.patrol_points = [(610.0, 65.0), (610.0, 200.0)]
    obs.x, obs.y = 610.0, 65.0
    obs.target_wp_index = 1
    env.player.x, env.player.y = 610.0, 200.0
    env.step((0.0, 0.0), 0.01)
    assert env.state == StealthEnvironment.STATE_DETECTED, "Player stepping out from obstacle was NOT detected!"
    assert obs.last_los_info["is_visible"] == True, "Player should be visible in clear corridor!"
    print("[PASS] Test 10: Stepping out from cover into visible space triggers immediate detection.")

    # -------------------------------------------------------------
    # Test 11: Reaching the goal triggers ESCAPED state
    # -------------------------------------------------------------
    env.reset()
    # Place player on top of goal
    env.player.x = env.goal_pos[0]
    env.player.y = env.goal_pos[1]
    env.step((0.0, 0.0), 0.01)
    assert env.state == StealthEnvironment.STATE_ESCAPED, "Touching goal did not trigger ESCAPED state!"
    print("[PASS] Test 11: Reaching the goal successfully triggers ESCAPED state.")

    # -------------------------------------------------------------
    # Test 12: Pressing 'R' (reset) correctly restores initial state
    # -------------------------------------------------------------
    env.reset()
    assert env.state == StealthEnvironment.STATE_PLAYING, "Reset state is not PLAYING"
    assert env.detected_by is None, "detected_by not cleared"
    assert math.isclose(env.player.x, PLAYER_START_POS[0], abs_tol=1e-3), "Player X not reset"
    assert math.isclose(env.player.y, PLAYER_START_POS[1], abs_tol=1e-3), "Player Y not reset"
    for obs in env.observers:
        start_wp = obs.patrol_points[0]
        assert math.isclose(obs.x, start_wp[0], abs_tol=1e-3), f"{obs.name} X not reset"
        assert math.isclose(obs.y, start_wp[1], abs_tol=1e-3), f"{obs.name} Y not reset"
    print("[PASS] Test 12: Environment reset strictly restores clean initial conditions.")

    # -------------------------------------------------------------
    # Test 13: Debug mode toggles and render functions without error
    # -------------------------------------------------------------
    test_surf = pygame.Surface((1100, 700))
    env.render(test_surf, debug=False, fps=60.0)
    env.render(test_surf, debug=True, fps=60.0)
    print("[PASS] Test 13: Debug rendering and normal rendering render without errors.")

    # -------------------------------------------------------------
    # Test 14: Determinism test - two runs produce identical observer paths
    # -------------------------------------------------------------
    env1 = StealthEnvironment()
    env2 = StealthEnvironment()

    trajectory1 = []
    trajectory2 = []

    # Run 120 steps on env1
    for _ in range(120):
        env1.step((0.0, 0.0), 0.033)
        trajectory1.append([(obs.x, obs.y, obs.facing_angle) for obs in env1.observers])

    # Run 120 steps on env2
    for _ in range(120):
        env2.step((0.0, 0.0), 0.033)
        trajectory2.append([(obs.x, obs.y, obs.facing_angle) for obs in env2.observers])

    for step_i in range(120):
        for obs_i in range(3):
            t1 = trajectory1[step_i][obs_i]
            t2 = trajectory2[step_i][obs_i]
            assert math.isclose(t1[0], t2[0], abs_tol=1e-9), f"Step {step_i} Obs {obs_i} X mismatch"
            assert math.isclose(t1[1], t2[1], abs_tol=1e-9), f"Step {step_i} Obs {obs_i} Y mismatch"
            assert math.isclose(t1[2], t2[2], abs_tol=1e-9), f"Step {step_i} Obs {obs_i} Angle mismatch"

    print("[PASS] Test 14: Strict determinism verified across independent identical runs.")

    # -------------------------------------------------------------
    # Test 15: Elapsed-Time Timer Verification
    # -------------------------------------------------------------
    env = StealthEnvironment()
    # 1. Starts at 0
    assert env.elapsed_time == 0.0, "Timer did not start at 0.0"

    # 2. Counts upward only while PLAYING
    env.step((0.0, 0.0), 0.5)
    assert math.isclose(env.elapsed_time, 0.5, abs_tol=1e-3), f"Timer did not increment to 0.5s: {env.elapsed_time}"

    # 3. Resets to 0 with reset() ('R')
    env.reset()
    assert env.elapsed_time == 0.0, "Timer did not reset to 0.0 on reset"

    # 4. Freezes when DETECTED
    obs = env.observers[0]
    front_dist = 80.0
    env.player.x = obs.x + math.cos(obs.facing_angle) * front_dist
    env.player.y = obs.y + math.sin(obs.facing_angle) * front_dist
    env.step((0.0, 0.0), 0.1)
    assert env.state == StealthEnvironment.STATE_DETECTED
    frozen_time_detected = env.elapsed_time
    assert frozen_time_detected > 0.0
    # Further steps while DETECTED must not advance timer
    for _ in range(10):
        env.step((0.0, 0.0), 0.1)
    assert env.elapsed_time == frozen_time_detected, "Timer failed to freeze upon DETECTION!"

    # 5. Freezes when ESCAPED
    env.reset()
    env.player.x, env.player.y = env.goal_pos[0], env.goal_pos[1]
    env.step((0.0, 0.0), 0.25)
    assert env.state == StealthEnvironment.STATE_ESCAPED
    frozen_time_escaped = env.elapsed_time
    assert math.isclose(frozen_time_escaped, 0.25, abs_tol=1e-3)
    for _ in range(10):
        env.step((0.0, 0.0), 0.1)
    assert env.elapsed_time == frozen_time_escaped, "Timer failed to freeze upon ESCAPE!"

    print("[PASS] Test 15: Elapsed-time timer starts at 0, counts while PLAYING, resets with R, and freezes on DETECTED/ESCAPED.")

    print("=" * 60)
    print("ALL ACCEPTANCE & TIMER TESTS PASSED PERFECTLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_acceptance_tests()
