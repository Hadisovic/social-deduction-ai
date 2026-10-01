"""
Automated verification suite for Stage 1 Spawn & Goal Randomization (Phase 3A extension).
Validates all 17 requirements specified in Section L.
"""

import math
import os
import sys

# Ensure headless execution
os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from config import (
    ARENA_WIDTH,
    ARENA_HEIGHT,
    ROOM_MARGIN,
    STAGE1_SPAWN_MARGIN,
    STAGE1_MIN_START_GOAL_DISTANCE,
    PLAYER_RADIUS,
    GOAL_RADIUS,
    PLAYER_START_POS,
    GOAL_POS,
    STAGE1_STEP_PENALTY,
    RL_DT,
    PROGRESS_CHECK_INTERVAL,
)
from rl_environment import StealthGymEnv
from test_stealth_env import run_acceptance_tests
from test_rl_env import run_phase_2_tests
from test_ppo_infrastructure import run_infrastructure_tests


def run_randomization_tests():
    print("=" * 70)
    print("RUNNING STAGE 1 RANDOMIZATION VERIFICATION SUITE (SECTION L)")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Test 1: Two consecutive Stage 1 resets normally produce different positions
    # -----------------------------------------------------------------
    env1 = StealthGymEnv(stage=1, render_mode=None)
    env1.reset(seed=42)
    p1 = (env1.env.player.x, env1.env.player.y)
    g1 = (env1.env.goal_pos[0], env1.env.goal_pos[1])

    env1.reset()  # Unseeded subsequent reset advances RNG sequence
    p2 = (env1.env.player.x, env1.env.player.y)
    g2 = (env1.env.goal_pos[0], env1.env.goal_pos[1])

    assert p1 != p2 or g1 != g2, f"Consecutive resets produced identical positions: p={p1}, g={g1}"
    print(f"[PASS] Test 1: Consecutive Stage 1 resets produce distinct layouts:")
    print(f"       Episode 1: Player {p1}, Goal {g1}")
    print(f"       Episode 2: Player {p2}, Goal {g2}")

    # -----------------------------------------------------------------
    # Test 2: Player remains within valid arena bounds across many episodes
    # -----------------------------------------------------------------
    min_x = ROOM_MARGIN + STAGE1_SPAWN_MARGIN
    max_x = 1100 - ROOM_MARGIN - STAGE1_SPAWN_MARGIN
    min_y = ROOM_MARGIN + STAGE1_SPAWN_MARGIN
    max_y = 700 - ROOM_MARGIN - STAGE1_SPAWN_MARGIN

    for ep in range(100):
        env1.reset()
        px, py = env1.env.player.x, env1.env.player.y
        assert min_x <= px <= max_x, f"Player X {px} out of bounds [{min_x}, {max_x}] on ep {ep}"
        assert min_y <= py <= max_y, f"Player Y {py} out of bounds [{min_y}, {max_y}] on ep {ep}"
        # Verify bounding rect doesn't penetrate walls
        assert px - PLAYER_RADIUS >= ROOM_MARGIN, f"Player left edge clips wall: {px - PLAYER_RADIUS}"
        assert px + PLAYER_RADIUS <= 1100 - ROOM_MARGIN, f"Player right edge clips wall: {px + PLAYER_RADIUS}"
        assert py - PLAYER_RADIUS >= ROOM_MARGIN, f"Player top edge clips wall: {py - PLAYER_RADIUS}"
        assert py + PLAYER_RADIUS <= 700 - ROOM_MARGIN, f"Player bottom edge clips wall: {py + PLAYER_RADIUS}"
    print(f"[PASS] Test 2: Player remains strictly within valid bounds over 100 random resets.")

    # -----------------------------------------------------------------
    # Test 3: Goal remains within valid arena bounds across many episodes
    # -----------------------------------------------------------------
    for ep in range(100):
        env1.reset()
        gx, gy = env1.env.goal_pos[0], env1.env.goal_pos[1]
        assert min_x <= gx <= max_x, f"Goal X {gx} out of bounds [{min_x}, {max_x}] on ep {ep}"
        assert min_y <= gy <= max_y, f"Goal Y {gy} out of bounds [{min_y}, {max_y}] on ep {ep}"
        assert gx - GOAL_RADIUS >= ROOM_MARGIN, f"Goal left edge clips wall: {gx - GOAL_RADIUS}"
        assert gx + GOAL_RADIUS <= 1100 - ROOM_MARGIN, f"Goal right edge clips wall: {gx + GOAL_RADIUS}"
        assert gy - GOAL_RADIUS >= ROOM_MARGIN, f"Goal top edge clips wall: {gy - GOAL_RADIUS}"
        assert gy + GOAL_RADIUS <= 700 - ROOM_MARGIN, f"Goal bottom edge clips wall: {gy + GOAL_RADIUS}"
    print(f"[PASS] Test 3: Goal remains strictly within valid bounds over 100 random resets.")

    # -----------------------------------------------------------------
    # Test 4: Player and goal do not overlap
    # -----------------------------------------------------------------
    min_separation = PLAYER_RADIUS + GOAL_RADIUS
    for ep in range(100):
        env1.reset()
        d = math.hypot(env1.env.player.x - env1.env.goal_pos[0], env1.env.player.y - env1.env.goal_pos[1])
        assert d > min_separation, f"Player and goal overlap on ep {ep}: dist={d:.2f} <= {min_separation}"
    print(f"[PASS] Test 4: Player and goal never overlap across 100 resets.")

    # -----------------------------------------------------------------
    # Test 5: Minimum start-goal distance is respected
    # -----------------------------------------------------------------
    for ep in range(100):
        env1.reset()
        d = math.hypot(env1.env.player.x - env1.env.goal_pos[0], env1.env.player.y - env1.env.goal_pos[1])
        assert d >= STAGE1_MIN_START_GOAL_DISTANCE, (
            f"Distance {d:.2f} < STAGE1_MIN_START_GOAL_DISTANCE ({STAGE1_MIN_START_GOAL_DISTANCE}) on ep {ep}"
        )
    print(f"[PASS] Test 5: Minimum distance (>= {STAGE1_MIN_START_GOAL_DISTANCE}px) strictly respected over 100 resets.")

    # -----------------------------------------------------------------
    # Test 6: Same initial seed reproduces the same sequence of Stage 1 layouts
    # -----------------------------------------------------------------
    env_a = StealthGymEnv(stage=1, render_mode=None)
    env_b = StealthGymEnv(stage=1, render_mode=None)

    env_a.reset(seed=123)
    env_b.reset(seed=123)

    for i in range(10):
        pa = (env_a.env.player.x, env_a.env.player.y)
        pb = (env_b.env.player.x, env_b.env.player.y)
        ga = (env_a.env.goal_pos[0], env_a.env.goal_pos[1])
        gb = (env_b.env.goal_pos[0], env_b.env.goal_pos[1])
        assert pa == pb, f"Player pos mismatch at sequence step {i}: {pa} vs {pb}"
        assert ga == gb, f"Goal pos mismatch at sequence step {i}: {ga} vs {gb}"
        env_a.reset()
        env_b.reset()
    print("[PASS] Test 6: Seeded reset produces identical, bit-exact sequence of layouts.")

    # -----------------------------------------------------------------
    # Test 7: Different seeds produce different sequences
    # -----------------------------------------------------------------
    env_a.reset(seed=123)
    env_b.reset(seed=999)
    pa = (env_a.env.player.x, env_a.env.player.y)
    pb = (env_b.env.player.x, env_b.env.player.y)
    ga = (env_a.env.goal_pos[0], env_a.env.goal_pos[1])
    gb = (env_b.env.goal_pos[0], env_b.env.goal_pos[1])
    assert pa != pb or ga != gb, "Different seeds produced identical layouts"
    print("[PASS] Test 7: Different seeds produce distinct layout sequences.")

    # -----------------------------------------------------------------
    # Test 8: Observation player coordinates match randomized spawn
    # -----------------------------------------------------------------
    obs, _ = env1.reset()
    norm_px = env1.env.player.x / ARENA_WIDTH
    norm_py = env1.env.player.y / ARENA_HEIGHT
    assert math.isclose(obs[0], norm_px, abs_tol=1e-5), f"obs[0] {obs[0]} != norm_px {norm_px}"
    assert math.isclose(obs[1], norm_py, abs_tol=1e-5), f"obs[1] {obs[1]} != norm_py {norm_py}"
    print("[PASS] Test 8: Observation player coordinates accurately reflect randomized spawn.")

    # -----------------------------------------------------------------
    # Test 9: Observation goal relative vector matches randomized goal
    # -----------------------------------------------------------------
    expected_dx = (env1.env.goal_pos[0] - env1.env.player.x) / ARENA_WIDTH
    expected_dy = (env1.env.goal_pos[1] - env1.env.player.y) / ARENA_HEIGHT
    assert math.isclose(obs[2], expected_dx, abs_tol=1e-5), f"obs[2] {obs[2]} != expected_dx {expected_dx}"
    assert math.isclose(obs[3], expected_dy, abs_tol=1e-5), f"obs[3] {obs[3]} != expected_dy {expected_dy}"
    print("[PASS] Test 9: Observation goal relative vector matches randomized goal.")

    # -----------------------------------------------------------------
    # Test 10: best_distance_to_goal resets correctly
    # -----------------------------------------------------------------
    obs, info = env1.reset()
    actual_dist = math.hypot(env1.env.player.x - env1.env.goal_pos[0], env1.env.player.y - env1.env.goal_pos[1])
    assert math.isclose(env1.best_distance_to_goal, actual_dist, abs_tol=1e-5), (
        f"best_distance_to_goal {env1.best_distance_to_goal} != actual_dist {actual_dist}"
    )
    assert math.isclose(info["initial_distance_to_goal"], actual_dist, abs_tol=1e-5)
    print(f"[PASS] Test 10: best_distance_to_goal correctly initialized to new randomized distance ({actual_dist:.1f}px).")

    # -----------------------------------------------------------------
    # Test 11: Stage 1 -0.01 step penalty still works
    # -----------------------------------------------------------------
    _, r_step, _, _, _ = env1.step(np.array([0.0, 0.0], dtype=np.float32))
    assert math.isclose(r_step, STAGE1_STEP_PENALTY, abs_tol=1e-5), (
        f"Stage 1 step reward {r_step} != STAGE1_STEP_PENALTY {STAGE1_STEP_PENALTY}"
    )
    print(f"[PASS] Test 11: Stage 1 step penalty ({STAGE1_STEP_PENALTY}) active on randomized layout.")

    # -----------------------------------------------------------------
    # Test 12: Existing progress reward still works
    # -----------------------------------------------------------------
    env1.reset()
    init_d = env1.best_distance_to_goal
    # Move directly towards goal
    dx = env1.env.goal_pos[0] - env1.env.player.x
    dy = env1.env.goal_pos[1] - env1.env.player.y
    mag = math.hypot(dx, dy)
    action = np.array([dx / mag, dy / mag], dtype=np.float32)

    steps_per_interval = int(round(PROGRESS_CHECK_INTERVAL / RL_DT))
    total_r = 0.0
    for _ in range(steps_per_interval):
        _, r, _, _, _ = env1.step(action)
        total_r += r

    assert env1.best_distance_to_goal < init_d, "best_distance_to_goal failed to improve"
    assert total_r > 0.0, f"Expected positive net progress reward, got {total_r}"
    print(f"[PASS] Test 12: Directional advancement towards randomized goal awards positive progress reward (+{total_r:.2f}).")

    # -----------------------------------------------------------------
    # Test 13: Stage 3 remains unchanged (fixed spawn and goal)
    # -----------------------------------------------------------------
    env3 = StealthGymEnv(stage=3, render_mode=None)
    for _ in range(5):
        env3.reset()
        assert env3.env.player.pos == PLAYER_START_POS, f"Stage 3 player pos modified: {env3.env.player.pos}"
        assert env3.env.goal_pos == GOAL_POS, f"Stage 3 goal pos modified: {env3.env.goal_pos}"
    print("[PASS] Test 13: Stage 3 preserved with fixed initial player and goal positions.")

    # -----------------------------------------------------------------
    # Test 14: Existing Phase 1 tests still pass
    # -----------------------------------------------------------------
    print("\n--- Running Phase 1 Acceptance Suite ---")
    run_acceptance_tests()
    print("[PASS] Test 14: Phase 1 tests passed completely.")

    # -----------------------------------------------------------------
    # Test 15: Existing Phase 2 tests still pass
    # -----------------------------------------------------------------
    print("\n--- Running Phase 2 RL Environment Suite ---")
    run_phase_2_tests()
    print("[PASS] Test 15: Phase 2 tests passed completely.")

    # -----------------------------------------------------------------
    # Test 16: PPO infrastructure tests still pass
    # -----------------------------------------------------------------
    print("\n--- Running Phase 3A PPO Infrastructure Suite ---")
    run_infrastructure_tests()
    print("[PASS] Test 16: PPO infrastructure tests passed completely.")

    # -----------------------------------------------------------------
    # Test 17: Gymnasium checker still passes
    # -----------------------------------------------------------------
    print("\n--- Running Gymnasium Environment Check ---")
    check_env(StealthGymEnv(stage=1, render_mode=None))
    print("[PASS] Test 17: Gymnasium environment checker passes cleanly.")

    print("=" * 70)
    print("ALL 17 STAGE 1 RANDOMIZATION VERIFICATION TESTS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_randomization_tests()
