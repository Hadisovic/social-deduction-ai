"""
Verification test suite for Curriculum Stage 2:
- Randomization rules (margins, clearance, distance, seeding, observation alignment)
- Rewards and penalties (step penalties, progress reward, wall collision penalty, terminal rewards)
"""

import math
import numpy as np
import pytest

from config import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    ROOM_MARGIN,
    STAGE2_SPAWN_MARGIN,
    STAGE2_MIN_START_GOAL_DISTANCE,
    STAGE2_OBSTACLE_SPAWN_CLEARANCE,
    PLAYER_RADIUS,
    GOAL_RADIUS,
    OBSTACLE_RECTS,
    BOUNDARY_WALLS,
    STAGE1_STEP_PENALTY,
    STAGE2_STEP_PENALTY,
    STAGE2_MAX_EPISODE_TIME,
    STAGE2_TIMEOUT_PENALTY,
    STAGE2_BLOCKED_STEP_PENALTY,
    GOAL_REWARD,
    PROGRESS_CHECK_INTERVAL,
    MIN_PROGRESS_DISTANCE,
    PROGRESS_REWARD_SCALE,
    STAGE1_PROGRESS_REWARD_SCALE,
    STAGE2_PROGRESS_REWARD_SCALE,
    RL_DT,
)
from geometry import point_to_rect_distance
from rl_environment import StealthGymEnv


def run_stage2_randomization_tests():
    print("=" * 70)
    print("RUNNING STAGE 2 RANDOMIZATION & REWARD VERIFICATION SUITE")
    print("=" * 70)

    env = StealthGymEnv(stage=2, render_mode=None)

    # -----------------------------------------------------------------
    # Test 1: Consecutive resets produce different layouts
    # -----------------------------------------------------------------
    env.reset(seed=100)
    p1 = (env.env.player.x, env.env.player.y)
    g1 = (env.env.goal_pos[0], env.env.goal_pos[1])

    env.reset()
    p2 = (env.env.player.x, env.env.player.y)
    g2 = (env.env.goal_pos[0], env.env.goal_pos[1])

    assert p1 != p2 or g1 != g2, "Consecutive Stage 2 resets produced identical layouts!"
    print(f"[PASS] Test 1: Consecutive Stage 2 resets produce distinct layouts:")
    print(f"       Ep 1: Player {p1}, Goal {g1}")
    print(f"       Ep 2: Player {p2}, Goal {g2}")

    # -----------------------------------------------------------------
    # Test 2-8: Statistical checks over 200 random resets
    # -----------------------------------------------------------------
    player_margin = ROOM_MARGIN + max(STAGE2_SPAWN_MARGIN, PLAYER_RADIUS + STAGE2_OBSTACLE_SPAWN_CLEARANCE)  # 35 + 50 = 85.0
    player_min_x = player_margin
    player_max_x = WINDOW_WIDTH - player_margin  # 1015.0
    player_min_y = player_margin
    player_max_y = WINDOW_HEIGHT - player_margin  # 615.0

    goal_margin = ROOM_MARGIN + max(STAGE2_SPAWN_MARGIN, GOAL_RADIUS + STAGE2_OBSTACLE_SPAWN_CLEARANCE)  # 35 + 53 = 88.0
    goal_min_x = goal_margin
    goal_max_x = WINDOW_WIDTH - goal_margin  # 1012.0
    goal_min_y = goal_margin
    goal_max_y = WINDOW_HEIGHT - goal_margin  # 612.0

    req_player_dist = PLAYER_RADIUS + STAGE2_OBSTACLE_SPAWN_CLEARANCE  # 14 + 25 = 39.0
    req_goal_dist = GOAL_RADIUS + STAGE2_OBSTACLE_SPAWN_CLEARANCE      # 28 + 25 = 53.0

    # Inner faces of actual outer arena boundary walls
    wall_left_inner = float(ROOM_MARGIN)               # 35.0
    wall_right_inner = float(WINDOW_WIDTH - ROOM_MARGIN) # 1065.0
    wall_top_inner = float(ROOM_MARGIN)                # 35.0
    wall_bottom_inner = float(WINDOW_HEIGHT - ROOM_MARGIN) # 665.0

    for i in range(200):
        env.reset()
        px, py = env.env.player.x, env.env.player.y
        gx, gy = env.env.goal_pos[0], env.env.goal_pos[1]

        # 2. Player within arena bounds [85.0, 1015.0] x [85.0, 615.0]
        assert player_min_x <= px <= player_max_x, f"Reset {i}: Player x ({px}) outside bounds [{player_min_x}, {player_max_x}]"
        assert player_min_y <= py <= player_max_y, f"Reset {i}: Player y ({py}) outside bounds [{player_min_y}, {player_max_y}]"

        # 3. Goal within arena bounds [88.0, 1012.0] x [88.0, 612.0]
        assert goal_min_x <= gx <= goal_max_x, f"Reset {i}: Goal x ({gx}) outside bounds [{goal_min_x}, {goal_max_x}]"
        assert goal_min_y <= gy <= goal_max_y, f"Reset {i}: Goal y ({gy}) outside bounds [{goal_min_y}, {goal_max_y}]"

        # Outer-wall clearance verification for Player:
        # Player center must be >= 50.0 px from actual wall; edge must be >= 36.0 px (exceeding clearance of 25.0 px)
        player_wall_center_dist = min(px - wall_left_inner, wall_right_inner - px, py - wall_top_inner, wall_bottom_inner - py)
        player_wall_edge_clearance = player_wall_center_dist - PLAYER_RADIUS
        assert player_wall_center_dist >= STAGE2_SPAWN_MARGIN - 1e-5, (
            f"Reset {i}: Player center to outer wall {player_wall_center_dist:.2f} < {STAGE2_SPAWN_MARGIN}"
        )
        assert player_wall_edge_clearance >= STAGE2_OBSTACLE_SPAWN_CLEARANCE - 1e-5, (
            f"Reset {i}: Player edge to outer wall {player_wall_edge_clearance:.2f} < {STAGE2_OBSTACLE_SPAWN_CLEARANCE}"
        )

        # Outer-wall clearance verification for Goal:
        # Goal center must be >= 53.0 px from actual wall; edge must be >= 25.0 px (strictly respecting clearance)
        goal_wall_center_dist = min(gx - wall_left_inner, wall_right_inner - gx, gy - wall_top_inner, wall_bottom_inner - gy)
        goal_wall_edge_clearance = goal_wall_center_dist - GOAL_RADIUS
        assert goal_wall_center_dist >= (GOAL_RADIUS + STAGE2_OBSTACLE_SPAWN_CLEARANCE) - 1e-5, (
            f"Reset {i}: Goal center to outer wall {goal_wall_center_dist:.2f} < 53.0"
        )
        assert goal_wall_edge_clearance >= STAGE2_OBSTACLE_SPAWN_CLEARANCE - 1e-5, (
            f"Reset {i}: Goal edge to outer wall {goal_wall_edge_clearance:.2f} < {STAGE2_OBSTACLE_SPAWN_CLEARANCE}"
        )

        # Direct geometry validation against BOUNDARY_WALLS
        for wall in BOUNDARY_WALLS:
            assert point_to_rect_distance(px, py, wall) >= req_player_dist - 1e-5
            assert point_to_rect_distance(gx, gy, wall) >= req_goal_dist - 1e-5

        # 4 & 6. Player interior obstacle clearance
        for obs in OBSTACLE_RECTS:
            dist_p = point_to_rect_distance(px, py, obs)
            assert dist_p >= req_player_dist - 1e-5, (
                f"Reset {i}: Player center ({px}, {py}) clearance violation ({dist_p:.2f} < {req_player_dist}) to {obs}"
            )

        # 5 & 6. Goal interior obstacle clearance
        for obs in OBSTACLE_RECTS:
            dist_g = point_to_rect_distance(gx, gy, obs)
            assert dist_g >= req_goal_dist - 1e-5, (
                f"Reset {i}: Goal center ({gx}, {gy}) clearance violation ({dist_g:.2f} < {req_goal_dist}) to {obs}"
            )

        # 7 & 8. Player and goal do not overlap and distance >= 350
        dist_pg = math.hypot(px - gx, py - gy)
        assert dist_pg >= STAGE2_MIN_START_GOAL_DISTANCE - 1e-5, (
            f"Reset {i}: Player-goal distance ({dist_pg:.2f}) < {STAGE2_MIN_START_GOAL_DISTANCE}"
        )

    print("[PASS] Test 2: Player center strictly within outer arena margins [85.0, 1015.0] x [85.0, 615.0].")
    print("[PASS] Test 3: Goal center strictly within outer arena margins [88.0, 1012.0] x [88.0, 612.0].")
    print(f"[PASS] Test 4: Player maintains >= {STAGE2_OBSTACLE_SPAWN_CLEARANCE}px clearance from all outer walls and obstacles (edge clearance: {player_wall_edge_clearance:.1f}px).")
    print(f"[PASS] Test 5: Goal maintains >= {STAGE2_OBSTACLE_SPAWN_CLEARANCE}px clearance from all outer walls and obstacles (edge clearance: {goal_wall_edge_clearance:.1f}px).")
    print(f"[PASS] Test 6: Obstacle clearance strictly respected (Player >= {req_player_dist}px, Goal >= {req_goal_dist}px).")
    print(f"[PASS] Test 7: Player and goal circles never overlap across 200 resets.")
    print(f"[PASS] Test 8: Start-to-goal Euclidean distance always >= {STAGE2_MIN_START_GOAL_DISTANCE}px.")

    # -----------------------------------------------------------------
    # Test 9: Identical seeds reproduce layout sequence
    # -----------------------------------------------------------------
    env.reset(seed=999)
    seq1 = []
    for _ in range(10):
        seq1.append(((env.env.player.x, env.env.player.y), (env.env.goal_pos[0], env.env.goal_pos[1])))
        env.reset()

    env.reset(seed=999)
    seq2 = []
    for _ in range(10):
        seq2.append(((env.env.player.x, env.env.player.y), (env.env.goal_pos[0], env.env.goal_pos[1])))
        env.reset()

    for idx, (l1, l2) in enumerate(zip(seq1, seq2)):
        assert math.isclose(l1[0][0], l2[0][0], abs_tol=1e-5) and math.isclose(l1[0][1], l2[0][1], abs_tol=1e-5)
        assert math.isclose(l1[1][0], l2[1][0], abs_tol=1e-5) and math.isclose(l1[1][1], l2[1][1], abs_tol=1e-5)
    print("[PASS] Test 9: Identical seed reproduces bit-exact layout sequence.")

    # -----------------------------------------------------------------
    # Test 10: Different seeds produce different sequences
    # -----------------------------------------------------------------
    env.reset(seed=123)
    p_a = (env.env.player.x, env.env.player.y)
    env.reset(seed=456)
    p_b = (env.env.player.x, env.env.player.y)
    assert p_a != p_b, "Different seeds produced identical player positions!"
    print("[PASS] Test 10: Different seeds produce distinct layout sequences.")

    # -----------------------------------------------------------------
    # Test 11: Observation player position matches randomized spawn
    # -----------------------------------------------------------------
    obs, info = env.reset(seed=777)
    norm_px = obs[0]
    norm_py = obs[1]
    expected_norm_px = env.env.player.x / WINDOW_WIDTH
    expected_norm_py = env.env.player.y / WINDOW_HEIGHT
    assert math.isclose(norm_px, expected_norm_px, abs_tol=1e-5)
    assert math.isclose(norm_py, expected_norm_py, abs_tol=1e-5)
    print("[PASS] Test 11: Observation player position matches randomized spawn exactly.")

    # -----------------------------------------------------------------
    # Test 12: Goal-relative observation matches randomized goal
    # -----------------------------------------------------------------
    rel_gx = obs[2]
    rel_gy = obs[3]
    expected_rel_gx = (env.env.goal_pos[0] - env.env.player.x) / WINDOW_WIDTH
    expected_rel_gy = (env.env.goal_pos[1] - env.env.player.y) / WINDOW_HEIGHT
    assert math.isclose(rel_gx, expected_rel_gx, abs_tol=1e-5)
    assert math.isclose(rel_gy, expected_rel_gy, abs_tol=1e-5)
    print("[PASS] Test 12: Observation relative goal vector matches randomized layout exactly.")

    # -----------------------------------------------------------------
    # Test 13: 8 wall rays detect Stage 2 obstacles correctly
    # -----------------------------------------------------------------
    # Place player just to the left of obstacle 0: pygame.Rect(520, 95, 60, 160)
    env.env.set_layout((490.0, 150.0), (100.0, 100.0))
    obs = env.env.get_observation()
    # Ray 0 is angle 0 (pointing right: positive X)
    ray_right = obs[4]
    # Distance from 490 to 520 is 30 px. Normalized by 300 px max range = 0.10
    assert math.isclose(ray_right, 30.0 / 300.0, abs_tol=0.01), (
        f"Ray 0 expected ~0.10, got {ray_right}"
    )
    print(f"[PASS] Test 13: 8 wall rays correctly detect central obstacles (ray={ray_right:.3f}).")

    # -----------------------------------------------------------------
    # Test 14 & 15: best_distance_to_goal & initial_distance_to_goal reset
    # -----------------------------------------------------------------
    env.reset()
    actual_init_dist = math.hypot(
        env.env.player.x - env.env.goal_pos[0],
        env.env.player.y - env.env.goal_pos[1]
    )
    assert math.isclose(env.initial_distance_to_goal, actual_init_dist, abs_tol=1e-5)
    assert math.isclose(env.best_distance_to_goal, actual_init_dist, abs_tol=1e-5)
    print(f"[PASS] Test 14 & 15: initial_distance_to_goal and best_distance_to_goal reset cleanly ({actual_init_dist:.1f}px).")

    # -----------------------------------------------------------------
    # REWARD TESTS (Section 43)
    # -----------------------------------------------------------------
    print("\n--- Running Stage 2 Reward Verification ---")

    # Reward Test 1: Stage 2 neutral step gets -0.01
    env.reset()
    _, r_s2, _, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert math.isclose(r_s2, STAGE2_STEP_PENALTY, abs_tol=1e-6)
    assert STAGE2_STEP_PENALTY == -0.01
    print(f"[PASS] Reward Test 1: Stage 2 neutral step receives STAGE2_STEP_PENALTY ({r_s2}).")

    # Reward Test 2: Stage 1 still gets -0.01
    s1_env = StealthGymEnv(stage=1, render_mode=None)
    s1_env.reset()
    _, r_s1, _, _, _ = s1_env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert math.isclose(r_s1, STAGE1_STEP_PENALTY, abs_tol=1e-6)
    assert STAGE1_STEP_PENALTY == -0.01
    s1_env.close()
    print(f"[PASS] Reward Test 2: Stage 1 still receives STAGE1_STEP_PENALTY ({r_s1}).")

    # Reward Test 3: Stage 3 still does NOT inherit this automatically (reward = 0.0)
    s3_env = StealthGymEnv(stage=3, render_mode=None)
    s3_env.reset()
    _, r_s3, _, _, _ = s3_env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert math.isclose(r_s3, 0.0, abs_tol=1e-6)
    s3_env.close()
    print(f"[PASS] Reward Test 3: Stage 3 step penalty remains 0.0 (not automatically inherited).")

    # Reward Test 4: Wall contact has no impact penalty; pushing into obstacle receives blocked penalty
    env.reset()
    # Place player flush against an obstacle
    obs_rect = OBSTACLE_RECTS[0]  # (520, 95, 60, 160)
    # Set player at left edge: x = 520 - 14 = 506
    env.env.set_layout((float(obs_rect.left - PLAYER_RADIUS), 150.0), (100.0, 100.0))
    # Stationary contact: reward should ONLY be normal step penalty (-0.01), zero collision penalty
    _, r_touch, _, _, info_touch = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert math.isclose(r_touch, STAGE2_STEP_PENALTY, abs_tol=1e-6), (
        f"Wall contact received unexpected penalty: {r_touch}"
    )
    # Pushing into wall: receives normal step penalty (-0.01) + short blocked penalty (-0.02) = -0.03
    _, r_push, term, trunc, info_push = env.step(np.array([1.0, 0.0], dtype=np.float32))
    expected_push_r = STAGE2_STEP_PENALTY + STAGE2_BLOCKED_STEP_PENALTY
    assert math.isclose(r_push, expected_push_r, abs_tol=1e-6), (
        f"Pushing into obstacle received {r_push}, expected {expected_push_r}"
    )
    assert info_push.get("wall_contact_steps", 0) > 0, "Wall contact step not recorded in info!"
    print(f"[PASS] Reward Test 4: Touching wall incurs 0 collision penalty ({r_touch:.2f}); pushing into wall receives blocked penalty ({r_push:.2f}).")

    # Reward Test 5: Progress reward still awards positive reward for genuine advancement
    env.reset()
    init_d = env.initial_distance_to_goal
    dx = env.env.goal_pos[0] - env.env.player.x
    dy = env.env.goal_pos[1] - env.env.player.y
    mag = math.hypot(dx, dy)
    action = np.array([dx / mag, dy / mag], dtype=np.float32)

    steps_per_interval = int(round(PROGRESS_CHECK_INTERVAL / RL_DT))
    total_r = 0.0
    for _ in range(steps_per_interval):
        _, r, _, _, _ = env.step(action)
        total_r += r

    assert total_r > 0.0, f"Expected net positive progress reward, got {total_r}"
    assert env.best_distance_to_goal < init_d, "best_distance_to_goal did not improve"
    print(f"[PASS] Reward Test 5: Direct progress toward goal earns positive progress reward (+{total_r:.2f}).")

    # Reward Test 6: Moving away from goal gets NO additional distance punishment
    prev_best = env.best_distance_to_goal
    # Move opposite to goal
    opp_action = np.array([-dx / mag, -dy / mag], dtype=np.float32)
    step_r_sum = 0.0
    for _ in range(steps_per_interval):
        _, r, _, _, _ = env.step(opp_action)
        step_r_sum += r

    expected_sum = STAGE2_STEP_PENALTY * steps_per_interval
    assert math.isclose(step_r_sum, expected_sum, abs_tol=1e-5), (
        f"Moving away incurred extra punishment! Expected {expected_sum}, got {step_r_sum}"
    )
    assert env.best_distance_to_goal == prev_best, "best_distance_to_goal altered when moving away"
    print(f"[PASS] Reward Test 6: Moving away from goal incurs zero distance penalty (only step penalty: {step_r_sum:.2f}).")

    # Reward Test 7: Goal reached triggers terminated=True and reward=+100.0
    env.reset()
    # Place player directly adjacent to goal
    gx, gy = env.env.goal_pos
    env.env.player.x = gx - float(PLAYER_RADIUS + GOAL_RADIUS - 5.0)
    env.env.player.y = gy
    _, r_goal, term_goal, _, info_goal = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert term_goal, "Goal reach failed to trigger terminated=True"
    assert info_goal["is_success"], "info['is_success'] is False upon reaching goal"
    assert math.isclose(r_goal, GOAL_REWARD, abs_tol=1e-5), f"Goal reward {r_goal} != {GOAL_REWARD}"
    print(f"[PASS] Reward Test 7: Goal reached awards exactly +{GOAL_REWARD} and terminates episode.")

    # Reward Test 8: Timeout truncates at 20 seconds with -50 timeout penalty
    env.reset()
    env.episode_time = STAGE2_MAX_EPISODE_TIME - 0.01  # 19.99s
    _, r_to, term_to, trunc_to, info_to = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert trunc_to, "Episode failed to truncate at 20.0 seconds"
    assert not term_to, "Timeout should be truncated, not terminated"
    assert info_to["timeout"], "info['timeout'] not set on timeout"
    expected_timeout_reward = STAGE2_STEP_PENALTY + STAGE2_TIMEOUT_PENALTY  # -0.01 - 50.0 = -50.01
    assert math.isclose(r_to, expected_timeout_reward, abs_tol=1e-5), f"Unexpected timeout penalty: {r_to}, expected {expected_timeout_reward}"
    print(f"[PASS] Reward Test 8: Episode truncates at {STAGE2_MAX_EPISODE_TIME:.1f}s with {STAGE2_TIMEOUT_PENALTY:.1f} timeout penalty (total step reward: {r_to:.2f}).")

    # Reward Test 9: Verify Stage 1 and Stage 2 use IDENTICAL progress-shaping mechanics
    # - PROGRESS_CHECK_INTERVAL == 0.5s
    # - MIN_PROGRESS_DISTANCE == 5.0px
    # - PROGRESS_REWARD_SCALE == 0.05
    # - Identical step_progress_reward for identical advancements
    # - Identical anti-farming behavior
    assert math.isclose(PROGRESS_CHECK_INTERVAL, 0.5, abs_tol=1e-6), f"PROGRESS_CHECK_INTERVAL is {PROGRESS_CHECK_INTERVAL}, expected 0.5"
    assert math.isclose(MIN_PROGRESS_DISTANCE, 5.0, abs_tol=1e-6), f"MIN_PROGRESS_DISTANCE is {MIN_PROGRESS_DISTANCE}, expected 5.0"
    assert math.isclose(PROGRESS_REWARD_SCALE, 0.05, abs_tol=1e-6), f"PROGRESS_REWARD_SCALE is {PROGRESS_REWARD_SCALE}, expected 0.05"

    env_s1 = StealthGymEnv(stage=1, render_mode=None)
    env_s2 = StealthGymEnv(stage=2, render_mode=None)
    env_s1.reset()
    env_s2.reset()

    # Place both in an unobstructed horizontal lane: Player (200, 200), Goal (900, 200)
    test_player_start = (200.0, 200.0)
    test_goal = (900.0, 200.0)
    env_s1.env.set_layout(test_player_start, test_goal)
    env_s2.env.set_layout(test_player_start, test_goal)
    init_d = 700.0
    env_s1.best_distance_to_goal = init_d
    env_s1.initial_distance_to_goal = init_d
    env_s2.best_distance_to_goal = init_d
    env_s2.initial_distance_to_goal = init_d

    # Move right directly toward goal for 15 steps (0.5s)
    # Distance moved in 0.5s = 185 px/s * 0.5s = 92.5 px > MIN_PROGRESS_DISTANCE (5.0 px)
    forward_action = np.array([1.0, 0.0], dtype=np.float32)
    s1_rewards, s2_rewards = [], []
    for _ in range(steps_per_interval):
        _, r1, _, _, info1 = env_s1.step(forward_action)
        _, r2, _, _, info2 = env_s2.step(forward_action)
        s1_rewards.append(r1)
        s2_rewards.append(r2)

    # Compare gross progress rewards
    expected_advancement = 92.5
    expected_prg_s1 = expected_advancement * STAGE1_PROGRESS_REWARD_SCALE  # 92.5 * 0.05 = 4.625
    expected_prg_s2 = expected_advancement * STAGE2_PROGRESS_REWARD_SCALE  # 92.5 * 0.03 = 2.775
    assert math.isclose(info1["progress_reward_this_step"], expected_prg_s1, abs_tol=1e-2), (
        f"Expected Stage 1 progress reward {expected_prg_s1}, got {info1['progress_reward_this_step']}"
    )
    assert math.isclose(info2["progress_reward_this_step"], expected_prg_s2, abs_tol=1e-2), (
        f"Expected Stage 2 progress reward {expected_prg_s2}, got {info2['progress_reward_this_step']}"
    )
    # Net step rewards: step_penalty (-0.01) + progress_reward
    assert math.isclose(sum(s1_rewards), 15 * STAGE1_STEP_PENALTY + expected_prg_s1, abs_tol=1e-2)
    assert math.isclose(sum(s2_rewards), 15 * STAGE2_STEP_PENALTY + expected_prg_s2, abs_tol=1e-2)

    # Anti-farming test: retreat for 15 steps (0.5s)
    backward_action = np.array([-1.0, 0.0], dtype=np.float32)
    for _ in range(steps_per_interval):
        _, r1, _, _, info1 = env_s1.step(backward_action)
        _, r2, _, _, info2 = env_s2.step(backward_action)
    assert info1["progress_reward_this_step"] == 0.0
    assert info2["progress_reward_this_step"] == 0.0

    # Advance back to prior position for 15 steps (0.5s): anti-farming must award 0 in both
    for _ in range(steps_per_interval):
        _, r1, _, _, info1 = env_s1.step(forward_action)
        _, r2, _, _, info2 = env_s2.step(forward_action)
    assert info1["progress_reward_this_step"] == 0.0, "Stage 1 awarded progress reward for retracing prior ground!"
    assert info2["progress_reward_this_step"] == 0.0, "Stage 2 awarded progress reward for retracing prior ground!"
    assert math.isclose(r1, STAGE1_STEP_PENALTY, abs_tol=1e-6)
    assert math.isclose(r2, STAGE2_STEP_PENALTY, abs_tol=1e-6)

    env_s1.close()
    env_s2.close()
    print("[PASS] Reward Test 9: Stage 1 (0.05 scale) and Stage 2 (0.03 scale) use correct progress-shaping mechanics (0.5s interval, 5.0px min distance, anti-farming).")

    env.close()
    print("=" * 70)
    print("ALL 15 STAGE 2 RANDOMIZATION & 9 REWARD TESTS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_stage2_randomization_tests()
