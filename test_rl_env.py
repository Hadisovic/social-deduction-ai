"""
Comprehensive verification test suite for Phase 2:
Reinforcement Learning Gymnasium Environment (StealthGymEnv).
Covers all 35 verification requirements specified for Phase 2.
"""

import math
import os
import sys

# Ensure headless execution
os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from config import (
    ARENA_WIDTH,
    ARENA_HEIGHT,
    RL_DT,
    ACTION_DEADZONE,
    RAY_COUNT,
    RAY_MAX_DISTANCE,
    OBSERVATION_SIZE,
    GOAL_REWARD,
    DETECTION_PENALTY,
    PROGRESS_CHECK_INTERVAL,
    MIN_PROGRESS_DISTANCE,
    PROGRESS_REWARD_SCALE,
    MAX_EPISODE_TIME,
    PLAYER_SPEED,
    PLAYER_START_POS,
    GOAL_POS,
    OBSTACLE_RECTS,
    STAGE1_STEP_PENALTY,
    STAGE2_STEP_PENALTY,
    STAGE2_MAX_EPISODE_TIME,
    STAGE2_TIMEOUT_PENALTY,
)
from rl_environment import StealthGymEnv
from test_stealth_env import run_acceptance_tests


def run_phase_2_tests():
    print("=" * 70)
    print("RUNNING PHASE 2 RL ENVIRONMENT VERIFICATION SUITE")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Test 1: RL environment initializes without errors
    # -----------------------------------------------------------------
    env = StealthGymEnv(stage=1)
    assert env is not None, "Failed to instantiate StealthGymEnv"
    print("[PASS] Test 1: RL environment initializes without errors.")

    # -----------------------------------------------------------------
    # Test 2: action_space is continuous Box shape (2,)
    # -----------------------------------------------------------------
    assert isinstance(env.action_space, spaces.Box), "action_space is not spaces.Box"
    assert env.action_space.shape == (2,), f"Expected shape (2,), got {env.action_space.shape}"
    assert np.allclose(env.action_space.low, -1.0), "action_space low is not -1.0"
    assert np.allclose(env.action_space.high, 1.0), "action_space high is not 1.0"
    assert env.action_space.dtype == np.float32, "action_space dtype is not float32"
    print("[PASS] Test 2: action_space is continuous Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32).")

    # -----------------------------------------------------------------
    # Test 3: observation_space matches returned observation
    # -----------------------------------------------------------------
    obs, info = env.reset()
    assert env.observation_space.shape == obs.shape, f"Shape mismatch: {env.observation_space.shape} vs {obs.shape}"
    assert env.observation_space.contains(obs), "Initial observation not contained in observation_space"
    print("[PASS] Test 3: observation_space matches returned observation.")

    # -----------------------------------------------------------------
    # Test 4: observation has exactly 43 values
    # -----------------------------------------------------------------
    assert len(obs) == 43, f"Expected exactly 43 observation values, got {len(obs)}"
    assert obs.shape == (43,), f"Expected shape (43,), got {obs.shape}"
    print("[PASS] Test 4: Observation has exactly 43 float values.")

    # -----------------------------------------------------------------
    # Test 5: observation dtype is np.float32
    # -----------------------------------------------------------------
    assert obs.dtype == np.float32, f"Expected np.float32, got {obs.dtype}"
    print("[PASS] Test 5: Observation dtype is strictly np.float32.")

    # -----------------------------------------------------------------
    # Test 6: Every returned observation is contained in observation_space
    # -----------------------------------------------------------------
    env4 = StealthGymEnv(stage=4)
    obs, _ = env4.reset()
    assert env4.observation_space.contains(obs)
    for _ in range(40):
        action = env4.action_space.sample()
        obs, _, term, trunc, _ = env4.step(action)
        assert env4.observation_space.contains(obs), f"Observation out of bounds: {obs}"
        if term or trunc:
            obs, _ = env4.reset()
    print("[PASS] Test 6: Every step observation is strictly contained in observation_space.")

    # -----------------------------------------------------------------
    # Test 7: [0, 0] action leaves player stationary
    # -----------------------------------------------------------------
    env.reset()
    init_x, init_y = env.env.player.pos
    obs, r, _, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert env.env.player.x == init_x and env.env.player.y == init_y, "Player moved on zero action"
    print("[PASS] Test 7: [0, 0] action leaves player stationary.")

    # -----------------------------------------------------------------
    # Test 8: Arbitrary diagonal continuous action works
    # -----------------------------------------------------------------
    env.reset()
    init_x, init_y = env.env.player.pos
    # Action diagonally up-right (dx > 0, dy < 0)
    diag_action = np.array([0.7, -0.7], dtype=np.float32)
    env.step(diag_action)
    assert env.env.player.x > init_x, "Player did not move right"
    assert env.env.player.y < init_y, "Player did not move up"
    print("[PASS] Test 8: Arbitrary diagonal continuous action translates to correct movement.")

    # -----------------------------------------------------------------
    # Test 9: Action magnitude does not alter player speed beyond constant speed
    # -----------------------------------------------------------------
    env.reset()
    env.env.player.x, env.env.player.y = 300.0, 300.0
    start_pos = env.env.player.pos
    # Small action: magnitude 0.2 (above 0.05 deadzone)
    env.step(np.array([0.2, 0.0], dtype=np.float32))
    dist_small = math.hypot(env.env.player.x - start_pos[0], env.env.player.y - start_pos[1])

    env.reset()
    env.env.player.x, env.env.player.y = 300.0, 300.0
    start_pos = env.env.player.pos
    # Full action: magnitude 1.0
    env.step(np.array([1.0, 0.0], dtype=np.float32))
    dist_full = math.hypot(env.env.player.x - start_pos[0], env.env.player.y - start_pos[1])

    expected_dist = PLAYER_SPEED * RL_DT
    assert math.isclose(dist_small, expected_dist, abs_tol=1e-4), f"Small action dist {dist_small} != {expected_dist}"
    assert math.isclose(dist_full, expected_dist, abs_tol=1e-4), f"Full action dist {dist_full} != {expected_dist}"
    assert math.isclose(dist_small, dist_full, abs_tol=1e-6), "Action magnitude altered player speed!"
    print(f"[PASS] Test 9: Direction normalization verified (constant speed = {PLAYER_SPEED}px/s).")

    # -----------------------------------------------------------------
    # Test 10: Action deadzone prevents tiny output jitter
    # -----------------------------------------------------------------
    env.reset()
    init_pos = env.env.player.pos
    # Tiny action below ACTION_DEADZONE (0.05)
    tiny_action = np.array([0.02, -0.02], dtype=np.float32)  # magnitude ~0.028 < 0.05
    env.step(tiny_action)
    assert env.env.player.pos == init_pos, "Tiny sub-deadzone action produced unwanted movement!"
    print("[PASS] Test 10: Action deadzone (0.05) prevents tiny neural-network jitter.")

    # -----------------------------------------------------------------
    # Test 11: Goal relative X/Y changes correctly as player moves
    # -----------------------------------------------------------------
    obs1, _ = env.reset()
    # Move player right toward goal
    obs2, _, _, _, _ = env.step(np.array([1.0, 0.0], dtype=np.float32))
    # Relative dx = (goal_x - player_x) / ARENA_WIDTH
    assert obs2[2] < obs1[2], f"Goal relative dx did not decrease: {obs1[2]} -> {obs2[2]}"
    assert math.isclose(obs2[3], obs1[3], abs_tol=1e-5), "Goal relative dy unexpectedly changed"
    print("[PASS] Test 11: Goal relative vector updates accurately with player movement.")

    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    # Test 12: All 16 wall raycasts return valid normalized distances
    # -----------------------------------------------------------------
    obs, _ = env.reset()
    wall_rays = obs[4:20]
    assert len(wall_rays) == 16, f"Expected 16 wall rays, got {len(wall_rays)}"
    for idx, r in enumerate(wall_rays):
        assert 0.0 <= r <= 1.0, f"Ray {idx} distance out of [0, 1] range: {r}"
    print("[PASS] Test 12: All 16 wall raycasts return valid normalized values in [0.0, 1.0].")

    # -----------------------------------------------------------------
    # Test 13: Raycasts detect nearby obstacles/boundaries correctly
    # -----------------------------------------------------------------
    env.reset()
    # Position player at (50, 50). Top wall is at y=35 (dist=15px). Left wall is at x=35 (dist=15px).
    env.env.player.x, env.env.player.y = 50.0, 50.0
    obs = env.env.get_observation()
    # Ray 8 is 180 deg (Left): should hit left boundary wall at dist 15px
    ray_left = obs[4 + 8]
    # Ray 12 is 270 deg (Up): should hit top boundary wall at dist 15px
    ray_up = obs[4 + 12]
    expected_norm = 15.0 / RAY_MAX_DISTANCE
    assert math.isclose(ray_left, expected_norm, abs_tol=0.01), f"Left ray mismatch: {ray_left} vs {expected_norm}"
    assert math.isclose(ray_up, expected_norm, abs_tol=0.01), f"Up ray mismatch: {ray_up} vs {expected_norm}"
    print(f"[PASS] Test 13: Raycasts detect boundaries with exact geometry (norm={expected_norm:.3f}).")

    # -----------------------------------------------------------------
    # Test 14: Observer relative positions are correct
    # -----------------------------------------------------------------
    env4 = StealthGymEnv(stage=4)
    obs, _ = env4.reset()
    for i in range(3):
        base = 20 + i * 7
        rel_x = obs[base + 1]
        rel_y = obs[base + 2]
        expected_rel_x = (env4.env.observers[i].x - env4.env.player.x) / ARENA_WIDTH
        expected_rel_y = (env4.env.observers[i].y - env4.env.player.y) / ARENA_HEIGHT
        assert math.isclose(rel_x, expected_rel_x, abs_tol=1e-5), f"Obs {i} rel_x mismatch"
        assert math.isclose(rel_y, expected_rel_y, abs_tol=1e-5), f"Obs {i} rel_y mismatch"
    print("[PASS] Test 14: Observer relative positions match world coordinates exactly.")

    # -----------------------------------------------------------------
    # Test 15: Observer facing sin/cos representation is correct
    # -----------------------------------------------------------------
    for i in range(3):
        base = 20 + i * 7
        fx = obs[base + 3]
        fy = obs[base + 4]
        expected_fx = math.cos(env4.env.observers[i].facing_angle)
        expected_fy = math.sin(env4.env.observers[i].facing_angle)
        assert math.isclose(fx, expected_fx, abs_tol=1e-5), f"Obs {i} facing_x mismatch"
        assert math.isclose(fy, expected_fy, abs_tol=1e-5), f"Obs {i} facing_y mismatch"
        assert math.isclose(fx**2 + fy**2, 1.0, abs_tol=1e-5), f"Obs {i} facing vector not unit length"
    print("[PASS] Test 15: Observer heading represented as continuous unit vector (cos, sin).")

    # -----------------------------------------------------------------
    # Test 16: Inactive observer slots are zeroed and active flag is zero
    # -----------------------------------------------------------------
    env1 = StealthGymEnv(stage=1)
    obs1, _ = env1.reset()
    # In Stage 1, all 3 observer slots (indices 20 to 40) must be zero
    assert np.allclose(obs1[20:41], 0.0), f"Stage 1 observer slots not all zero: {obs1[20:41]}"
    print("[PASS] Test 16: Inactive observer slots are strictly zero-filled with active flag = 0.")

    # -----------------------------------------------------------------
    # Test 17: Active observer slots have active flag one
    # -----------------------------------------------------------------
    obs4, _ = env4.reset()
    for i in range(3):
        base = 20 + i * 7
        assert obs4[base + 0] == 1.0, f"Observer {i} active flag != 1.0 in Stage 4"
    print("[PASS] Test 17: Active observers have active flag = 1.0.")

    # -----------------------------------------------------------------
    # Test 18: Stage 1 contains no active observers and no central obstacles
    # -----------------------------------------------------------------
    assert len(env1.env.active_observers) == 0, "Stage 1 has active observers"
    assert len(env1.env.obstacles) == 0, "Stage 1 has central obstacles"
    print("[PASS] Test 18: Stage 1 verified (no observers, no central obstacles).")

    # -----------------------------------------------------------------
    # Test 19: Stage 2 enables obstacles but no observers
    # -----------------------------------------------------------------
    env2 = StealthGymEnv(stage=2)
    assert len(env2.env.active_observers) == 0, "Stage 2 has active observers"
    assert len(env2.env.obstacles) == len(OBSTACLE_RECTS), "Stage 2 missing central obstacles"
    print("[PASS] Test 19: Stage 2 verified (obstacles enabled, no observers).")

    # -----------------------------------------------------------------
    # Test 20: Stage 3 has exactly one active observer
    # -----------------------------------------------------------------
    env3 = StealthGymEnv(stage=3)
    assert len(env3.env.active_observers) == 1, f"Stage 3 expected 1 observer, got {len(env3.env.active_observers)}"
    assert env3.env.active_observers[0].id == 1, "Stage 3 active observer is not Observer 1"
    assert len(env3.env.obstacles) == len(OBSTACLE_RECTS), "Stage 3 missing obstacles"
    obs3, _ = env3.reset()
    assert obs3[20] == 1.0, "Observer 1 not marked active"
    assert obs3[27] == 0.0, "Observer 2 not zeroed"
    assert obs3[34] == 0.0, "Observer 3 not zeroed"
    print("[PASS] Test 20: Stage 3 verified (exactly 1 active observer, slots 2 & 3 zeroed).")

    # -----------------------------------------------------------------
    # Test 21: Stage 4 has exactly three active observers
    # -----------------------------------------------------------------
    assert len(env4.env.active_observers) == 3, "Stage 4 does not have 3 active observers"
    assert len(env4.env.obstacles) == len(OBSTACLE_RECTS), "Stage 4 missing obstacles"
    print("[PASS] Test 21: Stage 4 verified (all 3 observers active, full arena).")

    # -----------------------------------------------------------------
    # Test 22: Reaching goal produces large positive reward and terminated=True
    # -----------------------------------------------------------------
    env.reset()
    env.env.player.x, env.env.player.y = env.env.goal_pos[0], env.env.goal_pos[1]
    obs, reward, term, trunc, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert term == True, "Goal reach did not terminate episode"
    assert trunc == False, "Goal reach was marked truncated instead of terminated"
    assert reward >= GOAL_REWARD, f"Reward {reward} < GOAL_REWARD {GOAL_REWARD}"
    assert info["success"] == True, "info['success'] not set to True"
    print(f"[PASS] Test 22: Goal reach triggers terminated=True and reward=+{reward:.1f}.")

    # -----------------------------------------------------------------
    # Test 23: Detection produces large negative reward and terminated=True
    # -----------------------------------------------------------------
    env4.reset()
    obs0 = env4.env.observers[0]
    # Place player right in front of Observer 1
    front_dist = 80.0
    env4.env.player.x = obs0.x + math.cos(obs0.facing_angle) * front_dist
    env4.env.player.y = obs0.y + math.sin(obs0.facing_angle) * front_dist
    obs, reward, term, trunc, info = env4.step(np.array([0.0, 0.0], dtype=np.float32))
    assert term == True, "Detection did not terminate episode"
    assert trunc == False, "Detection was marked truncated instead of terminated"
    assert reward <= DETECTION_PENALTY, f"Reward {reward} > DETECTION_PENALTY {DETECTION_PENALTY}"
    assert info["detected"] == True, "info['detected'] not set to True"
    print(f"[PASS] Test 23: Detection triggers terminated=True and penalty={reward:.1f}.")

    # -----------------------------------------------------------------
    # Test 24: Timeout produces truncated=True rather than terminated=True
    # -----------------------------------------------------------------
    env4.reset()
    # Fast forward time to just before timeout
    env4.episode_time = MAX_EPISODE_TIME - (RL_DT * 0.5)
    obs, reward, term, trunc, info = env4.step(np.array([0.0, 0.0], dtype=np.float32))
    assert trunc == True, "Timeout did not truncate episode"
    assert term == False, "Timeout was marked terminated instead of truncated"
    assert info["timeout"] == True, "info['timeout'] not set to True"
    assert reward == 0.0, f"Timeout incorrectly generated penalty: {reward}"
    print("[PASS] Test 24: Episode timeout produces truncated=True with no penalty.")

    # -----------------------------------------------------------------
    # Test 25: Moving farther from the goal does NOT generate a distance penalty
    # -----------------------------------------------------------------
    env4.reset()
    # Move player away from goal (Left / Down) for 1 second of simulation time
    steps = int(1.0 / RL_DT)
    total_retreat_reward = 0.0
    for _ in range(steps):
        _, r, _, _, _ = env4.step(np.array([-1.0, 1.0], dtype=np.float32))
        total_retreat_reward += r
        assert r >= 0.0, f"Negative penalty generated while retreating: {r}"
    assert total_retreat_reward == 0.0, "Non-zero reward received while moving away from goal"
    print("[PASS] Test 25: Moving away from goal incurs zero distance penalty.")

    # -----------------------------------------------------------------
    # Test 26: Progress reward is given only for genuinely new best distance
    # -----------------------------------------------------------------
    env4.reset()
    steps_per_interval = int(round(PROGRESS_CHECK_INTERVAL / RL_DT))
    total_progress_reward = 0.0
    for _ in range(steps_per_interval):
        _, r, _, _, _ = env4.step(np.array([1.0, -1.0], dtype=np.float32))
        total_progress_reward += r
    assert total_progress_reward > 0.0, f"Expected positive progress reward, got {total_progress_reward}"
    print(f"[PASS] Test 26: New best distance generates positive progress reward (+{total_progress_reward:.2f}).")

    # -----------------------------------------------------------------
    # Test 27: Moving closer, then farther, then back to the SAME distance does NOT farm reward
    # -----------------------------------------------------------------
    env4.reset()
    # Step 1: Move forward 1 interval (15 steps) and record best distance
    fwd_reward = 0.0
    for _ in range(steps_per_interval):
        _, r, _, _, _ = env4.step(np.array([1.0, -1.0], dtype=np.float32))
        fwd_reward += r
    assert fwd_reward > 0.0
    recorded_best = env4.best_distance_to_goal

    # Step 2: Retreat backward 1 interval (15 steps)
    retreat_reward = 0.0
    for _ in range(steps_per_interval):
        _, r, _, _, _ = env4.step(np.array([-1.0, 1.0], dtype=np.float32))
        retreat_reward += r
    assert retreat_reward == 0.0, "Reward earned during retreat"

    # Step 3: Advance forward again, returning to the previous position
    return_reward = 0.0
    for _ in range(steps_per_interval):
        _, r, _, _, _ = env4.step(np.array([1.0, -1.0], dtype=np.float32))
        return_reward += r
    assert return_reward == 0.0, f"Exploit detected: farmed {return_reward} reward returning to old best!"
    assert math.isclose(env4.best_distance_to_goal, recorded_best, abs_tol=1e-3), "best_distance_to_goal corrupted"
    print("[PASS] Test 27: Anti-farming verified (oscillating back to prior best earns 0 reward).")

    # -----------------------------------------------------------------
    # Test 28: Exceeding previous best distance toward goal produces new progress reward
    # -----------------------------------------------------------------
    # Now continue moving forward past the old best for another full interval
    new_progress_reward = 0.0
    for _ in range(steps_per_interval):
        _, r, _, _, _ = env4.step(np.array([1.0, -1.0], dtype=np.float32))
        new_progress_reward += r
    assert new_progress_reward > 0.0, "Exceeding prior best failed to generate new progress reward"
    assert env4.best_distance_to_goal < recorded_best, "best_distance_to_goal was not updated"
    print(f"[PASS] Test 28: Exceeding prior best awards fresh progress reward (+{new_progress_reward:.2f}).")

    # -----------------------------------------------------------------
    # Test 29: Reset restores best_distance_to_goal correctly
    # -----------------------------------------------------------------
    # On Stage 1, best_distance_to_goal is initialized to the randomized player-goal distance
    env.reset()
    stage1_dist = math.hypot(
        env.env.player.x - env.env.goal_pos[0],
        env.env.player.y - env.env.goal_pos[1]
    )
    assert math.isclose(env.best_distance_to_goal, stage1_dist, abs_tol=1e-3), \
        f"Stage 1 reset did not initialize goal distance: {env.best_distance_to_goal} vs {stage1_dist}"
    # On Stage 2, best_distance_to_goal is initialized to the randomized player-goal distance
    env2.reset()
    stage2_dist = math.hypot(
        env2.env.player.x - env2.env.goal_pos[0],
        env2.env.player.y - env2.env.goal_pos[1]
    )
    assert math.isclose(env2.best_distance_to_goal, stage2_dist, abs_tol=1e-3), \
        f"Stage 2 reset did not initialize goal distance: {env2.best_distance_to_goal} vs {stage2_dist}"
    # On Stage 3+, fixed initial player and goal positions are restored
    env3.reset()
    expected_initial_dist = math.hypot(
        PLAYER_START_POS[0] - GOAL_POS[0],
        PLAYER_START_POS[1] - GOAL_POS[1]
    )
    assert math.isclose(env3.best_distance_to_goal, expected_initial_dist, abs_tol=1e-3), \
        f"Stage 3 reset did not restore fixed goal distance: {env3.best_distance_to_goal} vs {expected_initial_dist}"
    print(f"[PASS] Test 29: Reset initializes best_distance_to_goal correctly across stages.")

    # -----------------------------------------------------------------
    # Test 30: Reset restores timer/progress interval state
    # -----------------------------------------------------------------
    env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert env.episode_time > 0.0
    env.reset()
    assert env.episode_time == 0.0, "Episode time not reset to 0.0"
    assert env.time_since_progress_check == 0.0, "time_since_progress_check not reset to 0.0"
    print("[PASS] Test 30: Reset restores episode timers and progress intervals.")

    # -----------------------------------------------------------------
    # Test 31: Reset restores deterministic observer state
    # -----------------------------------------------------------------
    env4.reset()
    init_obs_positions = [obs.pos for obs in env4.env.observers]
    for _ in range(60):
        env4.step(np.array([0.0, 0.0], dtype=np.float32))
    env4.reset()
    for i, obs in enumerate(env4.env.observers):
        assert obs.pos == init_obs_positions[i], f"Observer {i} position not restored on reset"
    print("[PASS] Test 31: Reset deterministically restores observer positions and headings.")

    # -----------------------------------------------------------------
    # Test 32: Fixed RL timestep gives deterministic simulation behavior
    # -----------------------------------------------------------------
    env_a = StealthGymEnv(stage=4)
    env_b = StealthGymEnv(stage=4)
    obs_a, _ = env_a.reset(seed=42)
    obs_b, _ = env_b.reset(seed=42)
    assert np.allclose(obs_a, obs_b), "Initial observations differ"

    for step_num in range(60):
        action = np.array([0.6, -0.4], dtype=np.float32)
        oa, ra, ta, qa, _ = env_a.step(action)
        ob, rb, tb, qb, _ = env_b.step(action)
        assert np.allclose(oa, ob, atol=1e-6), f"Step {step_num} observation diverged"
        assert math.isclose(ra, rb, abs_tol=1e-7), f"Step {step_num} reward diverged"
        assert ta == tb and qa == qb, f"Step {step_num} termination status diverged"
    print("[PASS] Test 32: Fixed RL timestep produces bit-exact deterministic trajectories.")

    # -----------------------------------------------------------------
    # Test 33: Rendering disabled does not create gameplay differences
    # -----------------------------------------------------------------
    env_headless = StealthGymEnv(stage=4, render_mode=None)
    env_rgb = StealthGymEnv(stage=4, render_mode="rgb_array")
    o_head, _ = env_headless.reset(seed=123)
    o_rgb, _ = env_rgb.reset(seed=123)
    for _ in range(30):
        act = np.array([0.5, -0.5], dtype=np.float32)
        oh, rh, th, qh, _ = env_headless.step(act)
        or_, rr, tr, qr, _ = env_rgb.step(act)
        assert np.allclose(oh, or_, atol=1e-6), "Headless and render mode observations differ!"
        assert math.isclose(rh, rr, abs_tol=1e-7), "Headless and render mode rewards differ!"
    print("[PASS] Test 33: Headless and rendered modes execute with identical physics and outcomes.")

    # -----------------------------------------------------------------
    # Test 34: Existing manual environment tests still pass
    # -----------------------------------------------------------------
    print("Running Phase 1 baseline acceptance suite...")
    run_acceptance_tests()
    print("[PASS] Test 34: Existing manual environment tests still pass completely.")

    # -----------------------------------------------------------------
    # Test 35: Gymnasium environment checker passes
    # -----------------------------------------------------------------
    for stg in [1, 2, 3, 4]:
        env_direct = StealthGymEnv(stage=stg)
        check_env(env_direct)
        env_made = gym.make("StealthSandbox-v0", stage=stg)
        check_env(env_made.unwrapped)
    print("[PASS] Test 35: Gymnasium environment checker passes across all curriculum stages (both direct and gym.make).")

    # -----------------------------------------------------------------
    # Test 36: Stage 1 receives -0.01 for an otherwise neutral step
    # -----------------------------------------------------------------
    s1_env = StealthGymEnv(stage=1)
    s1_env.reset()
    _, s1_r, s1_term, s1_trunc, _ = s1_env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert math.isclose(s1_r, STAGE1_STEP_PENALTY, abs_tol=1e-6), (
        f"Stage 1 step reward was {s1_r}, expected STAGE1_STEP_PENALTY {STAGE1_STEP_PENALTY}"
    )
    assert not s1_term and not s1_trunc
    print(f"[PASS] Test 36: Stage 1 receives exactly {STAGE1_STEP_PENALTY} for a neutral step.")

    # -----------------------------------------------------------------
    # Test 37: Stage 2 receives STAGE2_STEP_PENALTY (-0.01)
    # -----------------------------------------------------------------
    s2_env = StealthGymEnv(stage=2)
    s2_env.reset()
    _, s2_r, s2_term, s2_trunc, _ = s2_env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert math.isclose(s2_r, -0.01, abs_tol=1e-6), (
        f"Stage 2 received unexpected penalty: {s2_r}, expected -0.01"
    )
    assert not s2_term and not s2_trunc
    print("[PASS] Test 37: Stage 2 receives Stage 2 step penalty (-0.01).")

    # -----------------------------------------------------------------
    # Test 38: Stage 3 does NOT receive the Stage 1 penalty
    # -----------------------------------------------------------------
    s3_env = StealthGymEnv(stage=3)
    s3_env.reset()
    _, s3_r, s3_term, s3_trunc, _ = s3_env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert math.isclose(s3_r, 0.0, abs_tol=1e-6), (
        f"Stage 3 received unexpected penalty: {s3_r}"
    )
    assert not s3_term and not s3_trunc
    print("[PASS] Test 38: Stage 3 does NOT receive the Stage 1 step penalty (reward is 0.0).")

    # -----------------------------------------------------------------
    # Test 39: Existing goal reward still works (+100)
    # -----------------------------------------------------------------
    for stg in [1, 2, 3]:
        g_env = StealthGymEnv(stage=stg)
        g_env.reset()
        g_env.env.player.x, g_env.env.player.y = g_env.env.goal_pos[0], g_env.env.goal_pos[1]
        _, g_r, g_term, g_trunc, g_info = g_env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert g_term == True, f"Stage {stg} goal did not terminate"
        assert g_trunc == False, f"Stage {stg} goal marked truncated"
        assert g_r >= GOAL_REWARD, f"Stage {stg} goal reward {g_r} < {GOAL_REWARD}"
        assert g_info["is_success"] == True, f"Stage {stg} is_success not True"
    print(f"[PASS] Test 39: Goal reward (+{GOAL_REWARD}) intact across stages.")

    # -----------------------------------------------------------------
    # Test 40: Existing progress reward still works
    # -----------------------------------------------------------------
    prg_env = StealthGymEnv(stage=1)
    prg_env.reset()
    init_dist = prg_env.best_distance_to_goal
    steps_per_interval = int(round(PROGRESS_CHECK_INTERVAL / RL_DT))
    total_r = 0.0
    for _ in range(steps_per_interval):
        _, r, _, _, _ = prg_env.step(np.array([1.0, -1.0], dtype=np.float32))
        total_r += r
    new_dist = prg_env.best_distance_to_goal
    progress_made = init_dist - new_dist
    expected_progress_reward = progress_made * PROGRESS_REWARD_SCALE
    expected_step_penalties = steps_per_interval * STAGE1_STEP_PENALTY
    expected_total = expected_progress_reward + expected_step_penalties
    assert math.isclose(total_r, expected_total, abs_tol=1e-3), (
        f"Progress reward calculation mismatch: got {total_r}, expected {expected_total}"
    )
    print(f"[PASS] Test 40: Progress reward (+0.05/pixel) intact and correctly combines with Stage 1 step penalty.")

    # -----------------------------------------------------------------
    # Test 41: Existing timeout behavior still works (truncation, no extra penalty)
    # -----------------------------------------------------------------
    for stg in [1, 2, 3]:
        t_env = StealthGymEnv(stage=stg)
        t_env.reset()
        max_t = STAGE2_MAX_EPISODE_TIME if stg == 2 else MAX_EPISODE_TIME
        t_env.episode_time = max_t - (RL_DT * 0.5)
        _, t_r, t_term, t_trunc, t_info = t_env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert t_trunc == True, f"Stage {stg} timeout not truncated"
        assert t_term == False, f"Stage {stg} timeout marked terminated"
        assert t_info["timeout"] == True, f"Stage {stg} info['timeout'] not True"
        if stg == 1:
            assert math.isclose(t_r, STAGE1_STEP_PENALTY, abs_tol=1e-5), (
                f"Stage 1 timeout reward unexpected: {t_r}"
            )
        elif stg == 2:
            assert math.isclose(t_r, STAGE2_STEP_PENALTY + STAGE2_TIMEOUT_PENALTY, abs_tol=1e-5), (
                f"Stage 2 timeout reward unexpected: {t_r}"
            )
        else:
            assert math.isclose(t_r, 0.0, abs_tol=1e-5), (
                f"Stage {stg} timeout reward unexpected: {t_r}"
            )
    print("[PASS] Test 41: Timeout behavior intact (Stage 1: 30s/no penalty, Stage 2: 20s/-50 penalty, Stage 3: 30s/no penalty).")

    print("=" * 70)
    print("ALL 41 RL VERIFICATION TESTS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_phase_2_tests()

