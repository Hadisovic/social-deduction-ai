"""
Verification test suite for Stage 2 Forced Obstacle Detours and Revisit/Looping Penalty.
Covers:
1. Forced-detour generation across >= 200 resets (direct player-goal line segment intersects >= 1 inflated obstacle)
2. Seeded layout reproducibility
3. Fallback layout validity and obstacle obstruction
4. Revisit / looping penalty system (visit counts, 3-step confirmation, -2.0 penalty on 3rd+ visit)
5. Grid boundary jitter protection
6. Loop behavior trajectory (A->B->A->B->A->B)
7. Legitimate navigation without penalty
8. Single-step reward composition and recovery refund exclusion
9. No Stage 1 leakage
10. Observation space remains 43 float32 values
11. Bounded path efficiency diagnostic (<= 100%)
"""

import math
import numpy as np
import pytest
import pygame

from config import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    ROOM_MARGIN,
    STAGE2_SPAWN_MARGIN,
    STAGE2_MIN_START_GOAL_DISTANCE,
    STAGE2_OBSTACLE_SPAWN_CLEARANCE,
    STAGE2_REVISIT_CELL_SIZE,
    STAGE2_REVISIT_CONFIRM_STEPS,
    STAGE2_REVISIT_PENALTY,
    STAGE2_FALLBACK_PLAYER_POS,
    STAGE2_FALLBACK_GOAL_POS,
    PLAYER_RADIUS,
    GOAL_RADIUS,
    OBSTACLE_RECTS,
    BOUNDARY_WALLS,
    STAGE1_STEP_PENALTY,
    STAGE2_STEP_PENALTY,
    RL_DT,
    OBSERVATION_SIZE,
)
from geometry import (
    does_segment_intersect_obstacles,
    is_point_clear_of_obstacles,
    segment_intersects_rect,
)
from environment import (
    sample_stage2_positions,
    StealthEnvironment,
)
from rl_environment import StealthGymEnv


class TestForcedObstacleDetour:
    """Test suite for Goal 1: Forced Obstacle Detour Generation."""

    def test_01_two_hundred_seeded_resets_all_require_detour(self):
        """Verify that across 200 consecutive resets, EVERY layout has direct path intersecting >= 1 inflated obstacle."""
        env = StealthGymEnv(stage=2)
        all_solid_rects = list(BOUNDARY_WALLS) + list(OBSTACLE_RECTS)
        n_resets = 200

        for seed in range(n_resets):
            obs, info = env.reset(seed=seed + 1000)
            px = env.env.player.x
            py = env.env.player.y
            gx = env.env.goal_pos[0]
            gy = env.env.goal_pos[1]

            # 1. Player clearance
            assert is_point_clear_of_obstacles(
                px, py, PLAYER_RADIUS, STAGE2_OBSTACLE_SPAWN_CLEARANCE, all_solid_rects
            ), f"Seed {seed}: Player clearance violated"

            # 2. Goal clearance
            assert is_point_clear_of_obstacles(
                gx, gy, GOAL_RADIUS, STAGE2_OBSTACLE_SPAWN_CLEARANCE, all_solid_rects
            ), f"Seed {seed}: Goal clearance violated"

            # 3. Minimum Euclidean separation >= 350 px
            dist = math.hypot(px - gx, py - gy)
            assert dist >= STAGE2_MIN_START_GOAL_DISTANCE - 1e-5, (
                f"Seed {seed}: Distance {dist:.2f} < {STAGE2_MIN_START_GOAL_DISTANCE}"
            )

            # 4. Direct start-to-goal segment MUST intersect at least one obstacle inflated by PLAYER_RADIUS
            is_blocked = does_segment_intersect_obstacles(
                (px, py), (gx, gy), env.env.obstacles, inflate_radius=PLAYER_RADIUS
            )
            assert is_blocked, (
                f"Seed {seed}: Direct path from player ({px:.1f}, {py:.1f}) to goal ({gx:.1f}, {gy:.1f}) "
                f"is NOT blocked by any inflated obstacle! Forced detour requirement failed."
            )

        env.close()

    def test_02_seeded_reproducibility_preserved(self):
        """Identical reset seeds must produce bit-exact identical layout sequences."""
        env1 = StealthGymEnv(stage=2)
        env2 = StealthGymEnv(stage=2)

        for seed in [42, 999, 12345]:
            obs1, _ = env1.reset(seed=seed)
            obs2, _ = env2.reset(seed=seed)

            assert env1.env.player.x == env2.env.player.x
            assert env1.env.player.y == env2.env.player.y
            assert env1.env.goal_pos == env2.env.goal_pos
            assert np.array_equal(obs1, obs2)

        env1.close()
        env2.close()

    def test_03_fallback_pair_valid_and_blocked(self):
        """Verify the deterministic fallback layout satisfies all clearance and forced detour rules."""
        all_solid = list(BOUNDARY_WALLS) + list(OBSTACLE_RECTS)
        px, py = STAGE2_FALLBACK_PLAYER_POS
        gx, gy = STAGE2_FALLBACK_GOAL_POS

        # Clearance
        assert is_point_clear_of_obstacles(px, py, PLAYER_RADIUS, STAGE2_OBSTACLE_SPAWN_CLEARANCE, all_solid)
        assert is_point_clear_of_obstacles(gx, gy, GOAL_RADIUS, STAGE2_OBSTACLE_SPAWN_CLEARANCE, all_solid)

        # Distance
        dist = math.hypot(px - gx, py - gy)
        assert dist >= STAGE2_MIN_START_GOAL_DISTANCE

        # Direct path blocked
        blocked = does_segment_intersect_obstacles(
            STAGE2_FALLBACK_PLAYER_POS,
            STAGE2_FALLBACK_GOAL_POS,
            OBSTACLE_RECTS,
            inflate_radius=PLAYER_RADIUS
        )
        assert blocked, "Fallback player-to-goal line segment must be blocked by central obstacle"

    def test_04_fallback_triggered_on_exhausted_attempts(self):
        """When sampling attempts fail, the verified fallback must be cleanly returned."""
        # Pass empty obstacle list to simulate failure to find a blocked path
        pos_p, pos_g = sample_stage2_positions(rng=None, obstacles=[])
        assert pos_p == (float(STAGE2_FALLBACK_PLAYER_POS[0]), float(STAGE2_FALLBACK_PLAYER_POS[1]))
        assert pos_g == (float(STAGE2_FALLBACK_GOAL_POS[0]), float(STAGE2_FALLBACK_GOAL_POS[1]))


class TestRevisitSystem:
    """Test suite for Goal 2: Revisit / Looping Penalty System."""

    def test_01_initial_cell_starts_at_visit_count_1_no_penalty(self):
        """At episode reset, spawn cell is visit #1 with 0 penalty."""
        env = StealthGymEnv(stage=2)
        obs, info = env.reset(seed=42)

        spawn_x, spawn_y = env.env.player.x, env.env.player.y
        expected_cell = (int(math.floor(spawn_x / STAGE2_REVISIT_CELL_SIZE)),
                         int(math.floor(spawn_y / STAGE2_REVISIT_CELL_SIZE)))

        assert env.current_confirmed_cell == expected_cell
        assert env.cell_visit_counts[expected_cell] == 1
        assert info["current_confirmed_cell"] == expected_cell
        assert info["cell_visits"] == 1
        assert info["revisit_penalty_this_step"] == 0.0
        env.close()

    def test_02_continuous_occupancy_does_not_repeat_visit_or_penalty(self):
        """Remaining inside the same cell for many steps counts as 1 visit and 0 penalty."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        cell = env.current_confirmed_cell

        # Stationary action inside deadzone
        stationary_action = np.array([0.0, 0.0], dtype=np.float32)
        for _ in range(50):
            _, reward, _, _, info = env.step(stationary_action)
            assert env.cell_visit_counts[cell] == 1
            assert info["cell_visits"] == 1
            assert info["revisit_penalty_this_step"] == 0.0
            assert math.isclose(reward, STAGE2_STEP_PENALTY)

        env.close()

    def test_03_grid_boundary_jitter_protection(self):
        """Alternating across cell boundary without 3 consecutive steps does NOT confirm entry."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)

        # Place player right near boundary between cell (10, 10) and (11, 10)
        # e.g. x = 329.0 (cell 10) and x = 331.0 (cell 11)
        cell_A = (10, 10)
        cell_B = (11, 10)
        env.env.player.x = 329.0
        env.env.player.y = 315.0
        env.current_confirmed_cell = cell_A
        env.cell_visit_counts = {cell_A: 1}
        env.candidate_cell = None
        env.candidate_cell_steps = 0

        # Jitter: A -> B (1 step) -> A (1 step) -> B (2 steps) -> A (1 step)
        # Sequence:
        # Step 1: at B
        env.env.player.x = 331.0
        env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert env.current_confirmed_cell == cell_A
        assert env.candidate_cell == cell_B
        assert env.candidate_cell_steps == 1

        # Step 2: back to A
        env.env.player.x = 329.0
        env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert env.current_confirmed_cell == cell_A
        assert env.candidate_cell is None
        assert env.candidate_cell_steps == 0

        # Step 3: to B (step 1)
        env.env.player.x = 331.0
        env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert env.candidate_cell_steps == 1

        # Step 4: stay in B (step 2)
        env.env.player.x = 331.5
        env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert env.candidate_cell_steps == 2
        assert env.current_confirmed_cell == cell_A  # Still not confirmed!

        # Step 5: back to A before step 3
        env.env.player.x = 329.0
        env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert env.current_confirmed_cell == cell_A
        assert env.candidate_cell is None
        assert cell_B not in env.cell_visit_counts  # B was never committed

        env.close()

    def test_04_three_step_confirmation_commits_entry(self):
        """Staying in candidate cell for 3 consecutive steps commits transition."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)

        cell_A = (10, 10)
        cell_B = (11, 10)
        env.env.player.x = 329.0
        env.env.player.y = 315.0
        env.current_confirmed_cell = cell_A
        env.cell_visit_counts = {cell_A: 1}
        env.candidate_cell = None
        env.candidate_cell_steps = 0

        # Step into B for 3 consecutive steps
        env.env.player.x = 331.0
        env.step(np.array([0.0, 0.0], dtype=np.float32))
        env.env.player.x = 331.5
        env.step(np.array([0.0, 0.0], dtype=np.float32))
        _, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))

        assert env.current_confirmed_cell == cell_B
        assert env.cell_visit_counts[cell_B] == 1
        assert info["current_confirmed_cell"] == cell_B
        assert info["cell_visits"] == 1
        assert info["revisit_penalty_this_step"] == 0.0  # Visit 1 is free

        env.close()

    def test_05_revisit_counts_and_penalty_schedule(self):
        """
        Test exact visit schedule:
        Visit 1: free (0.0)
        Visit 2: free (0.0)
        Visit 3: penalty (-2.0) once
        Visit 4: penalty (-2.0) once
        """
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)

        cell_A = (10, 10)  # x in [300, 330]
        cell_B = (15, 10)  # x in [450, 480]

        env.current_confirmed_cell = cell_A
        env.cell_visit_counts = {cell_A: 1}
        env.candidate_cell = None
        env.candidate_cell_steps = 0

        def transition_to(target_cell, center_x):
            env.env.player.x = center_x
            env.env.player.y = 315.0
            r_pen = 0.0
            for _ in range(STAGE2_REVISIT_CONFIRM_STEPS):
                _, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
                if info["revisit_penalty_this_step"] != 0.0:
                    r_pen = info["revisit_penalty_this_step"]
            return r_pen

        # Visit 1 to B: free
        pen = transition_to(cell_B, 465.0)
        assert env.current_confirmed_cell == cell_B
        assert env.cell_visit_counts[cell_B] == 1
        assert pen == 0.0

        # Visit 2 to A: free
        pen = transition_to(cell_A, 315.0)
        assert env.current_confirmed_cell == cell_A
        assert env.cell_visit_counts[cell_A] == 2
        assert pen == 0.0

        # Visit 2 to B: free
        pen = transition_to(cell_B, 465.0)
        assert env.current_confirmed_cell == cell_B
        assert env.cell_visit_counts[cell_B] == 2
        assert pen == 0.0

        # Visit 3 to A: -2.0 penalty!
        pen = transition_to(cell_A, 315.0)
        assert env.current_confirmed_cell == cell_A
        assert env.cell_visit_counts[cell_A] == 3
        assert pen == STAGE2_REVISIT_PENALTY  # -2.0

        # Remaining in A: 0 additional revisit penalty
        for _ in range(20):
            _, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
            assert info["revisit_penalty_this_step"] == 0.0

        # Visit 3 to B: -2.0 penalty!
        pen = transition_to(cell_B, 465.0)
        assert env.current_confirmed_cell == cell_B
        assert env.cell_visit_counts[cell_B] == 3
        assert pen == STAGE2_REVISIT_PENALTY  # -2.0

        # Visit 4 to A: -2.0 penalty!
        pen = transition_to(cell_A, 315.0)
        assert env.current_confirmed_cell == cell_A
        assert env.cell_visit_counts[cell_A] == 4
        assert pen == STAGE2_REVISIT_PENALTY  # -2.0

        env.close()

    def test_06_loop_trajectory_sanity_total_penalty(self):
        """
        Trajectory A -> B -> A -> B -> A -> B
        Visits:
          A: 1, B: 1  (0)
          A: 2, B: 2  (0)
          A: 3, B: 3  (-2 + -2 = -4)
        Total revisit penalty must be exactly -4.0.
        """
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)

        cell_A = (5, 5)
        cell_B = (6, 5)
        env.current_confirmed_cell = cell_A
        env.cell_visit_counts = {cell_A: 1}
        env.candidate_cell = None
        env.candidate_cell_steps = 0

        revisit_penalties_collected = []

        def move_to_cell(target_cell, px):
            env.env.player.x = px
            env.env.player.y = 165.0
            for _ in range(STAGE2_REVISIT_CONFIRM_STEPS):
                _, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
                if info["revisit_penalty_this_step"] < 0:
                    revisit_penalties_collected.append(info["revisit_penalty_this_step"])

        # 1. to B (visit 1)
        move_to_cell(cell_B, 195.0)
        # 2. to A (visit 2)
        move_to_cell(cell_A, 165.0)
        # 3. to B (visit 2)
        move_to_cell(cell_B, 195.0)
        # 4. to A (visit 3 -> -2)
        move_to_cell(cell_A, 165.0)
        # 5. to B (visit 3 -> -2)
        move_to_cell(cell_B, 195.0)

        assert sum(revisit_penalties_collected) == -4.0
        assert len(revisit_penalties_collected) == 2
        env.close()

    def test_07_recovery_does_not_refund_revisit_penalties(self):
        """Recovery refund must strictly refund ONLY blocked penalties, NEVER revisit penalties."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)

        # Incur a revisit penalty
        env.cell_visit_counts[(10, 10)] = 2
        env.current_confirmed_cell = (11, 10)
        env.candidate_cell = (10, 10)
        env.candidate_cell_steps = 2
        env.env.player.x = 315.0
        env.env.player.y = 315.0
        _, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert info["revisit_penalty_this_step"] == -2.0

        # Verify blocked_penalty_accumulated did NOT absorb the revisit penalty
        assert env.blocked_penalty_accumulated == 0.0

        # Now simulate a stuck state of 15 blocked steps
        env.consecutive_blocked_steps = 14
        env.blocked_penalty_accumulated = -0.28
        # Step 15 with blocked movement: player right edge (x + 14) is touching wall at x=520 -> x = 506.0
        env.env.player.x = 506.0
        env.env.player.y = 100.0
        env.prev_player_pos = (506.0, 100.0)
        # Command right into wall: actual displacement = 0
        _, _, _, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        assert env.is_stuck
        # Accumulated blocked penalties should only be -0.28 + -0.05 = -0.33
        assert math.isclose(env.blocked_penalty_accumulated, -0.33, abs_tol=1e-5)

        # Trigger genuine recovery: 10 unblocked steps + move >= 40px away in open space
        env.unblocked_steps = 9
        env.stuck_anchor_pos = (200.0, 200.0)
        env.env.player.x = 100.0  # 100px away from anchor, open space
        env.env.player.y = 200.0
        env.prev_player_pos = (100.0, 200.0)
        _, _, _, _, info = env.step(np.array([-1.0, 0.0], dtype=np.float32))

        # Recovery reward = 50% of 0.33 = +0.165 (does NOT refund the -2 revisit)
        assert math.isclose(info["recovery_reward"], 0.165, abs_tol=1e-3)

        env.close()

    def test_08_reset_clears_all_visit_counts(self):
        """Episode reset must clear all prior visit counts and establish new initial cell."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)

        # Accumulate some visits
        env.cell_visit_counts[(1, 1)] = 5
        env.cell_visit_counts[(2, 2)] = 3

        # Reset
        env.reset(seed=99)
        spawn_cell = env.current_confirmed_cell
        assert len(env.cell_visit_counts) == 1
        assert env.cell_visit_counts[spawn_cell] == 1
        assert (1, 1) not in env.cell_visit_counts or (1, 1) == spawn_cell
        env.close()


class TestStage1ImmunityAndCompatibility:
    """Verify that Stage 1 remains completely untouched and compatible."""

    def test_01_stage1_never_incurs_revisit_penalties(self):
        """Stage 1 does not track or penalize cell revisits."""
        env = StealthGymEnv(stage=1)
        env.reset(seed=42)

        # Rapidly loop around
        for _ in range(60):
            _, reward, _, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
            assert info["revisit_penalty_this_step"] == 0.0
            # Step reward should only be step penalty (-0.01) + optional progress reward
            assert reward >= STAGE1_STEP_PENALTY

        env.close()

    def test_02_stage1_has_no_forced_detour_requirement(self):
        """Stage 1 layouts do not require obstacle obstruction (there are no central obstacles)."""
        env = StealthGymEnv(stage=1)
        obs, _ = env.reset(seed=42)
        assert len(env.env.obstacles) == 0
        env.close()

    def test_03_observation_size_remains_strictly_43(self):
        """Observation size must remain exactly 43 float32 values."""
        env_s1 = StealthGymEnv(stage=1)
        env_s2 = StealthGymEnv(stage=2)

        obs1, _ = env_s1.reset(seed=1)
        obs2, _ = env_s2.reset(seed=2)

        assert obs1.shape == (43,)
        assert obs2.shape == (43,)
        assert obs1.dtype == np.float32
        assert obs2.dtype == np.float32

        assert env_s1.observation_space.shape == (43,)
        assert env_s2.observation_space.shape == (43,)

        env_s1.close()
        env_s2.close()

    def test_04_path_efficiency_diagnostic_bounded_at_one(self):
        """Path efficiency diagnostic must be strictly <= 1.0 (<= 100%)."""
        env = StealthGymEnv(stage=1)
        env.reset(seed=42)

        # Move directly forward
        for _ in range(30):
            _, _, _, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
            eff = info["path_efficiency"]
            assert 0.0 <= eff <= 1.0, f"Path efficiency {eff} exceeded 1.0!"

        env.close()
