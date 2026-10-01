"""
Unit and Integration Tests for the 43-Dimensional Observation Space (Phase 3B Redesign).

Validates all 13 items specified in Section 40:
1. Observation length exactly 43
2. Observation dtype is float32
3. observation_space.contains(obs) is True
4. Exactly 16 wall/obstacle rays exist (indices 4..19)
5. Rays are evenly spaced by 22.5 degrees (360 / 16)
6. Rays detect central obstacles
7. Rays detect outer arena boundaries
8. Player position indices (0..1) are normalized coordinates
9. Goal vector indices (2..3) are relative normalized coordinates
10. Observer slots (20..40) are formatted correctly (7 values each)
11. Inactive guard slots remain zero-filled
12. Stuck flag index (41) is correct
13. Stagnation progress index (42) stays bounded between 0.0 and 1.0
"""

import math
import numpy as np
import pytest
import pygame

from config import (
    ARENA_WIDTH,
    ARENA_HEIGHT,
    RAY_COUNT,
    RAY_MAX_DISTANCE,
    OBSERVATION_SIZE,
    ROOM_MARGIN,
    OBSTACLE_RECTS,
    BOUNDARY_WALLS,
    ALL_SOLID_RECTS,
)
from geometry import compute_wall_raycasts
from environment import StealthEnvironment
from rl_environment import StealthGymEnv


class TestObservation43Dim:
    """Test suite verifying the 43-dimensional observation vector format and raycasting."""

    def test_01_observation_length_exactly_43(self):
        """1. Observation length must be exactly 43."""
        env = StealthGymEnv(stage=2)
        obs, info = env.reset(seed=42)
        assert len(obs) == 43, f"Expected obs length 43, got {len(obs)}"
        assert obs.shape == (43,), f"Expected shape (43,), got {obs.shape}"

        step_obs, reward, term, trunc, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert len(step_obs) == 43
        assert step_obs.shape == (43,)
        env.close()

    def test_02_observation_dtype_float32(self):
        """2. Observation dtype must be float32."""
        env = StealthGymEnv(stage=2)
        obs, _ = env.reset(seed=123)
        assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
        step_obs, _, _, _, _ = env.step(np.array([1.0, 0.0], dtype=np.float32))
        assert step_obs.dtype == np.float32
        env.close()

    def test_03_observation_space_contains_observation(self):
        """3. observation_space must contain observation across resets and steps."""
        env = StealthGymEnv(stage=2)
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs), "observation_space.contains(obs) failed on reset"

        # Take multiple varied actions
        actions = [
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([-0.5, 0.5], dtype=np.float32),
            np.array([0.0, -1.0], dtype=np.float32),
        ]
        for act in actions:
            obs, _, _, _, _ = env.step(act)
            assert env.observation_space.contains(obs), f"observation_space does not contain step obs for action {act}"
        env.close()

    def test_04_sixteen_wall_rays_exist(self):
        """4. Exactly 16 wall rays exist spanning indices 4 to 19."""
        assert RAY_COUNT == 16, f"Expected RAY_COUNT=16 in config, got {RAY_COUNT}"
        env = StealthGymEnv(stage=2)
        obs, _ = env.reset(seed=42)
        ray_slice = obs[4:20]
        assert len(ray_slice) == 16, f"Expected 16 ray values in obs[4:20], got {len(ray_slice)}"
        # All ray values must be normalized in [0.0, 1.0]
        assert np.all(ray_slice >= 0.0) and np.all(ray_slice <= 1.0), "Wall rays must be in [0.0, 1.0]"
        env.close()

    def test_05_rays_evenly_spaced_22_5_degrees(self):
        """5. Rays must be evenly spaced by exactly 22.5 degrees (2 * pi / 16)."""
        angle_step = (2.0 * math.pi) / 16.0
        expected_step_deg = 360.0 / 16.0
        assert math.isclose(expected_step_deg, 22.5, rel_tol=1e-7)
        assert math.isclose(angle_step, math.radians(22.5), rel_tol=1e-7)

        # Cast rays in an empty arena with a single vertical wall at x=200
        origin = (100.0, 100.0)
        wall = [pygame.Rect(200, 0, 10, 300)]
        rays = compute_wall_raycasts(origin, 16, 300.0, wall)
        assert len(rays) == 16

        # Ray 0 is at angle 0 (pointing +x towards the wall at x=200)
        # Distance to wall edge (x=200) from x=100 is 100.0 px -> normalized: 100/300 = 0.3333
        assert math.isclose(rays[0], 100.0 / 300.0, rel_tol=1e-3)

        # Ray 4 is at angle 4 * 22.5 = 90 degrees (pointing +y: (0, 1))
        # No wall in +y direction within 300 px -> distance should be max_dist = 1.0
        assert math.isclose(rays[4], 1.0, rel_tol=1e-3)

    def test_06_rays_detect_central_obstacles(self):
        """6. Rays detect central obstacles."""
        raw_env = StealthEnvironment(stage=2)
        # Position player just to the left of the left horizontal obstacle:
        # Left horizontal wall is Rect(175, 320, 250, 60)
        # Place player at (150, 350)
        raw_env.player.x = 150.0
        raw_env.player.y = 350.0
        obs = raw_env.get_observation()
        # Ray 0 points at 0 deg (+x, directly hitting the left obstacle edge at x=175)
        # Expected distance = 175 - 150 = 25 px -> 25 / 300 = 0.0833
        ray_0 = obs[4]
        assert math.isclose(ray_0, 25.0 / 300.0, rel_tol=1e-2), f"Expected ~0.0833, got {ray_0}"

    def test_07_rays_detect_outer_boundaries(self):
        """7. Rays detect outer boundary walls."""
        raw_env = StealthEnvironment(stage=2)
        # Place player near the left boundary wall: left wall is Rect(0, 0, 35, 700)
        # Place player at (55, 350). Distance to left wall right edge (x=35) is 20 px.
        raw_env.player.x = 55.0
        raw_env.player.y = 350.0
        obs = raw_env.get_observation()
        # Ray at 180 degrees is index 8 (8 * 22.5 = 180 deg, pointing -x)
        # Distance = 55 - 35 = 20 px -> 20 / 300 = 0.0667
        ray_180 = obs[4 + 8]
        assert math.isclose(ray_180, 20.0 / 300.0, rel_tol=1e-2), f"Expected ~0.0667, got {ray_180}"

    def test_08_player_position_indices_correct(self):
        """8. Indices 0 and 1 are player normalized absolute coordinates [0, 1]."""
        raw_env = StealthEnvironment(stage=2)
        raw_env.player.x = 220.0
        raw_env.player.y = 350.0
        obs = raw_env.get_observation()
        assert math.isclose(obs[0], 220.0 / ARENA_WIDTH, rel_tol=1e-5)
        assert math.isclose(obs[1], 350.0 / ARENA_HEIGHT, rel_tol=1e-5)

    def test_09_goal_vector_indices_correct(self):
        """9. Indices 2 and 3 are goal relative coordinates in [-1, 1]."""
        raw_env = StealthEnvironment(stage=2)
        raw_env.set_layout((200.0, 300.0), (500.0, 600.0))
        obs = raw_env.get_observation()
        expected_dx = (500.0 - 200.0) / ARENA_WIDTH
        expected_dy = (600.0 - 300.0) / ARENA_HEIGHT
        assert math.isclose(obs[2], expected_dx, rel_tol=1e-5)
        assert math.isclose(obs[3], expected_dy, rel_tol=1e-5)

    def test_10_observer_slots_correct_and_indexed(self):
        """10. Observers 1, 2, 3 occupy indices 20..26, 27..33, 34..40 (7 floats each).
        Values: [active, rel_x, rel_y, facing_cos, facing_sin, norm_vision_range, norm_fov]
        """
        raw_env = StealthEnvironment(stage=4)  # Stage 4 activates all 3 observers
        obs = raw_env.get_observation()
        # Verify all 3 observers occupy exact 7-slot blocks with correct semantics
        for i, observer in enumerate(raw_env.observers):
            base_idx = 20 + i * 7
            # 1. active flag
            assert math.isclose(obs[base_idx + 0], 1.0), f"Observer {i+1} active flag at obs[{base_idx+0}] should be 1.0"
            # 2. relative X
            expected_rx = (observer.x - raw_env.player.x) / ARENA_WIDTH
            assert math.isclose(obs[base_idx + 1], expected_rx, rel_tol=1e-5), f"Observer {i+1} rel_x mismatch"
            # 3. relative Y
            expected_ry = (observer.y - raw_env.player.y) / ARENA_HEIGHT
            assert math.isclose(obs[base_idx + 2], expected_ry, rel_tol=1e-5), f"Observer {i+1} rel_y mismatch"
            # 4. facing cos
            expected_cos = math.cos(observer.facing_angle)
            assert math.isclose(obs[base_idx + 3], expected_cos, rel_tol=1e-5), f"Observer {i+1} facing_cos mismatch"
            # 5. facing sin
            expected_sin = math.sin(observer.facing_angle)
            assert math.isclose(obs[base_idx + 4], expected_sin, rel_tol=1e-5), f"Observer {i+1} facing_sin mismatch"
            # 6. normalized vision range
            expected_range = observer.vision_distance / ARENA_WIDTH
            assert math.isclose(obs[base_idx + 5], expected_range, rel_tol=1e-5), f"Observer {i+1} vision_range mismatch"
            # 7. normalized FOV
            expected_fov = observer.vision_angle_rad / math.pi
            assert math.isclose(obs[base_idx + 6], expected_fov, rel_tol=1e-5), f"Observer {i+1} fov mismatch"

    def test_11_inactive_guard_slots_remain_zero(self):
        """11. In Stage 1 and Stage 2, observer slots remain exactly zero-filled."""
        raw_env = StealthEnvironment(stage=2)
        obs = raw_env.get_observation()
        observer_slice = obs[20:41]
        assert np.all(observer_slice == 0.0), f"Observer slice in Stage 2 must be all zeros: {observer_slice}"

        raw_env1 = StealthEnvironment(stage=1)
        obs1 = raw_env1.get_observation()
        assert np.all(obs1[20:41] == 0.0), "Observer slice in Stage 1 must be all zeros"

    def test_12_stuck_flag_index_correct(self):
        """12. Index 41 is the stuck flag (0.0 when not stuck, 1.0 when stuck)."""
        env = StealthGymEnv(stage=2)
        obs, _ = env.reset(seed=42)
        assert obs[41] == 0.0, f"Expected stuck_flag=0.0 on reset, got {obs[41]}"

        # Artificially set is_stuck to True and verify get_observation output
        env.is_stuck = True
        obs_stuck = env.env.get_observation(stuck_flag=1.0, stagnation_progress=0.5)
        assert obs_stuck[41] == 1.0, f"Expected stuck_flag=1.0 at index 41, got {obs_stuck[41]}"
        env.close()

    def test_13_stagnation_progress_index_stays_between_0_and_1(self):
        """13. Index 42 is stagnation progress, staying bounded between 0.0 and 1.0."""
        env = StealthGymEnv(stage=2)
        obs, _ = env.reset(seed=42)
        assert obs[42] == 0.0, f"Expected stagnation_progress=0.0 on reset, got {obs[42]}"

        # Test various stagnation progress values
        for steps in [0, 15, 30, 60, 100]:
            prog = min(steps / 60.0, 1.0)
            obs_test = env.env.get_observation(stuck_flag=0.0, stagnation_progress=prog)
            assert 0.0 <= obs_test[42] <= 1.0
            assert math.isclose(obs_test[42], prog, rel_tol=1e-5)
        env.close()
