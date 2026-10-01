"""
Comprehensive Unit and Behavioral Tests for Stage 2 Stuck, Stagnation, and Recovery Systems.

Covers Sections 41, 42, 43, and 44:
- Section 41: Stuck State Tests (1..5)
- Section 42: Stagnation Tests (1..7)
- Section 43: Recovery Tests (1..9)
- Section 44: Reward Sanity Tests (A..F)
"""

import math
import numpy as np
import pytest

from config import (
    RL_DT,
    STAGE2_BLOCKED_STEP_PENALTY,
    STAGE2_PROLONGED_BLOCKED_PENALTY,
    STAGE2_PROLONGED_BLOCKED_STEPS,
    STAGE2_STAGNATION_STEPS,
    STAGE2_STAGNATION_RADIUS,
    STAGE2_STAGNATION_PENALTY,
    STAGE2_STAGNATION_ESCAPE_DISTANCE,
    STAGE2_RECOVERY_STEPS,
    STAGE2_RECOVERY_ESCAPE_DISTANCE,
    STAGE2_RECOVERY_REFUND_FRACTION,
    STAGE2_RECOVERY_MAX_REWARD,
    STAGE2_TIMEOUT_PENALTY,
    GOAL_REWARD,
)
from rl_environment import StealthGymEnv


class TestStuckState:
    """Section 41: Stuck State Tests."""

    def test_01_under_15_blocked_steps_not_stuck(self):
        """1. Less than 15 blocked steps: stuck_flag == 0.0."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        # Position player flush against left wall (x=49, radius=14, left wall at x=35)
        env.env.player.x = 49.0
        env.env.player.y = 350.0

        # Command movement into the wall (-1.0, 0.0) for 14 steps
        for step in range(14):
            obs, reward, term, trunc, info = env.step(np.array([-1.0, 0.0], dtype=np.float32))
            assert info["is_blocked"] is True
            assert info["consecutive_blocked_steps"] == step + 1
            assert info["stuck_flag"] == 0.0
            assert obs[41] == 0.0
            # Blocked steps 1-14 incur -0.02 blocked penalty
            assert math.isclose(info["blocked_penalty"], STAGE2_BLOCKED_STEP_PENALTY)
        env.close()

    def test_02_at_15_blocked_steps_stuck_flag_becomes_1(self):
        """2. At 15 consecutive blocked steps: stuck_flag == 1.0 and prolonged penalty fires."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        env.env.player.x = 49.0
        env.env.player.y = 350.0

        for step in range(14):
            env.step(np.array([-1.0, 0.0], dtype=np.float32))

        # 15th step
        obs, reward, term, trunc, info = env.step(np.array([-1.0, 0.0], dtype=np.float32))
        assert info["is_blocked"] is True
        assert info["consecutive_blocked_steps"] == 15
        assert info["is_stuck"] is True
        assert info["stuck_flag"] == 1.0
        assert obs[41] == 1.0
        # 15th step incurs prolonged blocked penalty (-0.05)
        assert math.isclose(info["blocked_penalty"], STAGE2_PROLONGED_BLOCKED_PENALTY)
        env.close()

    def test_03_genuine_recovery_eventually_clears_stuck_flag(self):
        """3. Genuine recovery clears stuck flag."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        # Move into left wall to become stuck
        env.env.player.x = 49.0
        env.env.player.y = 350.0
        for _ in range(15):
            env.step(np.array([-1.0, 0.0], dtype=np.float32))
        assert env.is_stuck is True

        # Now move right (+x, away from left wall into open arena)
        # Each step moves PLAYER_SPEED * RL_DT ≈ 6.1667 px
        # In 10 steps, moves ~61.67 px (> 40.0 px escape distance)
        for i in range(9):
            obs, _, _, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
            assert info["is_blocked"] is False
            # Still stuck until 10 steps satisfied!
            assert info["is_stuck"] is True
            assert obs[41] == 1.0

        # Step 10: 10th consecutive unblocked step AND > 40 px displacement
        obs10, reward10, _, _, info10 = env.step(np.array([1.0, 0.0], dtype=np.float32))
        assert info10["is_stuck"] is False
        assert info10["stuck_flag"] == 0.0
        assert obs10[41] == 0.0
        assert info10["recovery_reward"] > 0.0
        env.close()

    def test_04_one_movement_frame_does_not_clear_stuck_flag(self):
        """4. One single unblocked movement frame does NOT clear stuck flag."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        env.env.player.x = 49.0
        env.env.player.y = 350.0
        for _ in range(15):
            env.step(np.array([-1.0, 0.0], dtype=np.float32))
        assert env.is_stuck is True

        # 1 single unblocked step
        obs, _, _, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        assert info["is_blocked"] is False
        assert info["is_stuck"] is True
        assert obs[41] == 1.0
        assert info["unblocked_steps"] == 1
        assert info["recovery_reward"] == 0.0
        env.close()

    def test_05_episode_reset_clears_stuck_state(self):
        """5. Episode reset completely clears stuck state."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        env.env.player.x = 49.0
        env.env.player.y = 350.0
        for _ in range(16):
            env.step(np.array([-1.0, 0.0], dtype=np.float32))
        assert env.is_stuck is True

        obs, info = env.reset(seed=99)
        assert env.is_stuck is False
        assert env.consecutive_blocked_steps == 0
        assert env.unblocked_steps == 0
        assert env.blocked_penalty_accumulated == 0.0
        assert obs[41] == 0.0
        assert info["is_stuck"] is False
        assert info["stuck_flag"] == 0.0
        env.close()


class TestStagnation:
    """Section 42: Stagnation Tests."""

    def test_01_stagnant_under_60_steps_no_penalty(self):
        """1. Remaining within stagnation radius for <2s (59 steps) gives NO -25 penalty."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        # Stand still (magnitude 0.0 < deadzone)
        for step in range(59):
            obs, reward, term, trunc, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
            assert info["stagnation_steps"] == step + 1
            assert info["stagnation_penalty"] == 0.0
            assert info["stagnation_penalty_triggered"] is False
            assert obs[42] < 1.0
        env.close()

    def test_02_reaching_60_stagnant_steps_fires_exactly_one_minus_25(self):
        """2. Reaching 60 stagnant steps triggers exactly one -25.0 penalty."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        for _ in range(59):
            env.step(np.array([0.0, 0.0], dtype=np.float32))

        # 60th step
        obs, reward, term, trunc, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert info["stagnation_steps"] == 60
        assert math.isclose(info["stagnation_penalty"], STAGE2_STAGNATION_PENALTY)
        assert info["stagnation_penalty_triggered"] is True
        assert obs[42] == 1.0
        # Total step reward includes -0.01 step penalty and -25.0 stagnation penalty = -25.01
        assert math.isclose(reward, -25.01, rel_tol=1e-4)
        env.close()

    def test_03_remaining_stagnant_after_60_steps_does_not_fire_again(self):
        """3. Remaining stagnant for another 60 steps does NOT receive another -25 penalty."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        for _ in range(60):
            env.step(np.array([0.0, 0.0], dtype=np.float32))

        # Run another 60 steps in the same spot
        for step in range(60):
            obs, reward, term, trunc, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
            assert info["stagnation_penalty"] == 0.0
            # Step reward is just normal step penalty (-0.01), NOT -25.0
            assert math.isclose(reward, -0.01, rel_tol=1e-4)
            assert obs[42] == 1.0
        env.close()

    def test_04_moving_40px_away_rearms_stagnation(self):
        """4. Moving >=40 px away rearms stagnation system."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        for _ in range(60):
            env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert env.stagnation_penalty_triggered is True

        # Move right (+x) in open space: 7 steps = 43.16 px >= 40.0 px
        for _ in range(7):
            env.step(np.array([1.0, 0.0], dtype=np.float32))

        assert env.stagnation_penalty_triggered is False
        assert env.stagnation_steps == 0
        env.close()

    def test_05_later_independent_stagnation_fires_again(self):
        """5. Later independent stagnation can receive another -25 after rearming."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        # Event 1: stagnant for 60 steps
        for _ in range(60):
            env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert env.stagnation_penalty_triggered is True

        # Escape >= 40px away (7 steps = 43.16 px)
        for _ in range(7):
            env.step(np.array([1.0, 0.0], dtype=np.float32))
        assert env.stagnation_penalty_triggered is False
        assert env.stagnation_steps == 0

        # Stagnant again at new location for 60 steps
        for _ in range(59):
            env.step(np.array([0.0, 0.0], dtype=np.float32))

        obs, reward, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert math.isclose(info["stagnation_penalty"], STAGE2_STAGNATION_PENALTY)
        assert info["stagnation_penalty_triggered"] is True
        env.close()

    def test_06_escape_before_2sec_resets_counter_and_anchor(self):
        """6. Moving outside 10px radius before 2.0s resets stagnation counter and anchor."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        # Stand still for 30 steps
        for _ in range(30):
            env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert env.stagnation_steps == 30

        # Step 1: moves ~6.17 px (still <= 10 px)
        env.step(np.array([1.0, 0.0], dtype=np.float32))
        assert env.stagnation_steps == 31

        # Step 2: moves to ~12.33 px (> 10.0 px from anchor)
        env.step(np.array([1.0, 0.0], dtype=np.float32))

        # Counter must be reset from 31 to 0 because player left the 10px area
        assert env.stagnation_steps == 0
        assert env.stagnation_penalty_triggered is False
        env.close()

    def test_07_reset_clears_stagnation_state(self):
        """7. Reset clears all stagnation state."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        for _ in range(60):
            env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert env.stagnation_penalty_triggered is True

        obs, info = env.reset(seed=42)
        assert env.stagnation_steps == 0
        assert env.stagnation_penalty_triggered is False
        assert obs[42] == 0.0
        assert info["stagnation_steps"] == 0
        assert info["stagnation_progress"] == 0.0
        env.close()


class TestRecovery:
    """Section 43: Recovery Tests."""

    def test_01_blocked_under_15_steps_then_moving_away_no_recovery(self):
        """1. Blocked < 15 steps then moving away: no recovery reward."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        env.env.player.x = 49.0
        env.env.player.y = 350.0
        # Blocked for 10 steps (< 15)
        for _ in range(10):
            env.step(np.array([-1.0, 0.0], dtype=np.float32))
        assert env.is_stuck is False

        # Move away unblocked for 15 steps
        for _ in range(15):
            obs, reward, _, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
            assert info["recovery_reward"] == 0.0
        env.close()

    def test_02_stuck_followed_by_one_unblocked_frame_no_reward_yet(self):
        """2. Genuine stuck event followed by one unblocked frame: no reward yet."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        env.env.player.x = 49.0
        env.env.player.y = 350.0
        for _ in range(15):
            env.step(np.array([-1.0, 0.0], dtype=np.float32))
        assert env.is_stuck is True

        # 1 unblocked frame
        _, _, _, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        assert info["recovery_reward"] == 0.0
        assert info["is_stuck"] is True
        env.close()

    def test_03_genuine_stuck_plus_10_unblocked_plus_40px_triggers_once(self):
        """3. Genuine stuck + 10 unblocked steps + >=40px displacement triggers recovery exactly once."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        env.env.player.x = 49.0
        env.env.player.y = 350.0
        for _ in range(15):
            env.step(np.array([-1.0, 0.0], dtype=np.float32))

        # 9 unblocked steps
        for _ in range(9):
            _, _, _, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
            assert info["recovery_reward"] == 0.0

        # 10th step
        _, _, _, _, info10 = env.step(np.array([1.0, 0.0], dtype=np.float32))
        assert info10["recovery_reward"] > 0.0

        # 11th step: recovery should NOT trigger again
        _, _, _, _, info11 = env.step(np.array([1.0, 0.0], dtype=np.float32))
        assert info11["recovery_reward"] == 0.0
        env.close()

    def test_04_recovery_refunds_50_percent_blocked_penalties(self):
        """4. Recovery reward equals exactly 50% of accumulated blocked penalties."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        env.env.player.x = 49.0
        env.env.player.y = 350.0

        # Blocked for 20 steps:
        # Steps 1-14: 14 * (-0.02) = -0.28
        # Steps 15-20: 6 * (-0.05) = -0.30
        # Total blocked penalties = -0.58
        for _ in range(20):
            env.step(np.array([-1.0, 0.0], dtype=np.float32))

        expected_blocked_penalty = 14 * (-0.02) + 6 * (-0.05)  # -0.58
        assert math.isclose(env.blocked_penalty_accumulated, expected_blocked_penalty, rel_tol=1e-5)

        # Move 10 steps to recover
        for _ in range(9):
            env.step(np.array([1.0, 0.0], dtype=np.float32))
        _, _, _, _, info10 = env.step(np.array([1.0, 0.0], dtype=np.float32))

        expected_refund = abs(expected_blocked_penalty) * 0.50  # 0.29
        assert math.isclose(info10["recovery_reward"], expected_refund, rel_tol=1e-5)
        env.close()

    def test_05_recovery_capped_at_2_0(self):
        """5. Recovery reward never exceeds +2.0 even after prolonged blocking."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        env.env.player.x = 49.0
        env.env.player.y = 350.0

        # Blocked for 100 steps:
        # 14 * -0.02 + 86 * -0.05 = -0.28 + -4.30 = -4.58
        # 50% of 4.58 is 2.29, but capped at 2.0!
        for _ in range(100):
            env.step(np.array([-1.0, 0.0], dtype=np.float32))

        assert abs(env.blocked_penalty_accumulated) * 0.50 > 2.0

        for _ in range(9):
            env.step(np.array([1.0, 0.0], dtype=np.float32))
        _, _, _, _, info10 = env.step(np.array([1.0, 0.0], dtype=np.float32))

        assert math.isclose(info10["recovery_reward"], STAGE2_RECOVERY_MAX_REWARD)
        assert info10["recovery_reward"] == 2.0
        env.close()

    def test_06_stagnation_penalty_never_refunded(self):
        """6. -25 stagnation penalty is NEVER refunded by recovery reward."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        env.env.player.x = 49.0
        env.env.player.y = 350.0

        # Push wall for 60 steps: both prolonged blocked and stagnation -25 occur!
        for _ in range(60):
            env.step(np.array([-1.0, 0.0], dtype=np.float32))

        # Expected blocked penalty: 14 * -0.02 + 46 * -0.05 = -0.28 + -2.30 = -2.58
        # Stagnation penalty = -25.0
        assert math.isclose(env.blocked_penalty_accumulated, -2.58, rel_tol=1e-5)

        for _ in range(9):
            env.step(np.array([1.0, 0.0], dtype=np.float32))
        _, _, _, _, info10 = env.step(np.array([1.0, 0.0], dtype=np.float32))

        # Recovery reward must be 50% of 2.58 = 1.29, NOT including the -25!
        assert math.isclose(info10["recovery_reward"], 1.29, rel_tol=1e-4)
        env.close()

    def test_07_timeout_never_refunded(self):
        """7. -50 timeout is NEVER refunded."""
        # Timeout occurs at truncation, ending the episode, so recovery cannot refund timeout
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        env.env.player.x = 49.0
        env.env.player.y = 350.0
        # Step until timeout (600 steps)
        trunc = False
        while not trunc:
            obs, reward, term, trunc, info = env.step(np.array([-1.0, 0.0], dtype=np.float32))
        assert trunc is True
        assert info["timeout"] is True
        assert math.isclose(info["timeout_penalty"], STAGE2_TIMEOUT_PENALTY)
        assert info["recovery_reward"] == 0.0
        env.close()

    def test_08_same_event_cannot_pay_recovery_twice(self):
        """8. Same stuck event cannot pay recovery twice."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        env.env.player.x = 49.0
        env.env.player.y = 350.0
        for _ in range(15):
            env.step(np.array([-1.0, 0.0], dtype=np.float32))

        # Move 10 steps to recover
        for _ in range(9):
            env.step(np.array([1.0, 0.0], dtype=np.float32))
        _, _, _, _, info10 = env.step(np.array([1.0, 0.0], dtype=np.float32))
        assert info10["recovery_reward"] > 0.0

        # Continue moving for another 20 unblocked steps
        for _ in range(20):
            _, _, _, _, info_next = env.step(np.array([1.0, 0.0], dtype=np.float32))
            assert info_next["recovery_reward"] == 0.0
        env.close()

    def test_09_anti_farming_proof_net_reward_strictly_negative(self):
        """9. Mathematical proof and empirical verification that stuck/recovery cycle is strictly negative."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        env.env.player.x = 49.0
        env.env.player.y = 350.0

        cycle_reward = 0.0
        # 15 blocked steps
        for _ in range(15):
            _, r, _, _, _ = env.step(np.array([-1.0, 0.0], dtype=np.float32))
            cycle_reward += r

        # 10 recovery steps
        for _ in range(10):
            _, r, _, _, _ = env.step(np.array([1.0, 0.0], dtype=np.float32))
            cycle_reward += r

        # Total cycle reward MUST be strictly negative (punishment > refund)
        # Blocked: 14 * -0.02 + 1 * -0.05 = -0.33
        # Step penalties: 25 steps * -0.01 = -0.25
        # Refund: +0.165
        # Net sum <= -0.415 < 0
        assert cycle_reward < 0.0, f"Net cycle reward should be negative, got {cycle_reward}"
        env.close()


class TestRewardSanity:
    """Section 44: Reward Sanity Tests."""

    def test_A_running_directly_into_wall_and_timing_out(self):
        """A. Running directly into wall and timing out produces strong negative return."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        env.env.player.x = 49.0
        env.env.player.y = 350.0
        total_reward = 0.0
        trunc = False
        while not trunc:
            _, r, _, trunc, _ = env.step(np.array([-1.0, 0.0], dtype=np.float32))
            total_reward += r

        # Includes -50 timeout, -25 stagnation, prolonged blocked penalties, step penalties
        assert total_reward < -80.0, f"Expected strong negative return (< -80), got {total_reward}"
        env.close()

    def test_B_getting_stuck_then_recovering_better_than_staying_stuck(self):
        """B. Getting stuck then recovering has negative stuck return, but better than staying stuck."""
        # Case 1: Stay stuck for 25 steps
        env1 = StealthGymEnv(stage=2)
        env1.reset(seed=42)
        env1.env.player.x = 49.0
        env1.env.player.y = 350.0
        r_stay_stuck = 0.0
        for _ in range(25):
            _, r, _, _, _ = env1.step(np.array([-1.0, 0.0], dtype=np.float32))
            r_stay_stuck += r
        env1.close()

        # Case 2: 15 steps stuck + 10 steps recovery
        env2 = StealthGymEnv(stage=2)
        env2.reset(seed=42)
        env2.env.player.x = 49.0
        env2.env.player.y = 350.0
        r_recover = 0.0
        for _ in range(15):
            _, r, _, _, _ = env2.step(np.array([-1.0, 0.0], dtype=np.float32))
            r_recover += r
        for _ in range(10):
            _, r, _, _, _ = env2.step(np.array([1.0, 0.0], dtype=np.float32))
            r_recover += r
        env2.close()

        # Recovering is strictly better than continuing to push the wall
        assert r_recover > r_stay_stuck, f"Recovering ({r_recover}) should be better than staying stuck ({r_stay_stuck})"

    def test_C_successful_efficient_navigation_large_positive_return(self):
        """C. Efficient obstacle navigation gives large positive episode return (> +80)."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        # Position player close to goal with direct unobstructed line
        env.env.set_layout((900.0, 105.0), (990.0, 105.0))
        total_reward = 0.0
        done = False
        while not done:
            # Move directly +x towards goal
            _, r, term, trunc, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
            total_reward += r
            done = term or trunc

        assert info["is_success"] is True
        assert total_reward > 80.0, f"Expected large positive return, got {total_reward}"
        env.close()

    def test_D_sideways_detour_no_distance_punishment(self):
        """D. Sideways detour receives 0 progress reward, NOT negative punishment."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        # Position player at (700, 150) in clear arena, goal at (700, 50)
        env.env.set_layout((700.0, 150.0), (700.0, 50.0))
        dist = math.hypot(700.0 - 700.0, 150.0 - 50.0)
        env.best_distance_to_goal = dist
        env.initial_distance_to_goal = dist

        # Move sideways in +x direction (perpendicular to goal direction) for 0.6 seconds (18 steps)
        # Goal is at (700, 50), moving sideways increases Euclidean distance slightly
        step_rewards = []
        for _ in range(18):
            _, r, _, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
            step_rewards.append(r)

        # Step penalty is -0.01. Progress reward should be 0.0 (NO negative progress punishment)
        for r in step_rewards:
            assert math.isclose(r, -0.01, rel_tol=1e-3), f"Expected -0.01 step penalty without progress penalty, got {r}"
        env.close()

    def test_E_repeatedly_staying_in_one_area_triggers_stagnation(self):
        """E. Repeatedly staying in one tiny area triggers -25 stagnation event."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        penalties = []
        for _ in range(65):
            _, r, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
            if info["stagnation_penalty"] < 0:
                penalties.append(info["stagnation_penalty"])

        assert len(penalties) == 1
        assert math.isclose(penalties[0], STAGE2_STAGNATION_PENALTY)
        env.close()

    def test_F_stagnation_cannot_fire_every_frame(self):
        """F. Stagnation cannot fire -25 every frame after firing."""
        env = StealthGymEnv(stage=2)
        env.reset(seed=42)
        stagnation_events = 0
        for _ in range(120):
            _, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
            if info["stagnation_penalty"] < -1.0:
                stagnation_events += 1

        assert stagnation_events == 1, f"Expected exactly 1 stagnation event, got {stagnation_events}"
        env.close()
