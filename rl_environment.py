"""
Reinforcement Learning Gymnasium environment for the 2D Stealth Sandbox.
Wraps the existing StealthEnvironment without modifying its core mechanics.
Provides:
  - Continuous action space Box(-1.0, 1.0, shape=(2,))
  - 33-dimensional float32 numerical observation space
  - Configurable 5-stage curriculum learning
  - Non-exploitable new-best progress reward calculation
  - Clear episode termination and timeout truncation
  - Headless and human render modes
"""

import math
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from config import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    ARENA_WIDTH,
    ARENA_HEIGHT,
    RL_DT,
    ACTION_DEADZONE,
    OBSERVATION_SIZE,
    GOAL_REWARD,
    DETECTION_PENALTY,
    PROGRESS_CHECK_INTERVAL,
    MIN_PROGRESS_DISTANCE,
    PROGRESS_REWARD_SCALE,
    STAGE1_PROGRESS_REWARD_SCALE,
    STAGE2_PROGRESS_REWARD_SCALE,
    MAX_EPISODE_TIME,
    STAGE1_STEP_PENALTY,
    STAGE2_STEP_PENALTY,
    STAGE2_MAX_EPISODE_TIME,
    STAGE2_TIMEOUT_PENALTY,
    STAGE2_BLOCKED_STEP_PENALTY,
    STAGE2_PROLONGED_BLOCKED_PENALTY,
    STAGE2_BLOCKED_RATIO_THRESHOLD,
    STAGE2_PROLONGED_BLOCKED_STEPS,
    STAGE2_STAGNATION_TIME,
    STAGE2_STAGNATION_STEPS,
    STAGE2_STAGNATION_RADIUS,
    STAGE2_STAGNATION_PENALTY,
    STAGE2_STAGNATION_ESCAPE_DISTANCE,
    STAGE2_RECOVERY_STEPS,
    STAGE2_RECOVERY_ESCAPE_DISTANCE,
    STAGE2_RECOVERY_REFUND_FRACTION,
    STAGE2_RECOVERY_MAX_REWARD,
    STAGE2_REVISIT_CELL_SIZE,
    STAGE2_REVISIT_CONFIRM_STEPS,
    STAGE2_REVISIT_PENALTY,
    STAGE2_REVISIT_FREE_VISITS,
    PLAYER_SPEED,
    PLAYER_START_POS,
    GOAL_POS,
)
from environment import (
    StealthEnvironment,
    sample_stage1_positions,
    sample_stage2_positions,
)


class StealthGymEnv(gym.Env):
    """
    Gymnasium-compatible continuous 2D Stealth Environment.
    
    Observation Space (43 float32 values):
      [0:2]   Player normalized position (x, y) in [0.0, 1.0]
      [2:4]   Goal relative position (dx, dy) in [-1.0, 1.0]
      [4:20]  16 wall/obstacle raycast distances in [0.0, 1.0] (every 22.5 deg)
      [20:27] Observer 1: [active, rel_x, rel_y, facing_x, facing_y, norm_vision_range, norm_fov]
      [27:34] Observer 2: [active, rel_x, rel_y, facing_x, facing_y, norm_vision_range, norm_fov]
      [34:41] Observer 3: [active, rel_x, rel_y, facing_x, facing_y, norm_vision_range, norm_fov]
      [41]    Stuck flag: 0.0 (not stuck) or 1.0 (stuck)
      [42]    Stagnation progress: min(stagnation_steps / 60.0, 1.0) in [0.0, 1.0]

    Action Space:
      Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
      [dx, dy] continuous direction vector.
      Magnitudes below ACTION_DEADZONE (0.05) indicate stationary waiting.
      Magnitudes above ACTION_DEADZONE are normalized to unit direction with constant speed.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, stage: int = 1, render_mode: Optional[str] = None):
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render_mode '{render_mode}'. Valid modes: {self.metadata['render_modes']}")

        self.stage = int(stage)
        self.render_mode = render_mode

        # 1. Action Space: Continuous 2D vector in [-1.0, 1.0]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # 2. Observation Space: 43 float32 values bounded in [-1.5, 1.5]
        self.observation_space = spaces.Box(
            low=-1.5,
            high=1.5,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32
        )

        # 3. Underlying deterministic simulation environment
        self.env = StealthEnvironment(stage=self.stage)

        # 4. Simulation and Reward Tracking
        self.episode_time = 0.0
        self.time_since_progress_check = 0.0
        self.best_distance_to_goal = 0.0
        self.initial_distance_to_goal = 0.0
        self.distance_travelled = 0.0
        self.prev_player_pos = (0.0, 0.0)
        self.wall_contact_steps = 0
        self.total_episode_steps = 0

        # Stuck & Recovery Tracking
        self.consecutive_blocked_steps = 0
        self.is_stuck = False
        self.stuck_anchor_pos: Optional[Tuple[float, float]] = None
        self.blocked_penalty_accumulated = 0.0
        self.unblocked_steps = 0
        self.recovery_reward_this_step = 0.0

        # Stagnation Tracking
        self.stagnation_anchor: Optional[Tuple[float, float]] = None
        self.stagnation_steps = 0
        self.stagnation_penalty_triggered = False
        self.stagnation_penalty_this_step = 0.0

        # Revisit / Looping Tracking (Stage 2)
        self.current_confirmed_cell: Optional[Tuple[int, int]] = None
        self.cell_visit_counts: Dict[Tuple[int, int], int] = {}
        self.candidate_cell: Optional[Tuple[int, int]] = None
        self.candidate_cell_steps: int = 0
        self.revisit_penalty_this_step: float = 0.0
        self.spawn_pos: Tuple[float, float] = (0.0, 0.0)

        # 5. Rendering Resources
        self.screen = None
        self.clock = None
        self.overlay_callback = None

    def set_curriculum_stage(self, stage: int):
        """Switch curriculum stage (1 to 5) dynamically without recreating the environment."""
        self.stage = int(stage)
        self.env.set_curriculum_stage(self.stage)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.
        Restores initial positions, deterministic patrols, timer, and progress tracker.
        In Stage 1: Randomizes player spawn and goal positions reproducibly via self.np_random.
        In Stage 2: Randomizes player spawn and goal positions with obstacle clearance.
        In Stages 3-5: Preserves fixed player start and goal positions.
        """
        super().reset(seed=seed)

        if options and "stage" in options:
            self.set_curriculum_stage(options["stage"])

        self.env.reset()

        if self.stage == 1:
            p_pos, g_pos = sample_stage1_positions(self.np_random)
            self.env.set_layout(p_pos, g_pos)
        elif self.stage == 2:
            p_pos, g_pos = sample_stage2_positions(self.np_random, self.env.obstacles)
            self.env.set_layout(p_pos, g_pos)
        else:
            self.env.set_layout(PLAYER_START_POS, GOAL_POS)

        self.episode_time = 0.0
        self.time_since_progress_check = 0.0
        self.wall_contact_steps = 0
        self.total_episode_steps = 0

        # Reset stuck & recovery state
        self.consecutive_blocked_steps = 0
        self.is_stuck = False
        self.stuck_anchor_pos = None
        self.blocked_penalty_accumulated = 0.0
        self.unblocked_steps = 0
        self.recovery_reward_this_step = 0.0

        # Reset stagnation state
        self.stagnation_anchor = (float(self.env.player.x), float(self.env.player.y))
        self.stagnation_steps = 0
        self.stagnation_penalty_triggered = False
        self.stagnation_penalty_this_step = 0.0

        # Reset revisit / looping state (Stage 2)
        if self.stage == 2:
            spawn_cell = (
                int(math.floor(self.env.player.x / STAGE2_REVISIT_CELL_SIZE)),
                int(math.floor(self.env.player.y / STAGE2_REVISIT_CELL_SIZE))
            )
            self.current_confirmed_cell = spawn_cell
            self.cell_visit_counts = {spawn_cell: 1}
            self.candidate_cell = None
            self.candidate_cell_steps = 0
            self.revisit_penalty_this_step = 0.0
        else:
            self.current_confirmed_cell = None
            self.cell_visit_counts = {}
            self.candidate_cell = None
            self.candidate_cell_steps = 0
            self.revisit_penalty_this_step = 0.0

        self.spawn_pos = (float(self.env.player.x), float(self.env.player.y))
        current_dist = math.hypot(
            self.env.player.x - self.env.goal_pos[0],
            self.env.player.y - self.env.goal_pos[1]
        )
        self.best_distance_to_goal = current_dist
        self.initial_distance_to_goal = current_dist
        self.distance_travelled = 0.0
        self.prev_player_pos = (self.env.player.x, self.env.player.y)

        obs = self.env.get_observation(stuck_flag=0.0, stagnation_progress=0.0)
        info = self._build_info(
            success=False,
            detected=False,
            timeout=False,
            step_progress_reward=0.0
        )

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one RL simulation step with fixed timestep RL_DT.
        
        Args:
            action: 2D array [dx, dy] in [-1.0, 1.0].
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # 1. Action Deadzone and Direction Normalization
        ax = float(action[0])
        ay = float(action[1])
        mag = math.hypot(ax, ay)

        if mag < ACTION_DEADZONE:
            move_dir = (0.0, 0.0)
            expected_dist = 0.0
        else:
            # Normalize direction so network chooses direction, preserving constant player speed
            move_dir = (ax / mag, ay / mag)
            expected_dist = PLAYER_SPEED * RL_DT

        # 2. Advance simulation by fixed RL_DT
        prev_x, prev_y = self.env.player.x, self.env.player.y
        self.env.step(move_dir, RL_DT)
        actual_dist = math.hypot(self.env.player.x - prev_x, self.env.player.y - prev_y)
        self.episode_time += RL_DT
        self.time_since_progress_check += RL_DT
        self.distance_travelled += actual_dist
        self.prev_player_pos = (self.env.player.x, self.env.player.y)
        self.total_episode_steps += 1
        if getattr(self.env.player, "is_colliding", False):
            self.wall_contact_steps += 1

        # 3. Blocked Movement & Stuck Tracking
        is_blocked = (mag >= ACTION_DEADZONE and actual_dist < (expected_dist * STAGE2_BLOCKED_RATIO_THRESHOLD))
        blocked_penalty = 0.0
        recovery_reward = 0.0

        if is_blocked:
            self.consecutive_blocked_steps += 1
            self.unblocked_steps = 0

            # Stuck escalation at 15 steps (0.5s at 30Hz)
            if self.consecutive_blocked_steps >= STAGE2_PROLONGED_BLOCKED_STEPS:
                if not self.is_stuck:
                    self.is_stuck = True
                    self.stuck_anchor_pos = (float(self.env.player.x), float(self.env.player.y))
                if self.stage == 2:
                    blocked_penalty = STAGE2_PROLONGED_BLOCKED_PENALTY  # -0.05
            else:
                if self.stage == 2:
                    blocked_penalty = STAGE2_BLOCKED_STEP_PENALTY  # -0.02

            if self.stage == 2:
                self.blocked_penalty_accumulated += blocked_penalty
        else:
            self.consecutive_blocked_steps = 0
            if self.is_stuck:
                self.unblocked_steps += 1
                dist_from_stuck_anchor = (
                    math.hypot(
                        self.env.player.x - self.stuck_anchor_pos[0],
                        self.env.player.y - self.stuck_anchor_pos[1]
                    )
                    if self.stuck_anchor_pos is not None
                    else 0.0
                )
                # Genuine recovery check:
                # 1. was stuck (is_stuck is True)
                # 2. unblocked
                # 3. remains unblocked >= 10 consecutive steps
                # 4. moved >= 40 px from stuck anchor
                if self.unblocked_steps >= STAGE2_RECOVERY_STEPS and dist_from_stuck_anchor >= STAGE2_RECOVERY_ESCAPE_DISTANCE:
                    if self.stage == 2:
                        recovery_reward = min(
                            STAGE2_RECOVERY_MAX_REWARD,
                            abs(self.blocked_penalty_accumulated) * STAGE2_RECOVERY_REFUND_FRACTION
                        )
                    # Clear stuck state
                    self.is_stuck = False
                    self.stuck_anchor_pos = None
                    self.blocked_penalty_accumulated = 0.0
                    self.unblocked_steps = 0
            else:
                self.blocked_penalty_accumulated = 0.0
                self.unblocked_steps = 0

        self.recovery_reward_this_step = recovery_reward

        # 4. Stagnation Tracking
        stagnation_penalty = 0.0
        if self.stagnation_anchor is None:
            self.stagnation_anchor = (float(self.env.player.x), float(self.env.player.y))

        dist_from_stagnation_anchor = math.hypot(
            self.env.player.x - self.stagnation_anchor[0],
            self.env.player.y - self.stagnation_anchor[1]
        )

        if not self.stagnation_penalty_triggered:
            if dist_from_stagnation_anchor <= STAGE2_STAGNATION_RADIUS:
                self.stagnation_steps += 1
                if self.stagnation_steps >= STAGE2_STAGNATION_STEPS:
                    if self.stage == 2:
                        stagnation_penalty = STAGE2_STAGNATION_PENALTY  # -25.0
                    self.stagnation_penalty_triggered = True
            else:
                # Meaningfully left the 10px area before penalty fired: reset anchor and counter
                self.stagnation_anchor = (float(self.env.player.x), float(self.env.player.y))
                self.stagnation_steps = 0
        else:
            # Stagnation penalty already fired; check for genuine escape >= 40 px to rearm
            if dist_from_stagnation_anchor >= STAGE2_STAGNATION_ESCAPE_DISTANCE:
                self.stagnation_penalty_triggered = False
                self.stagnation_steps = 0
                self.stagnation_anchor = (float(self.env.player.x), float(self.env.player.y))
            else:
                # Still within 40px; penalty fired once, so no additional penalty
                pass

        self.stagnation_penalty_this_step = stagnation_penalty

        # 4.3 Stage 2 Revisit / Anti-Loop Tracking
        revisit_penalty = 0.0
        if self.stage == 2:
            raw_cell = (
                int(math.floor(self.env.player.x / STAGE2_REVISIT_CELL_SIZE)),
                int(math.floor(self.env.player.y / STAGE2_REVISIT_CELL_SIZE))
            )
            if raw_cell == self.current_confirmed_cell:
                self.candidate_cell = None
                self.candidate_cell_steps = 0
            else:
                if raw_cell == self.candidate_cell:
                    self.candidate_cell_steps += 1
                else:
                    self.candidate_cell = raw_cell
                    self.candidate_cell_steps = 1

                if self.candidate_cell_steps >= STAGE2_REVISIT_CONFIRM_STEPS:
                    self.current_confirmed_cell = self.candidate_cell
                    new_visits = self.cell_visit_counts.get(self.current_confirmed_cell, 0) + 1
                    self.cell_visit_counts[self.current_confirmed_cell] = new_visits
                    if new_visits >= 3:
                        revisit_penalty = STAGE2_REVISIT_PENALTY  # -2.0
                    self.candidate_cell = None
                    self.candidate_cell_steps = 0

        self.revisit_penalty_this_step = revisit_penalty

        # 5. Reward and Terminal Condition Evaluation
        reward = 0.0
        terminated = False
        truncated = False
        success = False
        detected = False
        timeout = False
        step_progress_reward = 0.0

        # Check Success (Goal reached)
        if self.env.state == StealthEnvironment.STATE_ESCAPED:
            reward += GOAL_REWARD
            terminated = True
            success = True
            self.consecutive_blocked_steps = 0
        # Check Failure (Detected by active observer)
        elif self.env.state == StealthEnvironment.STATE_DETECTED:
            reward += DETECTION_PENALTY
            terminated = True
            detected = True
            self.consecutive_blocked_steps = 0
        else:
            # Stage-specific step penalty and events
            if self.stage == 1:
                reward += STAGE1_STEP_PENALTY
            elif self.stage == 2:
                reward += STAGE2_STEP_PENALTY
                reward += blocked_penalty
                reward += stagnation_penalty
                reward += recovery_reward
                reward += revisit_penalty

            # Compute Euclidean distance to goal
            current_dist = math.hypot(
                self.env.player.x - self.env.goal_pos[0],
                self.env.player.y - self.env.goal_pos[1]
            )

            # Evaluate progress reward at fixed simulation time intervals
            if self.time_since_progress_check >= (PROGRESS_CHECK_INTERVAL - 1e-5):
                self.time_since_progress_check = 0.0
                if current_dist < (self.best_distance_to_goal - MIN_PROGRESS_DISTANCE):
                    progress_amount = self.best_distance_to_goal - current_dist
                    scale = STAGE2_PROGRESS_REWARD_SCALE if self.stage == 2 else STAGE1_PROGRESS_REWARD_SCALE
                    step_progress_reward = scale * progress_amount
                    reward += step_progress_reward
                    self.best_distance_to_goal = current_dist

            # Check Maximum Episode Duration (Truncation per stage)
            max_time = STAGE2_MAX_EPISODE_TIME if self.stage == 2 else MAX_EPISODE_TIME
            if self.episode_time >= (max_time - 1e-5):
                truncated = True
                timeout = True
                if self.stage == 2:
                    reward += STAGE2_TIMEOUT_PENALTY
                self.consecutive_blocked_steps = 0

        # Build 43-element observation
        stuck_flag = 1.0 if self.is_stuck else 0.0
        stagnation_progress = min(float(self.stagnation_steps) / float(STAGE2_STAGNATION_STEPS), 1.0)
        obs = self.env.get_observation(stuck_flag=stuck_flag, stagnation_progress=stagnation_progress)

        info = self._build_info(
            success=success,
            detected=detected,
            timeout=timeout,
            step_progress_reward=step_progress_reward,
            is_blocked=is_blocked,
            blocked_penalty=blocked_penalty,
            recovery_reward=recovery_reward,
            stagnation_penalty=stagnation_penalty,
            stuck_flag=stuck_flag,
            stagnation_progress=stagnation_progress,
        )

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def _build_info(
        self,
        success: bool,
        detected: bool,
        timeout: bool,
        step_progress_reward: float,
        is_blocked: bool = False,
        blocked_penalty: float = 0.0,
        recovery_reward: float = 0.0,
        stagnation_penalty: float = 0.0,
        stuck_flag: float = 0.0,
        stagnation_progress: float = 0.0,
    ) -> Dict[str, Any]:
        """Construct diagnostic information dictionary."""
        current_dist = math.hypot(
            self.env.player.x - self.env.goal_pos[0],
            self.env.player.y - self.env.goal_pos[1]
        )
        straight_line_dist = math.hypot(
            self.env.player.x - self.spawn_pos[0],
            self.env.player.y - self.spawn_pos[1]
        )
        path_eff = (
            min(1.0, straight_line_dist / max(self.distance_travelled, 1e-5))
            if self.distance_travelled > 0
            else 1.0
        )
        wall_contact_fraction = (
            float(self.wall_contact_steps) / float(max(1, self.total_episode_steps))
        )
        return {
            "episode_time": float(self.episode_time),
            "elapsed_time": float(self.episode_time),
            "distance_to_goal": float(current_dist),
            "best_distance_to_goal": float(self.best_distance_to_goal),
            "initial_distance_to_goal": float(self.initial_distance_to_goal),
            "distance_travelled": float(self.distance_travelled),
            "path_efficiency": float(path_eff),
            "success": bool(success),
            "is_success": bool(success),
            "detected": bool(detected),
            "timeout": bool(timeout),
            "curriculum_stage": int(self.stage),
            "active_observers": len(self.env.active_observers),
            "progress_reward_this_step": float(step_progress_reward),
            "wall_contact_steps": int(self.wall_contact_steps),
            "wall_contact_fraction": float(wall_contact_fraction),
            "is_blocked": bool(is_blocked),
            "consecutive_blocked_steps": int(self.consecutive_blocked_steps),
            "blocked_penalty": float(blocked_penalty),
            "is_stuck": bool(self.is_stuck),
            "stuck_flag": float(stuck_flag),
            "unblocked_steps": int(self.unblocked_steps),
            "blocked_penalty_accumulated": float(self.blocked_penalty_accumulated),
            "stagnation_steps": int(self.stagnation_steps),
            "stagnation_progress": float(stagnation_progress),
            "stagnation_penalty_triggered": bool(self.stagnation_penalty_triggered),
            "stagnation_penalty": float(stagnation_penalty),
            "recovery_reward": float(recovery_reward),
            "revisit_penalty_this_step": float(self.revisit_penalty_this_step),
            "current_confirmed_cell": self.current_confirmed_cell,
            "cell_visits": int(self.cell_visit_counts.get(self.current_confirmed_cell, 0)) if self.current_confirmed_cell else 0,
            "timeout_penalty": float(STAGE2_TIMEOUT_PENALTY if (timeout and self.stage == 2) else 0.0),
        }

    def render(self):
        """Render the environment for human visualization or off-screen image capture."""
        if self.render_mode is None:
            return None

        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                pygame.display.init()
                self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
                pygame.display.set_caption(f"Stealth Sandbox - RL Mode (Stage {self.stage})")
                self.clock = pygame.time.Clock()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.close()
                    return None

            self.env.render(self.screen, debug=False, fps=self.metadata["render_fps"])
            if self.overlay_callback is not None:
                self.overlay_callback(self.screen)
            pygame.display.flip()

            if self.clock is not None:
                self.clock.tick(self.metadata["render_fps"])
            return None

        elif self.render_mode == "rgb_array":
            surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            self.env.render(surface, debug=False, fps=self.metadata["render_fps"])
            # Return RGB array with shape (H, W, 3)
            return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))

    def close(self):
        """Release display and Pygame resources."""
        if self.screen is not None:
            pygame.display.quit()
            self.screen = None
            self.clock = None


# Register with Gymnasium
try:
    gym.register(
        id="StealthSandbox-v0",
        entry_point="rl_environment:StealthGymEnv",
        max_episode_steps=int(round(MAX_EPISODE_TIME / RL_DT)),
    )
except gym.error.RegistrationError:
    pass

