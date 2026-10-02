"""
skeld_environment.py
Gymnasium-compatible Reinforcement Learning environment for The Skeld navigation.

Physics and RL observations operate strictly in logical WORLD coordinates (1600.0 x 895.45 px).
Rendering transforms world space to display screen pixels (1100 x 700 px) with aspect-ratio
preserving letterboxing via world_to_screen().

Observation space: Proposed Structured Observation V1 (22 float32):
  [0:2]   Player position normalized: (x / WORLD_W, y / WORLD_H)
  [2:4]   Relative goal vector: (dx / WORLD_W, dy / WORLD_H) clamped to [-1, 1]
  [4:20]  16 radial wall raycasts normalized: [0.0, 1.0] (360 deg at 22.5 deg intervals)
  [20]    Stuck flag: 0.0 or 1.0 (triggers after 15 prolonged blocked steps)
  [21]    Stagnation progress: [0.0, 1.0] (measures lack of net displacement)

Action space:
  Box(-1.0, 1.0, shape=(2,), dtype=np.float32) -- continuous (dx, dy) velocity direction

Goal Modes Supported:
  - 'task' (or task_goal_mode=True): random valid room spawn -> authentic TaskDestination
  - 'task_to_task': authentic TaskDestination -> another authentic TaskDestination
  - 'room_to_room': random room spawn -> random position in different room
"""

import math
import random
from collections import defaultdict
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from skeld_config import (
    SKELD_WORLD_WIDTH, SKELD_WORLD_HEIGHT,
    SKELD_WINDOW_WIDTH, SKELD_WINDOW_HEIGHT,
    SKELD_WINDOW_TITLE,
    SKELD_COLOR_BG, SKELD_COLOR_FLOOR, SKELD_COLOR_WALL,
    SKELD_COLOR_WALL_BORDER, SKELD_COLOR_ROOM_LABEL,
    SKELD_COLOR_VENT, SKELD_COLOR_DOOR,
    SKELD_COLOR_PLAYER, SKELD_COLOR_PLAYER_BDR, SKELD_COLOR_PLAYER_CORE,
    SKELD_COLOR_TASK, SKELD_COLOR_TASK_ACTIVE,
    SKELD_COLOR_GOAL, SKELD_COLOR_GOAL_GLOW,
    SKELD_COLOR_HUD_BG, SKELD_COLOR_HUD_TEXT, SKELD_COLOR_HUD_ACCENT,
    SKELD_COLOR_SUCCESS, SKELD_COLOR_ALERT, SKELD_COLOR_NAV_PATH,
    SKELD_PLAYER_RADIUS, SKELD_PLAYER_SPEED, SKELD_GOAL_RADIUS,
    SKELD_INTERACTION_RADIUS,
    SKELD_RL_DT, SKELD_ACTION_DEADZONE,
    SKELD_MAX_EPISODE_TIME, SKELD_TIMEOUT_PENALTY, SKELD_GOAL_REWARD,
    SKELD_RAY_COUNT, SKELD_RAY_MAX_DISTANCE, SKELD_OBSERVATION_SIZE,
    SKELD_STEP_PENALTY, SKELD_PROGRESS_SCALE,
    SKELD_MIN_PROGRESS_DIST, SKELD_PROGRESS_CHECK_INT,
    SKELD_STAGNATION_STEPS, SKELD_STAGNATION_RADIUS,
    SKELD_STAGNATION_PENALTY, SKELD_STAGNATION_ESCAPE,
    SKELD_BLOCKED_THRESHOLD, SKELD_BLOCKED_PROLONGED,
    SKELD_SPAWN_MARGIN, SKELD_MIN_SPAWN_DIST,
    SKELD_ALL_SOLID_RECTS, SKELD_ALL_WALKABLE_AREAS,
    SKELD_ROOMS, SKELD_ROOM_MAP,
    SKELD_ALL_TASK_DESTINATIONS, TaskDestination,
    SKELD_ALL_VENTS, SKELD_SECURITY_CAMERAS,
    world_to_screen, screen_to_world,
    get_room_or_region, compute_map_aware_efficiency,
)
from geometry import (
    normalize_vector,
    compute_wall_raycasts,
    is_point_clear_of_obstacles,
)


# ---------------------------------------------------------------------------
# Ship floor boundary helper (dual-layer safety)
# ---------------------------------------------------------------------------
def is_point_in_ship_floor(x: float, y: float) -> bool:
    """Check if point (x, y) in world space is inside any defined walkable floor region."""
    pt = (int(round(x)), int(round(y)))
    for rect in SKELD_ALL_WALKABLE_AREAS:
        if rect.collidepoint(pt):
            return True
    return False


# ---------------------------------------------------------------------------
# Spawn sampling helpers
# ---------------------------------------------------------------------------
def _sample_point_in_room(
    room_rect: pygame.Rect,
    radius: float,
    margin: float,
    all_solids: List[pygame.Rect],
    rng,
    max_attempts: int = 50,
) -> Optional[Tuple[float, float]]:
    """Sample a random point inside room_rect with clearance from solids and room bounds."""
    m = int(radius + margin)
    x_lo = room_rect.left + m
    x_hi = room_rect.right - m
    y_lo = room_rect.top + m
    y_hi = room_rect.bottom - m
    if x_lo >= x_hi or y_lo >= y_hi:
        return None
    for _ in range(max_attempts):
        x = float(rng.uniform(x_lo, x_hi))
        y = float(rng.uniform(y_lo, y_hi))
        if is_point_clear_of_obstacles(x, y, radius, margin, all_solids):
            return x, y
    return None


def sample_skeld_positions(
    rng,
    goal_mode: str = "task",
) -> Tuple[Tuple[float, float], Tuple[float, float], Optional[TaskDestination]]:
    """
    Sample valid (player_pos, goal_pos, target_task) in logical WORLD coordinates.
    Modes:
      - 'task': random valid room spawn -> authentic TaskDestination
      - 'task_to_task': authentic TaskDestination -> another authentic TaskDestination
      - 'room_to_room': random room spawn -> random position in different room
    """
    rooms = list(SKELD_ROOMS)
    rng.shuffle(rooms)

    # 1. 'task_to_task' mode
    if goal_mode == "task_to_task" and len(SKELD_ALL_TASK_DESTINATIONS) >= 2:
        tasks = list(SKELD_ALL_TASK_DESTINATIONS)
        rng.shuffle(tasks)
        start_task = tasks[0]
        # Find destination task at least MIN_SPAWN_DIST away or in different room
        dest_task = None
        for t in tasks[1:]:
            d = math.hypot(t.world_x - start_task.world_x, t.world_y - start_task.world_y)
            if t.room != start_task.room or d >= SKELD_MIN_SPAWN_DIST:
                dest_task = t
                break
        if dest_task is None:
            dest_task = tasks[1]
        return start_task.world_pos, dest_task.world_pos, dest_task

    # 2. 'task' mode (default)
    if goal_mode == "task":
        # Pick random task destination
        tasks = list(SKELD_ALL_TASK_DESTINATIONS)
        rng.shuffle(tasks)
        target_task = tasks[0]
        goal_pos = target_task.world_pos

        # Pick player spawn in a different room
        player_pos = None
        for r in rooms:
            if r.name == target_task.room:
                continue
            pos = _sample_point_in_room(r.walkable_rect, SKELD_PLAYER_RADIUS, SKELD_SPAWN_MARGIN, SKELD_ALL_SOLID_RECTS, rng)
            if pos is not None:
                d = math.hypot(pos[0] - goal_pos[0], pos[1] - goal_pos[1])
                if d >= SKELD_MIN_SPAWN_DIST:
                    player_pos = pos
                    break

        if player_pos is None:
            # Fallback player room
            fallback_room = SKELD_ROOM_MAP["Cafeteria"] if target_task.room != "Cafeteria" else SKELD_ROOM_MAP["Security"]
            player_pos = fallback_room.center

        return player_pos, goal_pos, target_task

    # 3. 'room_to_room' mode
    player_pos, player_room = None, None
    for r in rooms:
        pos = _sample_point_in_room(r.walkable_rect, SKELD_PLAYER_RADIUS, SKELD_SPAWN_MARGIN, SKELD_ALL_SOLID_RECTS, rng)
        if pos is not None:
            player_pos = pos
            player_room = r
            break
    if player_pos is None:
        player_pos = SKELD_ROOM_MAP["Cafeteria"].center
        player_room = SKELD_ROOM_MAP["Cafeteria"]

    goal_pos = None
    rng.shuffle(rooms)
    for r in rooms:
        if r.name == player_room.name:
            continue
        pos = _sample_point_in_room(r.walkable_rect, SKELD_PLAYER_RADIUS, SKELD_SPAWN_MARGIN, SKELD_ALL_SOLID_RECTS, rng)
        if pos is not None:
            d = math.hypot(pos[0] - player_pos[0], pos[1] - player_pos[1])
            if d >= SKELD_MIN_SPAWN_DIST:
                goal_pos = pos
                break
    if goal_pos is None:
        goal_pos = SKELD_ROOM_MAP["Navigation"].center

    return player_pos, goal_pos, None


# ---------------------------------------------------------------------------
# Low-level physics movement (axis-separated sliding with dual-layer safety)
# ---------------------------------------------------------------------------
def _move_player(
    x: float, y: float, dx: float, dy: float,
    speed: float, dt: float,
    all_solid: List[pygame.Rect],
    radius: float,
    world_w: float = SKELD_WORLD_WIDTH,
    world_h: float = SKELD_WORLD_HEIGHT,
) -> Tuple[float, float, bool]:
    """
    Move (x, y) by (dx, dy)*speed*dt with axis-separated collision resolution.
    Maintains player inside ship floor and clear of solid wall rects.
    Returns (new_x, new_y, is_colliding).
    """
    nx, ny = normalize_vector(dx, dy)
    if nx == 0.0 and ny == 0.0:
        return x, y, False

    vx = nx * speed
    vy = ny * speed
    colliding = False
    r = int(math.ceil(radius))

    # --- X axis movement ---
    cand_x = x + vx * dt
    cand_x = max(radius, min(world_w - radius, cand_x))
    if cand_x != (x + vx * dt):
        colliding = True

    px = int(round(cand_x))
    player_rect = pygame.Rect(px - r, int(round(y)) - r, r * 2, r * 2)
    hit_x = False
    for srect in all_solid:
        if player_rect.colliderect(srect):
            hit_x = True
            colliding = True
            if vx > 0.0:
                cand_x = float(srect.left) - radius
            elif vx < 0.0:
                cand_x = float(srect.right) + radius
            player_rect = pygame.Rect(int(round(cand_x)) - r, int(round(y)) - r, r * 2, r * 2)

    # Floor boundary safety check
    if not is_point_in_ship_floor(cand_x, y):
        cand_x = x
        colliding = True

    # --- Y axis movement ---
    cand_y = y + vy * dt
    cand_y = max(radius, min(world_h - radius, cand_y))
    if cand_y != (y + vy * dt):
        colliding = True

    py = int(round(cand_y))
    player_rect = pygame.Rect(int(round(cand_x)) - r, py - r, r * 2, r * 2)
    hit_y = False
    for srect in all_solid:
        if player_rect.colliderect(srect):
            hit_y = True
            colliding = True
            if vy > 0.0:
                cand_y = float(srect.top) - radius
            elif vy < 0.0:
                cand_y = float(srect.bottom) + radius

    # Floor boundary safety check
    if not is_point_in_ship_floor(cand_x, cand_y):
        cand_y = y
        colliding = True

    return cand_x, cand_y, colliding


# ---------------------------------------------------------------------------
# Gymnasium Environment
# ---------------------------------------------------------------------------
class SkeldNavEnv(gym.Env):
    """
    The Skeld Navigation RL Environment.

    A standalone continuous 2D navigation environment modeled on The Skeld.
    Physics and state operate in logical WORLD coordinates (1600.0 x 895.45 px).
    Display renders to 1100 x 700 window with uniform scaling and letterboxing.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    STATE_PLAYING = "playing"
    STATE_GOAL    = "goal"
    STATE_TIMEOUT = "timeout"

    def __init__(
        self,
        render_mode: Optional[str] = None,
        task_goal_mode: bool = True,
        goal_mode: Optional[str] = None,
    ):
        """
        Args:
            render_mode: 'human' for display window, 'rgb_array' for headless image array.
            task_goal_mode: If True (default), sets goal_mode='task' for authentic task targets.
            goal_mode: Explicit goal sampling mode: 'task', 'task_to_task', or 'room_to_room'.
        """
        super().__init__()

        self.render_mode = render_mode
        if goal_mode is not None:
            self.goal_mode = goal_mode
        else:
            self.goal_mode = "task" if task_goal_mode else "room_to_room"

        self.task_goal_mode = (self.goal_mode == "task")

        # Spaces (22 float32 observation)
        obs_low  = np.zeros(SKELD_OBSERVATION_SIZE, dtype=np.float32)
        obs_high = np.ones(SKELD_OBSERVATION_SIZE,  dtype=np.float32)
        obs_low[2:4]  = -1.0  # relative goal can be negative
        obs_high[2:4] =  1.0
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        # Rendering surfaces
        self._screen: Optional[pygame.Surface] = None
        self._clock:  Optional[pygame.time.Clock] = None
        self._font_small  = None
        self._font_medium = None
        self._font_large  = None

        # Episode state (WORLD coordinates)
        self._player_x: float = 0.0
        self._player_y: float = 0.0
        self._goal_x:   float = 0.0
        self._goal_y:   float = 0.0
        self._facing:   float = -math.pi / 2.0
        self._state:    str   = self.STATE_PLAYING
        self._elapsed:  float = 0.0
        self._step_idx: int   = 0
        self._target_task: Optional[TaskDestination] = None

        # Metrics for path efficiency
        self._total_path_length: float = 0.0
        self._optimal_path_length: float = 0.0

        # Progress tracking
        self._best_dist:       float = float("inf")
        self._last_prog_check: float = 0.0

        # Stagnation
        self._stag_anchor:    Optional[Tuple[float, float]] = None
        self._stag_steps:     int   = 0
        self._stag_armed:     bool  = True
        self._stag_progress:  float = 0.0

        # Stuck detection
        self._blocked_steps:  int   = 0
        self._stuck_flag:     float = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Allow override of goal_mode via options
        current_mode = self.goal_mode
        if options and "goal_mode" in options:
            current_mode = options["goal_mode"]

        player_pos, goal_pos, target_task = sample_skeld_positions(self.np_random, goal_mode=current_mode)
        self._player_x, self._player_y = player_pos
        self._goal_x,   self._goal_y   = goal_pos
        self._target_task = target_task
        self._facing = -math.pi / 2.0

        self._state    = self.STATE_PLAYING
        self._elapsed  = 0.0
        self._step_idx = 0
        self._total_path_length = 0.0

        # Best distance for reward shaping
        self._best_dist       = math.hypot(self._player_x - self._goal_x, self._player_y - self._goal_y)
        self._last_prog_check = 0.0

        # Stagnation reset
        self._stag_anchor   = (self._player_x, self._player_y)
        self._stag_steps    = 0
        self._stag_armed    = True
        self._stag_progress = 0.0

        # Stuck reset
        self._blocked_steps = 0
        self._stuck_flag    = 0.0

        obs = self._build_observation()
        info = {
            "player_pos":  (self._player_x, self._player_y),
            "goal_pos":    (self._goal_x,   self._goal_y),
            "target_task": self._target_task.name if self._target_task else None,
            "room":        get_room_or_region(self._player_x, self._player_y),
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self._state == self.STATE_PLAYING, "Call reset() before step()"

        dx = float(action[0])
        dy = float(action[1])
        dt = SKELD_RL_DT

        # Deadzone filter
        mag = math.hypot(dx, dy)
        if mag < SKELD_ACTION_DEADZONE:
            dx, dy = 0.0, 0.0

        # Move player in world space
        prev_x, prev_y = self._player_x, self._player_y
        new_x, new_y, is_colliding = _move_player(
            self._player_x, self._player_y, dx, dy,
            SKELD_PLAYER_SPEED, dt, SKELD_ALL_SOLID_RECTS,
            SKELD_PLAYER_RADIUS
        )
        self._player_x, self._player_y = new_x, new_y

        step_dist = math.hypot(new_x - prev_x, new_y - prev_y)
        self._total_path_length += step_dist

        if mag >= SKELD_ACTION_DEADZONE:
            self._facing = math.atan2(dy, dx)

        self._elapsed  += dt
        self._step_idx += 1

        # Stuck detection
        if is_colliding and mag >= SKELD_ACTION_DEADZONE:
            expected_dist = SKELD_PLAYER_SPEED * dt
            ratio = step_dist / (expected_dist + 1e-9)
            if ratio < SKELD_BLOCKED_THRESHOLD:
                self._blocked_steps += 1
            else:
                self._blocked_steps = max(0, self._blocked_steps - 1)
        else:
            self._blocked_steps = max(0, self._blocked_steps - 1)

        self._stuck_flag = 1.0 if self._blocked_steps >= SKELD_BLOCKED_PROLONGED else 0.0

        # Stagnation tracking
        ax, ay = self._stag_anchor or (new_x, new_y)
        stag_dist = math.hypot(new_x - ax, new_y - ay)
        if stag_dist < SKELD_STAGNATION_RADIUS:
            self._stag_steps += 1
        else:
            self._stag_anchor = (new_x, new_y)
            self._stag_steps  = 0
            self._stag_armed  = True
        self._stag_progress = min(1.0, self._stag_steps / float(SKELD_STAGNATION_STEPS))

        # Reward calculation
        reward = SKELD_STEP_PENALTY
        cur_dist = math.hypot(new_x - self._goal_x, new_y - self._goal_y)

        # Progress reward shaping
        if self._elapsed - self._last_prog_check >= SKELD_PROGRESS_CHECK_INT:
            improvement = self._best_dist - cur_dist
            if improvement >= SKELD_MIN_PROGRESS_DIST:
                reward += improvement * SKELD_PROGRESS_SCALE
                self._best_dist = cur_dist
            self._last_prog_check = self._elapsed

        # Stagnation penalty
        if self._stag_steps >= SKELD_STAGNATION_STEPS and self._stag_armed:
            reward += SKELD_STAGNATION_PENALTY
            self._stag_armed = False

        # Terminations
        terminated = False
        truncated  = False

        # Goal reached condition (player radius + goal interaction radius)
        if cur_dist <= SKELD_GOAL_RADIUS + SKELD_PLAYER_RADIUS:
            reward += SKELD_GOAL_REWARD
            self._state = self.STATE_GOAL
            terminated = True
        elif self._elapsed >= SKELD_MAX_EPISODE_TIME:
            reward += SKELD_TIMEOUT_PENALTY
            self._state = self.STATE_TIMEOUT
            truncated = True

        obs  = self._build_observation()
        info = {
            "elapsed":          self._elapsed,
            "distance":         cur_dist,
            "stuck":            self._stuck_flag,
            "stagnation":       self._stag_progress,
            "total_path_len":   self._total_path_length,
            "current_region":   get_room_or_region(self._player_x, self._player_y),
            "target_task":      self._target_task.name if self._target_task else None,
        }
        return obs, float(reward), terminated, truncated, info

    def _build_observation(self) -> np.ndarray:
        obs = np.zeros(SKELD_OBSERVATION_SIZE, dtype=np.float32)

        # [0:2] Normalized player position in WORLD space
        obs[0] = self._player_x / float(SKELD_WORLD_WIDTH)
        obs[1] = self._player_y / float(SKELD_WORLD_HEIGHT)

        # [2:4] Relative goal vector clamped to [-1, 1]
        obs[2] = np.clip((self._goal_x - self._player_x) / float(SKELD_WORLD_WIDTH),  -1.0, 1.0)
        obs[3] = np.clip((self._goal_y - self._player_y) / float(SKELD_WORLD_HEIGHT), -1.0, 1.0)

        # [4:20] 16 radial wall raycasts in WORLD units
        rays = compute_wall_raycasts(
            (self._player_x, self._player_y),
            SKELD_RAY_COUNT,
            SKELD_RAY_MAX_DISTANCE,
            SKELD_ALL_SOLID_RECTS,
        )
        obs[4:20] = rays

        # [20] Stuck flag
        obs[20] = self._stuck_flag

        # [21] Stagnation progress
        obs[21] = self._stag_progress

        return obs

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode is None:
            return None
        self._ensure_display()
        self._draw()
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(self.metadata.get("render_fps", 30))
        elif self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(self._screen), axes=(1, 0, 2))
        return None

    def close(self):
        if self._screen is not None:
            pygame.display.quit()
            self._screen = None

    def _ensure_display(self):
        if self._screen is None:
            pygame.init()
            pygame.font.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((SKELD_WINDOW_WIDTH, SKELD_WINDOW_HEIGHT))
                pygame.display.set_caption(SKELD_WINDOW_TITLE)
            else:
                self._screen = pygame.Surface((SKELD_WINDOW_WIDTH, SKELD_WINDOW_HEIGHT))
            self._clock       = pygame.time.Clock()
            self._font_small  = pygame.font.SysFont("Consolas", 11)
            self._font_medium = pygame.font.SysFont("Trebuchet MS", 13, bold=True)
            self._font_large  = pygame.font.SysFont("Trebuchet MS", 20, bold=True)

    def _draw(self):
        surf = self._screen
        surf.fill(SKELD_COLOR_BG)

        # 1. Draw floor regions (transformed to screen)
        for frect in SKELD_ALL_WALKABLE_AREAS:
            sx, sy = world_to_screen(frect.left, frect.top)
            sw, sh = int(round(frect.width * (SKELD_WINDOW_WIDTH / SKELD_WORLD_WIDTH))), int(round(frect.height * (SKELD_WINDOW_WIDTH / SKELD_WORLD_WIDTH)))
            pygame.draw.rect(surf, SKELD_COLOR_FLOOR, pygame.Rect(sx, sy, sw, sh))

        # 2. Draw solid wall rectangles
        for srect in SKELD_ALL_SOLID_RECTS:
            sx, sy = world_to_screen(srect.left, srect.top)
            sw = max(1, int(round(srect.width * (SKELD_WINDOW_WIDTH / SKELD_WORLD_WIDTH))))
            sh = max(1, int(round(srect.height * (SKELD_WINDOW_WIDTH / SKELD_WORLD_WIDTH))))
            drect = pygame.Rect(sx, sy, sw, sh)
            pygame.draw.rect(surf, SKELD_COLOR_WALL, drect)
            pygame.draw.rect(surf, SKELD_COLOR_WALL_BORDER, drect, 1)

        # 3. Room labels
        if self._font_small:
            for room in SKELD_ROOMS:
                cx, cy = room.center
                scx, scy = world_to_screen(cx, cy)
                label = self._font_small.render(room.name, True, SKELD_COLOR_ROOM_LABEL)
                surf.blit(label, (scx - label.get_width() // 2, scy - label.get_height() // 2))

        # 4. Vent markers
        for v in SKELD_ALL_VENTS:
            vx, vy = v["world_pos"]
            svx, svy = world_to_screen(vx, vy)
            pygame.draw.rect(surf, SKELD_COLOR_VENT, pygame.Rect(svx - 4, svy - 4, 8, 8))
            pygame.draw.rect(surf, (0, 150, 140), pygame.Rect(svx - 4, svy - 4, 8, 8), 1)

        # 5. Task markers
        for t in SKELD_ALL_TASK_DESTINATIONS:
            stx, sty = world_to_screen(t.world_x, t.world_y)
            d = math.hypot(self._player_x - t.world_x, self._player_y - t.world_y)
            c = SKELD_COLOR_TASK_ACTIVE if d < 25.0 else SKELD_COLOR_TASK
            pygame.draw.circle(surf, c, (stx, sty), 4)
            pygame.draw.circle(surf, (180, 140, 0), (stx, sty), 4, 1)

        # 6. Goal
        sgx, sgy = world_to_screen(self._goal_x, self._goal_y)
        d_gr = int(round(SKELD_GOAL_RADIUS * (SKELD_WINDOW_WIDTH / SKELD_WORLD_WIDTH)))
        pygame.draw.circle(surf, SKELD_COLOR_GOAL_GLOW, (sgx, sgy), d_gr + 4, 2)
        pygame.draw.circle(surf, SKELD_COLOR_GOAL, (sgx, sgy), d_gr)
        pygame.draw.circle(surf, (200, 255, 220), (sgx, sgy), 4)

        # 7. Raycasts
        obs = self._build_observation()
        angle_step = (2.0 * math.pi) / SKELD_RAY_COUNT
        spx, spy = world_to_screen(self._player_x, self._player_y)
        for i in range(SKELD_RAY_COUNT):
            angle = i * angle_step
            r_norm = float(obs[4 + i])
            r_len = r_norm * SKELD_RAY_MAX_DISTANCE
            w_ex = self._player_x + math.cos(angle) * r_len
            w_ey = self._player_y + math.sin(angle) * r_len
            sex, sey = world_to_screen(w_ex, w_ey)
            col = (40, 80, 60) if r_norm > 0.95 else (80, 160, 80)
            pygame.draw.line(surf, col, (spx, spy), (sex, sey), 1)

        # 8. Player
        d_pr = max(4, int(round(SKELD_PLAYER_RADIUS * (SKELD_WINDOW_WIDTH / SKELD_WORLD_WIDTH))))
        pygame.draw.circle(surf, SKELD_COLOR_PLAYER_BDR, (spx, spy), d_pr + 2)
        pygame.draw.circle(surf, SKELD_COLOR_PLAYER,     (spx, spy), d_pr)
        pygame.draw.circle(surf, SKELD_COLOR_PLAYER_CORE,(spx, spy), max(2, d_pr // 3))
        # Facing nose
        nose_x = spx + int(math.cos(self._facing) * (d_pr + 4))
        nose_y = spy + int(math.sin(self._facing) * (d_pr + 4))
        pygame.draw.line(surf, SKELD_COLOR_PLAYER_BDR, (spx, spy), (nose_x, nose_y), 2)

        # 9. HUD in letterbox bar
        self._draw_hud(surf)

    def _draw_hud(self, surf: pygame.Surface):
        hud_h = 42
        hud_rect = pygame.Rect(0, SKELD_WINDOW_HEIGHT - hud_h, SKELD_WINDOW_WIDTH, hud_h)
        hud_surf = pygame.Surface((hud_rect.width, hud_rect.height), pygame.SRCALPHA)
        hud_surf.fill((15, 20, 32, 230))
        surf.blit(hud_surf, (hud_rect.x, hud_rect.y))
        pygame.draw.rect(surf, SKELD_COLOR_HUD_ACCENT, hud_rect, 1)

        if self._font_medium:
            dist = math.hypot(self._player_x - self._goal_x, self._player_y - self._goal_y)
            time_left = max(0.0, SKELD_MAX_EPISODE_TIME - self._elapsed)
            region = get_room_or_region(self._player_x, self._player_y)
            task_str = f"Task: {self._target_task.name}" if self._target_task else "Goal: Point"
            state_txt = {"playing": "NAVIGATING", "goal": "GOAL REACHED!", "timeout": "TIMEOUT"}
            state_color = {"playing": SKELD_COLOR_HUD_TEXT, "goal": SKELD_COLOR_SUCCESS, "timeout": SKELD_COLOR_ALERT}

            items = [
                (f"Step {self._step_idx:04d}", SKELD_COLOR_HUD_ACCENT),
                (f"Region: {region}", SKELD_COLOR_HUD_TEXT),
                (task_str, (255, 215, 50)),
                (f"Dist: {dist:.0f}px", SKELD_COLOR_HUD_TEXT),
                (f"Time: {time_left:.1f}s", SKELD_COLOR_HUD_TEXT),
                (state_txt.get(self._state, "?"), state_color.get(self._state, SKELD_COLOR_HUD_TEXT)),
            ]
            x = 10
            y = SKELD_WINDOW_HEIGHT - hud_h + 10
            for txt, color in items:
                rendered = self._font_medium.render(txt, True, color)
                surf.blit(rendered, (x, y))
                x += rendered.get_width() + 22
