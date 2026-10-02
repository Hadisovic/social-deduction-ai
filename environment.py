"""
Environment sandbox managing the arena layout, entities, physics updates,
deterministic reset, and decoupled rendering.
Designed to be directly convertible into an RL Gymnasium environment in future phases.
"""

import math
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import pygame

from config import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    PLAY_AREA,
    BOUNDARY_WALLS,
    OBSTACLE_RECTS,
    ALL_SOLID_RECTS,
    GOAL_POS,
    GOAL_RADIUS,
    OBSERVER_CONFIGS,
    COLOR_BG,
    COLOR_PLAY_AREA_BG,
    COLOR_GRID,
    COLOR_BOUNDARY_WALL,
    COLOR_OBSTACLE,
    COLOR_OBSTACLE_BORDER,
    COLOR_OBSTACLE_ACCENT,
    COLOR_GOAL,
    COLOR_GOAL_GLOW,
    COLOR_GOAL_CORE,
    COLOR_GOAL_TEXT,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    COLOR_ALERT,
    COLOR_SUCCESS,
    ARENA_WIDTH,
    ARENA_HEIGHT,
    RAY_COUNT,
    RAY_MAX_DISTANCE,
    OBSERVATION_SIZE,
    ROOM_MARGIN,
    STAGE1_SPAWN_MARGIN,
    STAGE1_MIN_START_GOAL_DISTANCE,
    STAGE2_SPAWN_MARGIN,
    STAGE2_MIN_START_GOAL_DISTANCE,
    STAGE2_OBSTACLE_SPAWN_CLEARANCE,
    STAGE2_FALLBACK_PLAYER_POS,
    STAGE2_FALLBACK_GOAL_POS,
    PLAYER_RADIUS,
    PLAYER_START_POS,
    GOAL_RADIUS,
    BOUNDARY_WALLS,
)
from geometry import (
    compute_wall_raycasts,
    is_point_clear_of_obstacles,
    does_segment_intersect_obstacles,
)
from player import Player
from observer import Observer


def sample_stage1_positions(rng=None) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Sample randomized player start position and goal position for Stage 1.
    Respects arena boundaries, STAGE1_SPAWN_MARGIN, and STAGE1_MIN_START_GOAL_DISTANCE.

    Args:
        rng: Optional numpy Generator (e.g. env.np_random). If None, uses np.random.

    Returns:
        ((player_x, player_y), (goal_x, goal_y))
    """
    min_x = ROOM_MARGIN + STAGE1_SPAWN_MARGIN
    max_x = WINDOW_WIDTH - ROOM_MARGIN - STAGE1_SPAWN_MARGIN
    min_y = ROOM_MARGIN + STAGE1_SPAWN_MARGIN
    max_y = WINDOW_HEIGHT - ROOM_MARGIN - STAGE1_SPAWN_MARGIN

    max_attempts = 100
    for _ in range(max_attempts):
        if rng is not None:
            px = float(rng.uniform(min_x, max_x))
            py = float(rng.uniform(min_y, max_y))
            gx = float(rng.uniform(min_x, max_x))
            gy = float(rng.uniform(min_y, max_y))
        else:
            px = float(np.random.uniform(min_x, max_x))
            py = float(np.random.uniform(min_y, max_y))
            gx = float(np.random.uniform(min_x, max_x))
            gy = float(np.random.uniform(min_y, max_y))

        dist = math.hypot(px - gx, py - gy)
        if dist >= STAGE1_MIN_START_GOAL_DISTANCE:
            return (px, py), (gx, gy)

    # Safe deterministic fallback if limit exceeded
    return (float(PLAYER_START_POS[0]), float(PLAYER_START_POS[1])), (float(GOAL_POS[0]), float(GOAL_POS[1]))


def sample_stage2_positions(
    rng=None,
    obstacles: Optional[List[pygame.Rect]] = None
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Sample randomized player start position and goal position for Stage 2.
    Respects:
      1. Outer arena bounds with STAGE2_SPAWN_MARGIN.
      2. Obstacle clearance: neither player nor goal may spawn inside, touching,
         or within STAGE2_OBSTACLE_SPAWN_CLEARANCE of any obstacle.
         - Player requires distance >= (PLAYER_RADIUS + STAGE2_OBSTACLE_SPAWN_CLEARANCE).
         - Goal requires distance >= (GOAL_RADIUS + STAGE2_OBSTACLE_SPAWN_CLEARANCE).
      3. Minimum Euclidean distance between player and goal >= STAGE2_MIN_START_GOAL_DISTANCE (350 px).
      4. Reachability: The arena obstacles (4 central cross elements) are completely disconnected
         from each other and from the outer boundary walls, leaving wide open corridors
         (minimum corridor width 60 px > 28 px player diameter). Thus, after player-radius inflation,
         the free navigable space is topologically a single connected component; any two valid
         clearance-conforming points are mutually reachable without specialized pathfinding.

    Args:
        rng: Optional numpy Generator (e.g. env.np_random). If None, uses np.random.
        obstacles: Optional list of obstacle pygame.Rects. If None, uses OBSTACLE_RECTS.

    Returns:
        ((player_x, player_y), (goal_x, goal_y))
    """
    if obstacles is None:
        obstacles = OBSTACLE_RECTS

    # Full list of solid rects including outer perimeter boundary walls
    all_solid_rects = list(BOUNDARY_WALLS) + list(obstacles)

    # Player bounds: Ensure player center satisfies outer margin and clearance from boundary walls
    player_margin = ROOM_MARGIN + max(STAGE2_SPAWN_MARGIN, PLAYER_RADIUS + STAGE2_OBSTACLE_SPAWN_CLEARANCE)
    player_min_x = player_margin
    player_max_x = WINDOW_WIDTH - player_margin
    player_min_y = player_margin
    player_max_y = WINDOW_HEIGHT - player_margin

    # Goal bounds: Ensure goal center satisfies outer margin and clearance from boundary walls
    goal_margin = ROOM_MARGIN + max(STAGE2_SPAWN_MARGIN, GOAL_RADIUS + STAGE2_OBSTACLE_SPAWN_CLEARANCE)
    goal_min_x = goal_margin
    goal_max_x = WINDOW_WIDTH - goal_margin
    goal_min_y = goal_margin
    goal_max_y = WINDOW_HEIGHT - goal_margin

    max_attempts = 500
    for _ in range(max_attempts):
        if rng is not None:
            px = float(rng.uniform(player_min_x, player_max_x))
            py = float(rng.uniform(player_min_y, player_max_y))
            gx = float(rng.uniform(goal_min_x, goal_max_x))
            gy = float(rng.uniform(goal_min_y, goal_max_y))
        else:
            px = float(np.random.uniform(player_min_x, player_max_x))
            py = float(np.random.uniform(player_min_y, player_max_y))
            gx = float(np.random.uniform(goal_min_x, goal_max_x))
            gy = float(np.random.uniform(goal_min_y, goal_max_y))

        # Check start-goal minimum distance
        dist = math.hypot(px - gx, py - gy)
        if dist < STAGE2_MIN_START_GOAL_DISTANCE:
            continue

        # Check player clearance against all solid obstacles (interior and boundary walls)
        if not is_point_clear_of_obstacles(
            px, py, PLAYER_RADIUS, STAGE2_OBSTACLE_SPAWN_CLEARANCE, all_solid_rects
        ):
            continue

        # Check goal clearance against all solid obstacles (interior and boundary walls)
        if not is_point_clear_of_obstacles(
            gx, gy, GOAL_RADIUS, STAGE2_OBSTACLE_SPAWN_CLEARANCE, all_solid_rects
        ):
            continue

        # Check that direct line segment from player to goal intersects >= 1 central obstacle
        # inflated by PLAYER_RADIUS on all sides (guaranteeing a real obstacle detour)
        if not does_segment_intersect_obstacles(
            (px, py), (gx, gy), obstacles, inflate_radius=PLAYER_RADIUS
        ):
            continue

        return (px, py), (gx, gy)

    # Safe deterministic fallback if finite retry attempts exhausted
    # (guaranteed clearances >=39px/53px, dist >=350px, and direct path blocked by central obstacle)
    return (
        (float(STAGE2_FALLBACK_PLAYER_POS[0]), float(STAGE2_FALLBACK_PLAYER_POS[1])),
        (float(STAGE2_FALLBACK_GOAL_POS[0]), float(STAGE2_FALLBACK_GOAL_POS[1]))
    )



class StealthEnvironment:
    """
    2D Stealth Simulation Sandbox.
    Coordinates Player, Observers, Obstacles, and Game State.
    """

    STATE_PLAYING = "PLAYING"
    STATE_DETECTED = "DETECTED"
    STATE_ESCAPED = "ESCAPED"

    def __init__(self, stage: int = 4):
        # Boundaries & obstacles
        self.play_area = PLAY_AREA
        self.boundary_walls = BOUNDARY_WALLS
        self.obstacles = list(OBSTACLE_RECTS)
        self.all_solid_rects = list(ALL_SOLID_RECTS)

        # Goal parameters
        self.goal_pos = (float(GOAL_POS[0]), float(GOAL_POS[1]))
        self.goal_radius = float(GOAL_RADIUS)

        # Entities
        self.player = Player()
        self.observers: List[Observer] = [Observer(cfg) for cfg in OBSERVER_CONFIGS]
        self.active_observers: List[Observer] = list(self.observers)
        self.curriculum_stage = stage
        self.set_curriculum_stage(stage)

        # Simulation state
        self.state = self.STATE_PLAYING
        self.detected_by: Optional[Observer] = None
        self.status_message = ""
        self.elapsed_time = 0.0

        # Pre-create fonts for rendering
        pygame.font.init()
        self.font_large = pygame.font.SysFont("Trebuchet MS", 42, bold=True)
        self.font_medium = pygame.font.SysFont("Trebuchet MS", 22, bold=True)
        self.font_small = pygame.font.SysFont("Trebuchet MS", 15)
        self.font_mono = pygame.font.SysFont("Consolas", 13)
        self.font_timer = pygame.font.SysFont("Consolas", 15, bold=True)

        self.reset()

    def set_curriculum_stage(self, stage: int):
        """
        Configure environment for Curriculum Learning stages:
        Stage 1: No central obstacles, no active observers.
        Stage 2: Central obstacles enabled, no active observers.
        Stage 3: Central obstacles enabled, exactly 1 active observer (Observer 1).
        Stage 4: Central obstacles enabled, all 3 active observers (full arena).
        Stage 5: Reserved for future randomization (currently identical to Stage 4).
        """
        self.curriculum_stage = int(stage)
        if self.curriculum_stage == 1:
            self.obstacles = []
            self.all_solid_rects = list(self.boundary_walls)
            self.active_observers = []
        elif self.curriculum_stage == 2:
            self.obstacles = list(OBSTACLE_RECTS)
            self.all_solid_rects = list(ALL_SOLID_RECTS)
            self.active_observers = []
        elif self.curriculum_stage == 3:
            self.obstacles = list(OBSTACLE_RECTS)
            self.all_solid_rects = list(ALL_SOLID_RECTS)
            self.active_observers = [self.observers[0]]
        else:  # Stage 4 & 5
            self.obstacles = list(OBSTACLE_RECTS)
            self.all_solid_rects = list(ALL_SOLID_RECTS)
            self.active_observers = list(self.observers)

    def reset(self):
        """
        Deterministically reset the entire environment to initial conditions.
        Running this guarantees 100% reproducible observer trajectories.
        """
        self.state = self.STATE_PLAYING
        self.detected_by = None
        self.status_message = ""
        self.elapsed_time = 0.0

        self.player.reset()
        for obs in self.observers:
            obs.reset()

    def set_layout(self, player_pos: Tuple[float, float], goal_pos: Tuple[float, float]):
        """Explicitly set player spawn and goal positions."""
        self.player.x = float(player_pos[0])
        self.player.y = float(player_pos[1])
        self.player.initial_pos = (float(player_pos[0]), float(player_pos[1]))
        self.goal_pos = (float(goal_pos[0]), float(goal_pos[1]))

    def set_stage1_layout(self, player_pos: Tuple[float, float], goal_pos: Tuple[float, float]):
        """Backward-compatible alias for set_layout."""
        self.set_layout(player_pos, goal_pos)

    def step(self, action_vector: Tuple[float, float], dt: float):
        """
        Advance simulation by dt seconds given an action vector (dx, dy).
        
        Args:
            action_vector: (dx, dy) movement intention.
            dt: Delta time in seconds.
        """
        if self.state != self.STATE_PLAYING:
            return

        self.elapsed_time += dt

        # 1. Update Player Movement (Axis-separated sliding + bounds clamping)
        self.player.move(action_vector, dt, self.obstacles, self.play_area)

        # 2. Update Active Observers along fixed patrol waypoints
        for obs in self.active_observers:
            obs.update(dt)

        # 3. Check Escape / Goal Condition
        dist_to_goal = math.hypot(
            self.player.x - self.goal_pos[0],
            self.player.y - self.goal_pos[1]
        )
        if dist_to_goal <= (self.player.radius + self.goal_radius):
            self.state = self.STATE_ESCAPED
            self.status_message = "ESCAPED! Target Destination Reached."
            return

        # 4. Check Active Observers Line-of-Sight Detection
        for obs in self.active_observers:
            if obs.can_see_player(self.player.pos, self.all_solid_rects):
                self.state = self.STATE_DETECTED
                self.detected_by = obs
                self.status_message = f"DETECTED by {obs.name}!"
                break

    # =========================================================================
    # RENDERING METHODS
    # =========================================================================

    def render(self, surface: pygame.Surface, debug: bool = False, fps: float = 60.0):
        """Render complete environment scene and UI."""
        # 1. Background
        surface.fill(COLOR_BG)

        # 2. Play Area Surface with Subtle Blueprint Grid
        pygame.draw.rect(surface, COLOR_PLAY_AREA_BG, self.play_area)
        self._draw_grid(surface)

        # 3. Goal Destination (Clean glowing concentric rings)
        self._draw_goal(surface)

        # 4. Debug: Patrol paths
        if debug:
            for obs in self.active_observers:
                obs.draw_debug(surface, self.player.pos)

        # 5. Observer Vision Cones (Translucent polygons clipped by walls)
        for obs in self.active_observers:
            obs.draw_vision_cone(surface, self.all_solid_rects)

        # 6. Obstacles and Perimeter Boundaries (Drawn over vision cones)
        self._draw_obstacles(surface)

        # 7. Entities
        self.player.draw(surface, debug=debug)
        for obs in self.active_observers:
            obs.draw(surface, debug=debug)

        # 8. Unobtrusive Top Bar Timer
        self._draw_timer(surface)

        # 9. Unobtrusive Footer Controls Bar
        self._draw_footer(surface)

        # 10. Game State Modal Banner (When ESCAPED or DETECTED)
        if self.state != self.STATE_PLAYING:
            self._draw_end_state_modal(surface)

        # 11. Debug Information HUD Overlay
        if debug:
            self._draw_debug_hud(surface, fps)

    def _draw_grid(self, surface: pygame.Surface):
        """Draw light background grid inside the playable room."""
        grid_size = 40
        for x in range(self.play_area.left, self.play_area.right, grid_size):
            pygame.draw.line(
                surface,
                COLOR_GRID,
                (x, self.play_area.top),
                (x, self.play_area.bottom),
                1
            )
        for y in range(self.play_area.top, self.play_area.bottom, grid_size):
            pygame.draw.line(
                surface,
                COLOR_GRID,
                (self.play_area.left, y),
                (self.play_area.right, y),
                1
            )

    def _draw_goal(self, surface: pygame.Surface):
        """Draw destination beacon with glowing concentric rings."""
        gx, gy = int(self.goal_pos[0]), int(self.goal_pos[1])
        gr = int(self.goal_radius)

        # Pulsing / glowing translucent aura
        pulse = math.sin(self.elapsed_time * 3.5) * 4.0
        outer_radius = int(gr + 8 + pulse)

        aura_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        pygame.draw.circle(aura_surf, COLOR_GOAL_GLOW, (gx, gy), outer_radius)
        surface.blit(aura_surf, (0, 0))

        # Main green ring
        pygame.draw.circle(surface, COLOR_GOAL, (gx, gy), gr, 3)
        # Inner subtle ring
        pygame.draw.circle(surface, COLOR_GOAL, (gx, gy), max(4, gr - 8), 1)
        # Center core
        pygame.draw.circle(surface, COLOR_GOAL_CORE, (gx, gy), 6)

        # Label "GOAL"
        label_surf = self.font_small.render("GOAL", True, COLOR_GOAL_TEXT)
        label_rect = label_surf.get_rect(center=(gx, gy - gr - 14))
        surface.blit(label_surf, label_rect)

    def _draw_obstacles(self, surface: pygame.Surface):
        """Render outer boundary walls and central cross obstacles."""
        # Outer boundary walls
        for wall in self.boundary_walls:
            pygame.draw.rect(surface, COLOR_BOUNDARY_WALL, wall)

        # Interior cross obstacles
        for obs in self.obstacles:
            # Main obstacle block
            pygame.draw.rect(surface, COLOR_OBSTACLE, obs)
            # Crisp border
            pygame.draw.rect(surface, COLOR_OBSTACLE_BORDER, obs, 2)
            # Subtle interior bevel line for architectural feel
            inner_rect = obs.inflate(-6, -6)
            if inner_rect.width > 0 and inner_rect.height > 0:
                pygame.draw.rect(surface, COLOR_OBSTACLE_ACCENT, inner_rect, 1)

    def _draw_timer(self, surface: pygame.Surface):
        """Display elapsed time unobtrusively in seconds on the top boundary bar."""
        timer_str = f"TIME: {self.elapsed_time:.1f}s"
        text_surf = self.font_timer.render(timer_str, True, (225, 235, 245))
        rect = text_surf.get_rect(center=(WINDOW_WIDTH // 2, 17))
        surface.blit(text_surf, rect)

    def _draw_footer(self, surface: pygame.Surface):
        """Display subtle status and control cues."""
        text = "WASD / Arrows to Move   |   R: Reset Arena   |   F1: Toggle Debug Overlay"
        text_surf = self.font_small.render(text, True, COLOR_TEXT_MUTED)
        rect = text_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 17))
        surface.blit(text_surf, rect)

    def _draw_end_state_modal(self, surface: pygame.Surface):
        """Display stylish modal overlay for DETECTED or ESCAPED states."""
        # Semi-transparent backdrop overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((15, 20, 28, 175))
        surface.blit(overlay, (0, 0))

        # Banner card dimensions
        card_w = 540
        card_h = 225
        card_x = (WINDOW_WIDTH - card_w) // 2
        card_y = (WINDOW_HEIGHT - card_h) // 2
        card_rect = pygame.Rect(card_x, card_y, card_w, card_h)

        # Card body
        pygame.draw.rect(surface, (255, 255, 255), card_rect, border_radius=12)

        if self.state == self.STATE_DETECTED:
            card_border_color = COLOR_ALERT
            title_text = "DETECTED"
            title_color = COLOR_ALERT
            obs_name = self.detected_by.name if self.detected_by else "Observer"
            sub_text = f"You were spotted by {obs_name}"
            sub_color = self.detected_by.color if self.detected_by else COLOR_TEXT
        else:
            card_border_color = COLOR_SUCCESS
            title_text = "ESCAPED"
            title_color = COLOR_SUCCESS
            sub_text = "Target destination reached successfully!"
            sub_color = COLOR_SUCCESS

        # Card border
        pygame.draw.rect(surface, card_border_color, card_rect, 3, border_radius=12)

        # Render Title
        title_surf = self.font_large.render(title_text, True, title_color)
        title_rect = title_surf.get_rect(center=(WINDOW_WIDTH // 2, card_y + 42))
        surface.blit(title_surf, title_rect)

        # Render Subtitle
        sub_surf = self.font_medium.render(sub_text, True, sub_color)
        sub_rect = sub_surf.get_rect(center=(WINDOW_WIDTH // 2, card_y + 92))
        surface.blit(sub_surf, sub_rect)

        # Render Elapsed Time (Frozen)
        time_surf = self.font_mono.render(f"Elapsed Time: {self.elapsed_time:.1f}s", True, COLOR_TEXT_MUTED)
        time_rect = time_surf.get_rect(center=(WINDOW_WIDTH // 2, card_y + 132))
        surface.blit(time_surf, time_rect)

        # Render Restart Instruction
        restart_surf = self.font_small.render("Press  [ R ]  to restart round", True, COLOR_TEXT_MUTED)
        restart_rect = restart_surf.get_rect(center=(WINDOW_WIDTH // 2, card_y + 178))
        surface.blit(restart_surf, restart_rect)

    def _draw_debug_hud(self, surface: pygame.Surface, fps: float):
        """Render comprehensive debug panel with telemetry and observer status."""
        panel_w = 400
        panel_h = 240
        panel_x = 45
        panel_y = 45

        # Glass panel
        hud_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        hud_surf.fill((20, 25, 35, 215))
        surface.blit(hud_surf, (panel_x, panel_y))
        pygame.draw.rect(
            surface,
            (70, 85, 110),
            pygame.Rect(panel_x, panel_y, panel_w, panel_h),
            1
        )

        lines = [
            f"[DEBUG OVERLAY (F1)]   FPS: {fps:.1f}",
            f"State: {self.state}   Sim Time: {self.elapsed_time:.2f}s",
            f"Player Pos: ({self.player.x:.1f}, {self.player.y:.1f})   Heading: {math.degrees(self.player.facing_angle):.0f}°",
            "-" * 48,
        ]

        for obs in self.observers:
            los_info = obs.last_los_info
            if los_info:
                dist = los_info["distance"]
                in_fov = "YES" if los_info["in_angle"] else "NO"
                los = "CLEAR" if not los_info["blocked_by_obstacle"] else "BLOCKED"
                vis = "VISIBLE" if los_info["is_visible"] else "HIDDEN"
                status_str = f"D:{dist:.0f} | FOV:{in_fov} | LOS:{los} => {vis}"
            else:
                status_str = "Idle"
            lines.append(f"{obs.name[:11]}: {status_str}")

        lines.append("-" * 48)
        lines.append(f"Obstacles: {len(self.obstacles)} | Outer Bounds: {len(self.boundary_walls)}")

        curr_y = panel_y + 12
        for line in lines:
            line_surf = self.font_mono.render(line, True, (230, 235, 245))
            surface.blit(line_surf, (panel_x + 12, curr_y))
            curr_y += 18

    def get_observation(self, stuck_flag: float = 0.0, stagnation_progress: float = 0.0) -> np.ndarray:
        """
        Build and return the 43-element float32 observation vector.
        Layout:
          0-1:   Player normalized position: [player_x / ARENA_WIDTH, player_y / ARENA_HEIGHT]
          2-3:   Goal relative position: [(goal_x - player_x) / ARENA_WIDTH, (goal_y - player_y) / ARENA_HEIGHT]
          4-19:  Wall raycast distances: 16 normalized distances in [0.0, 1.0] (every 22.5 deg)
          20-26: Observer 1: [active, rel_x, rel_y, facing_x, facing_y, norm_vision_range, norm_fov]
          27-33: Observer 2: [active, rel_x, rel_y, facing_x, facing_y, norm_vision_range, norm_fov]
          34-40: Observer 3: [active, rel_x, rel_y, facing_x, facing_y, norm_vision_range, norm_fov]
          41:    Stuck flag: 0.0 (not stuck) or 1.0 (stuck)
          42:    Stagnation progress: min(stagnation_steps / 60.0, 1.0) in [0.0, 1.0]
        """
        obs = np.zeros(OBSERVATION_SIZE, dtype=np.float32)

        # 0-1: Player normalized position (global map context)
        obs[0] = np.float32(self.player.x / ARENA_WIDTH)
        obs[1] = np.float32(self.player.y / ARENA_HEIGHT)

        # 2-3: Goal position relative to player
        obs[2] = np.float32((self.goal_pos[0] - self.player.x) / ARENA_WIDTH)
        obs[3] = np.float32((self.goal_pos[1] - self.player.y) / ARENA_HEIGHT)

        # 4-19: 16 wall/obstacle awareness raycasts (22.5 deg intervals)
        wall_rays = compute_wall_raycasts(
            self.player.pos,
            RAY_COUNT,
            RAY_MAX_DISTANCE,
            self.all_solid_rects
        )
        for i, dist in enumerate(wall_rays):
            obs[4 + i] = np.float32(dist)

        # 20-40: Three observers (7 values each; zeroed if inactive)
        for i, observer in enumerate(self.observers):
            base_idx = 20 + i * 7
            if observer in self.active_observers:
                obs[base_idx + 0] = np.float32(1.0)  # active flag
                obs[base_idx + 1] = np.float32((observer.x - self.player.x) / ARENA_WIDTH)
                obs[base_idx + 2] = np.float32((observer.y - self.player.y) / ARENA_HEIGHT)
                obs[base_idx + 3] = np.float32(math.cos(observer.facing_angle))
                obs[base_idx + 4] = np.float32(math.sin(observer.facing_angle))
                obs[base_idx + 5] = np.float32(observer.vision_distance / ARENA_WIDTH)
                obs[base_idx + 6] = np.float32(observer.vision_angle_rad / math.pi)
            else:
                # Inactive observer slot remains zero-filled
                pass

        # 41: Stuck flag (0.0 or 1.0)
        obs[41] = np.float32(stuck_flag)

        # 42: Stagnation progress (0.0 to 1.0)
        obs[42] = np.float32(stagnation_progress)

        return obs

