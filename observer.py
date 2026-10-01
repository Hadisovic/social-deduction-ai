"""
Observer entity with fixed patrol paths, deterministic waypoint following,
raycast-based vision cone rendering, and decoupled geometric line-of-sight detection.
"""

import math
from typing import Tuple, List, Dict, Any, Optional
import pygame

from config import (
    OBSERVER_RADIUS,
    VISION_DISTANCE,
    VISION_ANGLE_DEG,
    VISION_ANGLE_RAD,
    VISION_CONE_ALPHA,
    VISION_RAY_COUNT,
    COLOR_OBSERVER_CORE,
    COLOR_DEBUG_RAY_CLEAR,
    COLOR_DEBUG_RAY_BLOCKED,
    COLOR_DEBUG_RAY_IDLE,
    COLOR_DEBUG_PATH,
    COLOR_DEBUG_WAYPOINT,
    COLOR_DEBUG_COLLIDER,
)
from geometry import (
    cast_ray_against_rects,
    check_line_of_sight,
)


class Observer:
    """
    Patrolling observer that scans for the escapee with a directional vision cone.
    """

    def __init__(self, config: Dict[str, Any]):
        self.id = config["id"]
        self.name = config["name"]
        self.color = config["color"]
        self.speed = float(config["speed"])
        self.radius = float(OBSERVER_RADIUS)

        # Patrol waypoints
        self.patrol_points: List[Tuple[float, float]] = [
            (float(p[0]), float(p[1])) for p in config["patrol_points"]
        ]
        self.initial_target_wp_index = int(config.get("initial_waypoint_index", 1))

        # Vision parameters
        self.vision_distance = float(VISION_DISTANCE)
        self.vision_angle_deg = float(VISION_ANGLE_DEG)
        self.vision_angle_rad = float(VISION_ANGLE_RAD)

        # Dynamic state
        self.x = 0.0
        self.y = 0.0
        self.target_wp_index = self.initial_target_wp_index
        self.facing_angle = 0.0

        # Last frame LOS check debug info
        self.last_los_info: Optional[Dict[str, Any]] = None

        # Set initial position and orientation
        self.reset()

    def reset(self):
        """Deterministically reset observer position, waypoint target, and facing angle."""
        start_wp = self.patrol_points[0]
        self.x = start_wp[0]
        self.y = start_wp[1]
        self.target_wp_index = self.initial_target_wp_index

        next_wp = self.patrol_points[self.target_wp_index]
        dx = next_wp[0] - self.x
        dy = next_wp[1] - self.y
        self.facing_angle = math.atan2(dy, dx)
        self.last_los_info = None

    @property
    def pos(self) -> Tuple[float, float]:
        """Current (x, y) coordinates."""
        return self.x, self.y

    def get_bounding_rect(self) -> pygame.Rect:
        """Bounding box for collision/debug."""
        return pygame.Rect(
            int(self.x - self.radius),
            int(self.y - self.radius),
            int(self.radius * 2),
            int(self.radius * 2)
        )

    def update(self, dt: float):
        """
        Advance observer along fixed patrol waypoints using constant speed.
        Sub-step loop ensures deterministic and smooth waypoint transitions.
        """
        move_dist = self.speed * dt

        while move_dist > 0.0:
            target_wp = self.patrol_points[self.target_wp_index]
            dx = target_wp[0] - self.x
            dy = target_wp[1] - self.y
            dist = math.hypot(dx, dy)

            if dist < 1e-4:
                # Instantly at waypoint; advance to next
                self.target_wp_index = (self.target_wp_index + 1) % len(self.patrol_points)
                continue

            self.facing_angle = math.atan2(dy, dx)

            if move_dist < dist:
                self.x += (dx / dist) * move_dist
                self.y += (dy / dist) * move_dist
                move_dist = 0.0
            else:
                self.x = target_wp[0]
                self.y = target_wp[1]
                move_dist -= dist
                self.target_wp_index = (self.target_wp_index + 1) % len(self.patrol_points)

    def can_see_player(
        self,
        player_pos: Tuple[float, float],
        obstacles: List[pygame.Rect]
    ) -> bool:
        """
        Mathematical line-of-sight detection.
        Pure geometric calculation completely independent of rendering:
        1. Distance <= vision_distance
        2. Angular offset <= FOV / 2
        3. Ray from observer to player does not intersect any obstacle rectangle.
        """
        is_visible, info = check_line_of_sight(
            (self.x, self.y),
            self.facing_angle,
            player_pos,
            self.vision_distance,
            self.vision_angle_rad,
            obstacles
        )
        self.last_los_info = info
        return is_visible

    def draw_vision_cone(
        self,
        surface: pygame.Surface,
        obstacles: List[pygame.Rect]
    ):
        """
        Render translucent vision cone polygon.
        Casts rays to clip naturally against solid obstacles and walls.
        """
        half_fov = self.vision_angle_rad / 2.0
        start_angle = self.facing_angle - half_fov
        end_angle = self.facing_angle + half_fov

        # Origin point of the cone
        cone_points = [(self.x, self.y)]

        # Raycast fan across the FOV arc
        for i in range(VISION_RAY_COUNT + 1):
            fraction = i / float(VISION_RAY_COUNT)
            ray_angle = start_angle + fraction * (end_angle - start_angle)
            dir_x = math.cos(ray_angle)
            dir_y = math.sin(ray_angle)

            hit_dist = cast_ray_against_rects(
                (self.x, self.y),
                (dir_x, dir_y),
                self.vision_distance,
                obstacles
            )

            cone_points.append((self.x + dir_x * hit_dist, self.y + dir_y * hit_dist))

        # Draw translucent polygon on an alpha surface
        cone_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        cone_color = (self.color[0], self.color[1], self.color[2], VISION_CONE_ALPHA)
        pygame.draw.polygon(cone_surface, cone_color, cone_points)

        # Draw vision perimeter arc edge
        edge_color = (self.color[0], self.color[1], self.color[2], 110)
        if len(cone_points) > 2:
            pygame.draw.lines(cone_surface, edge_color, False, cone_points[1:], 2)
            # Edge boundaries from origin
            pygame.draw.line(cone_surface, edge_color, cone_points[0], cone_points[1], 1)
            pygame.draw.line(cone_surface, edge_color, cone_points[0], cone_points[-1], 1)

        surface.blit(cone_surface, (0, 0))

    def draw(self, surface: pygame.Surface, debug: bool = False):
        """Render observer body and directional gaze."""
        center = (int(self.x), int(self.y))
        int_radius = int(self.radius)

        # Outer border
        pygame.draw.circle(surface, (30, 35, 45), center, int_radius + 2)
        # Main colored body
        pygame.draw.circle(surface, self.color, center, int_radius)
        # Eye / Pupil highlight
        pupil_dist = int_radius * 0.45
        pupil_x = self.x + math.cos(self.facing_angle) * pupil_dist
        pupil_y = self.y + math.sin(self.facing_angle) * pupil_dist
        pygame.draw.circle(surface, COLOR_OBSERVER_CORE, (int(pupil_x), int(pupil_y)), max(3, int_radius // 3))

        # Directional pointer
        pointer_x = self.x + math.cos(self.facing_angle) * (int_radius + 4)
        pointer_y = self.y + math.sin(self.facing_angle) * (int_radius + 4)
        pygame.draw.line(surface, (30, 35, 45), center, (int(pointer_x), int(pointer_y)), 3)

        if debug:
            pygame.draw.rect(surface, COLOR_DEBUG_COLLIDER, self.get_bounding_rect(), 1)

    def draw_debug(self, surface: pygame.Surface, player_pos: Tuple[float, float]):
        """Render patrol path, waypoints, and LOS test vector to player."""
        # 1. Draw patrol loop lines
        if len(self.patrol_points) > 1:
            pygame.draw.lines(surface, COLOR_DEBUG_PATH, True, self.patrol_points, 1)

        # 2. Draw waypoint dots with indices
        font = pygame.font.SysFont("Consolas", 11)
        for idx, pt in enumerate(self.patrol_points):
            is_current = (idx == self.target_wp_index)
            color = self.color if is_current else COLOR_DEBUG_WAYPOINT
            radius = 5 if is_current else 3
            pygame.draw.circle(surface, color, (int(pt[0]), int(pt[1])), radius)
            text_surf = font.render(str(idx + 1), True, (40, 50, 60))
            surface.blit(text_surf, (int(pt[0]) + 6, int(pt[1]) - 6))

        # 3. Draw line to player indicating LOS status
        if self.last_los_info is not None:
            if self.last_los_info["is_visible"]:
                line_color = COLOR_DEBUG_RAY_CLEAR
                width = 2
            elif self.last_los_info["in_angle"] and self.last_los_info["blocked_by_obstacle"]:
                line_color = COLOR_DEBUG_RAY_BLOCKED
                width = 2
            else:
                line_color = COLOR_DEBUG_RAY_IDLE
                width = 1

            line_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            pygame.draw.line(
                line_surf,
                line_color,
                (int(self.x), int(self.y)),
                (int(player_pos[0]), int(player_pos[1])),
                width
            )
            surface.blit(line_surf, (0, 0))
