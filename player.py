"""
Player (Escapee) entity with smooth continuous movement, diagonal normalization,
axis-separated wall sliding collision detection, and clean rendering.
"""

import math
from typing import Tuple, List
import pygame

from config import (
    PLAYER_START_POS,
    PLAYER_RADIUS,
    PLAYER_SPEED,
    COLOR_PLAYER,
    COLOR_PLAYER_BORDER,
    COLOR_PLAYER_CORE,
    COLOR_DEBUG_COLLIDER,
)
from geometry import normalize_vector


class Player:
    """
    The escapee character controlled manually via keyboard or future agent actions.
    """

    def __init__(self, start_pos: Tuple[float, float] = PLAYER_START_POS, radius: float = PLAYER_RADIUS, speed: float = PLAYER_SPEED):
        self.initial_pos = (float(start_pos[0]), float(start_pos[1]))
        self.x = self.initial_pos[0]
        self.y = self.initial_pos[1]
        self.radius = float(radius)
        self.speed = float(speed)
        self.facing_angle = -math.pi / 2.0  # Initially facing upward
        self.is_colliding = False

    def reset(self):
        """Reset player state deterministically to starting position."""
        self.x = self.initial_pos[0]
        self.y = self.initial_pos[1]
        self.facing_angle = -math.pi / 2.0
        self.is_colliding = False

    @property
    def pos(self) -> Tuple[float, float]:
        """Current (x, y) position as a tuple."""
        return self.x, self.y

    def get_bounding_rect(self) -> pygame.Rect:
        """Get axis-aligned bounding box for collision detection."""
        return pygame.Rect(
            int(self.x - self.radius),
            int(self.y - self.radius),
            int(self.radius * 2),
            int(self.radius * 2)
        )

    def move(
        self,
        direction: Tuple[float, float],
        dt: float,
        obstacles: List[pygame.Rect],
        bounds: pygame.Rect
    ):
        """
        Move the player along a direction vector with constant speed.
        
        Args:
            direction: (dx, dy) raw input vector.
            dt: Delta time in seconds.
            obstacles: List of obstacle rectangles.
            bounds: Outer playable area boundary rectangle.
        """
        dx, dy = direction
        nx, ny = normalize_vector(dx, dy)

        if nx == 0.0 and ny == 0.0:
            return

        # Update facing angle
        self.facing_angle = math.atan2(ny, nx)

        # Velocity components
        vx = nx * self.speed
        vy = ny * self.speed

        # -----------------------------------------------------------------
        # AXIS-SEPARATED MOVEMENT & COLLISION RESOLUTION (Allows wall-sliding)
        # -----------------------------------------------------------------

        # Reset collision diagnostic flag for this step
        self.is_colliding = False

        # 1. Horizontal Movement & Collision
        pre_x = self.x + vx * dt
        # Clamp against room boundaries
        self.x = max(bounds.left + self.radius, min(bounds.right - self.radius, pre_x))
        if self.x != pre_x:
            self.is_colliding = True

        # Check and resolve against obstacle rectangles on X axis
        player_rect = self.get_bounding_rect()
        for obs in obstacles:
            if player_rect.colliderect(obs):
                self.is_colliding = True
                if vx > 0.0:
                    self.x = obs.left - self.radius
                elif vx < 0.0:
                    self.x = obs.right + self.radius
                player_rect = self.get_bounding_rect()

        # 2. Vertical Movement & Collision
        pre_y = self.y + vy * dt
        # Clamp against room boundaries
        self.y = max(bounds.top + self.radius, min(bounds.bottom - self.radius, pre_y))
        if self.y != pre_y:
            self.is_colliding = True

        # Check and resolve against obstacle rectangles on Y axis
        player_rect = self.get_bounding_rect()
        for obs in obstacles:
            if player_rect.colliderect(obs):
                self.is_colliding = True
                if vy > 0.0:
                    self.y = obs.top - self.radius
                elif vy < 0.0:
                    self.y = obs.bottom + self.radius
                player_rect = self.get_bounding_rect()

    def draw(self, surface: pygame.Surface, debug: bool = False):
        """Render the player character cleanly on the target surface."""
        center = (int(self.x), int(self.y))
        int_radius = int(self.radius)

        # Draw outer boundary
        pygame.draw.circle(surface, COLOR_PLAYER_BORDER, center, int_radius + 2)
        # Draw main player body
        pygame.draw.circle(surface, COLOR_PLAYER, center, int_radius)
        # Draw glowing inner core
        pygame.draw.circle(surface, COLOR_PLAYER_CORE, center, max(2, int_radius // 3))

        # Draw directional nose / facing pointer
        nose_len = self.radius + 4
        nose_x = self.x + math.cos(self.facing_angle) * nose_len
        nose_y = self.y + math.sin(self.facing_angle) * nose_len
        pygame.draw.line(surface, COLOR_PLAYER_BORDER, center, (int(nose_x), int(nose_y)), 3)

        if debug:
            # Draw collision bounding box
            pygame.draw.rect(surface, COLOR_DEBUG_COLLIDER, self.get_bounding_rect(), 1)
