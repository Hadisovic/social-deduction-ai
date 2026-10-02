"""
skeld_navigation.py
Occupancy grid, validation pathfinder, and spatial reachability verification for The Skeld.

This module provides programmatic validation over the ACTUAL WALKABLE GEOMETRY:
- 4-pixel resolution occupancy grid.
- Accounts for PLAYER_RADIUS collision clearance.
- 8-connected A* pathfinding for optimal walkable path length and visual route display.
- Connected component analysis.

CRITICAL:
This pathfinder is for validation, diagnostics, and developer visual inspection ONLY.
It MUST NOT be exposed to PPO observations or reward functions.
"""

import math
import heapq
import collections
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
import pygame

from skeld_config import (
    SKELD_WORLD_WIDTH, SKELD_WORLD_HEIGHT,
    SKELD_PLAYER_RADIUS,
    SKELD_ALL_WALKABLE_AREAS, SKELD_ALL_SOLID_RECTS,
    SKELD_ROOM_FLOORS, SKELD_ALL_TASK_DESTINATIONS,
    TaskDestination,
)


class SkeldOccupancyGrid:
    """
    Occupancy grid over logical WORLD space for physical walkability validation.
    Cell is walkable IF AND ONLY IF:
      1. Its center is within at least one defined walkable floor region.
      2. Its center maintains >= PLAYER_RADIUS clearance from all solid wall primitives.
    """

    def __init__(self, cell_size: int = 4, player_radius: float = SKELD_PLAYER_RADIUS):
        self.cell_size = cell_size
        self.player_radius = player_radius
        self.grid_w = int(math.ceil(SKELD_WORLD_WIDTH / cell_size))
        self.grid_h = int(math.ceil(SKELD_WORLD_HEIGHT / cell_size))
        
        # Binary grid: True = walkable with player clearance, False = solid / exterior
        self.grid = np.zeros((self.grid_h, self.grid_w), dtype=bool)
        self._build_grid()
        self._compute_components()

    def _build_grid(self):
        cs = self.cell_size
        pr = int(math.ceil(self.player_radius))

        # Step 1: Mark cells inside defined walkable floor areas
        for rect in SKELD_ALL_WALKABLE_AREAS:
            gx0 = max(0, rect.left // cs)
            gx1 = min(self.grid_w, int(math.ceil(rect.right / cs)))
            gy0 = max(0, rect.top // cs)
            gy1 = min(self.grid_h, int(math.ceil(rect.bottom / cs)))
            self.grid[gy0:gy1, gx0:gx1] = True

        # Step 2: Clear cells that violate player radius clearance from solid walls
        pr_sq = self.player_radius * self.player_radius
        for srect in SKELD_ALL_SOLID_RECTS:
            gx0 = max(0, (srect.left - pr) // cs)
            gx1 = min(self.grid_w, int(math.ceil((srect.right + pr) / cs)))
            gy0 = max(0, (srect.top - pr) // cs)
            gy1 = min(self.grid_h, int(math.ceil((srect.bottom + pr) / cs)))

            for gy in range(gy0, gy1):
                cy = gy * cs + cs / 2.0
                clamped_y = max(float(srect.top), min(float(srect.bottom), cy))
                dy = cy - clamped_y
                for gx in range(gx0, gx1):
                    if not self.grid[gy, gx]:
                        continue
                    cx = gx * cs + cs / 2.0
                    clamped_x = max(float(srect.left), min(float(srect.right), cx))
                    dx = cx - clamped_x
                    if (dx * dx + dy * dy) < pr_sq:
                        self.grid[gy, gx] = False

    def _compute_components(self):
        """Find connected components via 4-connected BFS."""
        visited = np.zeros_like(self.grid, dtype=bool)
        self.components: List[List[Tuple[int, int]]] = []

        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                if self.grid[gy, gx] and not visited[gy, gx]:
                    comp = []
                    q = collections.deque([(gx, gy)])
                    visited[gy, gx] = True
                    while q:
                        cx, cy = q.popleft()
                        comp.append((cx, cy))
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                                if self.grid[ny, nx] and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    q.append((nx, ny))
                    self.components.append(comp)

        # Sort descending by size
        self.components.sort(key=len, reverse=True)
        self.main_component_set: Set[Tuple[int, int]] = set(self.components[0]) if self.components else set()

    def is_world_pos_walkable(self, wx: float, wy: float) -> bool:
        """Check if world position (wx, wy) falls on a walkable grid cell."""
        gx = int(round(wx)) // self.cell_size
        gy = int(round(wy)) // self.cell_size
        if 0 <= gy < self.grid_h and 0 <= gx < self.grid_w:
            return bool(self.grid[gy, gx])
        return False

    def find_nearest_walkable_cell(self, wx: float, wy: float, max_radius_px: float = 40.0) -> Optional[Tuple[int, int]]:
        """Find the nearest walkable grid cell to (wx, wy) within max_radius_px."""
        gx = int(round(wx)) // self.cell_size
        gy = int(round(wy)) // self.cell_size
        if 0 <= gy < self.grid_h and 0 <= gx < self.grid_w and self.grid[gy, gx]:
            return (gx, gy)

        r_cells = int(math.ceil(max_radius_px / self.cell_size))
        best_d = float("inf")
        best_cell = None
        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= ny < self.grid_h and 0 <= nx < self.grid_w and self.grid[ny, nx]:
                    d = math.hypot(dx * self.cell_size, dy * self.cell_size)
                    if d < best_d:
                        best_d = d
                        best_cell = (nx, ny)
        return best_cell

    def astar_path(
        self, start_pos: Tuple[float, float], goal_pos: Tuple[float, float]
    ) -> Tuple[bool, float, List[Tuple[float, float]]]:
        """
        8-connected A* search over actual walkable geometry.
        Returns:
            (path_found: bool, optimal_length_px: float, waypoints: List[(wx, wy)])
        """
        start_cell = self.find_nearest_walkable_cell(start_pos[0], start_pos[1])
        goal_cell = self.find_nearest_walkable_cell(goal_pos[0], goal_pos[1])

        if not start_cell or not goal_cell:
            return False, 0.0, []

        sx, sy = start_cell
        gx, gy = goal_cell

        if (sx, sy) == (gx, gy):
            return True, 0.0, [start_pos, goal_pos]

        cs = self.cell_size
        open_set = []
        heapq.heappush(open_set, (0.0, 0.0, sx, sy, [(sx, sy)]))
        visited: Dict[Tuple[int, int], float] = {}

        # 8 directions: cardinal (cost 1.0) and diagonal (cost sqrt(2))
        neighbors = [
            (-1,  0, 1.0), ( 1,  0, 1.0), ( 0, -1, 1.0), ( 0,  1, 1.0),
            (-1, -1, math.sqrt(2)), ( 1, -1, math.sqrt(2)),
            (-1,  1, math.sqrt(2)), ( 1,  1, math.sqrt(2)),
        ]

        while open_set:
            f, g, cx, cy, path = heapq.heappop(open_set)

            if (cx, cy) == (gx, gy):
                # Convert grid path to world coordinates
                waypoints = [(start_pos[0], start_pos[1])]
                for px, py in path[1:-1]:
                    waypoints.append((px * cs + cs / 2.0, py * cs + cs / 2.0))
                waypoints.append((goal_pos[0], goal_pos[1]))

                length = 0.0
                for i in range(len(waypoints) - 1):
                    length += math.hypot(
                        waypoints[i+1][0] - waypoints[i][0],
                        waypoints[i+1][1] - waypoints[i][1]
                    )
                return True, length, waypoints

            if (cx, cy) in visited and visited[(cx, cy)] <= g:
                continue
            visited[(cx, cy)] = g

            for dx, dy, cost in neighbors:
                nx, ny = cx + dx, cy + dy
                if 0 <= ny < self.grid_h and 0 <= nx < self.grid_w and self.grid[ny, nx]:
                    ng = g + cost * cs
                    if (nx, ny) not in visited or ng < visited[(nx, ny)]:
                        h = math.hypot((gx - nx) * cs, (gy - ny) * cs)
                        heapq.heappush(open_set, (ng + h, ng, nx, ny, path + [(nx, ny)]))

        return False, 0.0, []


# Global singleton instance for validation and inspector overlay
_GLOBAL_NAV_GRID: Optional[SkeldOccupancyGrid] = None

def get_skeld_nav_grid() -> SkeldOccupancyGrid:
    global _GLOBAL_NAV_GRID
    if _GLOBAL_NAV_GRID is None:
        _GLOBAL_NAV_GRID = SkeldOccupancyGrid(cell_size=4, player_radius=SKELD_PLAYER_RADIUS)
    return _GLOBAL_NAV_GRID
