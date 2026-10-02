"""
inspect_skeld.py
Interactive visual inspector for the Skeld navigation environment.

Operates in logical WORLD space, rendered to DISPLAY space via uniform scale + letterboxing.

Required Toggles:
  R               : Toggle 16 radial raycasts
  C               : Toggle collision / walkability geometry overlay
  T               : Toggle task destination markers
  L               : Toggle room & region labels
  V               : Toggle vent markers
  G               : Toggle internal validation A* route path overlay
  H               : Toggle HUD display
  Tab / N         : Cycle goal through the 40 authentic task destinations
  Mouse Left Click: Set goal directly to clicked world position
  SPACE           : Reset to a new random episode
  ESC / Q         : Quit

HUD displays:
  - World coordinate (px, py) & Screen coordinate (sx, sy)
  - True spatial room / region membership
  - Nearest task destination & distance
  - Current goal destination & distance
  - Player radius & speed
  - Ray sensor count & range
  - Optimal validation route length
  - FPS counter
"""

import sys
import math
import time
from typing import Optional, List, Tuple

import pygame

from skeld_config import (
    SKELD_WORLD_WIDTH, SKELD_WORLD_HEIGHT,
    SKELD_WINDOW_WIDTH, SKELD_WINDOW_HEIGHT, SKELD_WINDOW_TITLE,
    SKELD_COLOR_BG, SKELD_COLOR_FLOOR, SKELD_COLOR_WALL, SKELD_COLOR_WALL_BORDER,
    SKELD_COLOR_ROOM_LABEL, SKELD_COLOR_VENT, SKELD_COLOR_DOOR,
    SKELD_COLOR_PLAYER, SKELD_COLOR_PLAYER_BDR, SKELD_COLOR_PLAYER_CORE,
    SKELD_COLOR_TASK, SKELD_COLOR_TASK_ACTIVE,
    SKELD_COLOR_GOAL, SKELD_COLOR_GOAL_GLOW,
    SKELD_COLOR_HUD_BG, SKELD_COLOR_HUD_TEXT, SKELD_COLOR_HUD_ACCENT,
    SKELD_COLOR_SUCCESS, SKELD_COLOR_ALERT, SKELD_COLOR_NAV_PATH,
    SKELD_PLAYER_RADIUS, SKELD_PLAYER_SPEED, SKELD_GOAL_RADIUS,
    SKELD_RAY_COUNT, SKELD_RAY_MAX_DISTANCE,
    SKELD_ALL_SOLID_RECTS, SKELD_ALL_WALKABLE_AREAS,
    SKELD_ROOMS, SKELD_ALL_TASK_DESTINATIONS, TaskDestination,
    SKELD_ALL_VENTS,
    world_to_screen, screen_to_world,
    get_room_or_region,
)
from skeld_environment import SkeldNavEnv, _move_player
from skeld_navigation import get_skeld_nav_grid
from geometry import compute_wall_raycasts


def get_nearest_task(wx: float, wy: float) -> Tuple[Optional[TaskDestination], float]:
    """Find nearest task destination to world position (wx, wy)."""
    best_t = None
    best_d = float("inf")
    for t in SKELD_ALL_TASK_DESTINATIONS:
        d = math.hypot(wx - t.world_x, wy - t.world_y)
        if d < best_d:
            best_d = d
            best_t = t
    return best_t, best_d


def main():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((SKELD_WINDOW_WIDTH, SKELD_WINDOW_HEIGHT))
    pygame.display.set_caption(f"INSPECT: {SKELD_WINDOW_TITLE}")
    clock = pygame.time.Clock()

    font_s = pygame.font.SysFont("Consolas", 11)
    font_m = pygame.font.SysFont("Trebuchet MS", 13, bold=True)
    font_l = pygame.font.SysFont("Trebuchet MS", 18, bold=True)

    # Initialize environment in task goal mode
    env = SkeldNavEnv(render_mode=None, task_goal_mode=True)
    obs, info = env.reset(seed=0)
    px, py = info["player_pos"]
    gx, gy = info["goal_pos"]
    facing = -math.pi / 2.0

    task_idx = 0

    # Inspection toggles
    show_rays     = True
    show_col      = False
    show_tasks    = True
    show_labels   = True
    show_vents    = True
    show_nav_path = True
    show_hud      = True

    # Nav grid for validation A* path overlay
    nav_grid = get_skeld_nav_grid()
    val_path: List[Tuple[float, float]] = []
    val_path_len: float = 0.0
    val_path_found: bool = False
    last_path_calc = 0.0

    running = True
    frame   = 0
    fps     = 0.0
    t_last  = time.perf_counter()

    while running:
        dt = clock.tick(60) / 1000.0
        frame += 1
        if frame % 20 == 0:
            now = time.perf_counter()
            fps = 20.0 / max(0.001, now - t_last)
            t_last = now

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    obs, info = env.reset()
                    px, py = info["player_pos"]
                    gx, gy = info["goal_pos"]
                    val_path = []
                elif event.key == pygame.K_r:
                    show_rays = not show_rays
                elif event.key == pygame.K_c:
                    show_col = not show_col
                elif event.key == pygame.K_t:
                    show_tasks = not show_tasks
                elif event.key == pygame.K_l:
                    show_labels = not show_labels
                elif event.key == pygame.K_v:
                    show_vents = not show_vents
                elif event.key == pygame.K_g:
                    show_nav_path = not show_nav_path
                elif event.key == pygame.K_h:
                    show_hud = not show_hud
                elif event.key in (pygame.K_TAB, pygame.K_n):
                    # Cycle task goal
                    task_idx = (task_idx + 1) % len(SKELD_ALL_TASK_DESTINATIONS)
                    t_dest = SKELD_ALL_TASK_DESTINATIONS[task_idx]
                    gx, gy = t_dest.world_x, t_dest.world_y
                    val_path = []
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Set goal to clicked point in world space
                mx, my = event.pos
                gx, gy = screen_to_world(mx, my)
                val_path = []

        # --- Movement (WASD / Arrows in World Space) ---
        keys = pygame.key.get_pressed()
        dx, dy = 0.0, 0.0
        if keys[pygame.K_LEFT]  or keys[pygame.K_a]: dx -= 1.0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: dx += 1.0
        if keys[pygame.K_UP]    or keys[pygame.K_w]: dy -= 1.0
        if keys[pygame.K_DOWN]  or keys[pygame.K_s]: dy += 1.0
        if dx != 0.0 or dy != 0.0:
            px, py, _ = _move_player(px, py, dx, dy, SKELD_PLAYER_SPEED, dt,
                                     SKELD_ALL_SOLID_RECTS, SKELD_PLAYER_RADIUS)
            facing = math.atan2(dy, dx)

        # Update validation path periodically
        now = time.perf_counter()
        if show_nav_path and (now - last_path_calc > 0.15 or not val_path):
            val_path_found, val_path_len, val_path = nav_grid.astar_path((px, py), (gx, gy))
            last_path_calc = now

        # --- DRAWING ---
        screen.fill(SKELD_COLOR_BG)

        # 1. Draw floor areas
        for frect in SKELD_ALL_WALKABLE_AREAS:
            sx, sy = world_to_screen(frect.left, frect.top)
            sw = max(1, int(round(frect.width * (SKELD_WINDOW_WIDTH / SKELD_WORLD_WIDTH))))
            sh = max(1, int(round(frect.height * (SKELD_WINDOW_WIDTH / SKELD_WORLD_WIDTH))))
            pygame.draw.rect(screen, SKELD_COLOR_FLOOR, pygame.Rect(sx, sy, sw, sh))

        # 2. Draw solid walls
        for srect in SKELD_ALL_SOLID_RECTS:
            sx, sy = world_to_screen(srect.left, srect.top)
            sw = max(1, int(round(srect.width * (SKELD_WINDOW_WIDTH / SKELD_WORLD_WIDTH))))
            sh = max(1, int(round(srect.height * (SKELD_WINDOW_WIDTH / SKELD_WORLD_WIDTH))))
            drect = pygame.Rect(sx, sy, sw, sh)
            pygame.draw.rect(screen, SKELD_COLOR_WALL, drect)
            pygame.draw.rect(screen, SKELD_COLOR_WALL_BORDER, drect, 1)

        # 3. Collision / Walkability overlay
        if show_col:
            for frect in SKELD_ALL_WALKABLE_AREAS:
                sx, sy = world_to_screen(frect.left, frect.top)
                sw = max(1, int(round(frect.width * (SKELD_WINDOW_WIDTH / SKELD_WORLD_WIDTH))))
                sh = max(1, int(round(frect.height * (SKELD_WINDOW_WIDTH / SKELD_WORLD_WIDTH))))
                pygame.draw.rect(screen, (30, 80, 50), pygame.Rect(sx, sy, sw, sh), 1)

        # 4. Room & region labels
        if show_labels:
            for room in SKELD_ROOMS:
                cx, cy = room.center
                scx, scy = world_to_screen(cx, cy)
                lbl = font_s.render(room.name, True, SKELD_COLOR_ROOM_LABEL)
                screen.blit(lbl, (scx - lbl.get_width() // 2, scy - lbl.get_height() // 2))

        # 5. Vent markers
        if show_vents:
            for v in SKELD_ALL_VENTS:
                vx, vy = v["world_pos"]
                svx, svy = world_to_screen(vx, vy)
                pygame.draw.rect(screen, SKELD_COLOR_VENT, pygame.Rect(svx - 5, svy - 5, 10, 10))
                pygame.draw.rect(screen, (0, 150, 140), pygame.Rect(svx - 5, svy - 5, 10, 10), 1)
                vlbl = font_s.render("V", True, (0, 200, 180))
                screen.blit(vlbl, (svx - 3, svy - 5))

        # 6. Task destination markers
        if show_tasks:
            for t in SKELD_ALL_TASK_DESTINATIONS:
                stx, sty = world_to_screen(t.world_x, t.world_y)
                d = math.hypot(px - t.world_x, py - t.world_y)
                c = SKELD_COLOR_TASK_ACTIVE if d < 25.0 else SKELD_COLOR_TASK
                pygame.draw.circle(screen, c, (stx, sty), 4)
                pygame.draw.circle(screen, (180, 140, 0), (stx, sty), 4, 1)
                if d < 45.0:
                    tlbl = font_s.render(f"{t.name} ({t.id})", True, (255, 230, 120))
                    screen.blit(tlbl, (stx + 7, sty - 6))

        # 7. Validation A* path overlay (developer-only route visualization)
        if show_nav_path and val_path and len(val_path) >= 2:
            screen_pts = [world_to_screen(wx, wy) for wx, wy in val_path]
            pygame.draw.lines(screen, SKELD_COLOR_NAV_PATH, False, screen_pts, 2)
            # Breadcrumb dots
            for pt in screen_pts[1:-1:3]:
                pygame.draw.circle(screen, (255, 140, 220), pt, 2)

        # 8. 16 Raycasts
        spx, spy = world_to_screen(px, py)
        if show_rays:
            rays = compute_wall_raycasts((px, py), SKELD_RAY_COUNT, SKELD_RAY_MAX_DISTANCE, SKELD_ALL_SOLID_RECTS)
            angle_step = (2.0 * math.pi) / SKELD_RAY_COUNT
            for i, r_norm in enumerate(rays):
                angle = i * angle_step
                r_len = r_norm * SKELD_RAY_MAX_DISTANCE
                w_ex = px + math.cos(angle) * r_len
                w_ey = py + math.sin(angle) * r_len
                sex, sey = world_to_screen(w_ex, w_ey)
                col = (40, 90, 60) if r_norm > 0.95 else (80, 180, 80)
                pygame.draw.line(screen, col, (spx, spy), (sex, sey), 1)
                pygame.draw.circle(screen, (255, 80, 80), (sex, sey), 2)

        # 9. Goal marker
        sgx, sgy = world_to_screen(gx, gy)
        d_gr = max(6, int(round(SKELD_GOAL_RADIUS * (SKELD_WINDOW_WIDTH / SKELD_WORLD_WIDTH))))
        pygame.draw.circle(screen, SKELD_COLOR_GOAL_GLOW, (sgx, sgy), d_gr + 4, 2)
        pygame.draw.circle(screen, SKELD_COLOR_GOAL, (sgx, sgy), d_gr)
        pygame.draw.circle(screen, (200, 255, 220), (sgx, sgy), 4)
        glbl = font_s.render("GOAL", True, (150, 255, 180))
        screen.blit(glbl, (sgx - glbl.get_width() // 2, sgy + d_gr + 2))

        # 10. Player entity
        d_pr = max(4, int(round(SKELD_PLAYER_RADIUS * (SKELD_WINDOW_WIDTH / SKELD_WORLD_WIDTH))))
        pygame.draw.circle(screen, SKELD_COLOR_PLAYER_BDR, (spx, spy), d_pr + 2)
        pygame.draw.circle(screen, SKELD_COLOR_PLAYER,     (spx, spy), d_pr)
        pygame.draw.circle(screen, SKELD_COLOR_PLAYER_CORE,(spx, spy), max(2, d_pr // 3))
        nose_x = spx + int(math.cos(facing) * (d_pr + 5))
        nose_y = spy + int(math.sin(facing) * (d_pr + 5))
        pygame.draw.line(screen, SKELD_COLOR_PLAYER_BDR, (spx, spy), (nose_x, nose_y), 2)

        # 11. HUD Overlay
        if show_hud:
            cur_region = get_room_or_region(px, py)
            nearest_t, nearest_d = get_nearest_task(px, py)
            goal_dist = math.hypot(px - gx, py - gy)
            nearest_t_name = f"{nearest_t.name} ({nearest_d:.0f}px)" if nearest_t else "None"
            path_str = f"{val_path_len:.0f}px (A* optimal)" if val_path_found else "No path"

            hud_lines = [
                f"World Pos: ({px:.1f}, {py:.1f}) | Screen: ({spx}, {spy})",
                f"Region: {cur_region}",
                f"Nearest Task: {nearest_t_name}",
                f"Goal Dist: {goal_dist:.0f}px | Val Route: {path_str}",
                f"Player Radius: {SKELD_PLAYER_RADIUS:.1f}px | Speed: {SKELD_PLAYER_SPEED:.0f}px/s",
                f"Sensors: {SKELD_RAY_COUNT} rays, max {SKELD_RAY_MAX_DISTANCE:.0f}px | FPS: {fps:.0f}",
                "",
                f"[R] Rays: {'ON' if show_rays else 'OFF'}  [C] Collision Geom: {'ON' if show_col else 'OFF'}",
                f"[T] Tasks: {'ON' if show_tasks else 'OFF'}  [L] Labels: {'ON' if show_labels else 'OFF'}",
                f"[V] Vents: {'ON' if show_vents else 'OFF'}  [G] Val Path: {'ON' if show_nav_path else 'OFF'}",
                "[Tab/N] Cycle Task  [Click] Set Goal  [SPACE] Reset  [ESC] Quit",
            ]

            hud_w = 420
            hud_h = len(hud_lines) * 14 + 16
            hud_surf = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
            hud_surf.fill((10, 15, 28, 220))
            screen.blit(hud_surf, (8, 8))
            pygame.draw.rect(screen, SKELD_COLOR_HUD_ACCENT, pygame.Rect(8, 8, hud_w, hud_h), 1)

            for j, line in enumerate(hud_lines):
                col = SKELD_COLOR_HUD_ACCENT if j == 0 else (SKELD_COLOR_SUCCESS if "Region:" in line else SKELD_COLOR_HUD_TEXT)
                rendered = font_s.render(line, True, col)
                screen.blit(rendered, (14, 12 + j * 14))

        pygame.display.flip()

    pygame.quit()
    env.close()


if __name__ == "__main__":
    main()
