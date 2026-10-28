"""Premium dark spectator UI renderer.

Draws the full spectator window: map, sidebar panels, timeline, neural
decision dashboard, player inspection, and training stats.

The renderer is a pure drawing layer — it receives state and returns a
pygame Surface.  It owns no game logic.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import pygame

from among_us_renderer import MapRenderer
from phase3_assets import character_image
from phase3_renderer import character_sprite, player_color, label_color
from spectator.playback import FrameState, EventMarker

# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------

# Dark navy background palette
BG_DEEP    = (6,  10,  18)
BG_PANEL   = (11, 18,  30)
BG_CARD    = (16, 27,  45)
BG_HOVER   = (22, 36,  58)
BG_LINE    = (28, 44,  68)

# Text
TEXT_PRIMARY  = (226, 237, 249)
TEXT_MUTED    = (120, 145, 175)
TEXT_DIM      = (70,  95,  128)
TEXT_ACCENT   = (100, 230, 200)
TEXT_WARN     = (255, 195,  60)
TEXT_DANGER   = (255,  90,  90)
TEXT_SUCCESS  = ( 80, 210, 130)

# Neon accent palette (restrained)
ACCENT_TEAL   = (100, 230, 200)
ACCENT_BLUE   = ( 80, 150, 255)
ACCENT_AMBER  = (255, 195,  60)
ACCENT_RED    = (255,  90,  90)
ACCENT_GREEN  = ( 80, 210, 130)
ACCENT_PURPLE = (175, 110, 255)

# Timeline
TL_BG       = (10, 16, 28)
TL_FILL     = (40, 65, 100)
TL_PROGRESS = (100, 230, 200)

# Probability bar fill
PROB_CHOSEN = (100, 230, 200)
PROB_OTHER  = (60,  95, 140)
PROB_BG     = (18,  28,  46)


# ---------------------------------------------------------------------------
# Panel layout constants (pixels)
# ---------------------------------------------------------------------------

LEFT_PANEL_W  = 260   # replay library / episode list
RIGHT_PANEL_W = 310   # player inspection + neural
TIMELINE_H    = 72    # bottom timeline bar
HEADER_H      = 56    # top header bar


@dataclass
class LayoutRects:
    header: pygame.Rect
    map_area: pygame.Rect
    left_panel: pygame.Rect
    right_panel: pygame.Rect
    timeline: pygame.Rect
    window_size: tuple[int, int]


def compute_layout(window_size: tuple[int, int]) -> LayoutRects:
    w, h = window_size
    header   = pygame.Rect(0, 0, w, HEADER_H)
    timeline = pygame.Rect(0, h - TIMELINE_H, w, TIMELINE_H)
    left     = pygame.Rect(0, HEADER_H, LEFT_PANEL_W, h - HEADER_H - TIMELINE_H)
    right    = pygame.Rect(w - RIGHT_PANEL_W, HEADER_H, RIGHT_PANEL_W, h - HEADER_H - TIMELINE_H)
    map_area = pygame.Rect(LEFT_PANEL_W, HEADER_H, w - LEFT_PANEL_W - RIGHT_PANEL_W, h - HEADER_H - TIMELINE_H)
    return LayoutRects(header, map_area, left, right, timeline, window_size)


# ---------------------------------------------------------------------------
# Font cache
# ---------------------------------------------------------------------------

class Fonts:
    _instance: "Fonts | None" = None

    def __init__(self):
        pygame.font.init()
        self.tiny   = pygame.font.SysFont("segoeui", 11)
        self.small  = pygame.font.SysFont("segoeui", 13)
        self.normal = pygame.font.SysFont("segoeui", 16)
        self.medium = pygame.font.SysFont("segoeui", 18, bold=True)
        self.large  = pygame.font.SysFont("segoeui", 22, bold=True)
        self.title  = pygame.font.SysFont("segoeui", 28, bold=True)
        self.mono   = pygame.font.SysFont("consolas", 12)

    @classmethod
    def get(cls) -> "Fonts":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_text(surf: pygame.Surface, text: str, pos: tuple[int, int],
              color=TEXT_PRIMARY, font=None, align="left",
              max_width: int | None = None) -> int:
    """Draw text and return width used."""
    f = font or Fonts.get().normal
    s = str(text)
    if max_width is not None:
        while s and f.size(s)[0] > max_width:
            s = s[:-1]
        if len(s) < len(str(text)):
            s = s[:-2] + "…"
    surf_text = f.render(s, True, color)
    rect = surf_text.get_rect()
    if align == "center":
        rect.centerx = pos[0]
        rect.top = pos[1]
    elif align == "right":
        rect.right = pos[0]
        rect.top = pos[1]
    else:
        rect.topleft = pos
    surf.blit(surf_text, rect)
    return surf_text.get_width()


def draw_rect_aa(surf: pygame.Surface, color: tuple, rect: pygame.Rect,
                 radius: int = 6, alpha: int = 255) -> None:
    """Draw a rounded rect with optional alpha."""
    if alpha < 255:
        tmp = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(tmp, (*color, alpha), (0, 0, rect.width, rect.height), border_radius=radius)
        surf.blit(tmp, rect.topleft)
    else:
        pygame.draw.rect(surf, color, rect, border_radius=radius)


def draw_separator(surf: pygame.Surface, y: int, x0: int, x1: int,
                   color=BG_LINE) -> None:
    pygame.draw.line(surf, color, (x0, y), (x1, y))


def draw_pill(surf: pygame.Surface, text: str, pos: tuple[int, int],
              color=ACCENT_TEAL, bg=BG_CARD, font=None) -> pygame.Rect:
    """Draw a small rounded tag/pill."""
    f = font or Fonts.get().tiny
    text_surf = f.render(text, True, color)
    padding = (8, 3)
    rect = pygame.Rect(pos[0], pos[1],
                       text_surf.get_width() + padding[0] * 2,
                       text_surf.get_height() + padding[1] * 2)
    pygame.draw.rect(surf, bg, rect, border_radius=8)
    pygame.draw.rect(surf, color, rect, 1, border_radius=8)
    surf.blit(text_surf, (rect.x + padding[0], rect.y + padding[1]))
    return rect


def draw_progress_bar(surf: pygame.Surface, rect: pygame.Rect, value: float,
                      color=ACCENT_TEAL, bg=PROB_BG, radius: int = 3) -> None:
    """Draw a horizontal progress bar with value in [0, 1]."""
    pygame.draw.rect(surf, bg, rect, border_radius=radius)
    if value > 0:
        fill = pygame.Rect(rect.x, rect.y, max(radius, round(rect.width * min(1.0, value))), rect.height)
        pygame.draw.rect(surf, color, fill, border_radius=radius)


def section_header(surf: pygame.Surface, label: str, x: int, y: int, w: int,
                   color=ACCENT_TEAL) -> int:
    """Draw a section divider with label.  Returns new y."""
    draw_text(surf, label, (x, y), color, Fonts.get().small)
    draw_separator(surf, y + 18, x, x + w, BG_LINE)
    return y + 24


# ---------------------------------------------------------------------------
# Player character drawing (reuses phase3_renderer logic)
# ---------------------------------------------------------------------------

def draw_character(surf: pygame.Surface, color_name: str, pos: tuple[int, int],
                   height: int, direction: str = "right", frame: int = 0,
                   body: bool = False, alpha: int = 255) -> None:
    sprite = character_image(color_name, height, direction, frame, body)
    if sprite is None:
        sprite = character_sprite(player_color(color_name), height, direction, frame, body)
    if alpha < 255:
        sprite = sprite.copy()
        sprite.set_alpha(alpha)
    surf.blit(sprite, sprite.get_rect(midbottom=(round(pos[0]), round(pos[1] + height * 0.07))))


# ---------------------------------------------------------------------------
# Main spectator renderer
# ---------------------------------------------------------------------------

class SpectatorRenderer:
    """Stateless drawing engine.  Call draw() with current application state."""

    def __init__(self, world, window_size: tuple[int, int] = (1600, 900)):
        self.fonts = Fonts.get()
        self._world = world
        self._window_size = window_size
        self._layout = compute_layout(window_size)
        map_sz = (self._layout.map_area.width, self._layout.map_area.height)
        self._map_renderer = MapRenderer(world, map_sz)
        self._anim_time: float = 0.0

    def resize(self, window_size: tuple[int, int]) -> None:
        self._window_size = window_size
        self._layout = compute_layout(window_size)
        map_sz = (self._layout.map_area.width, self._layout.map_area.height)
        self._map_renderer = MapRenderer(self._world, map_sz)

    def set_anim_time(self, t: float) -> None:
        self._anim_time = t

    # ------------------------------------------------------------------
    # Top-level draw
    # ------------------------------------------------------------------

    def draw(self, app_state: "SpectatorAppState") -> pygame.Surface:
        surf = pygame.Surface(self._window_size)
        surf.fill(BG_DEEP)
        lay = self._layout
        app_state._library_hitboxes=[]
        app_state._player_hitboxes={}
        app_state._speed_hitboxes=[]

        # Map area (background rendered first)
        self._draw_map_area(surf, lay.map_area, app_state)

        # Panels (drawn over background)
        self._draw_left_panel(surf, lay.left_panel, app_state)
        self._draw_right_panel(surf, lay.right_panel, app_state)
        self._draw_header(surf, lay.header, app_state)
        self._draw_timeline(surf, lay.timeline, app_state)

        return surf

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _draw_header(self, surf: pygame.Surface, rect: pygame.Rect,
                     app: "SpectatorAppState") -> None:
        pygame.draw.rect(surf, BG_PANEL, rect)
        pygame.draw.line(surf, BG_LINE, rect.bottomleft, rect.bottomright)
        f = self.fonts

        # Title
        draw_text(surf, "SOCIAL DEDUCTION AI", (18, 12), ACCENT_TEAL, f.large)
        draw_text(surf, "SPECTATOR & REPLAY STUDIO", (18, 36), TEXT_MUTED, f.small)

        # Center: current status
        cx = rect.centerx
        if app.frame_state:
            fs = app.frame_state
            phase_str = fs.phase.upper()
            draw_text(surf, phase_str, (cx - 80, 10), ACCENT_AMBER, f.medium, align="left")
            draw_text(surf,
                      f"t = {fs.sim_time:.1f}s   tick {fs.tick}",
                      (cx - 80, 32), TEXT_MUTED, f.small)
            draw_text(surf,
                      f"{app.playback_speed:g}×  {'▶ PLAYING' if app.playing else '‖ PAUSED'}",
                      (cx + 10, 21), ACCENT_BLUE if app.playing else TEXT_MUTED, f.small)
        else:
            draw_text(surf, "NO REPLAY LOADED", (cx - 80, 21), TEXT_MUTED, f.medium)

        # Right: fps + episode info
        ep = app.episode_info or {}
        ep_str = f"Episode {ep.get('episode', '—')}   seed {ep.get('seed', '—')}"
        draw_text(surf, ep_str, (rect.right - 200, 10), TEXT_MUTED, f.small, align="left")
        draw_text(surf, f"{app.fps:.0f} fps", (rect.right - 60, 10), TEXT_DIM, f.small)
        schema = ep.get('condition',ep.get('game_schema',''))
        if schema:
            draw_text(surf, schema, (rect.right - 200, 30), TEXT_DIM, f.small)

    # ------------------------------------------------------------------
    # Map area
    # ------------------------------------------------------------------

    def _draw_map_area(self, surf: pygame.Surface, rect: pygame.Rect,
                       app: "SpectatorAppState") -> None:
        # Blit the pre-rendered map background
        map_surf = pygame.Surface((rect.width, rect.height))
        map_surf.blit(self._map_renderer.background, (0, 0))
        m = self._map_renderer

        if app.frame_state:
            fs = app.frame_state
            opts = app.view_options

            # Optional overlays
            if opts.get("tasks"):
                for dest in self._world.destinations:
                    if dest.category == "task":
                        pygame.draw.circle(map_surf, (244, 208, 104),
                                           m.point(dest.position), 4, 1)

            if opts.get("collision"):
                for wall in self._world.data.get("walls", []):
                    m.line(map_surf, wall["points"], (255, 177, 96), 1)

            # Bodies
            if opts.get("bodies", True):
                for body in fs.bodies.values():
                    if body.reported:
                        continue
                    color_name = "red"
                    # Try to get victim colour from players
                    if body.victim_id in fs.players:
                        color_name = fs.players[body.victim_id].color
                    height = max(28, round(m.scale * 0.85))
                    sprite = character_image(color_name, height, "right", 0, body=True)
                    if sprite is None:
                        sprite = character_sprite(player_color(color_name), height, body=True)
                    point = m.point(body.position)
                    map_surf.blit(sprite, sprite.get_rect(midbottom=(point[0], point[1] + 3)))
                    draw_text(map_surf, "BODY", (point[0] - 16, point[1] + 7),
                              (255, 169, 172), self.fonts.tiny)

            # Players
            players_sorted = sorted(fs.players.values(), key=lambda p: (-p.position[1], p.player_id))
            labels: list[pygame.Rect] = []
            for player in players_sorted:
                ghost=player.ghost_position is not None and player.status!='alive'
                if player.status not in ("alive", "ejected") and not ghost:
                    continue
                color_rgb = player_color(player.color)
                point = m.point(player.ghost_position if ghost else player.position)

                # Route overlay
                if opts.get("routes"):
                    pass  # Could draw navigator route if available

                # Sight range
                if opts.get("visibility"):
                    sight = getattr(self._world, "sight_range", None) or 4.5
                    pygame.draw.circle(map_surf, color_rgb, point,
                                       round(sight * m.scale), 1)

                # Character sprite
                vx, vy = player.velocity
                moving = math.hypot(vx, vy) > 1e-5
                direction = ("up" if vy > 0 else "down") if abs(vy) > abs(vx) else \
                            ("left" if vx < 0 else "right")
                height = max(28, round(m.scale * 0.97))
                frame = int(self._anim_time * 8) % 4 if moving else 0
                alpha = 80 if player.status == "ejected" or ghost else 255
                draw_character(map_surf, player.color, point, height, direction, frame,
                               body=False, alpha=alpha)

                # Task progress bar
                if player.interaction_task:
                    bar_rect = pygame.Rect(point[0] - 17, point[1] + 8, 36, 5)
                    if player.task_progress is not None:
                        draw_progress_bar(map_surf, bar_rect, player.task_progress, ACCENT_TEAL)
                    else:
                        draw_text(map_surf,'working',(point[0]-17,point[1]+8),ACCENT_TEAL,self.fonts.tiny)

                # Label
                if opts.get("labels", True):
                    lbl_surf = self.fonts.small.render(player.player_id+(' (auto ghost)' if ghost else ''), True, TEXT_PRIMARY)
                    lbl_rect = lbl_surf.get_rect(topleft=(point[0] + 11, point[1] - 20)).inflate(6, 2)
                    while any(lbl_rect.colliderect(prior) for prior in labels):
                        lbl_rect.y -= 16
                    labels.append(lbl_rect)
                    pygame.draw.rect(map_surf, (10, 18, 30), lbl_rect, border_radius=3)
                    map_surf.blit(lbl_surf, (lbl_rect.x + 3, lbl_rect.y + 1))

                # Role overlay
                if opts.get("roles") and player.role:
                    role_color = ACCENT_RED if player.role == "impostor" else ACCENT_GREEN
                    pygame.draw.circle(map_surf, role_color, (point[0], point[1] - height - 4), 4)

                # Highlight selected player
                if app.selected_player == player.player_id:
                    pygame.draw.circle(map_surf, ACCENT_AMBER, point,
                                       max(18, round(m.scale * 0.5)), 2)

            # Meeting overlay
            if fs.phase in ("discussion", "voting"):
                overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                overlay.fill((5, 10, 20, 160))
                map_surf.blit(overlay, (0, 0))
                self._draw_meeting_panel(map_surf, rect, fs)

            # Result overlay
            if fs.result:
                self._draw_result_overlay(map_surf, rect, fs.result)

        surf.blit(map_surf, rect.topleft)
        pygame.draw.rect(surf, BG_LINE, rect, 1)

    def _draw_meeting_panel(self, surf: pygame.Surface, rect: pygame.Rect,
                             fs: FrameState) -> None:
        panel = pygame.Rect(rect.width // 2 - 300, 60, 600, 340)
        draw_rect_aa(surf, BG_PANEL, panel, radius=12, alpha=240)
        pygame.draw.rect(surf, ACCENT_BLUE, panel, 1, border_radius=12)

        draw_text(surf, f"MEETING  —  {fs.phase.upper()}", (panel.x + 24, panel.y + 16),
                  ACCENT_BLUE, self.fonts.large)

        # Participants
        from spectator.meeting import current_meeting
        meeting=current_meeting(fs)
        participants=meeting['participants']
        y = panel.y + 56
        x = panel.x + 24
        for pid in participants:
            if pid in fs.players:
                draw_character(surf, fs.players[pid].color, (x + 20, y + 36), 40)
                draw_text(surf, pid, (x, y + 44), label_color(fs.players[pid].color), self.fonts.tiny)
                x += 90
                if x > panel.right - 90:
                    break

        # Recent claims and votes
        claims=meeting['claims']
        votes=meeting['votes']

        y = panel.y + 120
        draw_text(surf, "CLAIMS", (panel.x + 24, y), ACCENT_TEAL, self.fonts.small)
        y += 20
        for c in claims[-4:]:
            speaker = c.get("speaker_id", c.get("player_id", "?"))
            kind_str = c.get("claim_kind", c.get("kind", "")).replace("_", " ")
            subject = c.get("subject_id", "")
            msg = f"{speaker}: {kind_str} {subject}"
            draw_text(surf, msg, (panel.x + 24, y), TEXT_MUTED, self.fonts.small,
                      max_width=panel.width - 48)
            y += 18

        draw_text(surf, "VOTES", (panel.x + 24, y + 8), ACCENT_TEAL, self.fonts.small)
        y += 28
        for v in votes[-4:]:
            voter = v.get("voter_id", "?")
            target = (v.get("target_id") or "SKIP") if v['published'] else 'SUBMITTED (HIDDEN)'
            draw_text(surf, f"{voter}  →  {target}", (panel.x + 24, y), TEXT_PRIMARY, self.fonts.small)
            y += 18

    def _draw_result_overlay(self, surf: pygame.Surface, rect: pygame.Rect,
                              result: dict) -> None:
        winner = result.get("winner")
        reason = result.get("reason", "").replace("_", " ")
        panel = pygame.Rect(rect.width // 2 - 260, rect.height // 2 - 70, 520, 140)
        bg_color = (10, 30, 18) if winner == "crewmate" else (30, 10, 10) if winner == "impostor" else BG_PANEL
        draw_rect_aa(surf, bg_color, panel, radius=14, alpha=240)
        accent = ACCENT_GREEN if winner == "crewmate" else ACCENT_RED if winner == "impostor" else ACCENT_AMBER
        pygame.draw.rect(surf, accent, panel, 2, border_radius=14)
        title = f"{winner.upper() if winner else 'DRAW'} VICTORY" if winner else "DRAW  /  TIME LIMIT"
        draw_text(surf, title, (panel.centerx, panel.y + 22), accent, self.fonts.title, align="center")
        draw_text(surf, reason.upper(), (panel.centerx, panel.y + 64), TEXT_PRIMARY,
                  self.fonts.medium, align="center")
        draw_text(surf, "SPACE to replay  |  N for next episode", (panel.centerx, panel.y + 96),
                  TEXT_MUTED, self.fonts.small, align="center")

    # ------------------------------------------------------------------
    # Left panel — episode library
    # ------------------------------------------------------------------

    def _draw_left_panel(self, surf: pygame.Surface, rect: pygame.Rect,
                         app: "SpectatorAppState") -> None:
        pygame.draw.rect(surf, BG_PANEL, rect)
        pygame.draw.line(surf, BG_LINE, rect.topright, rect.bottomright)

        y = rect.y + 12
        x = rect.x + 12
        w = rect.width - 24

        draw_text(surf, "REPLAY LIBRARY", (x, y), ACCENT_TEAL, self.fonts.medium)
        y += 26
        draw_separator(surf, y, x, x + w)
        y += 8

        # Watch directory
        watch_dir = app.watch_dir or "No directory set"
        draw_text(surf, "WATCH DIR", (x, y), TEXT_DIM, self.fonts.tiny)
        y += 14
        draw_text(surf, watch_dir, (x, y), TEXT_MUTED, self.fonts.tiny, max_width=w)
        y += 20

        # Mode indicator
        mode_label = {"follow_latest": "● FOLLOW LATEST", "auto_next": "▶ AUTO-NEXT", "manual": "◼ MANUAL"}.get(app.playback_mode, "—")
        mode_color = {
            "follow_latest": ACCENT_GREEN,
            "auto_next": ACCENT_TEAL,
            "manual": TEXT_MUTED,
        }.get(app.playback_mode, TEXT_DIM)
        draw_text(surf, mode_label, (x, y), mode_color, self.fonts.small)
        y += 22

        draw_separator(surf, y, x, x + w)
        y += 8

        # Episode list
        metrics=app.training_metrics or {}
        if metrics:
            draw_text(surf,str(metrics.get('condition','PPO training')),(x,y),ACCENT_PURPLE,self.fonts.small,max_width=w)
            y+=18
            from spectator.progress import progress_lines
            for line in progress_lines(metrics):
                draw_text(surf,line,(x,y),TEXT_PRIMARY,self.fonts.tiny,max_width=w)
                y+=16
            draw_text(surf,f"{metrics.get('stage','')}  elapsed {metrics.get('elapsed_seconds',0)/60:.1f} min",(x,y),TEXT_MUTED,self.fonts.tiny)
            y+=22
            d=metrics.get('diagnostics') or {}
            if d:
                draw_text(surf,f"loss {d.get('loss',0):.3f}   KL {d.get('approximate_kl',0):.5f}",(x,y),TEXT_MUTED,self.fonts.tiny)
                y+=18
        ep=app.episode_info or {}
        if ep.get('stream') is not None:
            draw_text(surf,f"Worker {ep.get('worker')} / stream {ep['stream']}",(x,y),ACCENT_TEAL,self.fonts.tiny)
            y+=20
        entries = app.library_entries or []
        visible_entries = entries[-20:]  # most recent 20
        for i, entry in enumerate(visible_entries):
            ep_num = entry.get("header", {}).get("episode", i)
            seed = entry.get("header", {}).get("scenario_seed", "?")
            complete = entry.get("complete", False)
            tags = entry.get("tags", [])
            result = entry.get("result") or {}
            winner = result.get("winner", "")

            is_current = entry.get("path") == app.current_replay_path
            bg = BG_HOVER if is_current else BG_CARD

            card_rect = pygame.Rect(x, y, w, 44)
            app._library_hitboxes.append((card_rect,str(entry['path'])))
            draw_rect_aa(surf, bg, card_rect, radius=5)
            if is_current:
                pygame.draw.rect(surf, ACCENT_TEAL, card_rect, 1, border_radius=5)

            icon_color = ACCENT_GREEN if winner == "crewmate" else ACCENT_RED if winner == "impostor" else TEXT_DIM
            draw_text(surf, "●", (x + 6, y + 6), icon_color, self.fonts.small)

            condition=entry.get('header',{}).get('config',{}).get('training_condition','Replay')
            short='A Original' if condition.startswith('Model A') else 'B Shaped' if condition.startswith('Model B') else 'Replay'
            draw_text(surf, f"{short} / ep {ep_num}", (x + 20, y + 4), TEXT_PRIMARY, self.fonts.tiny)
            draw_text(surf, f"seed {seed}", (x + 20, y + 22), TEXT_MUTED, self.fonts.tiny)

            status_str = "✓ complete" if complete else "⟳ recording…"
            status_color = TEXT_MUTED if complete else ACCENT_AMBER
            draw_text(surf, status_str, (x + 120, y + 4), status_color, self.fonts.tiny)

            if tags:
                tag_str = tags[0].replace("_", " ")[:16]
                draw_text(surf, f"[{tag_str}]", (x + 120, y + 22), ACCENT_PURPLE, self.fonts.tiny)

            if entry.get("favorite"):
                draw_text(surf, "★", (x + w - 14, y + 14), ACCENT_AMBER, self.fonts.small)

            y += 50
            if y > rect.bottom - 60:
                break

        # Help at bottom
        y = rect.bottom - 46
        draw_separator(surf, y, x, x + w)
        y += 8
        draw_text(surf, "F  browse dir   M  change mode", (x, y), TEXT_DIM, self.fonts.tiny)
        y += 16
        draw_text(surf, "★  favourite   DEL  clear library", (x, y), TEXT_DIM, self.fonts.tiny)

    # ------------------------------------------------------------------
    # Right panel — player inspection + neural
    # ------------------------------------------------------------------

    def _draw_right_panel(self, surf: pygame.Surface, rect: pygame.Rect,
                          app: "SpectatorAppState") -> None:
        pygame.draw.rect(surf, BG_PANEL, rect)
        pygame.draw.line(surf, BG_LINE, rect.topleft, rect.bottomleft)

        y = rect.y + 12
        x = rect.x + 12
        w = rect.width - 24

        # Player list + selection
        draw_text(surf, "PLAYERS", (x, y), ACCENT_TEAL, self.fonts.medium)
        y += 26

        fs = app.frame_state
        if not fs:
            draw_text(surf, "No replay loaded", (x, y + 20), TEXT_MUTED, self.fonts.small)
            return

        # Mini player list
        for pid, player in fs.players.items():
            is_sel = app.selected_player == pid
            bg = BG_HOVER if is_sel else BG_CARD
            row = pygame.Rect(x, y, w, 30)
            app._player_hitboxes[pid]=row
            draw_rect_aa(surf, bg, row, radius=4)
            if is_sel:
                pygame.draw.rect(surf, ACCENT_TEAL, row, 1, border_radius=4)

            draw_character(surf, player.color, (x + 18, y + 27), 28)
            lbl_c = label_color(player.color)
            draw_text(surf, pid, (x + 34, y + 5), lbl_c, self.fonts.small)
            status_c = ACCENT_GREEN if player.status == "alive" else ACCENT_RED if player.status == "dead" else TEXT_DIM
            draw_text(surf, player.status.upper(), (x + w - 55, y + 9), status_c, self.fonts.tiny)
            y += 34

        draw_separator(surf, y + 4, x, x + w)
        y += 12

        # Selected player detail
        sel = app.selected_player
        if sel and sel in fs.players:
            player = fs.players[sel]
            y = self._draw_player_detail(surf, x, y, w, player, fs, app)
        else:
            draw_text(surf, "Click a player to inspect", (x, y), TEXT_MUTED, self.fonts.small)
            y += 30

        # Neural decision panel
        draw_separator(surf, y + 4, x, x + w)
        y += 12
        y = self._draw_neural_panel(surf, x, y, w, app)

    def _draw_player_detail(self, surf, x, y, w, player, fs: FrameState,
                             app: "SpectatorAppState") -> int:
        draw_text(surf, "PLAYER DETAIL", (x, y), ACCENT_TEAL, self.fonts.small)
        y += 20

        rows = [
            ("Color",    player.color),
            ("Role",     player.role.upper() if app.view_options.get("roles") else "hidden"),
            ("Position", f"{player.position[0]:.2f}, {player.position[1]:.2f}"),
            ("Room",     self._world.region(player.position) if player.status == "alive" else "—"),
            ("Velocity", f"({player.velocity[0]:.2f}, {player.velocity[1]:.2f})"),
            ("Intention", player.intention[:22]),
            ("Cooldown",  f"{player.cooldown:.1f}s" if player.cooldown > 0.01 else "—"),
            ("Meetings",  str(player.meetings_remaining)),
            ("Own tasks", f"{player.tasks_completed}/{player.tasks_total}" if player.tasks_total is not None else 'unavailable'),
        ]
        for label, value in rows:
            draw_text(surf, label, (x, y), TEXT_DIM, self.fonts.tiny)
            draw_text(surf, str(value), (x + 88, y), TEXT_PRIMARY, self.fonts.tiny)
            y += 16

        # Controller type
        header = (app.episode_info or {})
        ctrl = header.get("controller_types", {}).get(player.player_id, "unavailable")
        ctrl_color = ACCENT_PURPLE if ctrl == "ppo" else TEXT_MUTED
        from spectator.progress import controller_label
        control=controller_label(ctrl,fs.sim_time,header.get('config',{}),ghost=player.ghost_position is not None)
        draw_pill(surf, control, (x, y), ctrl_color)
        y += 22

        # Recent events for this player
        recent = [e for e in reversed(fs.events_up_to_now)
                  if e.get("player_id") == player.player_id or
                  e.get("killer_id") == player.player_id or
                  e.get("reporter_id") == player.player_id or
                  e.get("voter_id") == player.player_id][:2]
        if recent:
            y += 4
            draw_text(surf, "RECENT EVENTS", (x, y), TEXT_DIM, self.fonts.tiny)
            y += 14
            for ev in reversed(recent):
                kind = ev.get("kind", "event").replace("_", " ")[:20]
                t = ev.get("time", ev.get("sim_time", 0))
                draw_text(surf, f"{t:5.1f}  {kind}", (x, y), TEXT_MUTED, self.fonts.tiny,
                          max_width=w)
                y += 14

        return y + 6

    def _draw_neural_panel(self, surf, x, y, w, app: "SpectatorAppState") -> int:
        if not app.view_options.get('private_debug',False):
            draw_text(surf,'D: selected actor private debug',(x,y),TEXT_DIM,self.fonts.small)
            return y+24
        draw_text(surf, "NEURAL DECISIONS", (x, y), ACCENT_TEAL, self.fonts.medium)
        y += 24

        fs = app.frame_state
        if not fs:
            return y

        sel = app.selected_player
        neural = None
        if sel and sel in fs.latest_neural:
            neural = fs.latest_neural[sel]

        if neural is None:
            draw_text(surf, "No neural data recorded", (x, y), TEXT_DIM, self.fonts.small)
            y += 22
            draw_text(surf, "This actor/recording has no", (x, y), TEXT_DIM, self.fonts.tiny)
            y += 16
            draw_text(surf, "available neural trace.", (x, y), TEXT_DIM, self.fonts.tiny)
            y += 24
            # Indicate where neural data would appear
            draw_text(surf, "When PPO training data is recorded,", (x, y), TEXT_DIM, self.fonts.tiny)
            y += 14
            draw_text(surf, "action probabilities and value", (x, y), TEXT_DIM, self.fonts.tiny)
            y += 14
            draw_text(surf, "estimates will appear here.", (x, y), TEXT_DIM, self.fonts.tiny)
            return y + 20

        # Action chosen
        from spectator.progress import update_label
        draw_text(surf,f"Decision t={neural.get('sim_time',0):.1f}s / {update_label(neural.get('policy_update','?'))}",(x,y),TEXT_DIM,self.fonts.tiny)
        y+=16
        chosen = neural.get("action_kind", "?")
        draw_text(surf, "CHOSEN ACTION", (x, y), TEXT_DIM, self.fonts.tiny)
        y += 14
        draw_rect_aa(surf, BG_CARD, pygame.Rect(x, y, w, 22), radius=4)
        draw_text(surf, chosen.upper().replace("_", " "), (x + 8, y + 4),
                  ACCENT_TEAL, self.fonts.small)
        target = neural.get("action_target")
        if target:
            draw_text(surf, f"→ {target}", (x + w - 60, y + 4), ACCENT_AMBER, self.fonts.tiny)
        y += 26

        # Action probabilities
        probs: dict = neural.get("action_probabilities", {})
        if probs:
            draw_text(surf, "ACTION PROBABILITIES (TOP 5)", (x, y), TEXT_DIM, self.fonts.tiny)
            y += 14
            sorted_probs = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            for action, prob in sorted_probs[:5]:
                action_str = str(action).replace("_", " ")[:18]
                is_chosen = str(action) == chosen
                bar_color = PROB_CHOSEN if is_chosen else PROB_OTHER
                bar = pygame.Rect(x, y, w, 14)
                draw_progress_bar(surf, bar, float(prob), bar_color)
                draw_text(surf, action_str, (x + 4, y + 1),
                          TEXT_PRIMARY if is_chosen else TEXT_MUTED, self.fonts.tiny)
                pct = f"{100*float(prob):.0f}%"
                draw_text(surf, pct, (x + w - 28, y + 1), TEXT_PRIMARY, self.fonts.tiny)
                y += 18
            y += 4

        # Value estimate
        suspicion=neural.get('suspicion_probabilities',{})
        draw_text(surf,'PRIVATE SUSPICION — '+str(sel),(x,y),ACCENT_AMBER,self.fonts.tiny)
        y+=16
        if not suspicion:
            draw_text(surf,'Not recorded in this replay',(x,y),TEXT_DIM,self.fonts.tiny);y+=16
        for pid,probability in sorted(suspicion.items()):
            draw_text(surf,f'{pid}: {100*float(probability):.1f}%',(x,y),TEXT_MUTED,self.fonts.small);y+=16
        value = neural.get("value_estimate")
        entropy = neural.get("policy_entropy")
        if value is not None:
            draw_text(surf, "Value estimate", (x, y), TEXT_DIM, self.fonts.tiny)
            draw_text(surf, f"{float(value):+.3f}", (x + 100, y),
                      ACCENT_AMBER if float(value) > 0 else ACCENT_RED, self.fonts.small)
            y += 18
        if entropy is not None:
            draw_text(surf, "Policy entropy", (x, y), TEXT_DIM, self.fonts.tiny)
            draw_text(surf, f"{float(entropy):.4f}", (x + 100, y), TEXT_MUTED, self.fonts.small)
            y += 18

        # Reward
        sel_reward = (fs.latest_reward or {}).get(sel or "")
        if sel_reward:
            draw_separator(surf, y + 2, x, x + w)
            y += 8
            draw_text(surf, "REWARD", (x, y), TEXT_DIM, self.fonts.tiny)
            rew = sel_reward.get("reward", 0)
            cumul = sel_reward.get("cumulative_reward", 0)
            rew_color = ACCENT_GREEN if rew > 0 else (TEXT_DANGER if rew < 0 else TEXT_MUTED)
            draw_text(surf, f"{rew:+.3f}", (x + 60, y), rew_color, self.fonts.small)
            y += 18
            draw_text(surf, "Cumulative", (x, y), TEXT_DIM, self.fonts.tiny)
            cum_color = ACCENT_GREEN if cumul > 0 else (TEXT_DANGER if cumul < 0 else TEXT_MUTED)
            draw_text(surf, f"{cumul:+.2f}", (x + 60, y), cum_color, self.fonts.small)
            y += 20

        return y + 4

    # ------------------------------------------------------------------
    # Timeline bar
    # ------------------------------------------------------------------

    def _draw_timeline(self, surf: pygame.Surface, rect: pygame.Rect,
                       app: "SpectatorAppState") -> None:
        pygame.draw.rect(surf, TL_BG, rect)
        pygame.draw.line(surf, BG_LINE, rect.topleft, rect.topright)

        x0, y0, tw, th = rect.x + 80, rect.y + 22, rect.width - 160, rect.height - 34
        scrub_rect = pygame.Rect(x0, y0, tw, 12)

        # Background track
        pygame.draw.rect(surf, TL_FILL, scrub_rect, border_radius=6)

        fs = app.frame_state
        if not fs:
            draw_text(surf, "No replay", (rect.x + 10, rect.y + 22), TEXT_DIM, self.fonts.small)
            return

        duration = app.player_duration or 1
        progress = min(1.0, fs.sim_time / duration) if duration > 0 else 0

        # Progress fill
        fill_w = max(12, round(tw * progress))
        pygame.draw.rect(surf, TL_PROGRESS, pygame.Rect(x0, y0, fill_w, 12), border_radius=6)

        # Event markers on timeline
        for marker in (app.event_markers or []):
            if duration > 0:
                mx = x0 + round(tw * marker.sim_time / duration)
                if x0 <= mx <= x0 + tw:
                    pygame.draw.line(surf, marker.color, (mx, y0 - 3), (mx, y0 + 15), 2)

        # Playhead thumb
        px = x0 + round(tw * progress)
        pygame.draw.circle(surf, TEXT_PRIMARY, (px, y0 + 6), 7)
        pygame.draw.circle(surf, BG_DEEP, (px, y0 + 6), 5)
        pygame.draw.circle(surf, TL_PROGRESS, (px, y0 + 6), 3)

        # Time labels
        draw_text(surf, f"{fs.sim_time:.1f}s", (x0 - 4, y0 + 16), TEXT_MUTED, self.fonts.tiny, align="right")
        draw_text(surf, f"{duration:.1f}s", (x0 + tw + 4, y0 + 16), TEXT_MUTED, self.fonts.tiny)

        # Store scrub rect for mouse interaction
        app._scrub_rect = scrub_rect

        # Speed controls
        app._speed_hitboxes=[]
        speeds = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        sx = rect.x + 10
        sy = rect.y + 8
        draw_text(surf, "SPEED", (sx, sy), TEXT_DIM, self.fonts.tiny)
        sy += 14
        for sp in speeds:
            is_sel = abs(app.playback_speed - sp) < 0.01
            sp_str = f"{sp:g}×"
            color = ACCENT_TEAL if is_sel else TEXT_DIM
            bg = BG_CARD if is_sel else BG_PANEL
            pill_rect = pygame.Rect(sx, sy, 32, 16)
            app._speed_hitboxes.append((pill_rect.copy(),sp))
            draw_rect_aa(surf, bg, pill_rect, radius=3)
            if is_sel:
                pygame.draw.rect(surf, ACCENT_TEAL, pill_rect, 1, border_radius=3)
            draw_text(surf, sp_str, (sx + 16, sy + 2), color, self.fonts.tiny, align="center")
            sx += 36

        # Help text at far right
        help_x = rect.right - 10
        draw_text(surf, "SPACE play/pause   ←→ step   </> events   N next   R replay", (help_x, rect.y + 8),
                  TEXT_DIM, self.fonts.tiny, align="right")
        draw_text(surf, "D private debug   Tab HUD   1 roles   G routes   V sight   T tasks   L labels   Esc quit",
                  (help_x, rect.y + 24), TEXT_DIM, self.fonts.tiny, align="right")


# ---------------------------------------------------------------------------
# Welcome / no-replay screen
# ---------------------------------------------------------------------------

def draw_welcome_screen(surf: pygame.Surface, fonts: Fonts,
                        window_size: tuple[int, int],
                        watch_dir: str | None = None) -> None:
    surf.fill(BG_DEEP)
    w, h = window_size
    cx = w // 2

    # Title
    draw_text(surf, "SOCIAL DEDUCTION AI", (cx, 80), ACCENT_TEAL, fonts.title, align="center")
    draw_text(surf, "SPECTATOR & REPLAY STUDIO", (cx, 120), TEXT_MUTED, fonts.large, align="center")

    # Status
    y = 200
    if watch_dir:
        draw_text(surf, f"Watching: {watch_dir}", (cx, y), TEXT_MUTED, fonts.normal, align="center")
        y += 30
        draw_text(surf, "Waiting for replay files…", (cx, y), TEXT_DIM, fonts.small, align="center")
    else:
        draw_text(surf, "No replay directory configured.", (cx, y), TEXT_DIM, fonts.normal, align="center")

    y = 300
    options = [
        ("F", "Browse for a replay folder"),
        ("O", "Open a specific replay file"),
        ("D", "Run a demo match (scripted)"),
        ("Esc", "Quit"),
    ]
    for key, label in options:
        key_surf = fonts.medium.render(f"  {key}  ", True, ACCENT_TEAL)
        key_rect = key_surf.get_rect()
        key_rect.right = cx - 10
        key_rect.top = y
        pygame.draw.rect(surf, BG_CARD, key_rect.inflate(0, 4), border_radius=4)
        pygame.draw.rect(surf, ACCENT_TEAL, key_rect.inflate(0, 4), 1, border_radius=4)
        surf.blit(key_surf, key_rect)
        draw_text(surf, label, (cx, y + 4), TEXT_PRIMARY, fonts.normal)
        y += 42

    # Credits
    draw_text(surf, "Social Deduction AI Research Project — Spectator v1.0",
              (cx, h - 30), TEXT_DIM, fonts.small, align="center")
