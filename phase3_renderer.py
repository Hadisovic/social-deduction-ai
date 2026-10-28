"""Spectator-only Pygame view of the actual Phase 3 world.

No renderer object is passed to a controller. The default view hides roles;
the role toggle affects pixels only. An optional local artwork pack supplies
characters/task previews, with original geometry as a portable fallback.
"""
from dataclasses import dataclass, asdict, is_dataclass
from functools import lru_cache
import math
import re

import pygame

from among_us_renderer import MapRenderer
from phase3_assets import character_image, task_image
from social_deduction.actor import Phase


def visibility_outline(game, player, origin=None, rays=120):
    """Spectator-only ray-clipped extent; distance shares the actor rule resolver.

    This is a sampled visual outline, not an observation or an exact visibility
    polygon. Entity/evidence eligibility uses the exact projector LOS test.
    """
    from shapely.geometry import LineString, Point
    from shapely.ops import nearest_points
    origin = origin or player.position
    sight = game.config.effective_sight_range(player.role)
    start = Point(origin)
    points = []
    for i in range(rays):
        angle = 2 * math.pi * i / rays
        target = (origin[0] + sight * math.cos(angle), origin[1] + sight * math.sin(angle))
        hit = game.map.ray_barriers.intersection(LineString((origin, target)))
        points.append(target if hit.is_empty else tuple(nearest_points(start, hit)[1].coords[0]))
    return tuple(points)


COLORS = {
    "red": (231, 70, 83), "blue": (71, 119, 232), "green": (65, 175, 102),
    "pink": (235, 122, 192), "orange": (240, 155, 63), "yellow": (237, 210, 87),
    "black": (79, 91, 109), "white": (224, 230, 238), "purple": (157, 102, 214),
    "brown": (156, 110, 78), "cyan": (75, 209, 217), "lime": (161, 214, 77),
    "maroon": (158, 56, 75), "rose": (244, 177, 194), "banana": (242, 234, 152),
    "gray": (139, 151, 164), "grey": (139, 151, 164), "tan": (202, 176, 132),
    "coral": (241, 132, 126),
}
INK = (13, 23, 36)
TEXT = (226, 237, 249)
MUTED = (139, 163, 184)
ACCENT = (105, 229, 197)


def player_color(name):
    return COLORS.get(str(name).lower(), (137, 179, 203))


def label_color(name):
    return tuple(round(channel * .6 + 255 * .4) for channel in player_color(name))


@lru_cache(maxsize=512)
def character_sprite(color, height=38, direction="right", frame=0, body=False):
    """Original silhouette: separate feet, rounded torso, backpack and visor.

    Feet are the bottom-center anchor. Frames depend only on simulation time,
    never wall-clock time. Render large then downsample for legible small actors.
    Bodies use a half-suit, separate boots and a white bone silhouette.
    """
    surf = pygame.Surface((80, 100), pygame.SRCALPHA)
    shade = tuple(round(channel * .61) for channel in color)
    light = tuple(min(255, round(channel * .84 + 39)) for channel in color)
    pygame.draw.ellipse(surf, (4, 9, 15, 135), (9, 83, 60, 13))
    if body:
        # Lower suit remains standing at the death position. No visor/head:
        # the former collapsed-visor oval was indistinguishable from a marker.
        pygame.draw.rect(surf, INK, (9, 54, 19, 28), border_radius=6)
        pygame.draw.rect(surf, shade, (13, 58, 12, 21), border_radius=4)
        for x in (24, 47):
            pygame.draw.rect(surf, INK, (x, 73, 21, 20), border_radius=6)
            pygame.draw.rect(surf, shade, (x+4, 75, 13, 14), border_radius=4)
        pygame.draw.rect(surf, INK, (19, 46, 51, 38), border_radius=11)
        pygame.draw.rect(surf, color, (23, 50, 43, 29), border_radius=8)
        pygame.draw.ellipse(surf, INK, (19, 42, 51, 19))
        pygame.draw.ellipse(surf, shade, (24, 47, 41, 10))
        pygame.draw.rect(surf, INK, (36, 28, 17, 25), border_radius=5)
        pygame.draw.circle(surf, INK, (37, 27), 10)
        pygame.draw.circle(surf, INK, (52, 27), 10)
        pygame.draw.rect(surf, (246,242,221), (40, 28, 9, 23), border_radius=3)
        pygame.draw.circle(surf, (246,242,221), (37, 27), 6)
        pygame.draw.circle(surf, (246,242,221), (52, 27), 6)
    else:
        stride = (0, 4, 0, -4)[frame % 4]
        # The side pack is behind the body, not an extra collision shape.
        pygame.draw.rect(surf, INK, (7, 35, 23, 42), border_radius=9)
        pygame.draw.rect(surf, shade, (11, 39, 17, 34), border_radius=6)
        for x, offset in ((25, stride), (46, -stride)):
            pygame.draw.rect(surf, INK, (x - 3, 68 + offset, 22, 25), border_radius=6)
            pygame.draw.rect(surf, shade, (x + 1, 68 + offset, 14, 20), border_radius=4)
        pygame.draw.rect(surf, INK, (19, 9, 50, 70), border_radius=23)
        pygame.draw.rect(surf, shade, (23, 13, 42, 62), border_radius=20)
        pygame.draw.ellipse(surf, color, (23, 12, 40, 53))
        pygame.draw.ellipse(surf, light, (30, 16, 24, 12))
        if direction == "up":
            pygame.draw.rect(surf, INK, (25, 36, 36, 31), border_radius=10)
            pygame.draw.rect(surf, shade, (29, 40, 28, 23), border_radius=7)
        else:
            visor_x = 29 if direction == "down" else 32
            pygame.draw.ellipse(surf, INK, (visor_x - 3, 24, 42, 29))
            pygame.draw.ellipse(surf, (100, 164, 187), (visor_x, 28, 35, 21))
            pygame.draw.ellipse(surf, (169, 222, 235), (visor_x + 2, 28, 30, 14))
            pygame.draw.line(surf, (236, 252, 255), (visor_x + 7, 31), (visor_x + 23, 31), 3)
    if direction == "left":
        surf = pygame.transform.flip(surf, True, False)
    if body:
        # Match the local artwork path's visible-height convention. Cropping
        # removes empty head space; body size is not a tiny fraction of a canvas.
        surf = surf.subsurface(surf.get_bounding_rect()).copy()
        target_height = max(18, round(height * .65))
        return pygame.transform.smoothscale(surf, (round(surf.get_width()*target_height/surf.get_height()), target_height))
    return pygame.transform.smoothscale(surf, (max(12, round(height * .8)), height))


@dataclass
class ViewOptions:
    hud: bool = True
    roles: bool = False
    routes: bool = False
    visibility: bool = False
    tasks: bool = False
    labels: bool = True
    collision: bool = False
    events: bool = True
    bodies: bool = True


def _value(value):
    return getattr(value, "value", value)


class Phase3Renderer:
    """Read-only spectator drawing. Owns no game randomness or simulation state."""

    def __init__(self, world, size=(1440, 900)):
        self.size = size
        self.map_width = size[0] - 352
        self.map_renderer = MapRenderer(world, (self.map_width, size[1]))
        self.world = world
        self.destinations = {d.id: d for d in world.destinations}
        pygame.font.init()
        self.small = pygame.font.SysFont("segoeui", 13)
        self.font = pygame.font.SysFont("segoeui", 16)
        self.title = pygame.font.SysFont("segoeui", 24, bold=True)
        self.large = pygame.font.SysFont("segoeui", 31, bold=True)

    def text(self, surf, text, pos, color=TEXT, font=None, max_width=None):
        font = font or self.font
        text = str(text)
        if max_width is not None:
            while text and font.size(text)[0] > max_width:
                text = text[:-2]
                if font.size(text + "...")[0] <= max_width:
                    text += "..."
                    break
        surf.blit(font.render(text, True, color), pos)

    def wrap(self, surf, text, pos, width, color=TEXT, font=None, line_height=20):
        font = font or self.small
        lines = []
        line = ""
        for word in str(text).split():
            if line and font.size(line + " " + word)[0] > width:
                lines.append(line)
                line = word
            else:
                line = (line + " " + word).strip()
        lines.append(line)
        for index, line in enumerate(lines):
            self.text(surf, line, (pos[0], pos[1] + index * line_height), color, font)
        return len(lines) * line_height

    def actor(self, surf, player, now, position=None, height=None, velocity=None):
        pos = self.map_renderer.point(player.position) if position is None else position
        vx, vy = player.velocity if velocity is None else velocity
        moving = math.hypot(vx, vy) > 1e-5
        direction = ("up" if vy > 0 else "down") if abs(vy) > abs(vx) else ("left" if vx < 0 else "right")
        height = height or max(32, round(self.map_renderer.scale * .97))
        frame = int(now * 8) % 4 if moving else 0
        body = _value(player.status) == "dead"
        sprite = character_image(player.color_name, height, direction, frame, body)
        if sprite is None:
            sprite = character_sprite(player_color(player.color_name), height, direction, frame, body)
        if _value(player.status) == "ejected":
            sprite = sprite.copy()
            sprite.set_alpha(80)
        surf.blit(sprite, sprite.get_rect(midbottom=(round(pos[0]), round(pos[1] + height * .07))))

    def draw(self, game, options=None, paused=False, speed=1., fps=0., visual_state=None, animation_time=None):
        options = options or ViewOptions()
        surf = pygame.Surface(self.size)
        surf.fill((8, 13, 22))
        surf.blit(self.map_renderer.background, (0, 0))
        m = self.map_renderer
        if options.tasks:
            for destination in self.world.destinations:
                if destination.category == "task":
                    pygame.draw.circle(surf, (244, 208, 104), m.point(destination.position), 4, 1)
        if options.collision:
            for wall in self.world.data["walls"]:
                m.line(surf, wall["points"], (255, 177, 96), 1)
            for obstacle in self.world.data["obstacles"]:
                m.line(surf, obstacle["points"], (245, 112, 143), 1, obstacle["closed"])
        if options.bodies:
            for body in game.bodies.values():
                if body.reported:
                    continue
                victim = game.players[body.victim_id]
                height = max(32, round(m.scale))
                sprite = character_image(victim.color_name, height, 'right', 0, body=True)
                if sprite is None:
                    sprite = character_sprite(player_color(victim.color_name), height, body=True)
                point = m.point(body.position)
                surf.blit(sprite, sprite.get_rect(midbottom=(point[0], point[1] + 3)))
                # The corpse is the marker. A surrounding circle belongs only
                # to the explicitly enabled collision/debug overlay.
                if options.collision:
                    pygame.draw.circle(surf, (245, 124, 129), point, max(10, round(m.scale * .4)), 1)
                self.text(surf, "BODY", (point[0] - 16, point[1] + 7), (255, 169, 172), self.small)
        players = sorted(game.players.values(), key=lambda p: (-p.position[1], p.player_id))
        labels = []
        for player in players:
            if _value(player.status) != "alive":
                continue
            color = player_color(player.color_name)
            visual = (visual_state or {}).get(player.player_id)
            point = m.point(visual[0] if visual else player.position)
            if options.routes and player.navigator.route:
                m.line(surf, player.navigator.route, color, 2)
            if options.visibility:
                if getattr(game.config, 'rules_version', 1) == 1:
                    pygame.draw.circle(surf, color, point, round(game.config.sight_range * m.scale), 1)
                elif game.phase is Phase.ROAMING:
                    m.line(surf, visibility_outline(game, player, visual[0] if visual else None), color, 1, True)
            self.actor(surf, player, game.time if animation_time is None else animation_time,
                       position=point, velocity=visual[1] if visual else None)
            if options.labels:
                rect = self.small.render(player.player_id, True, TEXT).get_rect(topleft=(point[0] + 11, point[1] - 20)).inflate(6, 2)
                while any(rect.colliderect(prior) for prior in labels):
                    rect.y -= 17
                labels.append(rect)
                pygame.draw.rect(surf, (12, 20, 33), rect, border_radius=3)
                self.text(surf, player.player_id, (rect.x + 3, rect.y + 1), label_color(player.color_name), self.small)
            if player.interaction_task:
                task = next((t for t in player.tasks if t.task_id == player.interaction_task), None)
                if task:
                    progress = min(1., max(0., task.progress))
                    pygame.draw.rect(surf, INK, (point[0] - 17, point[1] + 8, 36, 6), border_radius=2)
                    pygame.draw.rect(surf, ACCENT, (point[0] - 16, point[1] + 9, round(34 * progress), 4), border_radius=2)
            if options.collision:
                pygame.draw.circle(surf, TEXT, point, round(self.world.radius * m.scale), 1)
        self.header(surf, game, paused, speed, fps)
        if options.visibility and getattr(game.config, 'rules_version', 1) == 6:
            self.text(surf, 'SPECTATOR VISION / WALL-CLIPPED / NOT POLICY INPUT', (22, 75), MUTED, self.small)
        if options.hud:
            self.sidebar(surf, game, options)
        else:
            self.text(surf, "TAB restores spectator HUD", (self.map_width + 20, 100), MUTED)
        if game.phase in (Phase.DISCUSSION, Phase.VOTING):
            self.meeting(surf, game)
        if game.result is not None:
            self.result(surf, game)
        self.footer(surf, options)
        return surf

    def header(self, surf, game, paused, speed, fps):
        pygame.draw.rect(surf, (12, 20, 33), (0, 0, self.size[0], 69))
        self.text(surf, "THE SKELD", (22, 12), font=self.title)
        self.text(surf, "SCRIPTED SOCIAL DEDUCTION  /  PHASE 3" if getattr(game.config, 'rules_version', 1) == 1
                  else "PHASE 6 RULES / SCRIPTED PREVIEW", (23, 43), MUTED, self.small)
        phase = str(_value(game.phase)).upper()
        self.text(surf, f"{phase}    {'PAUSED' if paused else 'RUNNING'}", (350, 14), ACCENT)
        self.text(surf, f"Seed {game.seed}   |   {game.time:6.1f}s   |   {speed:g}x   |   {fps:.0f} fps", (350, 41), MUTED, self.small)

    def sidebar(self, surf, game, options):
        x = self.map_width
        pygame.draw.rect(surf, (15, 25, 40), (x, 69, 352, self.size[1] - 130))
        self.text(surf, "SPECTATOR", (x + 20, 84), ACCENT)
        self.text(surf, "PRIVILEGED ROLES ON" if options.roles else "Roles hidden  /  1 to reveal", (x + 20, 108), (245, 166, 116) if options.roles else MUTED, self.small)
        for index, player in enumerate(game.players.values()):
            y = 138 + index * 86
            pygame.draw.rect(surf, (22, 36, 53), (x + 12, y, 328, 78), border_radius=7)
            color = label_color(player.color_name)
            self.actor(surf, player, game.time, position=(x + 38, y + 62), height=43)
            status = str(_value(player.status)).upper()
            self.text(surf, f"{player.color_name} / {player.player_id}", (x + 64, y + 7), color)
            self.text(surf, str(_value(player.role)).upper() if options.roles else status, (x + 221, y + 9), MUTED, self.small, 103)
            self.text(surf, f"{status.lower()}  |  {self.world.region(player.position)}", (x + 64, y + 31), MUTED, self.small, 255)
            task = next((t for t in player.tasks if t.task_id == player.interaction_task), None)
            intention = f"Task in progress: {task.progress:.0%}" if task else self.intention_text(player.intention)
            self.text(surf, intention or "Choosing intention", (x + 64, y + 53), TEXT, self.small, 255)
        y = 584
        active = [(player, task) for player in game.players.values() for task in player.tasks
                  if task.task_id == player.interaction_task]
        if active:
            player, task = active[0]
            destination = next(d for d in self.world.destinations if d.id == task.console_id)
            preview = task_image(destination.name)
            self.text(surf, 'TASK IN PROGRESS  /  SPECTATOR', (x + 20, y), ACCENT, self.small)
            left = x + 20
            if preview is not None:
                surf.blit(preview, (left, y + 27))
                left += 122
            self.wrap(surf, self.readable(destination.name), (left, y + 30), x + 329 - left, TEXT)
            self.text(surf, f'{player.color_name}  |  {task.progress:.0%}', (left, y + 76), MUTED, self.small)
            self.text(surf, 'Timed interaction', (left, y + 98), MUTED, self.small)
            y += 146
        self.text(surf, "RECENT EVENTS  /  SPECTATOR", (x + 20, y), ACCENT, self.small)
        if options.events:
            events = [event for event in game.event_log if str(event.get("kind", event.get("type", ""))) not in ("sighting", "direct_sighting", "navigation", "navigate", "body_observation")]
            for event in events[-(3 if active else 8):]:
                y += 26
                self.text(surf, self.event_text(event), (x + 20, y), MUTED, self.small, 312)

    def event_text(self, event):
        kind = event.get("kind", event.get("type", "event"))
        time = event.get("time", event.get("simulation_time", 0))
        detail = event.get("payload", event.get("data", event))
        if not isinstance(detail, dict):
            detail = asdict(detail) if is_dataclass(detail) else {}
        identity = detail.get("player_id", detail.get("voter_id", detail.get("reporter_id", event.get("player_id", ""))))
        return f"{time:5.1f}  {str(kind).replace('_', ' ')} {identity}"

    @staticmethod
    def readable(name):
        return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name)

    def intention_text(self, intention):
        value = str(intention or 'wait')
        destination = self.destinations.get(value)
        if destination:
            return 'To ' + self.readable(destination.room) + ' / ' + self.readable(destination.name)
        if value.startswith('task:'):
            destination = self.destinations.get(value[5:])
            if destination:
                return 'Working: ' + self.readable(destination.name)
        if value.startswith('room:'):
            return 'To ' + self.readable(value[5:])
        if value.startswith('body:'):
            return 'Approaching a body'
        return {'wait': 'Waiting', 'location': 'Moving to observed position'}.get(value, value.replace('_', ' '))

    def meeting(self, surf, game):
        overlay = pygame.Surface((self.map_width, self.size[1] - 130), pygame.SRCALPHA)
        overlay.fill((7, 14, 24, 172))
        surf.blit(overlay, (0, 69))
        panel = pygame.Rect(42, 132, self.map_width - 84, 598)
        pygame.draw.rect(surf, (19, 33, 50), panel, border_radius=14)
        pygame.draw.rect(surf, (61, 99, 120), panel, 1, border_radius=14)
        self.text(surf, "MEETING  /  " + str(_value(game.phase)).upper(), (66, 151), ACCENT, self.title)
        self.text(surf, self.meeting_origin(game), (66, 187), TEXT, self.font)
        self.text(surf, "Claims are statements by speakers, not verified facts.", (66, 213), MUTED, self.small)
        for index, player_id in enumerate(game.context.participants):
            player = game.players[player_id]
            x = 106 + index * 155
            self.actor(surf, player, game.time, position=(x, 280), height=56)
            self.text(surf, player.color_name, (x - 28, 293), label_color(player.color_name), self.small)
        self.text(surf, "PUBLIC CLAIMS", (68, 340), ACCENT, self.small)
        self.text(surf, "PUBLIC VOTES", (self.map_width - 315, 340), ACCENT, self.small)
        publications = getattr(game, "publications", ())
        meeting_id = game.context.meeting_id
        meeting_tick = max((event.tick for event in publications
                            if getattr(event.payload, "meeting_id", None) == meeting_id
                            and hasattr(event.payload, "participants")), default=0)
        claims, votes = [], []
        for event in publications:
            data = asdict(event) if is_dataclass(event) else dict(event) if isinstance(event, dict) else {}
            payload = data.get("payload", data)
            kind = str(_value(data.get("kind", data.get("type", "")))).lower()
            if not isinstance(payload, dict):
                continue
            if data.get("tick", 0) < meeting_tick:
                continue
            if payload.get("meeting_id", meeting_id) != meeting_id:
                continue
            if "claim" in kind or "speaker_id" in payload:
                claims.append(payload)
            elif "vote" in kind or "voter_id" in payload:
                votes.append(payload)
        y = 372
        for claim in claims[-8:]:
            speaker = self.identity(game, claim.get("speaker_id", "?"))
            subject = self.identity(game, claim.get("subject_id", claim.get("subject", "")))
            kind = str(_value(claim.get("claim_type", claim.get("kind", "saw_player")))).replace("_", " ")
            room = claim.get("region", claim.get("room", claim.get("location", "")))
            tick = claim.get("claimed_tick", claim.get("asserted_tick", claim.get("asserted_time", "?")))
            asserted = f"{tick * game.config.dt:.1f}s" if isinstance(tick, (int, float)) else str(tick)
            sentence = f"{speaker} claims: {kind} {subject} {room} (at {asserted})"
            y += self.wrap(surf, sentence, (68, y), self.map_width - 440, line_height=18) + 10
            if y > 664:
                break
        if not claims:
            self.text(surf, "Waiting for players to share observations...", (68, y), MUTED, self.small)
        y = 372
        for vote in votes[-5:]:
            voter = self.identity(game, vote.get("voter_id", "?"))
            target = self.identity(game, vote.get("target_id")) if vote.get("target_id") is not None else "SKIP"
            self.text(surf, f"{voter}  >  {target}", (self.map_width - 315, y), TEXT, self.small)
            y += 30
        if not votes:
            self.text(surf, "Voting has not begun.", (self.map_width - 315, y), MUTED, self.small)
        self.text(surf, "NO ROLE REVEAL AFTER EJECTION", (68, 699), MUTED, self.small)

    @staticmethod
    def meeting_origin(game):
        # Read only the public trigger immediately preceding the current meeting.
        started = next((i for i in range(len(game.event_log)-1, -1, -1)
                        if game.event_log[i]['kind'] == 'meeting_started'
                        and game.event_log[i].get('meeting_id') == game.context.meeting_id), None)
        if started is None:
            return 'Meeting trigger unavailable'
        for event in reversed(game.event_log[:started]):
            if event['kind'] == 'meeting_started': break
            if event['kind'] == 'report':
                return (f"{Phase3Renderer.identity(game, event['reporter_id'])} reported "
                        f"{Phase3Renderer.identity(game, event['victim_id'])}'s body / {event['room']}")
            if event['kind'] == 'emergency_called':
                return f"{Phase3Renderer.identity(game, event['reporter_id'])} called an emergency meeting"
        return 'Meeting trigger unavailable'

    @staticmethod
    def identity(game, player_id):
        player = game.players.get(player_id)
        return player.color_name if player else str(player_id or "")

    def result(self, surf, game):
        panel = pygame.Rect(60, 334, self.map_width - 120, 182)
        pygame.draw.rect(surf, (14, 30, 43), panel, border_radius=14)
        pygame.draw.rect(surf, ACCENT, panel, 2, border_radius=14)
        winner = str(_value(game.result.winner)).upper() if game.result.winner is not None else "DRAW"
        self.text(surf, winner + (" VICTORY" if game.result.winner is not None else " / TIME LIMIT"), (86, 357), ACCENT, self.large)
        self.text(surf, game.result.reason.replace("_", " ").upper(), (86, 412), TEXT, self.font)
        self.text(surf, f"Finished at {game.result.simulation_time:.1f}s.  R: replay seed {game.seed}   N: new match", (86, 463), MUTED, self.font)

    def footer(self, surf, options):
        y = self.size[1] - 60
        pygame.draw.rect(surf, (12, 20, 33), (0, y, self.size[0], 60))
        self.text(surf, "SPACE  pause    RIGHT  single step    +/-  speed    R  replay seed    N  new seed    ESC  exit", (22, y + 8), TEXT, self.small)
        self.text(surf, "TAB  HUD    1  spectator roles    G  routes    V  sight radius    T  tasks    L  labels    C  collision    E  log    B  bodies", (22, y + 33), MUTED, self.small)
