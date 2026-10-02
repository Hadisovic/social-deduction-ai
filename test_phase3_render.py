"""Renderer smoke checks, entry-point checks and spectator separation."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pygame
import pytest

from among_us_map import get_map
from phase3_renderer import COLORS, Phase3Renderer, ViewOptions, character_sprite
from social_deduction.actor import Phase, PublicContext, Role, Meeting, Vote, Provenance
from social_deduction.phase3_api import ClaimKind, StructuredClaim, EventView


@pytest.fixture(scope="module")
def render_world():
    return get_map()


def render_fixture(world):
    colors = ["Red", "Blue", "Green", "Yellow", "Cyan"]
    starts = [destination.standing for destination in world.destinations[:5]]
    players = {}
    for index, color in enumerate(colors):
        player_id = f"p{index}"
        players[player_id] = SimpleNamespace(
            player_id=player_id, color_name=color, role=Role.IMPOSTOR if index == 4 else Role.CREWMATE,
            status="alive", position=starts[index], velocity=(1., 0.), tasks=[],
            intention="go_to_task", interaction_task=None, navigator=SimpleNamespace(route=()),
        )
    return SimpleNamespace(map=world, players=players, bodies={}, seed=12, time=10., tick=50,
                           phase=Phase.ROAMING, context=PublicContext(), result=None,
                           config=SimpleNamespace(dt=.2, sight_range=4.5),
                           event_log=[], publications=[])


def test_procedural_characters_all_colors_directions_and_body():
    # No asset loading/display initialization is needed, including optional art.
    images = []
    for color in COLORS.values():
        for direction in ("left", "right", "up", "down"):
            for frame in range(4):
                sprite = character_sprite(color, 40, direction, frame)
                assert sprite.get_size() == (32, 40)
                assert 0 < np.count_nonzero(pygame.surfarray.array_alpha(sprite)) < 32 * 40
        images.append(pygame.image.tobytes(character_sprite(color, 40), "RGBA"))
    assert len(set(images)) == len(COLORS) - 1  # gray/grey are intentional aliases
    assert pygame.image.tobytes(character_sprite(COLORS["red"], 40, body=True), "RGBA") != images[0]
    assert pygame.image.tobytes(character_sprite(COLORS["red"], 40, frame=0), "RGBA") != pygame.image.tobytes(character_sprite(COLORS["red"], 40, frame=1), "RGBA")


def test_renderer_uses_improved_map_and_role_toggle_only_changes_pixels(render_world):
    game = render_fixture(render_world)
    renderer = Phase3Renderer(render_world)
    original = repr(vars(game))
    normal = renderer.draw(game)
    privileged = renderer.draw(game, ViewOptions(roles=True))
    assert normal.get_size() == (1440, 900)
    assert renderer.map_renderer.world is render_world
    assert renderer.map_renderer.art.get_width() > 500
    assert repr(vars(game)) == original
    assert not np.array_equal(pygame.surfarray.array3d(normal), pygame.surfarray.array3d(privileged))
    for player in game.players.values():
        x, y = renderer.map_renderer.point(player.position)
        # A sprite/identity is actually drawn at the map transform, not only HUD.
        patch = pygame.Rect(x - 20, y - 36, 50, 50)
        assert pygame.image.tobytes(normal.subsurface(patch), "RGB") != pygame.image.tobytes(renderer.map_renderer.background.subsurface(patch), "RGB")


def test_renderer_meeting_claim_vote_body_and_terminal_states(render_world):
    game = render_fixture(render_world)
    renderer = Phase3Renderer(render_world)
    game.players["p0"].status = "dead"
    game.bodies = {"p0": SimpleNamespace(victim_id="p0", position=game.players["p0"].position, reported=False)}
    renderer.draw(game, ViewOptions(routes=True, visibility=True, tasks=True, collision=True))
    game.phase = Phase.VOTING
    game.context = PublicContext(Phase.VOTING, "m1", tuple(game.players)[1:])
    game.publications = [
        EventView(40, Provenance.PUBLIC, Meeting("m1", tuple(game.players)[1:])),
        EventView(41, Provenance.CLAIM, StructuredClaim(ClaimKind.SAW_PLAYER, "p1", "p2", "Electrical", 41, 30)),
        EventView(42, Provenance.PUBLIC, Vote("m1", "p1", "p2")),
    ]
    renderer.draw(game)
    game.phase = Phase.FINISHED
    game.result = SimpleNamespace(winner=Role.CREWMATE, reason="IMPOSTOR_EJECTED", simulation_time=10.)
    game.players["p4"].status = "ejected"
    renderer.draw(game)


def test_visual_entrypoint_defaults_need_no_flags():
    from run_phase3 import parser
    args = parser().parse_args([])
    assert args.speed == 1.
    assert args.max_frames is None


def test_visual_entrypoint_smoke_starts_actual_simulator(tmp_path):
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("run_phase3.py")),
         "--max-frames", "2", "--capture", str(tmp_path)],
        cwd=Path(__file__).parent, env=env, text=True, capture_output=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert list(tmp_path.glob("seed-*-spawn.png"))


def test_pause_step_role_route_replay_and_next_seed_controls(monkeypatch):
    import run_phase3
    import phase3_renderer
    monkeypatch.setenv('SDL_VIDEODRIVER', 'dummy')
    monkeypatch.setenv('SDL_AUDIODRIVER', 'dummy')
    keys = iter((pygame.K_SPACE, pygame.K_RIGHT, pygame.K_1, pygame.K_g,
                 pygame.K_r, pygame.K_n, pygame.K_ESCAPE))
    monkeypatch.setattr(pygame.event, 'get', lambda: [pygame.event.Event(pygame.KEYDOWN, key=next(keys))])
    frames = []
    original = phase3_renderer.Phase3Renderer.draw
    def capture(self, game, options, paused, speed, fps):
        frames.append((game.seed, game.tick, paused, options.roles, options.routes))
        return original(self, game, options, paused, speed, fps)
    monkeypatch.setattr(phase3_renderer.Phase3Renderer, 'draw', capture)
    assert run_phase3.main([]) == 0
    assert frames[0] == (7, 0, True, False, False)
    assert frames[1] == (7, 1, True, False, False)
    assert frames[2] == (7, 1, True, True, False)
    assert frames[3] == (7, 1, True, True, True)
    assert frames[4][:3] == (7, 0, False)
    assert frames[5][:3] == (8, 0, False)
