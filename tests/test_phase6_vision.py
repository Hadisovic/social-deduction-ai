from dataclasses import replace
import math
import numpy as np
import pytest
from shapely.geometry import LineString, Point as ShapePoint

from among_us_map import get_map
from navigation_service import NavigationPlanner
from phase3_engine import Body, GameConfig, Phase3Game
from phase3_runner import ScriptedMatch, canonical_bytes
from phase6.config import Experiment, verify_frozen
from social_deduction.actor import Phase, Point, Role
from social_deduction.phase3_api import WitnessedElimination, observation_to_dict
from social_deduction.phase3_observation import project_game, visible_between

ORIGIN = (-.7, -2.8)
NEAR = (-1.02, -4.8)
MIDDLE = (-1.2, -8.28)
FAR = (-1.38, -10.8)


@pytest.fixture(scope='module')
def planner():
    return NavigationPlanner(get_map(), cache_size=96)


@pytest.fixture
def game(planner):
    return Phase3Game(3, config=Experiment().game_config(), planner=planner)


def fixture_world(game, role, target):
    world = game.snapshot()
    own = next(p for p in world.players if p.role is role)
    other = next(p for p in world.players if p.identity != own.identity)
    players = tuple(replace(p, position=Point(*ORIGIN) if p is own else
                            Point(*target) if p is other else Point(99., 99.)) for p in world.players)
    return replace(world, players=players), own.identity.player_id, other.identity.player_id


@pytest.mark.parametrize('target,crew_sees,impostor_sees', [
    (NEAR, True, True), (MIDDLE, False, True), (FAR, False, False)])
def test_role_specific_player_ranges(game, target, crew_sees, impostor_sees):
    for role, expected in ((Role.CREWMATE, crew_sees), (Role.IMPOSTOR, impostor_sees)):
        world, actor, other = fixture_world(game, role, target)
        packet = project_game(world, actor, game.map, game.planner, game.settings_for(role))
        assert (other in {p.player_id for p in packet.visible_players}) == expected
        assert packet.settings.sight_range == game.config.effective_sight_range(role)
        assert packet.map.sight_range == packet.settings.sight_range
        assert b'private_cooldown' not in canonical_bytes(observation_to_dict(packet))


def blocker_pair(game, kind):
    records = game.map.data['walls' if kind == 'wall' else 'obstacles']
    barrier = game.map.wall_lines if kind == 'wall' else game.map.objects
    for record in records:
        vertices = record['points']
        for a, b in zip(vertices, vertices[1:] + vertices[:1]):
            length = math.dist(a, b)
            if length < .5:
                continue
            center = tuple((x + y) / 2 for x, y in zip(a, b))
            normal = (-(b[1] - a[1]) / length, (b[0] - a[0]) / length)
            for offset in (.5, 1., 1.5, 2.):
                p = tuple(x + offset * n for x, n in zip(center, normal))
                q = tuple(x - offset * n for x, n in zip(center, normal))
                if all(game.map.center_domain.covers(ShapePoint(x)) for x in (p, q)) and barrier.intersects(LineString((p, q))):
                    return p, q
    raise AssertionError(f'No controlled native {kind} pair')


@pytest.mark.parametrize('kind', ['wall', 'obstacle'])
def test_native_occlusion_blocks_both_roles_and_direct_evidence(game, kind):
    p, q = blocker_pair(game, kind)
    for role in Role:
        world, actor, other = fixture_world(game, role, q)
        world = replace(world, players=tuple(replace(x, position=Point(*p)) if x.identity.player_id == actor else x for x in world.players))
        packet = project_game(world, actor, game.map, game.planner, game.settings_for(role))
        assert other not in {x.player_id for x in packet.visible_players}
        assert not any(getattr(e.payload, 'player_id', None) == other for e in packet.evidence)
        assert not visible_between(game.map, p, q, game.config.effective_sight_range(role))


def test_body_visibility_and_report_radius_remain_separate(game):
    from social_deduction.truth import BodyTruth
    for role, expected in ((Role.CREWMATE, False), (Role.IMPOSTOR, True)):
        world, actor, victim = fixture_world(game, role, MIDDLE)
        world = replace(world, bodies=(BodyTruth(victim, Point(*MIDDLE), 'Cafeteria', 0, 'secret'),))
        obs = project_game(world, actor, game.map, game.planner, game.settings_for(role))
        assert bool(obs.visible_bodies) == expected
        assert not any(a.kind.value == 'report' for a in obs.actions)
        assert 'secret' not in str(observation_to_dict(obs))


def test_witness_filter_is_event_time_not_later_visibility(game):
    crew = [p for p in game.players.values() if p.role is Role.CREWMATE]
    killer = next(p for p in game.players.values() if p.role is Role.IMPOSTOR)
    victim, near, far, other = crew
    for player, point in ((killer, ORIGIN), (victim, (-.7, -3.)), (near, NEAR), (far, MIDDLE), (other, FAR)):
        player.motion.spawn(point)
    game._invalidate()
    game._eliminate(killer, victim)
    assert any(type(e.payload) is WitnessedElimination for e in game.observe(near.player_id).evidence)
    assert not any(type(e.payload) is WitnessedElimination for e in game.observe(far.player_id).evidence)
    far.motion.spawn(ORIGIN)
    near.motion.spawn(FAR)
    game.tick += 1
    game._invalidate()
    assert not any(type(e.payload) is WitnessedElimination for e in game.observe(far.player_id).evidence)
    history = [e for e in game.observe(near.player_id).evidence if type(e.payload) is WitnessedElimination]
    assert len(history) == 1 and history[0].tick == 0


def test_witnesses_use_each_observer_role_range_and_walls(game):
    # Controlled trusted-event fixture: temporarily assign one observer an
    # impostor role to isolate role-range dispatch, not a legal five-player match.
    crew = [p for p in game.players.values() if p.role is Role.CREWMATE]
    killer = next(p for p in game.players.values() if p.role is Role.IMPOSTOR)
    victim, far, extra, blocked = crew
    p, q = blocker_pair(game, 'wall')
    for player in game.players.values():
        player.motion.spawn(MIDDLE)
    killer.motion.spawn(ORIGIN)
    victim.motion.spawn((-.7, -3.))
    extra.role = Role.IMPOSTOR
    game._invalidate()
    game._eliminate(killer, victim)
    assert not game.direct_events[far.player_id]
    assert game.direct_events[extra.player_id]
    # Same event check with native wall occlusion at short distance.
    fresh = Phase3Game(3, config=game.config, planner=game.planner)
    killer = next(x for x in fresh.players.values() if x.role is Role.IMPOSTOR)
    victim, observer = [x for x in fresh.players.values() if x.role is Role.CREWMATE][:2]
    killer.motion.spawn(q); victim.motion.spawn(q); observer.motion.spawn(p)
    fresh._eliminate(killer, victim)
    assert not fresh.direct_events[observer.player_id]


def test_history_keeps_only_real_sightings_and_observation_cache_is_actor_local(game):
    from belief.memory import ActorMemory
    crew = [p for p in game.players.values() if p.role is Role.CREWMATE]
    own, other = crew[:2]
    own.motion.spawn(ORIGIN); other.motion.spawn(MIDDLE); game._invalidate()
    memory = ActorMemory(); memory.update(game.observe(own.player_id))
    assert other.player_id not in memory.last_seen
    other.motion.spawn(NEAR); game.tick += 1; game._invalidate()
    first = game.observe(own.player_id); memory.update(first)
    assert first is game.observe(own.player_id)
    other.motion.spawn(FAR); game.tick += 1; game._invalidate()
    memory.update(game.observe(own.player_id))
    assert memory.last_seen[other.player_id].tick == 1
    before = {pid: game.observe(pid) for pid in game.players}
    game._invalidate()
    assert before == {pid: game.observe(pid) for pid in game.players}


@pytest.mark.parametrize('status', ['dead', 'ejected'])
def test_eliminated_actors_have_no_local_vision_or_actions(game, status):
    for p in game.players.values():
        p.status = status
    game._invalidate()
    for pid in game.players:
        obs = game.observe(pid)
        assert not obs.visible_players and not obs.visible_bodies and not obs.actions


def test_overlay_uses_role_ranges_occlusion_and_cannot_change_policy_input(game, monkeypatch):
    from phase3_renderer import Phase3Renderer, ViewOptions, visibility_outline
    import phase3_renderer
    before = {pid: game.observe(pid) for pid in game.players}
    outlines = {pid: visibility_outline(game, p, rays=120) for pid, p in game.players.items()}
    for pid, points in outlines.items():
        p = game.players[pid]
        assert max(math.dist(p.position, q) for q in points) == pytest.approx(game.config.effective_sight_range(p.role))
        assert any(math.dist(p.position, q) < game.config.effective_sight_range(p.role) - .01 for q in points)
    called = []
    original = phase3_renderer.visibility_outline
    monkeypatch.setattr(phase3_renderer, 'visibility_outline', lambda g, p, origin=None: (called.append(p.player_id) or original(g, p, origin)))
    Phase3Renderer(game.map).draw(game, ViewOptions(visibility=True, roles=True))
    assert set(called) == set(game.players)
    game._invalidate()
    assert before == {pid: game.observe(pid) for pid in game.players}


def test_legacy_replay_hashes_config_roundtrip_and_release_compatibility(planner):
    match = ScriptedMatch(7, planner=planner); match.run()
    assert match.actor_trace_hash == '42b5d65e484869fe42e7ac19fce9b9e18d50eb09c64d5db53ba762f52ed180b1'
    assert match.truth_trace_hash == 'dc6fe1cc6002983acf19a187d86625b8bc5bebfff15a0225e3148b28d36ff6e0'
    doc = match.replay_document()
    assert 'rules_version' not in doc['config']
    assert GameConfig(**doc['config']) == GameConfig()
    assert GameConfig(**Experiment().game_config().to_dict()) == Experiment().game_config()
    verify_frozen()
    from phase5_policy import StrategicPolicy
    from phase5_runtime import checked_environment
    from phase6.config import ROOT
    _, metadata = StrategicPolicy.load(ROOT / 'artifacts/phase5/release/policy.pt')
    env = checked_environment(metadata); env.reset(seed=620000)
    assert env.match.game.config == GameConfig()


@pytest.mark.parametrize('kwargs', [dict(dt=None), dict(crewmate_sight_range=4.),
    dict(rules_version=2), dict(rules_version=6), dict(rules_version=6, crewmate_sight_range=0., impostor_sight_range=6.),
    dict(rules_version=6, crewmate_sight_range=1., impostor_sight_range=6.),
    dict(rules_version=6, crewmate_sight_range=4., impostor_sight_range=float('nan'))])
def test_invalid_rules_fail_closed(kwargs):
    with pytest.raises(ValueError):
        GameConfig(**kwargs)
