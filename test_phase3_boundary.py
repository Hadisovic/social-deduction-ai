"""Paired-world tests of the complete Phase 3 input, including action candidates."""
from dataclasses import FrozenInstanceError, replace
import math

import pytest

from among_us_map import get_map
from navigation_service import NavigationPlanner
from phase3_bots import ScriptedController
from phase3_engine import Phase3Game
from phase3_runner import canonical_bytes
from social_deduction.actor import (
    BodySeen, Ejection, Meeting, Phase, PlayerSeen, Point, Provenance,
    PublicContext, Report, Role, Vote,
)
from social_deduction.phase3_api import (
    ClaimKind, EventView, IntentKind, StructuredClaim, WitnessedElimination,
    legal_actions, observation_to_dict, validate_event, validate_observation,
)
from social_deduction.phase3_observation import project_game, visible_between
from social_deduction.truth import BodyTruth, InternalEvent, TaskTruth


@pytest.fixture(scope='module')
def planner():
    return NavigationPlanner(get_map(), cache_size=96)


@pytest.fixture
def scene(planner):
    game = Phase3Game(3, planner=planner)
    world = game.snapshot()
    own = next(p for p in world.players if p.role is Role.CREWMATE)
    far = next(d.standing for d in game.map.destinations if d.room == 'Reactor')
    players = tuple(p if p is own else replace(p, position=Point(*map(float, far)),
                    room='Reactor') for p in world.players)
    return game, replace(world, tick=20, players=players), own.identity.player_id


def project(scene, world=None, **kwargs):
    game, original, actor = scene
    return project_game(world or original, actor, game.map, game.planner, game.settings, **kwargs)


def change_others(world, actor, transform):
    return replace(world, players=tuple(p if p.identity.player_id == actor else transform(p)
                                       for p in world.players))


@pytest.mark.parametrize('hidden', ['roles', 'positions', 'death', 'tasks', 'cooldowns',
                                  'velocity', 'interactions', 'meetings', 'ordering', 'events'])
def test_hidden_world_changes_preserve_every_input_byte_and_option(scene, hidden):
    game, world, actor = scene
    if hidden == 'roles':
        others = [p for p in world.players if p.identity.player_id != actor]
        roles = [p.role for p in others][1:] + [others[0].role]
        altered = dict(zip((p.identity.player_id for p in others), roles))
        changed = change_others(world, actor, lambda p: replace(p, role=altered[p.identity.player_id]))
    elif hidden == 'positions':
        far = next(d.standing for d in game.map.destinations if d.room == 'Navigation')
        changed = change_others(world, actor, lambda p: replace(p, position=Point(*map(float, far)), room='Navigation'))
    elif hidden == 'death':
        victim = next(p for p in world.players if p.identity.player_id != actor and p.role is Role.CREWMATE)
        changed = change_others(world, actor, lambda p: replace(p, active=False) if p is victim else p)
        changed = replace(changed, bodies=(BodyTruth(victim.identity.player_id, victim.position,
                          victim.room, 19, 'hidden-killer'),))
    elif hidden == 'tasks':
        changed = change_others(world, actor, lambda p: replace(p, tasks=(TaskTruth('private-id','private-console', .71),)))
    elif hidden == 'cooldowns':
        changed = change_others(world, actor, lambda p: replace(p, private_cooldown=99.))
    elif hidden == 'velocity':
        changed = change_others(world, actor, lambda p: replace(p, velocity=Point(-9., 11.)))
    elif hidden == 'interactions':
        changed = change_others(world, actor, lambda p: replace(p, interacting=True))
    elif hidden == 'meetings':
        changed = change_others(world, actor, lambda p: replace(p, meetings_remaining=0))
    elif hidden == 'ordering':
        changed = replace(world, players=tuple(reversed(world.players)))
    else:
        changed = replace(world, internal_events=(InternalEvent(0, 'secret', 'p0'),
                                                 InternalEvent(999, 'future', 'p1')))
    before, after = project(scene), project(scene, changed)
    assert before == after
    assert canonical_bytes(before) == canonical_bytes(after)
    assert before.actions == after.actions == legal_actions(after)
    assert ScriptedController(19).decide(before) == ScriptedController(19).decide(after)


def test_visible_players_do_not_disclose_private_tasks_roles_or_cooldowns(scene):
    game, world, actor = scene
    own = next(p for p in world.players if p.identity.player_id == actor)
    near = change_others(world, actor, lambda p: replace(p, position=own.position, room=own.room))
    altered = change_others(near, actor, lambda p: replace(p,
        role=Role.IMPOSTOR if p.role is Role.CREWMATE else Role.CREWMATE,
        tasks=(TaskTruth('secret', 'hidden', .5),), private_cooldown=567.))
    packet = project(scene, near)
    assert len(packet.visible_players) == 4
    assert packet == project(scene, altered)
    assert all(type(p) is PlayerSeen for p in packet.visible_players)


def test_visible_body_omits_death_time_and_killer_and_hidden_body_order(scene):
    _, world, actor = scene
    own = next(p for p in world.players if p.identity.player_id == actor)
    others = [p for p in world.players if p.identity.player_id != actor]
    body = BodyTruth(others[0].identity.player_id, own.position, own.room, 4, others[1].identity.player_id)
    hidden = BodyTruth(others[2].identity.player_id, others[2].position, 'Reactor', 5, 'p0')
    before = project(scene, replace(world, bodies=(body, hidden)))
    after = project(scene, replace(world, bodies=(replace(hidden, death_tick=18),
                          replace(body, death_tick=17, killer_id='different'))))
    assert before == after
    assert before.visible_bodies == (BodySeen(body.victim_id, own.position, own.room),)
    assert any(a.kind is IntentKind.REPORT for a in before.actions)
    assert b'killer_id' not in canonical_bytes(before)


def test_own_role_and_cooldown_are_the_only_role_local_controls(scene):
    _, world, actor = scene
    own = next(p for p in world.players if p.identity.player_id == actor)
    crew = project(scene)
    changed = replace(world, players=tuple(replace(p, private_cooldown=500.) if p is own else p for p in world.players))
    assert crew == project(scene, changed) and crew.kill_cooldown is None
    def impostor_world(cooldown):
        return replace(world, players=tuple(replace(p, role=Role.IMPOSTOR, private_cooldown=cooldown)
            if p is own else replace(p, role=Role.CREWMATE, position=own.position, room=own.room)
            for p in world.players))
    ready, cooling = project(scene, impostor_world(0.)), project(scene, impostor_world(4.))
    assert ready.kill_cooldown == 0. and cooling.kill_cooldown == 4.
    assert sum(a.kind is IntentKind.KILL for a in ready.actions) == 4
    assert not any(a.kind is IntentKind.KILL for a in cooling.actions)


def test_native_wall_occlusion_hides_both_players_and_bodies(scene):
    game, world, actor = scene
    points = [tuple(map(float, d.standing)) for d in game.map.destinations]
    a, b = next((a,b) for a in points for b in points
                if .1 < math.dist(a,b) < game.settings.sight_range
                and not visible_between(game.map,a,b,game.settings.sight_range))
    victim = next(p for p in world.players if p.identity.player_id != actor)
    changed = replace(world, players=tuple(replace(p, position=Point(*a)) if p.identity.player_id == actor
                       else replace(p, position=Point(*b)) for p in world.players),
                      bodies=(BodyTruth(victim.identity.player_id, Point(*b), 'room', 2, 'p0'),))
    packet = project(scene, changed)
    assert packet.visible_players == packet.visible_bodies == ()
    assert not any(a.kind in (IntentKind.REPORT, IntentKind.GO_TO_BODY, IntentKind.KILL) for a in packet.actions)
    assert visible_between(game.map, a, a, game.settings.sight_range)
    assert not visible_between(game.map, a, b, .01)


@pytest.mark.parametrize('phase', [Phase.DISCUSSION, Phase.VOTING, Phase.FINISHED])
def test_meetings_and_finished_hide_local_geometry_and_living_actions(scene, phase):
    _, world, actor = scene
    changed = replace(world, context=PublicContext(phase, 'm1', tuple(p.identity.player_id for p in world.players), actor))
    packet = project(scene, changed)
    assert packet.visible_players == packet.visible_bodies == ()
    assert not any(a.kind in (IntentKind.KILL, IntentKind.REPORT, IntentKind.INTERACT,
                              IntentKind.GO_TO_TASK, IntentKind.GO_TO_LOCATION) for a in packet.actions)


def test_dead_actor_cannot_see_current_geometry_or_act(scene):
    _, world, actor = scene
    changed = replace(world, players=tuple(replace(p, active=False) if p.identity.player_id == actor else p for p in world.players))
    packet = project(scene, changed)
    assert packet.visible_players == packet.visible_bodies == packet.actions == ()


def test_future_public_events_are_filtered_before_payload_and_order_checks(scene):
    base = project(scene)
    malformed_future = EventView(999, Provenance.PUBLIC, object())
    assert base == project(scene, publications=(malformed_future,))
    known = EventView(5, Provenance.PUBLIC, Ejection('m0', None))
    assert project(scene, publications=(known,)) == project(scene, publications=(malformed_future, known))


def test_claim_and_direct_witness_have_distinct_authenticated_provenance(scene):
    _, world, actor = scene
    claim = EventView(12, Provenance.CLAIM, StructuredClaim(ClaimKind.SAW_ELIMINATION,
        actor, 'p0', 'Reactor', 12, 3))
    direct = EventView(9, Provenance.DIRECT, WitnessedElimination('p0', 'p1', Point(0.,0.), 'Cafeteria'))
    packet = project(scene, publications=(claim,), direct_events=(direct,))
    assert claim in packet.evidence and direct in packet.evidence
    with pytest.raises(ValueError):
        project(scene, publications=(direct,))
    with pytest.raises(ValueError):
        project(scene, direct_events=(claim,))
    with pytest.raises(ValueError):
        validate_event(replace(claim, provenance=Provenance.DIRECT), world.tick)
    with pytest.raises(ValueError):
        validate_event(replace(claim, tick=13), world.tick)


@pytest.mark.parametrize('payload', [object(), {'killer': 'p0'}, InternalEvent(1, 'kill', 'p0')])
def test_truth_and_mutable_objects_rejected_at_public_envelope(scene, payload):
    with pytest.raises(TypeError):
        project(scene, publications=(EventView(1, Provenance.PUBLIC, payload),))


def test_packets_reject_subclasses_mutable_fields_nonfinite_numbers_and_backrefs(scene):
    packet = project(scene)
    with pytest.raises(FrozenInstanceError):
        packet.tick = 6
    class SpyPoint(Point):
        pass
    for bad in (replace(packet, roster=list(packet.roster)),
                replace(packet, own=replace(packet.own, position=Point(float('nan'),0.))),
                replace(packet, own=replace(packet.own, position=SpyPoint(0.,0.))),
                replace(packet, evidence=(scene[0].snapshot(),))):
        with pytest.raises(TypeError):
            validate_observation(bad)
    with pytest.raises(TypeError):
        observation_to_dict(scene[0].snapshot())
    assert b'private_cooldown' not in canonical_bytes(observation_to_dict(packet))


def test_independent_role_assignment_cannot_change_others_private_tasks(planner, monkeypatch):
    import phase3_engine
    first = Phase3Game(13, planner=planner)
    original = phase3_engine.assign_roles
    def shifted(count, *, seed):
        roles = original(count, seed=seed)
        return roles[1:] + roles[:1]
    monkeypatch.setattr(phase3_engine, 'assign_roles', shifted)
    second = Phase3Game(13, planner=planner)
    for pid, p in first.players.items():
        q = second.players[pid]
        assert (p.position, p.color_name) == (q.position, q.color_name)
        if p.role is q.role is Role.CREWMATE:
            assert p.tasks == q.tasks
            assert first.observe(pid) == second.observe(pid)


def test_spectator_options_cannot_change_observations_or_controller_decisions(scene):
    from phase3_renderer import Phase3Renderer, ViewOptions
    game, _, _ = scene
    before = {p: game.observe(p) for p in game.players}
    renderer = Phase3Renderer(game.map)
    renderer.draw(game, ViewOptions(roles=True, routes=True, visibility=True,
                                   tasks=True, collision=True))
    game._invalidate()  # Compare a new projection, not just cached objects.
    after = {p: game.observe(p) for p in game.players}
    assert before == after
    for pid in before:
        assert ScriptedController(24).decide(before[pid]) == ScriptedController(24).decide(after[pid])
