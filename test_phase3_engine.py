"""Rule transitions, real physics integration and adversarial action validation."""
from dataclasses import asdict
import json
import math

import pytest

from among_us_map import get_map
from navigation_service import NavigationPlanner, NavigationService, NavStatus
from phase3_engine import GameConfig, Phase3Game, stream_seed
from phase3_runner import ScriptedMatch
from social_deduction.actor import Ejection, MatchEnded, Phase, Point, Role
from social_deduction.phase3_api import ClaimDraft, ClaimKind, Intent, IntentKind, StructuredClaim


@pytest.fixture(scope='module')
def planner():
    return NavigationPlanner(get_map(), cache_size=96)


@pytest.fixture
def game(planner):
    return Phase3Game(3, planner=planner)


def crew(game):
    return [p for p in game.players.values() if p.role is Role.CREWMATE]


def impostor(game):
    return next(p for p in game.players.values() if p.role is Role.IMPOSTOR)


def position(game, player, point):
    """Explicit fixture spawn, never an execution/recovery shortcut."""
    player.motion.spawn(point)
    game._invalidate()


def isolated_kill(game, witness=False):
    killer, victim = impostor(game), crew(game)[0]
    others = [p for p in game.players.values() if p not in (killer, victim)]
    for p, room in zip(others, ('Reactor', 'Navigation', 'Communications')):
        destination = next(d for d in game.map.destinations if d.room == room)
        position(game, p, destination.standing)
    position(game, killer, (-.7, -2.8))
    position(game, victim, (-1.4, -2.8))
    if witness:
        position(game, others[0], (-.7, -3.5))
    killer.cooldown = 0.
    game._invalidate()
    return killer, victim, others


def meeting(game):
    caller = crew(game)[0]
    console = next(d for d in game.map.destinations if d.name == 'EmergencyConsole')
    position(game, caller, console.standing)
    game.step({caller.player_id: Intent(IntentKind.CALL_MEETING, console.id)})
    assert game.phase is Phase.DISCUSSION


def until(game, phase):
    for _ in range(100):
        if game.phase is phase:
            return
        game.step()
    pytest.fail(f'Did not reach {phase}')


def test_initialization_streams_tasks_and_map(game, planner):
    assert len(game.players) == 5 and len(crew(game)) == 4
    assert len({p.color_name for p in game.players.values()}) == 5
    assert len({p.position for p in game.players.values()}) == 5
    assert len(game.tasks) == 8
    assert all(len(p.tasks) == 2 for p in crew(game))
    assert impostor(game).tasks == []
    assert all(isinstance(p.navigator, NavigationService) and p.navigator.planner is planner for p in game.players.values())
    assert all(game.planner.destinations[t.console_id].name != 'VentCleaning' for t in game.tasks.values())
    repeat = Phase3Game(3, planner=planner)
    assert game.snapshot() == repeat.snapshot() and game.event_log == repeat.event_log
    assert len({stream_seed(3, name) for name in ('identities', 'roles', 'colors', 'tasks', 'behavior')}) == 5


def test_roles_are_not_fixed_to_id_color_or_spawn(planner):
    ids, colors, slots = set(), set(), set()
    for seed in range(30):
        game = Phase3Game(seed, planner=planner)
        p = impostor(game)
        ids.add(p.player_id); colors.add(p.color_name)
        slots.add(next(e['internal_slot'] for e in game.event_log if e['kind'] == 'spawn' and e['player_id'] == p.player_id))
    assert len(ids) == 5 and len(slots) == 5 and len(colors) >= 7


def test_timed_task_completion_and_fixed_quota_win(game):
    player = crew(game)[0]; task = player.tasks[0]
    for other in game.tasks.values():
        other.progress = 1. if other is not task else 0.
    position(game, player, game.planner.destinations[task.console_id].standing)
    game.step({player.player_id: Intent(IntentKind.INTERACT, task.task_id)})
    assert 0 < task.progress < 1 and not game.is_terminal
    while not game.is_terminal:
        game.step()
    assert game.result.reason == 'TASKS_COMPLETED' and game.result.winner is Role.CREWMATE
    started = next(e['time'] for e in game.event_log if e['kind'] == 'task_start')
    ended = next(e['time'] for e in game.event_log if e['kind'] == 'task_completed')
    assert ended - started == pytest.approx(game.config.task_seconds)


def test_task_navigation_is_real_and_interruptible(game):
    player = crew(game)[0]; task = player.tasks[0]
    old = player.position
    game.step({player.player_id: Intent(IntentKind.GO_TO_TASK, task.console_id)})
    assert math.dist(old, player.position) <= game.config.speed * game.config.dt + 1e-9
    for _ in range(500):
        if player.navigator.status is NavStatus.SUCCESS:
            break
        before = player.position
        game.step()
        assert game.map.segment_clear(before, player.position)
        assert math.dist(before, player.position) <= game.config.speed * game.config.dt + 1e-9
    assert player.navigator.status is NavStatus.SUCCESS
    game.step({player.player_id: Intent(IntentKind.INTERACT, task.task_id)})
    partial = task.progress
    game.step({player.player_id: Intent(IntentKind.CANCEL_NAVIGATION)})
    for _ in range(4): game.step()
    assert task.progress == partial and player.interaction_task is None
    assert any(e['kind'] == 'task_interrupted' for e in game.event_log)


def test_fake_task_is_ambiguous_visible_animation_but_never_team_progress(game):
    player = impostor(game)
    console = next(d for d in game.map.destinations if d.category == 'task')
    position(game, player, console.standing)
    before = [t.progress for t in game.tasks.values()]
    game.step({player.player_id: Intent(IntentKind.FAKE_TASK, console.id)})
    assert player.interaction_task.startswith('fake:')
    assert next(p for p in game.snapshot().players if p.identity.player_id == player.player_id).interacting
    for _ in range(math.ceil(game.config.fake_task_seconds / game.config.dt)): game.step()
    assert player.interaction_task is None and before == [t.progress for t in game.tasks.values()]


@pytest.mark.parametrize('invalid', ['range', 'cooldown', 'victim', 'crew_actor'])
def test_illegal_eliminations_are_rejected(game, invalid):
    killer, victim, others = isolated_kill(game)
    actor, target = killer.player_id, victim.player_id
    if invalid == 'range': position(game, victim, others[0].position)
    elif invalid == 'cooldown': killer.cooldown = 10.
    elif invalid == 'victim': target = 'not-a-player'
    else: actor, target = victim.player_id, killer.player_id
    game._invalidate()
    game.step({actor: Intent(IntentKind.KILL, target)})
    assert game.metrics['eliminations'] == 0 and not game.bodies
    assert game.metrics['illegal_actions'] == 1


def test_elimination_body_and_only_direct_witness_disclosure(game):
    killer, victim, others = isolated_kill(game, witness=True)
    location = victim.position
    game.step({killer.player_id: Intent(IntentKind.KILL, victim.player_id)})
    assert victim.status == 'dead' and game.bodies[victim.player_id].position == location
    assert not game.publications  # Death is not a global announcement.
    assert game.direct_events[others[0].player_id] and not game.direct_events[others[1].player_id]
    assert game.observe(victim.player_id).actions == ()
    assert not game.observe(victim.player_id).visible_players
    assert game.observe(others[1].player_id).roster == game.observe(others[0].player_id).roster
    for _ in range(3): game.step()
    assert victim.player_id in game.bodies and victim.position == location


def test_hidden_death_does_not_reassign_tasks_until_public_meeting(game):
    killer, victim, others = isolated_kill(game)
    original = {p.player_id: tuple(t.task_id for t in p.tasks) for p in others}
    victim_tasks = {t.task_id for t in victim.tasks}
    game.step({killer.player_id: Intent(IntentKind.KILL, victim.player_id)})
    assert {p.player_id: tuple(t.task_id for t in p.tasks) for p in others} == original
    position(game, others[0], victim.position)
    game.step({others[0].player_id: Intent(IntentKind.REPORT, victim.player_id)})
    assert game.phase is Phase.DISCUSSION and not game.bodies
    assert victim_tasks <= {t.task_id for p in crew(game) if p.active for t in p.tasks}
    assert len(game.tasks) == game.initial_task_count == 8


def test_report_priority_cancels_all_motion_and_freezes_timers(game):
    killer, victim, others = isolated_kill(game)
    game.step({killer.player_id: Intent(IntentKind.KILL, victim.player_id)})
    position(game, others[0], victim.position)
    position(game, others[1], (-.7, -3.5))
    killer.cooldown = 0.; game._invalidate()
    remaining = others[2]
    game.step({remaining.player_id: Intent(IntentKind.GO_TO_TASK, remaining.tasks[0].console_id)})
    before = {pid: p.position for pid, p in game.players.items()}
    game.step({others[0].player_id: Intent(IntentKind.REPORT, victim.player_id),
               killer.player_id: Intent(IntentKind.KILL, others[1].player_id)})
    assert game.metrics['eliminations'] == 1 and others[1].active
    assert game.phase is Phase.DISCUSSION
    assert all(p.navigator.status is NavStatus.CANCELLED for p in game.players.values())
    cooldown = killer.cooldown
    for _ in range(3): game.step()
    assert before == {pid: p.position for pid, p in game.players.items()}
    assert killer.cooldown == cooldown


def test_invalid_report_and_recursive_meeting_rejected(game):
    caller = crew(game)[0]
    game.step({caller.player_id: Intent(IntentKind.REPORT, 'unknown')})
    assert game.phase is Phase.ROAMING and not game.publications
    meeting(game)
    count = game.metrics['meetings']
    game.step({caller.player_id: Intent(IntentKind.CALL_MEETING, 'unknown')})
    assert game.metrics['meetings'] == count and game.metrics['illegal_actions'] == 2


def test_claim_authentication_and_asserted_time_without_fact_check(game):
    meeting(game)
    speaker = game.context.speaker_id
    subject = next(pid for pid in game.players if pid != speaker)
    draft = ClaimDraft(ClaimKind.SUSPECT_PLAYER, subject, 'Electrical', 0)
    tick = game.tick
    game.step({speaker: Intent(IntentKind.CLAIM, claim=draft)})
    claim = next(e.payload for e in game.publications if isinstance(e.payload, StructuredClaim))
    assert claim.speaker_id == speaker and claim.delivered_tick == tick and claim.asserted_tick == 0
    assert claim.region == 'Electrical'  # False allegation is still a valid claim.
    game.step({speaker: Intent(IntentKind.CLAIM, claim=draft)})
    assert sum(isinstance(e.payload, StructuredClaim) for e in game.publications) == 1


@pytest.mark.parametrize('votes,expected', [
    (['blue', 'blue', 'red', 'red', None], None),
    (['blue', 'blue', None, None, 'red'], None),
    ([None] * 5, None),
    (['blue'] * 5, 'blue'),
    (['blue', 'blue', 'blue', None, 'red'], 'blue'),
])
def test_tie_skip_plurality_semantics(votes, expected):
    assert Phase3Game.tally_votes(dict(enumerate(votes))) == expected


def test_votes_public_single_use_and_impostor_ejection_terminal(game):
    meeting(game); until(game, Phase.VOTING)
    target = impostor(game).player_id
    voter = game.participants[0]
    game.step({voter: Intent(IntentKind.VOTE, 'invalid')})
    assert not game.votes
    game.step({voter: Intent(IntentKind.VOTE, target)})
    assert game.votes[voter] == target
    game.step({voter: Intent(IntentKind.SKIP)})
    assert game.votes[voter] == target and game.metrics['illegal_actions'] == 2
    game.step({pid: Intent(IntentKind.VOTE, target) for pid in game.participants if pid != voter})
    assert game.phase is Phase.VOTING  # Results remain inspectable until deadline.
    until(game, Phase.FINISHED)
    assert game.result.reason == 'IMPOSTOR_EJECTED'
    ejection = next(e.payload for e in game.publications if isinstance(e.payload, Ejection))
    assert asdict(ejection) == {'meeting_id': 'm1', 'player_id': target}
    assert set(asdict(game.publications[-1].payload)) == {'winner'}
    tick, log = game.tick, list(game.event_log)
    game.step({voter: Intent(IntentKind.GO_TO_ROOM, 'Electrical')})
    assert game.tick == tick and game.event_log == log


def test_skip_resolution_returns_to_same_positions_without_stale_motion(game):
    meeting(game); locations = {pid: p.position for pid, p in game.players.items()}
    until(game, Phase.VOTING)
    game.step({pid: Intent(IntentKind.SKIP) for pid in game.participants})
    until(game, Phase.ROAMING)
    assert locations == {pid: p.position for pid, p in game.players.items()}
    assert not game.is_terminal and all(p.interaction_task is None for p in game.players.values())
    assert any(e['kind'] == 'exploration_resumed' for e in game.event_log)


def test_crew_ejection_reassigns_quota_and_disables_entity(game):
    meeting(game); until(game, Phase.VOTING)
    victim = crew(game)[0]; tasks = {t.task_id for t in victim.tasks}
    game.step({pid: Intent(IntentKind.VOTE, victim.player_id) for pid in game.participants})
    until(game, Phase.ROAMING)
    assert victim.status == 'ejected' and not game.observe(victim.player_id).actions
    assert tasks <= {t.task_id for p in crew(game) if p.active for t in p.tasks}
    old = victim.position
    game.step({victim.player_id: Intent(IntentKind.GO_TO_ROOM, 'Electrical')})
    assert victim.position == old


def test_dead_targets_not_vote_candidates(game):
    killer, victim, others = isolated_kill(game)
    game.step({killer.player_id: Intent(IntentKind.KILL, victim.player_id)})
    position(game, others[0], victim.position)
    game.step({others[0].player_id: Intent(IntentKind.REPORT, victim.player_id)})
    until(game, Phase.VOTING)
    observer = others[0].player_id
    assert victim.player_id not in game.context.participants
    assert Intent(IntentKind.VOTE, victim.player_id) not in game.legal_actions(observer)
    game.step({observer: Intent(IntentKind.VOTE, victim.player_id)})
    assert observer not in game.votes


def test_parity_is_immediate_after_third_elimination(game):
    killer = impostor(game)
    for victim in crew(game)[:3]:
        position(game, killer, (-.7, -2.8)); position(game, victim, (-1.4, -2.8))
        killer.cooldown = 0.; game._invalidate()
        game.step({killer.player_id: Intent(IntentKind.KILL, victim.player_id)})
    assert game.is_terminal and game.result.winner is Role.IMPOSTOR and game.result.reason == 'PARITY'
    assert game.metrics['eliminations'] == 3


def test_timeout_draw_and_serializable_event_log(planner, tmp_path):
    game = Phase3Game(config=GameConfig(timeout=.8), planner=planner)
    for _ in range(20): game.step()
    assert game.tick == 4 and game.result.winner is None and game.result.reason == 'TIMEOUT'
    path = game.write_replay(tmp_path / 'trace.jsonl')
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert events[0]['seed'] == 0 and events[-1]['kind'] == 'match_end'
    assert events[-1]['reason'] == 'TIMEOUT'


def test_full_match_and_actor_trace_determinism_independent_of_render_schedule(planner):
    first = ScriptedMatch(2, planner=planner); second = ScriptedMatch(2, planner=planner)
    first.run()
    # Equivalent to a display consuming irregular batches of simulation ticks.
    while not second.game.is_terminal:
        for _ in range(7): second.step()
    assert first.trajectory_hash == second.trajectory_hash
    assert first.game.result == second.game.result
    assert first.game.metrics['illegal_actions'] == first.game.metrics['navigation_failures'] == 0


@pytest.mark.parametrize('kwargs', [{'dt': 0}, {'tasks_per_crew': 0}, {'timeout': float('inf')},
                                   {'kill_range': 10}, {'task_seconds': -1}])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError): GameConfig(**kwargs)
