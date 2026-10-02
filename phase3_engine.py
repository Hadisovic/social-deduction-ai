"""Trusted deterministic five-player rules. Policies never receive this module's state."""
from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
from random import Random
from time import perf_counter

from among_us_map import ROOMS, get_map
from among_us_map_simulation import AmongUsMapEnv
from navigation_service import ControllerConfig, NavigationPlanner, NavigationService, NavStatus, NavTarget
from social_deduction.actor import (
    Console, Ejection, Identity, MatchEnded, Meeting, Phase, Point, Provenance,
    PublicContext, PublicMap, Report, Role, Vote,
)
from social_deduction.truth import BodyTruth, PlayerTruth, TaskTruth, WorldTruth, assign_identities, assign_roles
from social_deduction.phase3_api import EventView, Intent, IntentKind, ObservationSettings, StructuredClaim, WitnessedElimination, intent_allowed
from social_deduction.phase3_observation import project_game, visible_between


def stream_seed(seed, domain):
    """Domain-separated streams; neither seed nor internal slot enters observations."""
    return int.from_bytes(hashlib.sha256(f'{int(seed)}:{domain}'.encode()).digest()[:16], 'big')


@dataclass(frozen=True)
class GameConfig:
    dt: float = .2
    decision_interval: float = .6
    speed: float = 2.5
    tasks_per_crew: int = 2
    task_seconds: float = 4.0
    fake_task_seconds: float = 4.0
    initial_kill_cooldown: float = 12.0
    kill_cooldown: float = 12.0
    sight_range: float = 4.5
    kill_range: float = 1.1
    report_range: float = 1.3
    discussion_seconds: float = 9.0
    voting_seconds: float = 6.0
    timeout: float = 240.0
    check_invariants: bool = True

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key in ('check_invariants', 'tasks_per_crew'):
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be positive and finite')
        if type(self.tasks_per_crew) is not int or not 1 <= self.tasks_per_crew <= 40:
            raise ValueError('tasks_per_crew must be an integer from 1 to 40')
        if self.kill_range > self.sight_range or self.report_range > self.sight_range:
            raise ValueError('Interaction ranges must fit within sight range')


@dataclass
class Task:
    task_id: str
    console_id: str
    duration: float
    progress: float = 0.0


@dataclass
class Player:
    player_id: str
    color_name: str
    role: Role
    motion: AmongUsMapEnv
    navigator: NavigationService
    status: str = 'alive'
    velocity: tuple[float, float] = (0., 0.)
    tasks: list[Task] = field(default_factory=list)
    intention: str = 'wait'
    navigation_target: str | None = None
    interaction_task: str | None = None
    interaction_remaining: float = 0.0
    cooldown: float = 0.0
    meetings_remaining: int = 1

    @property
    def position(self):
        return tuple(float(v) for v in self.motion.position)

    @property
    def active(self):
        return self.status == 'alive'


@dataclass
class Body:
    victim_id: str
    position: tuple[float, float]
    death_tick: int
    killer_id: str
    reported: bool = False


@dataclass(frozen=True)
class MatchResult:
    winner: Role | None
    reason: str
    simulation_time: float


def json_value(value):
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, '__dataclass_fields__'):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


class Phase3Game:
    """Fixed-tick rules/physics with an explicit observation boundary.

    step accepts intentions; no controller callback is stored here. The trusted
    runner samples observations before collecting any player's action.
    """
    def __init__(self, seed=0, config=None, planner=None):
        self.config = config or GameConfig()
        self.map = planner.map if planner is not None else get_map()
        self.planner = planner or NavigationPlanner(self.map, cache_size=96)
        self.settings = ObservationSettings(sight_range=self.config.sight_range,
            report_range=self.config.report_range, kill_range=self.config.kill_range, dt=self.config.dt)
        self.public_map = PublicMap(tuple(sorted(name for name, _ in self.map.regions)), (), (),
            tuple(Console(d.id, d.room, Point(*map(float, d.position)), d.name == 'EmergencyConsole')
                  for d in sorted(self.map.destinations, key=lambda d: d.id)),
            self.config.sight_range, self.config.report_range)
        self.reset(seed)

    def reset(self, seed=0):
        self.seed = int(seed)
        self.tick = 0
        self.phase = Phase.ROAMING
        self.phase_started_tick = 0
        self.result = None
        self.players = {}
        self.bodies = {}
        self.tasks = {}
        self.publications = []
        self.event_log = []
        self.direct_events = {f'p{i}': [] for i in range(5)}
        self._seen = {f'p{i}': set() for i in range(5)}
        self._observation_cache = {}
        self._snapshot_cache = None
        self._meeting_number = 0
        self.meeting_id = None
        self.participants = ()
        self.votes = {}
        self.claimed = set()
        self.metrics = {k: 0 for k in ('steps', 'eliminations', 'reports', 'meetings', 'votes',
            'task_completions', 'illegal_actions', 'navigation_failures')}
        self.metrics.update({k: 0.0 for k in ('navigation_seconds', 'visibility_seconds',
                                            'event_log_seconds', 'physics_seconds')})
        identities = assign_identities(5, seed=stream_seed(seed, 'identities'))
        roles = assign_roles(5, seed=stream_seed(seed, 'roles'))
        colors = ['Red', 'Blue', 'Green', 'Yellow', 'Purple', 'Orange', 'Black', 'Pink', 'White', 'Brown']
        Random(stream_seed(seed, 'colors')).shuffle(colors)
        spawn_points = [(-.7, -2.8), (-1.4, -2.8), (0., -2.8), (-2.1, -2.8), (-.7, -3.5)]
        destinations = sorted((d for d in self.map.destinations
                               if d.category == 'task' and d.name != 'VentCleaning'), key=lambda d: d.id)
        self._record('match_start', seed=self.seed, config=asdict(self.config), rules_version=1)
        for slot, (identity, role, color, position) in enumerate(zip(identities, roles, colors, spawn_points)):
            motion = AmongUsMapEnv(speed=self.config.speed)
            if motion.map is not self.map:
                motion.map = self.map  # Trusted immutable geometry, no alternate movement model.
            motion.spawn(position)
            navigator = NavigationService(self.planner, ControllerConfig(speed=self.config.speed))
            player = Player(identity.player_id, color, role, motion, navigator,
                            cooldown=self.config.initial_kill_cooldown if role is Role.IMPOSTOR else 0.)
            # Hypothetical assignments are keyed by public ID, independent of role
            # and which other players receive tasks. Never sequentially skip an
            # impostor slot in the task RNG or task-ID allocation.
            rng = Random(stream_seed(seed, 'tasks:' + player.player_id))
            selected = rng.sample(destinations, self.config.tasks_per_crew)
            if role is Role.CREWMATE:
                for number, destination in enumerate(selected):
                    identifier = hashlib.sha256(f'{stream_seed(seed,"task_ids")}:{player.player_id}:{number}'.encode()).hexdigest()[:16]
                    task = Task('t-' + identifier, destination.id, self.config.task_seconds)
                    player.tasks.append(task)
                    self.tasks[task.task_id] = task
            self.players[player.player_id] = player
            self._record('spawn', player_id=player.player_id, color=color, role=role,
                         position=position, internal_slot=slot)
        self.players = dict(sorted(self.players.items()))
        self.initial_task_count = len(self.tasks)
        self.assert_invariants()
        return self

    @property
    def time(self):
        return round(self.tick * self.config.dt, 9)

    @property
    def is_terminal(self):
        return self.result is not None

    @property
    def context(self):
        speaker = None
        if self.phase is Phase.DISCUSSION and self.participants:
            duration_ticks = max(1, math.ceil(self.config.discussion_seconds / self.config.dt))
            index = min(len(self.participants) - 1,
                        (self.tick - self.phase_started_tick) * len(self.participants) // duration_ticks)
            candidate = self.participants[index]
            if candidate not in self.claimed:
                speaker = candidate
        return PublicContext(self.phase, self.meeting_id, self.participants, speaker, tuple(sorted(self.votes)))

    def _invalidate(self):
        self._observation_cache.clear()
        self._snapshot_cache = None

    def snapshot(self):
        if self._snapshot_cache is None:
            players = tuple(PlayerTruth(Identity(p.player_id, p.color_name), p.role,
                Point(*p.position), self.map.region(p.position), p.active, Point(*p.velocity),
                p.interaction_task is not None,
                tuple(TaskTruth(t.task_id, t.console_id, t.progress) for t in p.tasks),
                p.cooldown, p.meetings_remaining) for p in self.players.values())
            bodies = tuple(BodyTruth(b.victim_id, Point(*b.position), self.map.region(b.position),
                b.death_tick, b.killer_id) for b in self.bodies.values() if not b.reported)
            # Opaque constant is not a seed/role fingerprint. A dataset runner can
            # attach its own public match UUID outside this seeded simulation.
            self._snapshot_cache = WorldTruth('match', self.tick, self.public_map, players, self.context, bodies)
        return self._snapshot_cache

    def observe(self, player_id):
        if player_id not in self.players:
            raise ValueError('Unknown player')
        if player_id not in self._observation_cache:
            started = perf_counter()
            p = self.players[player_id]
            observation = project_game(self.snapshot(), player_id, self.map, self.planner, self.settings,
                navigation_status=p.navigator.status.value, navigation_target=p.navigation_target,
                interaction_task=p.interaction_task, publications=tuple(self.publications),
                direct_events=tuple(self.direct_events[player_id]))
            self._observation_cache[player_id] = observation
            self.metrics['visibility_seconds'] += perf_counter() - started
        return self._observation_cache[player_id]

    def legal_actions(self, player_id):
        return self.observe(player_id).actions

    def _record(self, kind, *, channel='truth', **data):
        started = perf_counter()
        self.event_log.append({'tick': self.tick, 'time': self.time, 'kind': kind,
                               'channel': channel, **json_value(data)})
        self.metrics['event_log_seconds'] += perf_counter() - started

    def _publish(self, payload, kind, **data):
        provenance = Provenance.CLAIM if isinstance(payload, StructuredClaim) else Provenance.PUBLIC
        self.publications.append(EventView(self.tick, provenance, payload))
        self._record(kind, channel='public', **data)
        self._invalidate()

    def record_observations(self, observations):
        """Trusted runner logs appearance transitions, never exposes the truth log."""
        for pid, observation in sorted(observations.items()):
            current = set()
            for seen in observation.visible_players:
                key = ('player', seen.player_id, seen.room, seen.interacting)
                current.add(key)
                if key not in self._seen[pid]:
                    self._record('direct_sighting', channel='direct', observer_id=pid,
                                 player_id=seen.player_id, position=seen.position, room=seen.room,
                                 interacting=seen.interacting)
            for body in observation.visible_bodies:
                key = ('body', body.victim_id)
                current.add(key)
                if key not in self._seen[pid]:
                    self._record('body_observed', channel='direct', observer_id=pid,
                                 victim_id=body.victim_id, position=body.position, room=body.room)
            self._seen[pid] = current

    def _interrupt(self, p, reason):
        p.navigator.cancel()
        p.navigation_target = None
        p.velocity = (0., 0.)
        p.intention = 'wait'
        if p.interaction_task:
            self._record('task_interrupted', player_id=p.player_id,
                         task_id=p.interaction_task, reason=reason)
        p.interaction_task = None
        p.interaction_remaining = 0.

    def _navigate(self, p, target, description):
        self._interrupt(p, 'new_intention')
        p.navigation_target = description
        p.intention = description
        started = perf_counter()
        status = p.navigator.navigate(p.position, target)
        self.metrics['navigation_seconds'] += perf_counter() - started
        self._record('navigate', player_id=p.player_id, target=description,
                     position=p.position, status=status.value)
        if status not in (NavStatus.MOVING, NavStatus.SUCCESS):
            self.metrics['navigation_failures'] += 1
            self._record('navigation_failed', player_id=p.player_id, status=status.value)

    def _eliminate(self, killer, victim):
        for observer in self.players.values():
            if observer.active and visible_between(self.map, observer.position, killer.position, self.config.sight_range) and visible_between(self.map, observer.position, victim.position, self.config.sight_range):
                event = EventView(self.tick, Provenance.DIRECT,
                    WitnessedElimination(killer.player_id, victim.player_id, Point(*victim.position), self.map.region(victim.position)))
                self.direct_events[observer.player_id].append(event)
                self._record('elimination_witnessed', channel='direct', observer_id=observer.player_id,
                             killer_id=killer.player_id, victim_id=victim.player_id)
        self._interrupt(victim, 'eliminated')
        victim.status = 'dead'
        self.bodies[victim.player_id] = Body(victim.player_id, victim.position, self.tick, killer.player_id)
        killer.cooldown = self.config.kill_cooldown
        self.metrics['eliminations'] += 1
        self._record('elimination', killer_id=killer.player_id, victim_id=victim.player_id, position=victim.position)
        self._invalidate()
        self._check_win()

    def _reassign_publicly_absent_tasks(self):
        recipients = sorted((p for p in self.players.values() if p.active and p.role is Role.CREWMATE), key=lambda p: p.player_id)
        if not recipients:
            return
        for former in self.players.values():
            if former.active:
                continue
            for task in tuple(former.tasks):
                if task.progress >= 1:
                    continue
                recipient = min(recipients, key=lambda p: (sum(t.progress < 1 for t in p.tasks), p.player_id))
                former.tasks.remove(task)
                recipient.tasks.append(task)
                self._record('task_reassigned', task_id=task.task_id, recipient_id=recipient.player_id)

    def _start_meeting(self, reporter, body=None):
        if self.phase is not Phase.ROAMING or not reporter.active:
            return
        if body is not None:
            self.metrics['reports'] += 1
            self._publish(Report(reporter.player_id, body.victim_id, self.map.region(body.position)),
                'report', reporter_id=reporter.player_id, victim_id=body.victim_id, room=self.map.region(body.position))
        else:
            reporter.meetings_remaining -= 1
            self._record('emergency_called', channel='public', reporter_id=reporter.player_id)
        self._record('report_transition', channel='public')
        for p in self.players.values():
            self._interrupt(p, 'meeting')
        self._meeting_number += 1
        self.meeting_id = f'm{self._meeting_number}'
        self.participants = tuple(p.player_id for p in self.players.values() if p.active)
        self.claimed = set()
        self.votes = {}
        self.phase = Phase.DISCUSSION
        self.phase_started_tick = self.tick
        self.metrics['meetings'] += 1
        self._publish(Meeting(self.meeting_id, self.participants), 'meeting_started',
                      meeting_id=self.meeting_id, participants=self.participants)
        # Only now does public absence authorize private orphan reassignment.
        self._reassign_publicly_absent_tasks()
        self.bodies.clear()  # A meeting resolves all bodies, including unreported ones.
        self._invalidate()

    def _vote(self, voter_id, target_id):
        if voter_id in self.votes:
            return
        self.votes[voter_id] = target_id
        self.metrics['votes'] += 1
        self._publish(Vote(self.meeting_id, voter_id, target_id), 'vote',
                      meeting_id=self.meeting_id, voter_id=voter_id, target_id=target_id)

    @staticmethod
    def tally_votes(votes):
        """Unique player plurality must strictly beat skip; ties never eject."""
        counts = {}
        for target in votes.values():
            counts[target] = counts.get(target, 0) + 1
        if not counts:
            return None
        highest = max(counts.values())
        winners = [target for target, count in counts.items() if count == highest]
        return winners[0] if len(winners) == 1 and winners[0] is not None else None

    def _resolve_meeting(self):
        self._record('meeting_resolution', channel='public', meeting_id=self.meeting_id)
        target = self.tally_votes(self.votes)
        if target is not None:
            p = self.players[target]
            self._interrupt(p, 'ejected')
            p.status = 'ejected'
        self._publish(Ejection(self.meeting_id, target), 'ejection', meeting_id=self.meeting_id, player_id=target)
        if self._check_win():
            return
        self._reassign_publicly_absent_tasks()
        self.phase = Phase.ROAMING
        self.phase_started_tick = self.tick
        self._record('exploration_resumed', channel='public', meeting_id=self.meeting_id)
        self.meeting_id = None
        self.participants = ()
        self.votes = {}
        self.claimed = set()
        self._invalidate()

    def _finish(self, winner, reason):
        if self.is_terminal:
            return
        for p in self.players.values():
            self._interrupt(p, 'match_finished')
        self.result = MatchResult(winner, reason, self.time)
        self.phase = Phase.FINISHED
        self._publish(MatchEnded(winner), 'match_end', winner=winner, reason=reason)

    def _check_win(self):
        living = [p for p in self.players.values() if p.active]
        crew = sum(p.role is Role.CREWMATE for p in living)
        impostors = sum(p.role is Role.IMPOSTOR for p in living)
        if impostors == 0:
            self._finish(Role.CREWMATE, 'IMPOSTOR_EJECTED')
        elif impostors >= crew:
            self._finish(Role.IMPOSTOR, 'PARITY')
        elif all(t.progress >= 1 for t in self.tasks.values()):
            self._finish(Role.CREWMATE, 'TASKS_COMPLETED')
        return self.is_terminal

    def _apply_exploration(self, p, action, observation):
        kind = action.kind
        if kind is IntentKind.CANCEL_NAVIGATION:
            self._interrupt(p, 'cancelled')
        elif kind is IntentKind.GO_TO_TASK:
            self._navigate(p, NavTarget.task(action.target), action.target)
        elif kind is IntentKind.GO_TO_ROOM:
            self._navigate(p, NavTarget.room(action.target), 'room:' + action.target)
        elif kind is IntentKind.GO_TO_LOCATION:
            self._navigate(p, NavTarget.location((action.position.x, action.position.y)), 'location')
        elif kind is IntentKind.GO_TO_BODY:
            body = next(b for b in observation.visible_bodies if b.victim_id == action.target)
            self._navigate(p, NavTarget.location((body.position.x, body.position.y), radius=.3), 'body:' + body.victim_id)
        elif kind is IntentKind.INTERACT:
            task = next(t for t in p.tasks if t.task_id == action.target)
            if p.interaction_task != task.task_id:
                self._interrupt(p, 'task_start')
                p.interaction_task = task.task_id
                p.intention = 'task:' + task.console_id
                self._record('task_start', player_id=p.player_id, task_id=task.task_id, console_id=task.console_id)
        elif kind is IntentKind.FAKE_TASK:
            self._interrupt(p, 'fake_task_start')
            p.interaction_task = 'fake:' + action.target
            p.interaction_remaining = self.config.fake_task_seconds
            p.intention = 'task:' + action.target
            self._record('fake_task_start', player_id=p.player_id, console_id=action.target)

    def step(self, actions=None):
        if self.is_terminal:
            return self.result
        actions = actions or {}
        self.metrics['steps'] += 1
        # Snapshot every proposed actor before applying any action. In particular,
        # iteration cannot give a later player knowledge of an earlier action.
        observations = {pid: self.observe(pid) for pid in sorted(actions) if pid in self.players}
        valid = {}
        for pid, action in sorted(actions.items()):
            if action is None or (isinstance(action, Intent) and action.kind is IntentKind.WAIT):
                continue  # No-op remains harmless in any phase.
            if pid not in observations or not intent_allowed(observations[pid], action):
                self.metrics['illegal_actions'] += 1
                self._record('illegal_action', player_id=pid)
            else:
                valid[pid] = action
        if self.phase is Phase.ROAMING:
            # Reports/emergencies at tick-start precede eliminations. Stable public
            # ID breaks simultaneous-report ties, without consulting roles.
            for pid, action in valid.items():
                if action.kind is IntentKind.REPORT:
                    self._start_meeting(self.players[pid], self.bodies[action.target])
                    break
                if action.kind is IntentKind.CALL_MEETING:
                    self._start_meeting(self.players[pid])
                    break
            if self.phase is Phase.ROAMING:
                for pid, action in valid.items():
                    if action.kind is IntentKind.KILL:
                        self._eliminate(self.players[pid], self.players[action.target])
                        if self.is_terminal:
                            return self.result
                for pid, action in valid.items():
                    if self.players[pid].active:
                        self._apply_exploration(self.players[pid], action, observations[pid])
        elif self.phase is Phase.DISCUSSION:
            for pid, action in valid.items():
                if action.kind is IntentKind.CLAIM:
                    draft = action.claim
                    claim = StructuredClaim(draft.kind, pid, draft.subject_id, draft.region, self.tick, draft.asserted_tick)
                    self.claimed.add(pid)
                    self._publish(claim, 'claim', claim_kind=draft.kind, speaker_id=pid,
                                  subject_id=draft.subject_id, region=draft.region, asserted_tick=draft.asserted_tick)
        elif self.phase is Phase.VOTING:
            for pid, action in valid.items():
                if action.kind in (IntentKind.VOTE, IntentKind.SKIP):
                    self._vote(pid, action.target if action.kind is IntentKind.VOTE else None)

        if self.phase is Phase.ROAMING:
            self._advance_physics()
        self.tick += 1
        self._invalidate()
        if self.phase is Phase.ROAMING:
            self._advance_tasks()
            self._check_win()
        elif self.phase is Phase.DISCUSSION:
            if self.tick - self.phase_started_tick >= math.ceil(self.config.discussion_seconds / self.config.dt):
                self.phase = Phase.VOTING
                self.phase_started_tick = self.tick
                self._record('voting_started', channel='public', meeting_id=self.meeting_id)
        elif self.phase is Phase.VOTING:
            if self.tick - self.phase_started_tick >= math.ceil(self.config.voting_seconds / self.config.dt):
                for pid in self.participants:
                    if pid not in self.votes:
                        self._vote(pid, None)
                # Keep the public voting window open for its fixed duration so
                # cast votes can be inspected; no policy can cast a second vote.
                self._resolve_meeting()
        if not self.is_terminal and self.tick >= math.ceil(self.config.timeout / self.config.dt):
            self._finish(None, 'TIMEOUT')
        self._invalidate()
        if self.config.check_invariants:
            self.assert_invariants()
        return self.result

    def _advance_physics(self):
        for p in self.players.values():
            if not p.active:
                continue
            p.cooldown = round(max(0., p.cooldown - self.config.dt), 9)
            old = p.position
            previous_status = p.navigator.status
            started = perf_counter()
            displacement = p.navigator.command(old, self.config.dt)
            self.metrics['navigation_seconds'] += perf_counter() - started
            if p.navigator.awaiting_feedback:
                started = perf_counter()
                new, hit = p.motion.advance_motion(displacement, self.config.dt)
                self.metrics['physics_seconds'] += perf_counter() - started
                started = perf_counter()
                p.navigator.feedback(new, hit)
                self.metrics['navigation_seconds'] += perf_counter() - started
            p.velocity = tuple((new - old_value) / self.config.dt for new, old_value in zip(p.position, old))
            if previous_status is NavStatus.MOVING and p.navigator.status not in (NavStatus.MOVING, NavStatus.SUCCESS):
                self.metrics['navigation_failures'] += 1
                self._record('navigation_failed', player_id=p.player_id, status=p.navigator.status.value)

    def _advance_tasks(self):
        for p in self.players.values():
            if not p.active or p.interaction_task is None:
                continue
            if p.interaction_task.startswith('fake:'):
                console_id = p.interaction_task[5:]
                if not self.planner.arrival_region(NavTarget.task(console_id)).contains(p.position):
                    self._interrupt(p, 'left_interaction_region')
                    continue
                p.interaction_remaining = max(0., p.interaction_remaining - self.config.dt)
                if p.interaction_remaining <= 1e-9:
                    self._record('fake_task_end', player_id=p.player_id, console_id=console_id)
                    p.interaction_task = None
                    p.intention = 'wait'
                continue
            task = self.tasks[p.interaction_task]
            if not self.planner.arrival_region(NavTarget.task(task.console_id)).contains(p.position):
                self._interrupt(p, 'left_interaction_region')
                continue
            task.progress = min(1., task.progress + self.config.dt / task.duration)
            if task.progress >= 1 - 1e-10:
                task.progress = 1.
                self.metrics['task_completions'] += 1
                self._record('task_completed', player_id=p.player_id, task_id=task.task_id, console_id=task.console_id)
                p.interaction_task = None
                p.intention = 'wait'

    def assert_invariants(self):
        assert len(self.players) == 5
        assert sum(p.role is Role.IMPOSTOR for p in self.players.values()) == 1
        assert sum(p.role is Role.CREWMATE for p in self.players.values()) == 4
        assert len(self.tasks) == self.initial_task_count
        assigned = [t.task_id for p in self.players.values() for t in p.tasks]
        assert len(assigned) == len(set(assigned)) and set(assigned) == set(self.tasks)
        assert all(0 <= t.progress <= 1 for t in self.tasks.values())
        assert all(self.map.contains(p.position) for p in self.players.values())
        assert all(p.status in ('alive', 'dead', 'ejected') for p in self.players.values())
        assert all(p.navigator.status is not NavStatus.MOVING and p.interaction_task is None
                   for p in self.players.values() if not p.active or self.phase is not Phase.ROAMING)
        assert all(not self.players[b.victim_id].active and b.victim_id == key for key, b in self.bodies.items())
        assert set(self.votes) <= set(self.participants)
        assert all(target is None or target in self.participants for target in self.votes.values())
        assert self.is_terminal == (self.phase is Phase.FINISHED)

    def write_replay(self, path):
        """Privileged replay for audit only, never a controller input."""
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as handle:
            for event in self.event_log:
                handle.write(json.dumps(event, sort_keys=True, separators=(',', ':')) + '\n')
        return path
