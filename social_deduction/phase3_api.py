"""Immutable actor-facing Phase 3 values; no simulator or training dependency.

The Phase 1 fixture API remains frozen. This additive contract supports native
Skeld geometry, role-local impostor controls and a finite meeting protocol.
"""

from dataclasses import asdict, dataclass, fields
from enum import Enum
from functools import lru_cache
import math
from types import UnionType
from typing import get_args, get_origin, get_type_hints

from .actor import (
    BodySeen, Console, Ejection, Identity, MatchEnded, Meeting, Phase,
    PlayerSeen, Point, Provenance, PublicContext, PublicMap, Report, Role,
    SelfView, TaskView, Vote, Wall,
)


class IntentKind(str, Enum):
    WAIT = 'wait'
    GO_TO_TASK = 'go_to_task'
    GO_TO_ROOM = 'go_to_room'
    GO_TO_LOCATION = 'go_to_location'
    GO_TO_BODY = 'go_to_body'
    CANCEL_NAVIGATION = 'cancel_navigation'
    INTERACT = 'interact'
    FAKE_TASK = 'fake_task'
    KILL = 'kill'
    REPORT = 'report'
    CALL_MEETING = 'call_meeting'
    CLAIM = 'claim'
    VOTE = 'vote'
    SKIP = 'skip'


class ClaimKind(str, Enum):
    SAW_PLAYER = 'saw_player'
    WAS_IN_REGION = 'was_in_region'
    FOUND_BODY = 'found_body'
    SAW_PLAYER_NEAR_BODY = 'saw_player_near_body'
    SUSPECT_PLAYER = 'suspect_player'
    DEFEND_PLAYER = 'defend_player'
    WAS_AT_TASK = 'was_at_task'
    SAW_ELIMINATION = 'saw_elimination'


@dataclass(frozen=True, slots=True)
class ClaimDraft:
    kind: ClaimKind
    subject_id: str | None
    region: str
    asserted_tick: int


@dataclass(frozen=True, slots=True)
class StructuredClaim:
    kind: ClaimKind
    speaker_id: str
    subject_id: str | None
    region: str
    delivered_tick: int
    asserted_tick: int


@dataclass(frozen=True, slots=True)
class WitnessedElimination:
    killer_id: str
    victim_id: str
    position: Point
    region: str


EventPayload = (PlayerSeen | BodySeen | Report | Meeting | Vote | Ejection |
                MatchEnded | StructuredClaim | WitnessedElimination)


@dataclass(frozen=True, slots=True)
class EventView:
    tick: int
    provenance: Provenance
    payload: EventPayload


@dataclass(frozen=True, slots=True)
class Intent:
    kind: IntentKind
    target: str | None = None
    position: Point | None = None
    claim: ClaimDraft | None = None


@dataclass(frozen=True, slots=True)
class ObservationSettings:
    sight_range: float = 4.5
    report_range: float = 1.3
    kill_range: float = 1.1
    dt: float = .2

    def __post_init__(self):
        for value in (self.sight_range, self.report_range, self.kill_range, self.dt):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('Observation distances and timestep must be positive and finite')


@dataclass(frozen=True, slots=True)
class DestinationView:
    id: str
    name: str
    room: str
    position: Point
    standing: Point
    radius: float
    category: str


@dataclass(frozen=True, slots=True)
class Phase3Observation:
    match_id: str
    tick: int
    time_seconds: float
    own: SelfView
    roster: tuple[Identity, ...]
    map: PublicMap
    context: PublicContext
    visible_players: tuple[PlayerSeen, ...]
    visible_bodies: tuple[BodySeen, ...]
    evidence: tuple[EventView, ...]
    destinations: tuple[DestinationView, ...]
    settings: ObservationSettings
    interactable_tasks: tuple[str, ...] = ()
    interactable_destinations: tuple[str, ...] = ()
    kill_cooldown: float | None = None
    navigation_status: str = 'idle'
    navigation_target: str | None = None
    interaction_task: str | None = None
    actions: tuple[Intent, ...] = ()


_VALUE_TYPES = frozenset((
    Point, Wall, Console, PublicMap, Identity, TaskView, SelfView, PlayerSeen,
    BodySeen, Report, Meeting, Vote, Ejection, MatchEnded, PublicContext,
    ClaimDraft, StructuredClaim, WitnessedElimination, EventView, Intent,
    ObservationSettings, DestinationView, Phase3Observation,
))


@lru_cache(maxsize=None)
def _annotations(cls):
    return get_type_hints(cls)


def _validate(value, expected):
    """Exact-type allowlist: reject subclasses, mutable containers and metadata."""
    origin, args = get_origin(expected), get_args(expected)
    if origin is UnionType:
        for option in args:
            try:
                _validate(value, option)
                return
            except TypeError:
                pass
        raise TypeError(f'Invalid union member: {type(value).__name__}')
    if origin is tuple:
        if type(value) is not tuple:
            raise TypeError('Actor collections must be tuples')
        if len(args) == 2 and args[1] is Ellipsis:
            for item in value:
                _validate(item, args[0])
        else:
            if len(value) != len(args):
                raise TypeError('Invalid tuple shape')
            for item, item_type in zip(value, args):
                _validate(item, item_type)
    elif expected is float:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise TypeError('Expected finite number')
    elif type(value) is not expected:
        raise TypeError(f'Expected {expected.__name__}, got {type(value).__name__}')
    elif expected in _VALUE_TYPES:
        annotations = _annotations(expected)
        for field in fields(value):
            _validate(getattr(value, field.name), annotations[field.name])


def validate_event(event: EventView, tick: int) -> None:
    _validate(event, EventView)
    if not 0 <= event.tick <= tick:
        raise ValueError('Evidence is outside observation time')
    expected = (Provenance.CLAIM if type(event.payload) is StructuredClaim else
                Provenance.DIRECT if type(event.payload) in
                (PlayerSeen, BodySeen, WitnessedElimination) else Provenance.PUBLIC)
    if event.provenance is not expected:
        raise ValueError('Evidence provenance does not match payload')
    if type(event.payload) is StructuredClaim and event.payload.delivered_tick != event.tick:
        raise ValueError('Claim delivery time must match its public envelope')


def validate_observation(observation: Phase3Observation) -> None:
    _validate(observation, Phase3Observation)
    if observation.tick < 0 or observation.time_seconds < 0:
        raise ValueError('Negative observation time')
    if observation.own.role is Role.CREWMATE and observation.kill_cooldown is not None:
        raise ValueError('Crew observations cannot carry kill cooldowns')
    if observation.own.role is Role.IMPOSTOR and observation.kill_cooldown is None:
        raise ValueError('Impostors require their own cooldown')
    if observation.kill_cooldown is not None and observation.kill_cooldown < 0:
        raise ValueError('Negative own cooldown')
    for event in observation.evidence:
        validate_event(event, observation.tick)
    own_task_ids = {task.task_id for task in observation.own.tasks if task.progress < 1}
    destination_ids = {destination.id for destination in observation.destinations}
    if not set(observation.interactable_tasks) <= own_task_ids:
        raise ValueError('Interaction candidates must reference incomplete own tasks')
    if not set(observation.interactable_destinations) <= destination_ids:
        raise ValueError('Interaction candidates must reference public destinations')


def validate_intent(intent: Intent) -> None:
    _validate(intent, Intent)
    if intent.kind is IntentKind.CLAIM:
        if intent.claim is None:
            raise ValueError('A submitted claim requires a typed draft')
    elif intent.claim is not None:
        raise ValueError('Only claim actions carry a draft')
    if intent.kind is IntentKind.GO_TO_LOCATION:
        if intent.position is None:
            raise ValueError('A location action requires a finite position')
    elif intent.position is not None:
        raise ValueError('Only location actions carry a position')


def legal_actions(observation: Phase3Observation) -> tuple[Intent, ...]:
    """Derive every option from actor-local, observed or public immutable values.

    CLAIM and GO_TO_LOCATION denote parameterized action families. The submitted
    intention must supply a typed draft/position, validated separately by engine.
    """
    own, context = observation.own, observation.context
    if not own.active or context.phase is Phase.FINISHED:
        return ()
    if context.phase is Phase.DISCUSSION:
        return ((Intent(IntentKind.CLAIM),) if context.speaker_id == own.player_id
                and own.player_id in context.participants else ())
    if context.phase is Phase.VOTING:
        if own.player_id not in context.participants or own.player_id in context.voted_ids:
            return ()
        return tuple(Intent(IntentKind.VOTE, player_id) for player_id in
                     sorted(context.participants)) + (Intent(IntentKind.SKIP),)

    options = [Intent(IntentKind.WAIT), Intent(IntentKind.CANCEL_NAVIGATION),
               Intent(IntentKind.GO_TO_LOCATION)]
    options.extend(Intent(IntentKind.GO_TO_ROOM, room) for room in sorted(observation.map.rooms))
    options.extend(Intent(IntentKind.GO_TO_TASK, destination.id)
                   for destination in sorted(observation.destinations, key=lambda d: d.id))
    options.extend(Intent(IntentKind.GO_TO_BODY, body.victim_id)
                   for body in sorted(observation.visible_bodies, key=lambda b: b.victim_id))
    options.extend(Intent(IntentKind.INTERACT, task_id)
                   for task_id in sorted(observation.interactable_tasks))
    nearby = set(observation.interactable_destinations)
    if own.role is Role.IMPOSTOR:
        options.extend(Intent(IntentKind.FAKE_TASK, destination.id)
                       for destination in sorted(observation.destinations, key=lambda d: d.id)
                       if destination.id in nearby and destination.category == 'task')
        if observation.kill_cooldown == 0:
            options.extend(Intent(IntentKind.KILL, player.player_id)
                           for player in sorted(observation.visible_players, key=lambda p: p.player_id)
                           if math.hypot(own.position.x - player.position.x,
                                         own.position.y - player.position.y) <= observation.settings.kill_range)
    options.extend(Intent(IntentKind.REPORT, body.victim_id)
                   for body in sorted(observation.visible_bodies, key=lambda b: b.victim_id)
                   if math.hypot(own.position.x - body.position.x,
                                 own.position.y - body.position.y) <= observation.settings.report_range)
    if own.meetings_remaining > 0:
        options.extend(Intent(IntentKind.CALL_MEETING, destination.id)
                       for destination in sorted(observation.destinations, key=lambda d: d.id)
                       if destination.id in nearby and destination.name == 'EmergencyConsole')
    return tuple(options)


def observation_to_dict(observation: Phase3Observation) -> dict:
    """Serializable actor packet; this function cannot accept a truth snapshot."""
    validate_observation(observation)
    return asdict(observation)


def intent_allowed(observation: Phase3Observation, intent: Intent) -> bool:
    """Pure candidate check; stale actions may still fail safely at execution.

    Claim syntax and public time/identity vocabulary are checked, never whether
    its alleged event really happened. Geometry bounds are an engine check for
    parameterized locations, based on public static geometry alone.
    """
    try:
        validate_intent(intent)
    except (TypeError, ValueError):
        return False
    options = legal_actions(observation)
    if intent.kind is IntentKind.GO_TO_LOCATION:
        return intent.target is None and Intent(IntentKind.GO_TO_LOCATION) in options
    if intent.kind is IntentKind.CLAIM:
        if intent.target is not None or Intent(IntentKind.CLAIM) not in options:
            return False
        claim = intent.claim
        player_ids = {identity.player_id for identity in observation.roster}
        regions = set(observation.map.rooms) | {destination.room for destination in observation.destinations}
        regions.update(('', 'Hallway'))
        return ((claim.subject_id is None or claim.subject_id in player_ids)
                and claim.region in regions and 0 <= claim.asserted_tick <= observation.tick)
    return intent in options
