"""Actor-safe immutable values. Never import truth or training into this module."""

from dataclasses import dataclass, fields
from enum import Enum
import math
from types import UnionType
from typing import get_args, get_origin, get_type_hints


class Role(str, Enum):
    CREWMATE = "crewmate"
    IMPOSTOR = "impostor"


class Phase(str, Enum):
    ROAMING = "roaming"
    DISCUSSION = "discussion"
    VOTING = "voting"
    FINISHED = "finished"


class Provenance(str, Enum):
    DIRECT = "direct"
    PUBLIC = "public"
    CLAIM = "claim"


class ActionKind(str, Enum):
    WAIT = "wait"
    NAVIGATE = "navigate"
    FOLLOW = "follow"
    INTERACT = "interact"
    REPORT = "report"
    CALL_MEETING = "call_meeting"
    CLAIM = "claim"
    VOTE = "vote"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Wall:
    left: float
    top: float
    right: float
    bottom: float


@dataclass(frozen=True, slots=True)
class Console:
    console_id: str
    room: str
    position: Point
    meeting: bool = False


@dataclass(frozen=True, slots=True)
class PublicMap:
    rooms: tuple[str, ...]
    connections: tuple[tuple[str, str], ...]
    walls: tuple[Wall, ...]
    consoles: tuple[Console, ...]
    sight_range: float = 220.0
    interaction_range: float = 25.0


@dataclass(frozen=True, slots=True)
class Identity:
    player_id: str
    display_name: str


@dataclass(frozen=True, slots=True)
class TaskView:
    task_id: str
    console_id: str
    progress: float


@dataclass(frozen=True, slots=True)
class SelfView:
    player_id: str
    role: Role
    position: Point
    room: str
    active: bool
    tasks: tuple[TaskView, ...]
    meetings_remaining: int


@dataclass(frozen=True, slots=True)
class PlayerSeen:
    player_id: str
    position: Point
    room: str
    velocity: Point
    interacting: bool


@dataclass(frozen=True, slots=True)
class BodySeen:
    victim_id: str
    position: Point
    room: str


@dataclass(frozen=True, slots=True)
class Report:
    reporter_id: str
    victim_id: str
    room: str


@dataclass(frozen=True, slots=True)
class Meeting:
    meeting_id: str
    participants: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Claim:
    speaker_id: str
    subject_id: str
    room: str
    claimed_tick: int


@dataclass(frozen=True, slots=True)
class Vote:
    meeting_id: str
    voter_id: str
    target_id: str | None  # None is a public skip.


@dataclass(frozen=True, slots=True)
class Ejection:
    meeting_id: str
    player_id: str | None  # No role; None means nobody ejected.


@dataclass(frozen=True, slots=True)
class MatchEnded:
    winner: Role | None  # None is a draw; no role roster is disclosed.


PublicPayload = Report | Meeting | Claim | Vote | Ejection | MatchEnded


@dataclass(frozen=True, slots=True)
class Evidence:
    observed_tick: int
    provenance: Provenance
    payload: PlayerSeen | BodySeen | PublicPayload


@dataclass(frozen=True, slots=True)
class PublicContext:
    phase: Phase = Phase.ROAMING
    meeting_id: str | None = None
    participants: tuple[str, ...] = ()
    speaker_id: str | None = None
    voted_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    target: str | None = None


@dataclass(frozen=True, slots=True)
class ActorObservation:
    match_id: str
    tick: int
    own: SelfView
    roster: tuple[Identity, ...]
    map: PublicMap
    context: PublicContext
    visible_players: tuple[PlayerSeen, ...]
    visible_bodies: tuple[BodySeen, ...]
    evidence: tuple[Evidence, ...]
    actions: tuple[Action, ...] = ()


@dataclass(frozen=True, slots=True)
class Knowledge:
    match_id: str
    player_id: str
    history: tuple[ActorObservation, ...] = ()


def _validate(value, expected):
    """Reject mutable/extra/privileged payloads, including subclasses with fields."""
    origin = get_origin(expected)
    args = get_args(expected)
    if origin is UnionType:
        for option in args:
            try:
                _validate(value, option)
                return
            except TypeError:
                pass
        raise TypeError(f"Invalid union member: {type(value).__name__}")
    if origin is tuple:
        if type(value) is not tuple:
            raise TypeError("Actor collections must be tuples")
        if len(args) == 2 and args[1] is Ellipsis:
            for item in value:
                _validate(item, args[0])
        else:
            if len(value) != len(args):
                raise TypeError("Invalid tuple shape")
            for item, item_type in zip(value, args):
                _validate(item, item_type)
    elif expected is float:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise TypeError("Expected finite number")
    elif type(value) is not expected:
        raise TypeError(f"Expected {expected.__name__}, got {type(value).__name__}")
    elif expected in _VALUE_TYPES:
        annotations = get_type_hints(expected)
        for field in fields(value):
            _validate(getattr(value, field.name), annotations[field.name])


_VALUE_TYPES = frozenset((
    Point, Wall, Console, PublicMap, Identity, TaskView, SelfView, PlayerSeen,
    BodySeen, Report, Meeting, Claim, Vote, Ejection, MatchEnded, Evidence,
    PublicContext, Action, ActorObservation, Knowledge,
))


def validate_observation(observation: ActorObservation) -> None:
    _validate(observation, ActorObservation)
    for event in observation.evidence:
        expected = (Provenance.CLAIM if type(event.payload) is Claim else
                    Provenance.DIRECT if type(event.payload) in (PlayerSeen, BodySeen)
                    else Provenance.PUBLIC)
        if event.provenance is not expected:
            raise ValueError("Evidence provenance does not match payload")
        if not 0 <= event.observed_tick <= observation.tick:
            raise ValueError("Evidence is outside observation time")


def remember(knowledge: Knowledge, observation: ActorObservation) -> Knowledge:
    """Minimal append-only foundation, not the Phase 4 memory/reasoning engine."""
    _validate(knowledge, Knowledge)
    validate_observation(observation)
    previous_tick = -1
    for prior in knowledge.history:
        validate_observation(prior)
        if (prior.match_id, prior.own.player_id) != (knowledge.match_id, knowledge.player_id):
            raise ValueError("Existing memory contains a different actor or match")
        if prior.tick <= previous_tick:
            raise ValueError("Existing memory is not chronological")
        previous_tick = prior.tick
    if (knowledge.match_id, knowledge.player_id) != (observation.match_id, observation.own.player_id):
        raise ValueError("Cannot mix actors or matches in memory")
    if knowledge.history and observation.tick <= knowledge.history[-1].tick:
        raise ValueError("Memory requires strictly increasing observation ticks")
    return Knowledge(knowledge.match_id, knowledge.player_id, knowledge.history + (observation,))
