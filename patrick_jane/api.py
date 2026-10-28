"""Detached actor values. No game, RNG, truth or training objects cross here."""
from dataclasses import dataclass
from social_deduction.phase3_api import Phase3Observation
from crewmate_v2.api import OwnTaskDetail


@dataclass(frozen=True)
class Event:
    kind: str
    subject: str = ''
    other: str = ''
    room: str = ''
    position: tuple[float, float] | None = None
    observed: float = 0.
    delivered: float = 0.
    source: str = ''
    provenance: str = 'public'
    detail: str = ''
    origin_source: str = ''


@dataclass(frozen=True)
class Observation:
    actor: Phase3Observation
    own_tasks: tuple[OwnTaskDetail, ...]
    task_bar: tuple[int, int]
    events: tuple[Event, ...]
    sabotage: str | None
    sabotage_remaining: float | None
    repair_locations: tuple[str, ...]
    repaired_locations: tuple[str, ...]
    can_sabotage: bool = False  # own impostor ability only
    teammates: tuple[str, ...] = ()  # legitimate impostor-only team knowledge
    own_route: tuple[tuple[float,float], ...] = ()


@dataclass(frozen=True)
class Action:
    kind: str = 'wait'
    target: str | None = None
    position: tuple[float, float] | None = None
    statement: Event | None = None


@dataclass(frozen=True)
class Choice:
    # Inactive branches are -1, so unused logits cannot enter PPO likelihoods.
    interaction:int=-1
    navigation:int=-1
    communication:int=-1
    voting:int=-1
    target:int=-1

