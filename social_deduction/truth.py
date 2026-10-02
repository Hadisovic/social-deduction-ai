"""Trusted simulator snapshots/fixtures. Never hand these objects to a policy."""

from dataclasses import dataclass
from random import Random

from .actor import Identity, Point, PublicContext, PublicMap, PublicPayload, Role


@dataclass(frozen=True, slots=True)
class TaskTruth:
    task_id: str
    console_id: str
    progress: float = 0.0


@dataclass(frozen=True, slots=True)
class PlayerTruth:
    identity: Identity
    role: Role
    position: Point
    room: str
    active: bool = True
    velocity: Point = Point(0.0, 0.0)
    interacting: bool = False
    tasks: tuple[TaskTruth, ...] = ()
    private_cooldown: float = 0.0
    meetings_remaining: int = 1


@dataclass(frozen=True, slots=True)
class BodyTruth:
    victim_id: str
    position: Point
    room: str
    death_tick: int
    killer_id: str


@dataclass(frozen=True, slots=True)
class Publication:
    """Public-only log, already authorized by the future simulator's rules."""
    tick: int
    payload: PublicPayload


@dataclass(frozen=True, slots=True)
class InternalEvent:
    tick: int
    kind: str
    subject_id: str


@dataclass(frozen=True, slots=True)
class WorldTruth:
    match_id: str
    tick: int
    map: PublicMap
    players: tuple[PlayerTruth, ...]
    context: PublicContext = PublicContext()
    bodies: tuple[BodyTruth, ...] = ()
    publications: tuple[Publication, ...] = ()
    internal_events: tuple[InternalEvent, ...] = ()


def assign_identities(count: int, *, seed: int) -> tuple[Identity, ...]:
    """Return internal-slot identities, using no role state or role RNG."""
    if count < 2:
        raise ValueError("At least two players required")
    rng = Random(seed)
    ids = list(range(count))
    names = [f"Explorer-{i}" for i in range(count)]
    rng.shuffle(ids)
    rng.shuffle(names)
    return tuple(Identity(f"p{number}", name) for number, name in zip(ids, names))


def assign_roles(count: int, *, seed: int, impostors: int = 1) -> tuple[Role, ...]:
    """Independent role RNG. Do not reuse the identity seed in match setup."""
    if not 0 < impostors < count:
        raise ValueError("Need both crew and impostors")
    selected = set(Random(seed).sample(range(count), impostors))
    return tuple(Role.IMPOSTOR if i in selected else Role.CREWMATE for i in range(count))
