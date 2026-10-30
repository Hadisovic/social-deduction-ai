"""Strict, immutable, JSON-editable settings shared by both Jane role projects.

Describing a setting does not implement its mechanic. Runtime adapters must pass
the separate compatibility gate. No existing environment defaults are changed.
"""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path

from jane import MAX_PLAYERS, SETTINGS_SCHEMA


OPTIONAL_ROLES = (
    "scientist", "engineer", "guardian_angel", "shapeshifter", "noisemaker",
    "tracker", "phantom", "detective", "viper",
)
RULESETS = ("historical-v2.2", "classic-skeld-v1-draft", "patrick-jane-a2-v1")


def _integer(name, value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def _number(name, value, low, high, step=None):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not low <= value <= high):
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")
    if step is not None and not math.isclose((value - low) / step,
                                             round((value - low) / step), abs_tol=1e-8):
        raise ValueError(f"{name} must use increments of {step}")


def _bools(instance, names):
    for name in names:
        if type(getattr(instance, name)) is not bool:
            raise ValueError(f"{name} must be boolean")


def _record(cls, value):
    if type(value) is not dict:
        raise ValueError(f"{cls.__name__} must be a JSON object")
    expected = {field.name for field in fields(cls)}
    if set(value) != expected:
        raise ValueError(f"{cls.__name__}: missing={sorted(expected - set(value))}; "
                         f"unknown={sorted(set(value) - expected)}")
    return cls(**value)


@dataclass(frozen=True)
class Lobby:
    players: int
    impostors: int
    player_speed: float
    crewmate_vision: float
    impostor_vision: float
    kill_cooldown_seconds: float
    kill_distance: str
    confirm_ejects: bool

    def __post_init__(self):
        _integer("players", self.players, 4, MAX_PLAYERS)
        _integer("impostors", self.impostors, 1, 3)
        # Conventional classic start minima; do not allow near-parity custom hacks.
        minimum = {1: 4, 2: 7, 3: 9}[self.impostors]
        if self.players < minimum:
            raise ValueError(f"{self.impostors} impostors require at least {minimum} players")
        _number("player_speed", self.player_speed, .5, 3., .25)
        _number("crewmate_vision", self.crewmate_vision, .25, 5., .25)
        _number("impostor_vision", self.impostor_vision, .25, 5., .25)
        _number("kill_cooldown_seconds", self.kill_cooldown_seconds, 10., 60., 2.5)
        if self.kill_distance not in ("short", "medium", "long"):
            raise ValueError("kill_distance must be short, medium or long")
        _bools(self, ("confirm_ejects",))


@dataclass(frozen=True)
class Meetings:
    emergency_meetings: int
    emergency_cooldown_seconds: int
    discussion_seconds: int
    voting_seconds: int
    anonymous_votes: bool

    def __post_init__(self):
        _integer("emergency_meetings", self.emergency_meetings, 0, 9)
        _integer("emergency_cooldown_seconds", self.emergency_cooldown_seconds, 0, 60)
        _integer("discussion_seconds", self.discussion_seconds, 0, 120)
        _integer("voting_seconds", self.voting_seconds, 0, 300)
        for name, step in (("emergency_cooldown_seconds", 5),
                           ("discussion_seconds", 15), ("voting_seconds", 15)):
            if getattr(self, name) % step:
                raise ValueError(f"{name} must use increments of {step}")
        _bools(self, ("anonymous_votes",))


@dataclass(frozen=True)
class Tasks:
    common: int
    long: int
    short: int
    visual_tasks: bool
    task_bar_updates: str
    pool_id: str

    def __post_init__(self):
        _integer("common", self.common, 0, 2)
        _integer("long", self.long, 0, 3)
        _integer("short", self.short, 0, 5)
        if self.task_bar_updates not in ("always", "meetings", "never"):
            raise ValueError("Unknown task-bar update mode")
        if self.pool_id not in ("historical-skeld-v2.2", "classic-skeld-v1-draft", "patrick-jane-skeld-v1"):
            raise ValueError("Unknown task pool")
        _bools(self, ("visual_tasks",))


@dataclass(frozen=True)
class Mechanics:
    vents: bool
    doors: bool
    lights: bool
    communications: bool
    oxygen: bool
    reactor: bool
    admin: bool
    security: bool
    impostor_ghosts: bool
    clean_vent: bool

    def __post_init__(self):
        _bools(self, tuple(field.name for field in fields(self)))
        if self.impostor_ghosts and not any((self.doors, self.lights,
                                          self.communications, self.oxygen, self.reactor)):
            raise ValueError("Impostor ghost assistance requires an enabled sabotage")
        if self.clean_vent and not self.vents:
            raise ValueError("Clean Vent needs the vent occupancy/eviction mechanic")

    def enabled(self):
        return frozenset(field.name for field in fields(self) if getattr(self, field.name))


@dataclass(frozen=True)
class Simulation:
    physics_step_seconds: float
    observation_interval_seconds: float
    match_cap_seconds: float
    stall_cap_seconds: float
    cap_policy: str

    def __post_init__(self):
        for name in ("physics_step_seconds", "observation_interval_seconds", "match_cap_seconds"):
            _number(name, getattr(self, name), .001, 1e6)
        _number("stall_cap_seconds", self.stall_cap_seconds, 0., 1e6)
        ratio = self.observation_interval_seconds / self.physics_step_seconds
        if ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-8):
            raise ValueError("Observation interval must be an integer number of physics steps")
        if self.cap_policy not in ("truncate", "historical-impostor-loss"):
            raise ValueError("Unknown cap policy")


@dataclass(frozen=True)
class RoleSetting:
    role: str
    max_count: int
    chance_percent: int

    def __post_init__(self):
        if self.role not in OPTIONAL_ROLES:
            raise ValueError(f"Unknown optional role: {self.role}")
        _integer("max_count", self.max_count, 0, MAX_PLAYERS)
        _integer("chance_percent", self.chance_percent, 0, 100)
        if self.chance_percent % 10:
            raise ValueError("Role chance must use increments of 10 percent")

    @property
    def enabled(self):
        return self.max_count > 0 and self.chance_percent > 0


@dataclass(frozen=True)
class GameSettings:
    schema: str
    profile_id: str
    ruleset_id: str
    reference_game_build: str | None
    map_id: str
    lobby: Lobby
    meetings: Meetings
    tasks: Tasks
    mechanics: Mechanics
    simulation: Simulation
    optional_roles: tuple[RoleSetting, ...]

    def __post_init__(self):
        if self.schema != SETTINGS_SCHEMA:
            raise ValueError("Unknown settings schema; migrate explicitly")
        if type(self.profile_id) is not str or not self.profile_id.strip():
            raise ValueError("profile_id must be nonempty")
        if self.ruleset_id not in RULESETS or self.map_id != "the_skeld":
            raise ValueError("Unknown ruleset or map; other maps need separate contracts")
        if self.reference_game_build is not None and (
                type(self.reference_game_build) is not str or not self.reference_game_build.strip()):
            raise ValueError("reference_game_build must be a build string or null")
        for name, cls in (("lobby", Lobby), ("meetings", Meetings), ("tasks", Tasks),
                          ("mechanics", Mechanics), ("simulation", Simulation)):
            if type(getattr(self, name)) is not cls:
                raise ValueError(f"{name} must be an immutable {cls.__name__}")
        if type(self.optional_roles) is not tuple or any(type(r) is not RoleSetting for r in self.optional_roles):
            raise ValueError("optional_roles must contain immutable role settings")
        names = [r.role for r in self.optional_roles]
        if len(names) != len(set(names)) or set(names) != set(OPTIONAL_ROLES):
            raise ValueError("Specify each optional role exactly once, disabled by default")
        if any(r.max_count > self.lobby.players for r in self.optional_roles):
            raise ValueError("Role count exceeds lobby population")
        expected_pool = {"historical-v2.2": "historical-skeld-v2.2",
                         "classic-skeld-v1-draft": "classic-skeld-v1-draft",
                         "patrick-jane-a2-v1": "patrick-jane-skeld-v1"}[self.ruleset_id]
        if self.tasks.pool_id != expected_pool:
            raise ValueError("Task pool and ruleset must have the same version")

    @property
    def crewmates(self):
        return self.lobby.players - self.lobby.impostors

    def to_dict(self):
        result = asdict(self)
        result["optional_roles"] = [asdict(role) for role in self.optional_roles]
        return result

    def fingerprint(self):
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def public_settings(self):
        # No trainer deadline, seed, opponent identity or selected policy is an actor input.
        result = self.to_dict()
        for key in ("simulation", "profile_id", "reference_game_build"):
            result.pop(key)
        return result

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict:
            raise ValueError("Settings must be a JSON object")
        value = dict(value)
        for name, record in (("lobby", Lobby), ("meetings", Meetings), ("tasks", Tasks),
                             ("mechanics", Mechanics), ("simulation", Simulation)):
            if name not in value:
                raise ValueError(f"Missing settings section: {name}")
            value[name] = _record(record, value[name])
        roles = value.get("optional_roles")
        if type(roles) is not list:
            raise ValueError("optional_roles must be a JSON array")
        value["optional_roles"] = tuple(_record(RoleSetting, role) for role in roles)
        return _record(cls, value)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_settings(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    return GameSettings.from_dict(value)


def bundle_receipt(settings, root=None):
    """Bind settings to the exact shared definitions, not just their version labels."""
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    names = ("settings.py", "compatibility.py", "patrick_jane.py", "catalog.json", "rulesets.json",
             "actor_contract.json", "network_reference.json",
             "upstream/patrick_jane_review.json", "upstream/patrick_jane_a2_default.json",
             "upstream/patrick_jane_architecture.json")
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    value = {"settings_sha256": settings.fingerprint(), "shared_files": files}
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return {**value, "bundle_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}
