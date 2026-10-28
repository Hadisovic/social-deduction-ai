"""Versioned public mechanics and locked initial learning configuration."""
from dataclasses import asdict, dataclass, field, fields, replace
import math
from crewmate_v2.config import LobbySettings


@dataclass(frozen=True)
class ClassicLobby(LobbySettings):
    """PJ-only profile; historical V2 settings and source hashes stay untouched."""
    schema: str = 'patrick-jane-classic-v2'
    emergency_cooldown: float = 15.
    initial_kill_cooldown: float = 10.

    def __post_init__(self):
        if self.schema != 'patrick-jane-classic-v2':
            raise ValueError('Unknown Patrick Jane lobby version')
        original = {f.name:getattr(self,f.name) for f in fields(LobbySettings)}
        original.update(schema=LobbySettings().schema,emergency_cooldown=0.)
        LobbySettings(**original)
        for key in ('emergency_cooldown','initial_kill_cooldown'):
            value=getattr(self,key)
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                raise ValueError(f'{key} must be nonnegative and finite')

    def engine_config(self, player_radius):
        return replace(super().engine_config(player_radius),initial_kill_cooldown=self.initial_kill_cooldown)


@dataclass(frozen=True)
class Population:
    crew: int = 10
    impostors: int = 2

    def __post_init__(self):
        if type(self.crew) is not int or type(self.impostors) is not int:
            raise ValueError('Integer population required')
        if not 1 <= self.impostors <= 3 or not self.impostors < self.crew <= 12 or not 4 <= self.total <= 15:
            raise ValueError('Need 1..3 impostors, more crew than impostors, at most 15 players')

    @property
    def total(self): return self.crew + self.impostors


@dataclass(frozen=True)
class SabotageSettings:
    """Explicit approximations, not a claim of retail timing fidelity."""
    cooldown: float = 30.
    initial_cooldown: float = 10.
    oxygen_seconds: float = 30.
    reactor_seconds: float = 30.
    lights_factor: float = .2

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in asdict(self).values()) or self.lights_factor > 1:
            raise ValueError('Invalid sabotage settings')


@dataclass(frozen=True)
class Experiment:
    population: Population = field(default_factory=Population)
    lobby: LobbySettings = field(default_factory=ClassicLobby)
    sabotage: SabotageSettings = field(default_factory=SabotageSettings)
    learning_rate: float = .0003
    gamma: float = .99
    gae_lambda: float = .95
    clip: float = .2
    entropy: float = .01
    epochs: int = 4
    max_grad_norm: float = .5
    sequence_length: int = 128
    event_capacity: int = 64
    evaluation_interval: int = 100_000
    evaluation_matches: int = 100

    def __post_init__(self):
        locked = dict(learning_rate=.0003,gamma=.99,gae_lambda=.95,clip=.2,entropy=.01,
                      epochs=4,max_grad_norm=.5,sequence_length=128)
        if any(getattr(self,k) != v for k,v in locked.items()):
            raise ValueError('Initial Patrick Jane PPO protocol is locked')
        if self.event_capacity < 1 or self.evaluation_matches < 1 or not 100_000 <= self.evaluation_interval <= 500_000:
            raise ValueError('Invalid capacity/evaluation registration')

    def to_dict(self): return asdict(self)


# New, domain-separated development partition. No Phase 6 final partition reader.
DEV_TRAIN = tuple(range(2_900_000, 2_901_000))
DEV_VALIDATION = tuple(range(2_910_000, 2_910_100))
ENGINEERING = tuple(range(2_920_000, 2_920_100))
