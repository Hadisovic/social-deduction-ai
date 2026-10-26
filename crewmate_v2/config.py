"""Public lobby constants and trainer-only reproducible mode selection."""
from dataclasses import asdict, dataclass
import math
import json
from pathlib import Path
from random import Random
from phase3_engine import GameConfig, stream_seed
from . import SCHEMA


@dataclass(frozen=True)
class LobbySettings:
    schema: str = SCHEMA
    player_speed: float = 1.
    confirm_ejects: bool = True
    anonymous_votes: bool = False
    task_bar_updates: str = 'always'
    emergency_meetings: int = 1
    emergency_cooldown: float = 0.
    discussion_seconds: float = 15.
    voting_seconds: float = 120.
    crewmate_vision: float = 1.
    impostor_vision: float = 1.5
    common_tasks: int = 1
    long_tasks: int = 1
    short_tasks: int = 2
    visual_tasks: bool = True
    kill_cooldown: float = 45.
    kill_distance_character_widths: float = 2.
    dt: float = .2
    decision_interval: float = .6
    timeout: float = 900.

    def __post_init__(self):
        if self.schema != SCHEMA:
            raise ValueError('Unknown lobby/environment version')
        for key in ('player_speed','discussion_seconds','voting_seconds','crewmate_vision',
                    'impostor_vision','kill_cooldown','kill_distance_character_widths','dt','decision_interval','timeout'):
            value=getattr(self,key)
            if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                raise ValueError(f'{key} must be positive and finite')
        if any(type(getattr(self,k)) is not bool for k in ('confirm_ejects','anonymous_votes','visual_tasks')):
            raise ValueError('Lobby switches must be boolean')
        if self.anonymous_votes or self.task_bar_updates!='always':
            raise ValueError('V2 supports visible vote identities and ALWAYS task bar only')
        if self.emergency_cooldown!=0 or type(self.emergency_cooldown) not in (int,float):
            raise ValueError('This version supports zero emergency cooldown')
        for k,value in (('emergency_meetings',1),('common_tasks',1),('long_tasks',1),('short_tasks',2)):
            if type(getattr(self,k)) is not int or getattr(self,k)!=value:
                raise ValueError(f'This lobby version requires {k}={value}')
        if not math.isclose(self.decision_interval/self.dt,round(self.decision_interval/self.dt),abs_tol=1e-9):
            raise ValueError('Decision interval must be an integer number of physics ticks')

    def engine_config(self, player_radius):
        return GameConfig(dt=self.dt,decision_interval=self.decision_interval,speed=2.5*self.player_speed,
            tasks_per_crew=4,task_seconds=4.,fake_task_seconds=4.,initial_kill_cooldown=self.kill_cooldown,
            kill_cooldown=self.kill_cooldown,kill_range=2*player_radius*self.kill_distance_character_widths,
            discussion_seconds=self.discussion_seconds,voting_seconds=self.voting_seconds,timeout=self.timeout,
            rules_version=6,crewmate_sight_range=4.5*self.crewmate_vision,
            impostor_sight_range=4.5*self.impostor_vision)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_file(cls,path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))


@dataclass(frozen=True)
class MatchMixture:
    """Trainer configuration; neither frequency nor selected bit is an actor input."""
    deduction_frequency: float = 0.

    def __post_init__(self):
        if type(self.deduction_frequency) not in (int,float) or not math.isfinite(self.deduction_frequency) or not 0<=self.deduction_frequency<=1:
            raise ValueError('Deduction frequency must lie in [0,1]')

    def select(self,seed):
        return Random(stream_seed(seed,'crewmate-v2:match-mode')).random()<self.deduction_frequency
