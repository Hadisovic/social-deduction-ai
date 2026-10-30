"""A2 registration: opt-in rewards/mechanics/opponents, unchanged neural PPO."""
from dataclasses import dataclass
from .lights_setup import LightsExperiment
from .config import Experiment


@dataclass(frozen=True)
class A2Experiment(LightsExperiment):
    a2_enabled: bool = True
    reward_regime: str = 'investigation-a2-v1'
    mechanics_regime: str = 'classic-skeld-a2-v1'
    opponent_regime: str = 'mixed-a2-v1'
    feature_schema: str = 'patrick-jane-events-a2-v1'
    task_commitment_mode: bool = True
    task_pursuit_mode: bool = True
    living_task_reward: float = .5
    observer_errors_fatal: bool = False
    doors_duration: float = 10.
    doors_cooldown: float = 30.
    vent_range: float = 1.
    vent_transition_seconds: float = .6
    upgraded_probability: float = .5

    def __post_init__(self):
        Experiment.__post_init__(self)
        if self.reward_regime not in ('investigation-a2-v1','task-pursuit-v1'):raise ValueError('Unknown reward control')
        if self.living_task_reward!=(.5 if self.reward_regime=='investigation-a2-v1' else 2.):raise ValueError('Reward regime/task coefficient mismatch')
        if self.mechanics_regime not in ('classic-skeld-a2-v1','historical'):raise ValueError('Unknown mechanics')
        if self.opponent_regime not in ('historical','upgraded-a2-v1','mixed-a2-v1'):raise ValueError('Unknown opponent regime')
        if self.mechanics_regime=='historical' and self.opponent_regime!='historical':raise ValueError('Upgraded opponents require doors/vents')
        if self.allowed_sabotages!=('lights',) or self.lights_penalty_per_second!=.005:raise ValueError('Lights-only profile required')
        if (self.doors_duration,self.doors_cooldown,self.vent_range,self.vent_transition_seconds)!=(10.,30.,1.,.6):raise ValueError('Unregistered mechanics settings')
        if self.upgraded_probability!=.5:raise ValueError('Opponent mixture is frozen at 50% per match')
        if not self.task_commitment_mode or not self.task_pursuit_mode or self.normal_match_timeout:raise ValueError('Task competence and legitimate endings required')


def coefficients(experiment):
    from .rewards import COEFFICIENTS
    return {**COEFFICIENTS,'task':experiment.living_task_reward,
        'task_victory':5. if experiment.reward_regime=='investigation-a2-v1' else 10.,
        'ejection_victory':10.,'vote_scale':.2 if experiment.reward_regime=='investigation-a2-v1' else 0.}
