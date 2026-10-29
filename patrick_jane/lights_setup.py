"""Owner-approved lights-only continuation, separate from the historical pilot."""
from dataclasses import dataclass
from .config import Experiment
from .rewards import RewardLedger


@dataclass(frozen=True)
class LightsExperiment(Experiment):
    profile: str = 'patrick-jane-lights-learning-v2'
    allowed_sabotages: tuple = ('lights',)
    lights_penalty_per_second: float = .005
    record_diagnostics: bool = True
    evaluation_matches: int = 8
    normal_match_timeout: bool = False
    technical_max_seconds: float = 1800.
    technical_stall_seconds: float = 300.

    def __post_init__(self):
        super().__post_init__()
        if self.profile != 'patrick-jane-lights-learning-v2':raise ValueError('Unknown lights profile')
        if len(set(self.allowed_sabotages))!=len(self.allowed_sabotages) or any(
                k not in ('lights','oxygen','reactor') for k in self.allowed_sabotages):
            raise ValueError('Unknown/duplicate sabotage capability')
        if self.lights_penalty_per_second != .005:raise ValueError('Registered lights penalty is locked at 0.005/s')
        if self.record_diagnostics is not True:raise ValueError('Lights accounting/diagnostics must remain enabled')
        if self.normal_match_timeout is not False:raise ValueError('Normal lights matches have no duration-based victory/end')
        import math
        if any(not math.isfinite(v) or v<=0 for v in (self.technical_max_seconds,self.technical_stall_seconds)):
            raise ValueError('Technical safety limits must be positive and finite')


class LightsLedger(RewardLedger):
    def __init__(self,game):
        super().__init__(game);self.dark_paid={p:0. for p in self.crew}

    def collect(self,game):
        result=super().collect(game)
        penalty={}
        for pid in self.crew:
            total=game.lights_actor_seconds.get(pid,0.)
            elapsed=max(0.,total-self.dark_paid[pid]);self.dark_paid[pid]=total
            if elapsed:
                amount=-game.experiment.lights_penalty_per_second*elapsed
                result[pid]['lights_unresolved_time']=amount
                penalty[pid]={'lights_unresolved_time':amount}
        self._record(game.time,penalty)
        return result
