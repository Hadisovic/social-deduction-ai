"""Unchanged V4 rewards/actions/network; replace only the inactive spawn frontier."""
from collections import Counter
from impostor_research.strategic_v2 import StrategicEnv
from impostor_options_v4.env import OptionsEnv as OriginalEnv,COMPONENTS
from impostor_options_v4.diagnostics import Audit
from .shaping import PursuitPotential
from . import SCHEMA,SHAPING


class OptionsEnv(OriginalEnv):
    def reset(self,**kwargs):
        self.pursuit=PursuitPotential(self.rewards.pursuit_scale,self.rewards.pursuit_min_progress)
        self.pending_pursuit=0.;self.reward_totals=Counter({k:0. for k in COMPONENTS})
        self.ready_elapsed=0.;self.fake_repeats=0;self.decision_journal=[]
        return StrategicEnv.reset(self,**kwargs)

    def summary(self):
        result=super().summary();result.update(schema=SCHEMA,shaping_version=SHAPING)
        return result


class DiagnosticEnv(Audit,OptionsEnv):
    pass
