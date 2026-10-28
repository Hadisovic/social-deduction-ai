"""Additive five-player coordinator; original collection/inference loop unchanged."""
from collections import Counter
from time import perf_counter
from belief.model import BeliefModel
from crewmate_v2.runtime import V2StrategicMatch
from crewmate_v2.fourcrew.match import FourCrewMatch,ActorView
from crewmate_v2.fourcrew.rewards import ActorLedger
from crewmate_v2.fourcrew.config import Rewards
from crewmate_v2.learning.opponents import ReporterController
from crewmate_v2.phase21.rewards import EligibleLedger
from phase3_runner import behavior_seed
from phase3_bots import BotStyle
from social_deduction.actor import Role
from .controller import LegalImpostor


def check_seed(seed):
    # Handoff-only development ranges. Original registered/final partitions stay sealed.
    if type(seed) is not int or not (2760000<=seed<=2760099 or 3600000<=seed<=3609999 or 3610000<=seed<=3610099):
        raise ValueError('Use engineering 2760000..99, handoff train 3600000..99 or validation 3610000..99')


class FrozenCrewMatch(FourCrewMatch):
    def __init__(self,seed=2760000,*,model='B',controller_factory=None,accounting='historical',opponent='baseline',
                 rotation=0,lobby=None,deadline=None):
        check_seed(seed)
        if model not in ('A','B') or accounting not in ('historical','eligible-v1') or opponent not in ('baseline','v1','v2'):
            raise ValueError('Unknown explicit handoff option')
        if controller_factory and opponent!='baseline':raise ValueError('Select an external controller or a scripted variant, not both')
        # Same constructor state as FourCrewMatch. A new seed allowlist is additive;
        # frozen sources and original study's partition validation are not patched.
        self.match=V2StrategicMatch(seed,lobby=lobby);self.game=self.match.game
        self.seed=seed;self.deadline=deadline or perf_counter()+600
        self.crew=tuple(sorted(pid for pid,p in self.game.players.items() if p.role is Role.CREWMATE))
        self.impostor=next(pid for pid,p in self.game.players.items() if p.role is Role.IMPOSTOR)
        if controller_factory:
            controller=controller_factory(seed=behavior_seed(seed,self.impostor),player_id=self.impostor)
        elif opponent=='baseline':controller=ReporterController(behavior_seed(seed,self.impostor),BotStyle('nearest','hunter','evidence'),rotation)
        elif opponent=='v1':
            from crewmate_v2.phase2.opponents import MixtureController
            controller=MixtureController(behavior_seed(seed,self.impostor))
        else:
            from crewmate_v2.phase21.opponents import CoherentMixture
            controller=CoherentMixture(behavior_seed(seed,self.impostor))
        self.match.controllers={self.impostor:LegalImpostor(controller,self.impostor)}
        belief=BeliefModel.load();belief.network.requires_grad_(False)
        self.views={pid:ActorView(self.game,pid,belief) for pid in self.crew}
        roles={pid:p.role for pid,p in self.game.players.items()}
        reward=Rewards.original() if model=='A' else Rewards()
        ledger=ActorLedger if accounting=='historical' else EligibleLedger
        self.ledgers={pid:ledger(pid,roles,reward,self.game.config.dt) for pid in self.crew}
        self.options={};self.selected={};self.starts={};self.histories={pid:[] for pid in self.crew}
        self.rows={pid:[] for pid in self.crew};self.actions={pid:Counter() for pid in self.crew}
        self.reporting={pid:Counter() for pid in self.crew}
        self.opportunities={pid:{k:set() for k in ('visible','range','legal','choice_legal','held_legal')} for pid in self.crew}
        self.max_candidates=0;self.policy_version=None;self.decisions=0
        self.handoff_options=dict(model=model,accounting=accounting,opponent=opponent,
                                  external_controller=bool(controller_factory))


def scripted_factory(*,seed,player_id):
    """Example external factory; replace its decide method with your NN adapter."""
    return ReporterController(seed,BotStyle('nearest','hunter','evidence'),0)
