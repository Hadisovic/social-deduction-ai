"""Legal reporter variation; no focal policy callback or hidden-role style selection."""
from random import Random
from phase3_runner import behavior_seed
from phase3_bots import BotStyle,CREW_VARIANTS,IMPOSTOR_VARIANTS,MEETING_VARIANTS
from phase4_scripts import StudyController
from social_deduction.phase3_api import IntentKind,intent_allowed
from crewmate_v2.runtime import V2StrategicMatch
from .rewards import RewardLedger

REPORT_STYLES=('prompt','inspect','bypass')


class ReporterController(StudyController):
    def __init__(self,seed,style,rotation=0):
        super().__init__(seed,style)
        index=Random(seed^0xD17E2517).randrange(3)
        self.reporting_style=REPORT_STYLES[(index+rotation)%3];self.first_body_sight={}

    def _observe(self,obs):
        super()._observe(obs)
        for body in obs.visible_bodies:self.first_body_sight.setdefault(body.victim_id,obs.time_seconds)

    def _report(self,obs):
        if self.reporting_style=='bypass':return None
        action=super()._report(obs)
        if action is None or self.reporting_style=='prompt' or action.kind is not IntentKind.REPORT:return action
        if obs.time_seconds-self.first_body_sight[action.target]>=6.-1e-9:return action
        for kind in (IntentKind.CANCEL_NAVIGATION,IntentKind.WAIT):
            if kind is IntentKind.CANCEL_NAVIGATION and obs.navigation_status!='moving' and obs.interaction_task is None:continue
            choices=self._available(obs,kind)
            if choices:return choices[0]
        return None


class LearningMatch(V2StrategicMatch):
    def __init__(self,*args,reward_config,opponents='original',rotation=0,deadline=None,**kwargs):
        super().__init__(*args,**kwargs)
        if opponents not in ('original','varied') or rotation not in (0,1,2):raise ValueError('Unknown opponent context')
        if opponents=='varied':
            for pid in self.controllers:
                rng=Random(behavior_seed(self.seed,pid)^0xB311EF)
                style=BotStyle(rng.choice(CREW_VARIANTS),rng.choice(IMPOSTOR_VARIANTS),rng.choice(MEETING_VARIANTS))
                self.controllers[pid]=ReporterController(behavior_seed(self.seed,pid),style,rotation)
        self.ledger=RewardLedger(self.focal,{pid:p.role for pid,p in self.game.players.items()},reward_config,self.game.config.dt)
        self.deadline=deadline
        self.report_visible=set();self.report_legal=set();self.report_legal_windows=0

    def tick(self,focal_intent=None):
        if self.game.is_terminal:return self.game.result
        if self.deadline is not None:
            from time import perf_counter
            if perf_counter()>self.deadline:raise TimeoutError('Learning/smoke wall ceiling reached')
        cursor=len(self.game.event_log)
        observation=self.game.observe(self.focal)
        legal=focal_intent is not None and intent_allowed(observation,focal_intent)
        if observation.own.active:
            self.report_visible.update(b.victim_id for b in observation.visible_bodies)
            reports=[a.target for a in observation.actions if a.kind is IntentKind.REPORT]
            self.report_legal.update(reports)
            if self.game.tick%self.decision_ticks==0:self.report_legal_windows+=bool(reports)
        result=super().tick(focal_intent)
        self.ledger.consume(self.game.event_log[cursor:],active=self.game.players[self.focal].active,
            terminal=self.game.is_terminal,winner=self.game.result.winner if self.game.is_terminal else None,
            intent=focal_intent,legal=legal)
        return result
