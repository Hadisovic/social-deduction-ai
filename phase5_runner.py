"""Trusted focal-policy orchestration. Truth never enters the policy encoder."""
from random import Random
from phase3_engine import GameConfig
from phase3_runner import ScriptedMatch,behavior_seed
from phase3_bots import BotStyle,CREW_VARIANTS,MEETING_VARIANTS
from phase4_scripts import StudyController
from social_deduction.actor import Role
from social_deduction.phase3_api import IntentKind


class StrategicMatch(ScriptedMatch):
    def __init__(self,seed,planner=None,config=None,heldout=False):
        super().__init__(seed,config or GameConfig(),planner)
        for pid in self.controllers:
            rng=Random(behavior_seed(seed,pid)^0xB311EF)
            style=BotStyle(rng.choice(CREW_VARIANTS),rng.choice(('patient',) if heldout else ('hunter','self_report')),
                           rng.choice(MEETING_VARIANTS))
            self.controllers[pid]=StudyController(behavior_seed(seed,pid),style)
        crew=sorted(pid for pid,p in self.game.players.items() if p.role is Role.CREWMATE)
        self.focal=Random(behavior_seed(seed,'focal')).choice(crew)

    def tick(self,focal_intent=None):
        game=self.game; actions={}
        if game.is_terminal:return game.result
        if game.tick%self.decision_ticks==0:
            observations={pid:game.observe(pid) for pid in sorted(self.controllers)}
            game.record_observations(observations)
            for pid,obs in observations.items():
                if pid==self.focal or not obs.own.active or not obs.actions:continue
                action=self.controllers[pid].decide(obs)
                if action.kind is not IntentKind.WAIT: actions[pid]=action
        if focal_intent is not None:actions[self.focal]=focal_intent
        return game.step(actions)
