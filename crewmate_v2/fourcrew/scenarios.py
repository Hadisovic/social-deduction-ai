"""Legal scripted setup, followed by private PPO control. No teleport or evidence injection."""
import math
from time import perf_counter
from social_deduction.actor import Point,Role
from social_deduction.phase3_api import Intent,IntentKind,intent_allowed
from .match import FourCrewMatch
from .rewards import ActorLedger

PREFIX_VERSION='legal-body-or-meeting-prefix-v1'
KILL_TICK=225                 # canonical 45s initial cooldown
HANDOFF_TICK=375              # 75s, same public clock for both reporter roles


class PrefixError(RuntimeError):pass


class ScenarioMatch(FourCrewMatch):
    def __init__(self,seed,*,context='ordinary',case_index=0,**kwargs):
        if context not in ('ordinary','body','meeting_crew','meeting_impostor'):raise ValueError('Unknown registered context')
        super().__init__(seed,**kwargs)
        self.context=context;self.case_index=case_index;self.prefix=None;self._prefix_record=None

    def _prefix_tick(self,actions=None):
        if perf_counter()>self.deadline:raise TimeoutError('Prefix wall ceiling')
        if self.game.is_terminal:raise PrefixError('Terminal during legal prefix')
        actions=actions or {}
        for pid,intent in actions.items():
            if not intent_allowed(self.game.observe(pid),intent):raise PrefixError('Illegal prefix intention')
        self.game.step(actions)
        for view in self.views.values():view.refresh()
        if self._prefix_record:self._prefix_record.after_step()

    def _drive(self,targets,latest_tick):
        last={}
        while True:
            actions={};arrived=True
            for pid,xy in targets.items():
                obs=self.game.observe(pid)
                if not obs.own.active:continue
                distance=math.dist((obs.own.position.x,obs.own.position.y),xy)
                if distance>.45:
                    arrived=False
                    if last.get(pid)!=xy or obs.navigation_status!='moving':
                        actions[pid]=Intent(IntentKind.GO_TO_LOCATION,position=Point(*xy));last[pid]=xy
                elif obs.navigation_status=='moving':actions[pid]=Intent(IntentKind.CANCEL_NAVIGATION)
            if arrived and not actions:return
            if self.game.tick>=latest_tick:raise PrefixError('Public route missed registered setup deadline')
            self._prefix_tick(actions)

    def prepare_prefix(self,recorder=None):
        if self.prefix is not None:raise PrefixError('Prefix already prepared')
        self._prefix_record=recorder
        if recorder:recorder.after_step()
        if self.context=='ordinary':
            self.prefix=dict(version=PREFIX_VERSION,context=self.context,handoff_tick=0,prefix_reports=0)
            return self.prefix
        game=self.game
        site=tuple(map(float,next(d.standing for d in game.map.destinations if d.name=='EmergencyConsole')))
        # Identity/role assignment is trusted scenario construction, never an actor feature.
        # Routes, positions, cooldowns and report availability are checked through own observations.
        victim=self.crew[self.case_index%4]
        others=[pid for pid in self.crew if pid!=victim]
        rooms=('Weapons','Admin','MedBay')
        parking=[tuple(map(float,next(d.standing for d in game.map.destinations if d.room==room and d.category=='task'))) for room in rooms]
        shift=self.case_index%3;parking=parking[shift:]+parking[:shift]
        targets={pid:xy for pid,xy in zip(others,parking)}
        targets.update({victim:site,self.impostor:site})
        self._drive(targets,KILL_TICK)
        while game.tick<KILL_TICK:self._prefix_tick()
        obs=game.observe(self.impostor)
        kill=next((a for a in obs.actions if a.kind is IntentKind.KILL and a.target==victim),None)
        if kill is None:raise PrefixError('Registered initial kill not legally available')
        self._prefix_tick({self.impostor:kill})
        self._drive({pid:site for pid in (*others,self.impostor)},HANDOFF_TICK)
        while game.tick<HANDOFF_TICK:self._prefix_tick()
        reporter=None
        if self.context.startswith('meeting_'):
            reporter=self.impostor if self.context=='meeting_impostor' else others[self.case_index%3]
            report=next((a for a in game.observe(reporter).actions if a.kind is IntentKind.REPORT),None)
            if report is None:raise PrefixError('Registered prefix report not legally available')
            self._prefix_tick({reporter:report})
        # Prefix events are legitimate history, but not decisions/rewards of a learned controller.
        for pid,ledger in list(self.ledgers.items()):
            fresh=ActorLedger(pid,ledger.roles,ledger.config,game.config.dt)
            fresh.alive=game.players[pid].active;fresh.death_paid=not fresh.alive
            self.ledgers[pid]=fresh
        self.prefix=dict(version=PREFIX_VERSION,context=self.context,handoff_tick=game.tick,
                         prefix_reports=int(reporter is not None),prefix_deaths=1)
        game._record('prefix_handoff',context=self.context,learned_control=True)
        return self.prefix

    def collect(self,*args,recorder=None,**kwargs):
        if self.prefix is None:self.prepare_prefix(recorder)
        rollout,summary=super().collect(*args,recorder=recorder,**kwargs)
        summary.update(context=self.context,prefix=self.prefix,
                       learned_control_seconds=self.game.time-self.prefix['handoff_tick']*self.game.config.dt)
        return rollout,summary
