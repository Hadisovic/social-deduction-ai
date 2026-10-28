"""Future-policy interface and unchanged strategic reward semantics; no training loop."""
from random import Random
import numpy as np
from gymnasium import spaces
from phase3_runner import ScriptedMatch, behavior_seed
from phase3_bots import BotStyle, CREW_VARIANTS, MEETING_VARIANTS
from phase4_scripts import StudyController
from phase5_features import GLOBAL_NAMES, PLAYER_FEATURES
from phase6.config import Experiment
from phase6.env import Phase6Env
from belief.memory import ActorMemory
from social_deduction.actor import Phase, Role
from social_deduction.phase3_api import IntentKind
from .api import ConfirmedEjection, VisualTaskProof, validate
from .config import LobbySettings, MatchMixture
from .engine import CrewmateV2Game
from . import SCHEMA


class PublicMemory:
    """One living actor's evidence only. Never reads an engine or ghost state."""
    def __init__(self):
        self.player_id=None;self.tick=-1;self.events=[];self._known=set()

    def update(self,observation):
        validate(observation)
        actor=observation.actor
        if self.player_id is not None and self.player_id!=actor.own.player_id:
            raise ValueError('Independent agent memory cannot change owners')
        if actor.tick<self.tick:raise ValueError('Memory cannot travel backwards')
        if not actor.own.active or actor.context.phase is Phase.FINISHED:return
        self.player_id=actor.own.player_id;self.tick=actor.tick
        for event in observation.events:
            if event not in self._known:self.events.append(event);self._known.add(event)

    def player_features(self,candidates):
        result=np.zeros((4,3),np.float32)
        for event in self.events:
            p=event.payload
            if p.player_id not in candidates:continue
            index=candidates.index(p.player_id)
            if type(p) is ConfirmedEjection:result[index,0 if p.role is Role.CREWMATE else 1]=1.
            elif type(p) is VisualTaskProof:result[index,2]=1.
        return result


class V2StrategicMatch(ScriptedMatch):
    def __init__(self,seed=0,*,lobby=None,mixture=None,planner=None,fast=True):
        self.game=CrewmateV2Game(seed,lobby=lobby,mixture=mixture,planner=planner,fast=fast)
        self._initialize_v2(seed)

    def _initialize_v2(self,seed):
        self._initialize(seed)
        for pid in self.controllers:
            rng=Random(behavior_seed(seed,pid)^0xB311EF)
            self.controllers[pid]=StudyController(behavior_seed(seed,pid),BotStyle(
                rng.choice(CREW_VARIANTS),rng.choice(('hunter','self_report')),rng.choice(MEETING_VARIANTS)))
        crew=sorted(pid for pid,p in self.game.players.items() if p.role is Role.CREWMATE)
        self.focal=Random(behavior_seed(seed,'focal')).choice(crew)

    def reset(self,seed=None):
        selected=self.seed if seed is None else seed
        self.game.reset(selected);self._initialize_v2(selected)
        return self

    def replay_document(self):
        record=super().replay_document()
        record.update(schema=SCHEMA+'-replay',lobby=self.game.lobby.to_dict(),
            deduction_frequency=self.game.mixture.deduction_frequency,
            deduction_only=self.game.deduction_only,execution=dict(lazy_projection=self.game.fast))
        return record

    def tick(self,focal_intent=None):
        game=self.game;actions={}
        if game.is_terminal:return game.result
        if game.tick%self.decision_ticks==0:
            observations={pid:game.observe(pid) for pid in sorted(self.controllers)}
            game.record_observations(observations)
            for pid,obs in observations.items():
                if pid==self.focal or not obs.own.active or not obs.actions:continue
                action=self.controllers[pid].decide(obs)
                if action.kind is not IntentKind.WAIT:actions[pid]=action
        if focal_intent is not None:actions[self.focal]=focal_intent
        return game.step(actions)


class CrewmateV2Env(Phase6Env):
    """New observation dimensions intentionally reject historical policy networks.

    Inherits the original time-aware options and rewards, including one -5 death.
    Ghost tasks are automatic task-only work; the dead policy never acts again.
    """
    def __init__(self,*,lobby=None,mixture=None,fast=True,planner=None,belief_model=None):
        self.lobby=lobby or LobbySettings();self.mixture=mixture or MatchMixture();self.fast=fast
        super().__init__(Experiment(),planner=planner,belief_model=belief_model)
        self.config=self.lobby.engine_config(self.planner.map.radius)
        self.observation_space.spaces['global']=spaces.Box(-1,1,(len(GLOBAL_NAMES)+2,),np.float32)
        self.observation_space.spaces['players']=spaces.Box(-1,1,(4,PLAYER_FEATURES+3),np.float32)
        self.observation_space.spaces['own_tasks_v2']=spaces.Box(0,1,(4,8),np.float32)

    def reset(self,*,seed=None,options=None):
        # Do not call the historical reset: it constructs a historical game.
        import gymnasium as gym
        gym.Env.reset(self,seed=seed)
        selected=int(self.np_random.integers(0,2**31)) if seed is None else int(seed)
        self.match=V2StrategicMatch(selected,lobby=self.lobby,mixture=self.mixture,planner=self.planner,fast=self.fast)
        self.focal=self.match.focal;self.memory=ActorMemory();self.public_memory=PublicMemory();self.pending=None
        self._previous_tick=-1;self._event_cursor=0;self._credited=set();self._eliminated=False
        self.finished=False;self.total_reward=0.;self.last_label='continue'
        self.completed_tasks=0;self.survival_time=None;self.visual_frames=[]
        self._observe()
        return self._packet(),{}

    def _observe(self):
        super()._observe()
        self.v2_observation=self.match.game.observe_v2(self.focal)
        self.public_memory.update(self.v2_observation)

    def _packet(self):
        packet=super()._packet()
        packet['global'][GLOBAL_NAMES.index('time')]=min(1.,self.observation.time_seconds/self.lobby.timeout)
        bar=self.v2_observation.task_bar
        packet['global']=np.concatenate((packet['global'],np.asarray([bar.completed/bar.total,bar.total/160.],np.float32)))
        packet['players']=np.concatenate((packet['players'],self.public_memory.player_features(self.memory.candidates)),axis=1)
        details={t.task_id:t for t in self.v2_observation.own_tasks}
        rows=[]
        for task in sorted(self.observation.own.tasks,key=lambda t:t.task_id):
            d=details[task.task_id]
            rows.append([float(d.category==c) for c in ('Common','Long','Short')]+[
                d.stage/6.,d.stages/6.,min(d.ready_in_seconds/60.,1.),min(d.stage_seconds/30.,1.),float(d.visual)])
        packet['own_tasks_v2']=np.asarray(rows,np.float32)
        self.packet=packet
        return packet
