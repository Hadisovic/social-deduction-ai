"""Strategic Gymnasium wrapper. Rewards and simulator remain trainer-only."""
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from among_us_map import get_map
from navigation_service import NavigationPlanner
from belief.memory import ActorMemory
from belief.features import current_memory
from belief.model import BeliefModel
from phase5_runner import StrategicMatch
from phase5_features import GLOBAL_NAMES,PLAYER_FEATURES,ACTION_FEATURES,MAX_ACTIONS,Choice,choices,encode,mechanical_intent
from social_deduction.phase3_api import intent_allowed,IntentKind
from social_deduction.actor import Role,Phase


class CrewmateStrategicEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,*,planner=None,belief_model=None,ablation=False,heldout=False,config=None,gamma=.99,render_trace=False):
        super().__init__()
        if not 0<gamma<=1:raise ValueError('gamma must be in (0,1]')
        self.planner=planner or NavigationPlanner(get_map(),cache_size=96)
        self.belief_model=belief_model or BeliefModel.load()
        self.belief_model.network.requires_grad_(False)
        self.ablation=ablation;self.heldout=heldout;self.config=config;self.gamma=gamma
        self.action_space=spaces.Discrete(MAX_ACTIONS)
        self.observation_space=spaces.Dict({
            'global':spaces.Box(-1,1,(len(GLOBAL_NAMES),),np.float32),
            'players':spaces.Box(-1,1,(4,PLAYER_FEATURES),np.float32),
            'actions':spaces.Box(-1,1,(MAX_ACTIONS,ACTION_FEATURES),np.float32),
            'targets':spaces.Box(-1,3,(MAX_ACTIONS,),np.int64),
            'mask':spaces.Box(0,1,(MAX_ACTIONS,),np.int8)})
        self.finished=True
        self.render_trace=render_trace
        self.visual_frames=[]

    def _visual_frame(self):
        # Spectator-only animation data; never included in packets or rewards.
        game=self.match.game
        return (game.time,{pid:(p.position,p.velocity) for pid,p in game.players.items()})

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        match_seed=int(self.np_random.integers(0,2**31)) if seed is None else int(seed)
        self.match=StrategicMatch(match_seed,self.planner,self.config,self.heldout)
        self.focal=self.match.focal;self.memory=ActorMemory();self.pending=None
        self._previous_tick=-1;self._event_cursor=0;self._credited=set();self._eliminated=False
        self.finished=False;self.total_reward=0.;self.last_label='continue'
        self.completed_tasks=0;self.survival_time=None
        self.visual_frames=[self._visual_frame()] if self.render_trace else []
        self._observe()
        return self._packet(),{}

    def _observe(self):
        self.observation=self.match.game.observe(self.focal)
        obs=self.observation
        if not self.ablation:self.memory.update(obs)
        elif obs.own.active and obs.context.phase is not Phase.FINISHED:
            self.memory=current_memory(obs,self._previous_tick)
        self._previous_tick=obs.tick

    def _packet(self):
        self.belief=self.belief_model.predict(self.memory)
        self.options=choices(self.observation,self.memory)
        self.packet=encode(self.observation,self.memory,self.belief,self.options,ablation=self.ablation)
        return self.packet

    def action_masks(self):return self.packet['mask'].astype(bool)

    def step(self,action):
        if self.finished:raise RuntimeError('Reset after a terminal match')
        if not self.action_space.contains(action):raise ValueError('Action outside Discrete space')
        # Generic Gym clients can sample padding without respecting masks. Such
        # selections are explicit no-submission transitions, never illegal intents.
        masked=int(action)>=len(self.options)
        packet,reward,done,truncated,info=self._advance(self.options[0 if masked else int(action)])
        info['masked_selection']=masked
        return packet,reward,done,truncated,info

    def step_scripted(self,intent):
        """Evaluation adapter only; executes the original script without remapping votes/claims."""
        if self.finished:raise RuntimeError('Reset after a terminal match')
        if intent.kind is not IntentKind.WAIT and not intent_allowed(self.observation,intent):
            raise ValueError('Script proposed an illegal intention')
        return self._advance(Choice('scripted',intent))

    def _advance(self,selected):
        if self.render_trace:self.visual_frames=[self._visual_frame()]
        self.last_label=selected.label
        if selected.kind!='continue':self.pending=selected if selected.kind=='task' else None
        start_tick=self.match.game.tick;reward=0.;undiscounted=0.;ticks=0
        initial_phase=self.observation.context.phase
        while True:
            game=self.match.game
            obs=self.observation
            chosen=selected if ticks==0 and selected.kind!='continue' else self.pending
            intent=mechanical_intent(chosen,obs)
            self.match.tick(intent)
            if self.render_trace:self.visual_frames.append(self._visual_frame())
            ticks+=1
            elapsed=(game.tick-start_tick)*game.config.dt
            # Terminal kills may end before tick increment; account at least dt.
            elapsed=max(elapsed,ticks*game.config.dt)
            r=-.005*game.config.dt/.6
            for event in game.event_log[self._event_cursor:]:
                if event.get('kind')=='task_completed' and event.get('player_id')==self.focal:
                    task_id=event['task_id']
                    if task_id not in self._credited:
                        self._credited.add(task_id);self.completed_tasks+=1;r+=1.
            self._event_cursor=len(game.event_log)
            if not game.observe(self.focal).own.active and not self._eliminated:
                self._eliminated=True;self.survival_time=game.time;r-=5.
            if game.is_terminal:
                r+=10. if game.result.winner is Role.CREWMATE else -10. if game.result.winner is Role.IMPOSTOR else -5.
            reward+=self.gamma**((elapsed-game.config.dt)/.6)*r;undiscounted+=r
            self._observe()
            if self.observation.context.phase is not initial_phase:self.pending=None
            if game.is_terminal:break
            if not self.observation.own.active:continue  # settle team outcome without ghost observations
            if ticks>=3 or self.observation.context.phase is not initial_phase:break
        self.finished=game.is_terminal;self.total_reward+=undiscounted
        packet=self._packet()
        info={'elapsed_seconds':elapsed,'discount':self.gamma**(elapsed/.6),'undiscounted_reward':undiscounted}
        if self.finished:
            info['episode']={'return':self.total_reward,'crew_win':game.result.winner is Role.CREWMATE,
                'reason':game.result.reason,'time':game.time,'own_tasks':self.completed_tasks,
                'survival':self.survival_time if self.survival_time is not None else game.time,
                'illegal_actions':game.metrics['illegal_actions'],'navigation_failures':game.metrics['navigation_failures']}
        return packet,float(reward),self.finished,False,info
