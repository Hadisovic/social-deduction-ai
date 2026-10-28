"""New trainer wrapper; original V2 source and disabled-reward behavior remain frozen."""
import math
import numpy as np
from gymnasium import spaces
from crewmate_v2.runtime import CrewmateV2Env,PublicMemory
from belief.memory import ActorMemory
from .config import RewardConfig
from .opponents import LearningMatch


class LearningEnv(CrewmateV2Env):
    def __init__(self,*,rewards=None,opponents='original',rotation=0,coalesce=False,deadline=None,**kwargs):
        if rewards is not None and type(rewards) is not RewardConfig:raise TypeError('Use a versioned RewardConfig or None')
        if type(coalesce) is not bool:raise ValueError('Coalescing switch must be boolean')
        self.rewards=rewards;self.opponents=opponents;self.rotation=rotation;self.coalesce=coalesce;self.deadline=deadline
        self._learning_ready=False
        super().__init__(**kwargs)
        self.observation_space.spaces['task_targets']=spaces.Box(-1,3,(96,),np.int64)

    def reset(self,*,seed=None,options=None):
        import gymnasium as gym
        gym.Env.reset(self,seed=seed)
        selected=int(self.np_random.integers(0,2**31)) if seed is None else int(seed)
        self.match=LearningMatch(selected,lobby=self.lobby,mixture=self.mixture,planner=self.planner,fast=self.fast,
            reward_config=self.rewards or RewardConfig.legacy(),opponents=self.opponents,rotation=self.rotation,deadline=self.deadline)
        self.focal=self.match.focal;self.memory=ActorMemory();self.public_memory=PublicMemory();self.pending=None
        self._previous_tick=-1;self._event_cursor=0;self._credited=set();self._eliminated=False
        self.finished=False;self.total_reward=0.;self.last_label='continue';self.completed_tasks=0;self.survival_time=None;self.visual_frames=[]
        self._learning_ready=True;self._observe()
        return self._packet(),{}

    def _packet(self):
        packet=super()._packet()
        rows={t.task_id:i for i,t in enumerate(sorted(self.observation.own.tasks,key=lambda t:t.task_id))}
        targets=np.full(96,-1,np.int64)
        for i,choice in enumerate(self.options):
            if choice.kind=='task':targets[i]=rows[choice.task_id]
        packet['task_targets']=targets
        return packet

    def _advance(self,selected):
        mark=len(self.match.ledger.frames)
        packet,original,done,truncated,info=super()._advance(selected)
        scored=self.match.ledger.rollup(mark,self.gamma)
        if not math.isclose(scored['elapsed_seconds'],info['elapsed_seconds'],abs_tol=1e-7):raise AssertionError('Reward/option clock mismatch')
        if self.rewards is None and not math.isclose(scored['reward'],original,abs_tol=1e-6):raise AssertionError('Legacy reward reproduction failed')
        reward=original if self.rewards is None else scored['reward']
        if self.rewards is not None:
            self.total_reward=sum(self.match.ledger.components.values())
            info.update(undiscounted_reward=scored['undiscounted_reward'])
            if done:info['episode']['return']=self.total_reward
        info.update(reward_components=scored['components'],discounted_reward_components=scored['discounted_components'],
            reward_counts=scored['counts'],learning_execution='v2-learning-options-v1')
        if done:
            info['episode'].update(alive_tasks=self.match.ledger.counts['alive_tasks'],ghost_tasks=self.match.ledger.counts['ghost_tasks'],
                physical_body_visible_unique=len(self.match.report_visible),physical_report_legal_unique=len(self.match.report_legal),
                scheduled_report_legal_windows=self.match.report_legal_windows,
                reward_components=dict(self.match.ledger.components),reward_counts=dict(self.match.ledger.counts))
        return packet,float(reward),done,truncated,info

    def step(self,action):
        mark=len(self.match.ledger.frames)
        packet,reward,done,truncated,info=super().step(action)
        masked=info['masked_selection']
        forced=0
        while self.coalesce and not done and len(self.options)==1:
            packet,_,done,truncated,info2=self._advance(self.options[0]);forced+=1
            info=info2
        if forced:
            scored=self.match.ledger.rollup(mark,self.gamma)
            info.update(elapsed_seconds=scored['elapsed_seconds'],discount=scored['discount'],
                undiscounted_reward=scored['undiscounted_reward'],reward_components=scored['components'],
                discounted_reward_components=scored['discounted_components'],reward_counts=scored['counts'])
            reward=scored['reward']
        info['coalesced_forced_windows']=forced
        info['masked_selection']=masked
        return packet,float(reward),done,truncated,info
