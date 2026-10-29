"""Actor-local V4.1 feature/option adapter. No engine, recorder or crew state here."""
from dataclasses import dataclass
from collections import Counter
import math
import hashlib
import numpy as np
import torch
from crewmate_v2.api import V2Observation,validate
from crewmate_v2.runtime import PublicMemory
from social_deduction.actor import Phase,Role
from social_deduction.phase3_api import Intent,IntentKind,intent_allowed
from .features import candidates,encode,MAX_ACTIONS,ACTION_COUNT
from .memory import ImpostorMemory
from .model import family
from .pursuit import PursuitPotential

@dataclass(frozen=True,slots=True)
class ActorContext:
    """Explicit v1 extension: legal V2 envelope + only this owner's fake timer."""
    envelope: V2Observation
    own_fake_remaining_seconds: float

    def checked(self,observation):
        if type(self) is not ActorContext:raise TypeError('Exact allowlisted context required')
        validate(self.envelope)
        if self.envelope.actor!=observation:raise ValueError('Stale/wrong-owner context')
        if type(self.own_fake_remaining_seconds) not in (float,int) or not math.isfinite(self.own_fake_remaining_seconds) or not 0<=self.own_fake_remaining_seconds<=4.000001:
            raise ValueError('Invalid own fake timer')
        return self

class Controller:
    def __init__(self,policy,*,seed,player_id,stochastic=False):
        self.policy=policy;self.player_id=player_id;self.stochastic=stochastic
        private_seed=int.from_bytes(hashlib.sha256(f'impostor-shared-v1:{int(seed)}:{player_id}'.encode()).digest()[:8],'big') & ((1<<63)-1)
        self.generator=torch.Generator().manual_seed(private_seed)
        self.memory=ImpostorMemory();self.public_memory=PublicMemory();self.pursuit=PursuitPotential()
        self.context=None;self.goal=None;self.lease=None;self.lease_time=0.;self.lease_phase=None
        self.seen=set();self.bodies=set();self.held_callbacks=0;self.decisions=0
        self.last_decision=None;self.actions=Counter();self.pending_pursuit=0.;self.fake_repeats=0

    def supply(self,context):
        if type(context) is not ActorContext:raise TypeError('Only ActorContext crosses the boundary')
        self.context=context

    def decide(self,obs):
        if obs.own.player_id!=self.player_id or obs.own.role is not Role.IMPOSTOR or not obs.own.active or obs.context.phase is Phase.FINISHED:
            raise ValueError('Wrong owner/role or inactive actor')
        context=self.context.checked(obs)
        self.memory.update(obs);self.public_memory.update(context.envelope)
        self.pending_pursuit+=self.pursuit.advance(obs)
        visible={p.player_id for p in obs.visible_players};bodies={b.victim_id for b in obs.visible_bodies}
        hold=(self.lease is not None and obs.context.phase is self.lease_phase is Phase.ROAMING
              and obs.time_seconds-self.lease_time<3.-1e-8 and obs.navigation_status=='moving'
              and not visible-self.seen and not bodies-self.bodies
              and not any(a.kind is IntentKind.KILL for a in obs.actions))
        if hold:
            self.held_callbacks+=1
            return Intent(IntentKind.WAIT)
        self.lease=None
        options=candidates(context.envelope,self.memory)
        packet=encode(context.envelope,self.memory,self.public_memory,options)
        x,y=obs.own.position.x,obs.own.position.y
        body_distance=min((math.hypot(b.position.x-x,b.position.y-y)/60 for b in obs.visible_bodies),default=0.)
        packet['extra']=np.asarray([float(bool(obs.visible_bodies)),min(body_distance,1.),self.pursuit.best,
            self.pursuit.credit/.05,(self.goal.x-x)/60 if self.goal else 0.,(self.goal.y-y)/60 if self.goal else 0.,
            float(self.goal is not None and obs.navigation_status=='moving'),float(any(a.kind is IntentKind.KILL for a in obs.actions))],np.float32)
        packet['global_'][3]=context.own_fake_remaining_seconds/4 if obs.interaction_task and obs.interaction_task.startswith('fake:') else 0.
        geometry=np.zeros((MAX_ACTIONS,3),np.float32);seen={p.player_id:p for p in obs.visible_players}
        for i,c in enumerate(options):
            target=seen.get(c.target)
            if target:
                distances=[math.hypot(p.position.x-target.position.x,p.position.y-target.position.y) for p in obs.visible_players if p.player_id!=target.player_id]
                geometry[i]=[1.,min(min(distances,default=20.)/20,1.),sum(d<=4.5 for d in distances)/3]
        packet['actions']=np.concatenate((packet['actions'],geometry),axis=1)
        if (packet['global_'].shape!=(35,) or packet['players'].shape!=(4,21) or packet['extra'].shape!=(8,)
            or packet['actions'].shape!=(256,51) or not packet['mask'].any()):raise ValueError('Feature schema mismatch')
        index,logp,value=self.policy.act(packet,generator=self.generator,deterministic=not self.stochastic)
        if not packet['mask'][index]:raise ValueError('Neural policy chose a masked action')
        with torch.no_grad():distribution,_,families,_,_=self.policy.evaluate({k:torch.as_tensor(v).unsqueeze(0) for k,v in packet.items()})
        choice=options[index];self.actions[choice.kind]+=1;self.decisions+=1
        labels=[f'{i:03d}: {c.kind} {c.target or (c.intent.target if c.intent else "") or c.region}' for i,c in enumerate(options)]
        self.last_decision=dict(tick=obs.tick,time_seconds=obs.time_seconds,packet=packet,index=index,logp=logp,value=value,
            kind=choice.kind,target=choice.target or (choice.intent.target if choice.intent else None),
            probabilities={label:float(distribution.probs[0,i]) for i,label in enumerate(labels)},
            family_probabilities=families.probs[0].tolist(),entropy=float(distribution.entropy()[0]),labels=labels)
        if family(choice.kind)=='travel':
            self.goal=choice.position;self.lease=choice;self.lease_time=obs.time_seconds;self.lease_phase=obs.context.phase
            self.seen=visible;self.bodies=bodies
        elif choice.kind=='cancel_navigation':self.goal=None
        action=choice.intent or Intent(IntentKind.WAIT)
        if action.kind is IntentKind.FAKE_TASK and obs.interaction_task=='fake:'+action.target:
            self.fake_repeats+=1;action=Intent(IntentKind.WAIT)
        if action.kind is not IntentKind.WAIT and not intent_allowed(obs,action):raise ValueError('Decoded stale/illegal action')
        return action
