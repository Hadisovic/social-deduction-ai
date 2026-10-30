"""Private, observation-bound curriculum state. Never emits executable actions."""
from collections import Counter
import math
import numpy as np


class Commitment:
    def __init__(self,owner):
        self.owner=owner;self.task=None;self.started=0.;self.stage=None;self.pause=None
        self.last_progress=0.;self.best={};self.work=-1.;self.failures=0;self.failure=None
        self.meeting=False;self.incubating=False;self.route_allowed=False;self.stats=Counter();self.events=[]

    def event(self,kind,**data):self.events.append(dict(kind='commitment_'+kind,player_id=self.owner,task_id=self.task,**data))

    def start(self,task,now):
        if self.task is not None:
            if self.task!=task:raise ValueError('Committed actor started a foreign task')
            return
        self.task=task;self.started=now;self.last_progress=now;self.work=-1.;self.stage=None
        self.pause=None;self.best={};self.failure=None;self.failures=0
        self.stats['commitment_starts']+=1;self.event('start')

    def close(self,reason,now):
        if self.task is None:return
        self.stats['commitment_ended_'+reason]+=1
        self.stats['commitment_duration_seconds']+=max(0,now-self.started)
        self.event('end',reason=reason,duration=max(0,now-self.started))
        self.task=None;self.pause=None;self.stage=None;self.failure=None;self.best={}

    def failed_route(self):
        if self.task:
            self.failures+=1;self.event('route_failure',attempt=self.failures)
            if self.failures>=3:self.fail('route_failures')

    def fail(self,reason):
        if self.failure is None:
            self.failure='commitment_'+reason;self.stats['commitment_technical_failures']+=1;self.event('failure',reason=reason)

    def selected(self,packet,choice):
        from .features import target_mask
        if packet.phase==2:count=int(packet.vote_mask.sum())
        elif packet.phase==1:
            count=int(packet.communication_mask[0])+sum(int(target_mask(packet,'communication',i).sum()) for i in range(1,6) if packet.communication_mask[i])
        else:
            count=0
            if packet.interaction_mask[0]:
                count+=int(packet.navigation_mask[0])
                count+=sum(int(target_mask(packet,'navigation',i).sum()) for i in (1,2) if packet.navigation_mask[i])
            count+=sum(int(target_mask(packet,'interaction',i).sum()) for i in range(1,5) if packet.interaction_mask[i])
        self.last_forced=count==1
        self.stats['forced_decisions' if count==1 else 'voluntary_decisions']+=1
        if self.task and packet.phase==0:
            self.stats['commitment_decisions']+=1
            self.stats['commitment_forced_decisions' if count==1 else 'commitment_voluntary_decisions']+=1
            if choice.target>=0:
                target=packet.candidates[choice.target]
                pause=(target.kind,target.key) if target.kind in ('body','repair') else None
                if pause!=self.pause:
                    if pause:self.stats['commitment_exception_pauses']+=1;self.event('exception_pause',exception=pause)
                    self.pause=pause

    def sync(self,obs,nav):
        if not self.task:return
        a=obs.actor;now=a.time_seconds
        if a.own.player_id!=self.owner:raise ValueError('Commitment owner mismatch')
        if not a.own.active:self.close('death',now);return
        tasks={t.task_id:t for t in a.own.tasks}
        if self.task not in tasks:self.fail('missing_owned_task');return
        t=tasks[self.task]
        if t.progress>=1:self.close('completed',now);return
        detail=next(d for d in obs.own_tasks if d.task_id==self.task)
        phase=a.context.phase.value
        if phase!='roaming':
            self.meeting=True;self.last_progress=now;return
        if self.meeting or self.stage!=detail.stage:
            nav.goal=None;nav.original=None;self.pause=None;self.best={};self.last_progress=now
            self.event('resume' if self.meeting else 'stage',stage=detail.stage)
        self.meeting=False;self.stage=detail.stage
        if self.pause:
            kind,key=self.pause
            available=any(b.victim_id==key for b in a.visible_bodies) if kind=='body' else obs.sabotage=='lights' and key not in obs.repaired_locations
            if not available:
                self.pause=None;nav.goal=None;nav.original=None;self.event('exception_return')
        if t.progress>self.work+1e-8:
            self.work=t.progress;self.last_progress=now;self.failures=0
        self.incubating=detail.ready_in_seconds>0
        if self.incubating:self.last_progress=now
        destination=next((d for d in a.destinations if d.id==t.console_id),None)
        if destination is None:self.fail('missing_console');return
        point=(destination.standing.x,destination.standing.y);key=('task',self.stage)
        if self.pause:
            kind,k=self.pause;key=self.pause
            if kind=='body':
                b=next(b for b in a.visible_bodies if b.victim_id==k);point=(b.position.x,b.position.y)
            else:
                d=next(d for d in a.destinations if d.id==k);point=(d.standing.x,d.standing.y)
        distance=math.dist((a.own.position.x,a.own.position.y),point)
        if key not in self.best:self.best[key]=distance
        elif distance<self.best[key]-.25:self.best[key]=distance;self.last_progress=now;self.failures=0
        self.route_allowed=nav.goal is not None and (nav.goal.key==self.task or self.pause and nav.goal.key==self.pause[1]) and nav.original is None
        if now-self.last_progress>=60:self.fail('no_progress')

    def constrain(self,packet):
        if not self.task or packet.phase!=0:return
        from .features import target_mask
        a=packet.observation.actor
        task=next((t for t in a.own.tasks if t.task_id==self.task),None)
        overrides={}
        for kind,choices in (('navigation',(1,2)),('interaction',(1,2,3,4))):
            for choice in choices:
                mask=target_mask(packet,kind,choice).copy()
                for i,t in enumerate(packet.candidates):
                    allowed=(kind=='navigation' and choice==2 and
                        (t.kind=='task' and t.key==self.task and task is not None and task.console_id not in a.interactable_destinations or
                         t.kind=='body' and not target_mask(packet,'interaction',2)[i] or t.kind=='repair' and packet.observation.sabotage=='lights' and not target_mask(packet,'interaction',3)[i]) or
                        kind=='interaction' and (choice==1 and t.kind=='task' and t.key==self.task or choice==2 and t.kind=='body' or choice==3 and t.kind=='repair'))
                    # Pursuit expands the uncommitted task head, not the committed
                    # interaction support. Travel/incubation keep their old masks.
                    if kind=='interaction' and choice==1:allowed=allowed and t.key in a.interactable_tasks
                    mask[i]=mask[i] and allowed
                overrides[kind,choice]=mask
        packet.commitment_masks=overrides
        packet.interaction_mask[:]=False
        for i in (1,2,3):packet.interaction_mask[i]=overrides['interaction',i].any()
        may_redirect=bool(packet.navigation_mask[2])
        packet.navigation_mask[:]=[False,False,may_redirect and overrides['navigation',2].any()]
        # Continue the current permitted route; at a ready console, task interaction
        # is mandatory unless a legal report/repair is selected. Incubation may wait.
        moving=a.navigation_status=='moving' and self.route_allowed
        if moving or (self.incubating and task is not None and task.console_id in a.interactable_destinations):packet.navigation_mask[0]=True
        if self.failure:
            packet.interaction_mask[:]=False;packet.navigation_mask[:]=[True,False,False]
        packet.interaction_mask[0]=packet.navigation_mask.any()
        if not packet.interaction_mask.any():
            self.fail('empty_legal_support');packet.interaction_mask[0]=True;packet.navigation_mask[0]=True
