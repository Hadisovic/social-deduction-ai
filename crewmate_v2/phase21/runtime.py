"""Explicit frozen-policy engineering bridge; no optimizer or historical edits."""
from collections import Counter
from dataclasses import asdict
from time import perf_counter
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from crewmate_v2.fourcrew.match import FourCrewMatch,ActorView
from crewmate_v2.learning.train import state_hash
from phase3_runner import behavior_seed
from social_deduction.actor import Phase
from social_deduction.phase3_api import WitnessedElimination,IntentKind
from .config import Features
from .social import ListenerState,normalize,decision_rules,confidence_order
from .rewards import EligibleLedger
from .opponents import CoherentMixture,local_proxy
from crewmate_v2.phase2.opponents import MixtureController


class SegmentFinished(Exception):pass


class DecisionAdapter(nn.Module):
    def __init__(self,base,match):
        super().__init__();self.base=base;self.match=match;self.queue=[]

    def forward(self,packet):
        distribution,values=self.base(packet)
        if len(self.queue)!=len(values):raise ValueError('Decision owner/batch mismatch')
        logits=distribution.logits.clone()
        for i,(pid,view) in enumerate(self.queue):
            obs=view.observation;raw=dict(view.belief.by_player)
            suspicion=self.match.listeners[pid].update(obs,view.memory,raw)
            allow,boost,reasons=decision_rules(obs,view.memory,view.options,suspicion,self.match.features)
            count=len(view.options);mask=torch.as_tensor(allow,device=logits.device)
            logits[i,:count]=torch.where(mask,logits[i,:count]+torch.as_tensor(boost,device=logits.device),torch.full_like(logits[i,:count],-1e9))
            self.match.pending_details[pid]=dict(actor=pid,tick=obs.tick,time=obs.time_seconds,phase=obs.context.phase.value,
                raw_belief=raw,eligible_base=normalize(raw,suspicion) if suspicion else {},adjusted_suspicion=suspicion,
                raw_probabilities=distribution.probs[i,:count].tolist(),allowed=allow.tolist(),boost=boost.tolist(),reasons=reasons,
                strong_candidates=[p for p,v in suspicion.items() if v>=self.match.features.strong_min],
                options=[dict(kind=c.kind,target=c.target_player) for c in view.options])
        self.queue=[]
        effective=Categorical(logits=logits)
        for i,pid in enumerate(self.match.batch_ids):self.match.pending_details[pid]['effective_probabilities']=effective.probs[i,:len(self.match.views[pid].options)].tolist()
        self.match.batch_ids=[]
        return effective,values


class MechanismView(ActorView):
    def _packet(self):
        packet=super()._packet()
        if len(self.options)>1 and self.bridge.adapter is not None:
            self.bridge.adapter.queue.append((self.focal,self));self.bridge.batch_ids.append(self.focal)
        return packet


class ExperimentalMatch(FourCrewMatch):
    @classmethod
    def adopt(cls,prepared,features=Features(),*,deadline=None,stop_after_meeting=False,max_roaming_seconds=None):
        """Take an OWNED prepared/cloned match; never mutate a historical match."""
        if prepared.policy_version is not None:raise ValueError('Cannot reuse a collected match')
        obj=cls.__new__(cls);obj.__dict__.update(prepared.__dict__)
        obj.features=features;obj.deadline=deadline or perf_counter()+120;obj.stop_after_meeting=stop_after_meeting
        obj.max_roaming_seconds=max_roaming_seconds;obj.segment_reason='MEETING_SEGMENT_END'
        obj.adapter=None;obj.batch_ids=[];obj.pending_details={};obj.decision_log=[];obj.order_log=[]
        obj.listeners={pid:ListenerState(pid,features) for pid in obj.crew};obj.witness_responses={};obj.order_frozen=set()
        obj.event_start=len(obj.game.event_log);obj.start_time=obj.game.time;obj.segment_done=False
        obj.physical_reports={pid:dict(first_witness=None,first_legal=None,legal_ticks=0,submissions=0,accepted=0,approaches=0,accepted_times=[]) for pid in obj.crew}
        obj.pre_step_active={pid:obj.game.players[pid].active for pid in obj.crew}
        for pid,view in list(obj.views.items()):
            new=MechanismView.__new__(MechanismView);new.__dict__.update(view.__dict__);new.bridge=obj;obj.views[pid]=new
        roles={pid:p.role for pid,p in obj.game.players.items()}
        for pid,old in list(obj.ledgers.items()):
            ledger=EligibleLedger(pid,roles,old.config,obj.game.config.dt)
            ledger.alive=obj.game.players[pid].active;ledger.death_paid=not ledger.alive;obj.ledgers[pid]=ledger
        if features.opponent=='v1':obj.match.controllers[obj.impostor]=MixtureController(behavior_seed(obj.seed,obj.impostor))
        elif features.opponent=='v2':obj.match.controllers[obj.impostor]=CoherentMixture(behavior_seed(obj.seed,obj.impostor))
        step=obj.game.step
        def observed_step(actions=None):
            if perf_counter()>obj.deadline:raise TimeoutError('Phase21 engineering deadline')
            actions=actions or {};obj.pre_step_active={pid:obj.game.players[pid].active for pid in obj.crew}
            for pid in obj.crew:
                if not obj.pre_step_active[pid]:continue
                obs=obj.game.observe(pid);row=obj.physical_reports[pid]
                witnesses=[e for e in obs.evidence if type(e.payload) is WitnessedElimination]
                if witnesses and row['first_witness'] is None:row['first_witness']=min(e.tick for e in witnesses)*obs.settings.dt
                if witnesses and any(a.kind is IntentKind.REPORT and a.target in {e.payload.victim_id for e in witnesses} for a in obs.actions):
                    if row['first_legal'] is None:row['first_legal']=obj.game.time
                    row['legal_ticks']+=1
                intent=actions.get(pid)
                if intent and intent.kind is IntentKind.REPORT:row['submissions']+=1
                if intent and intent.kind is IntentKind.GO_TO_BODY:row['approaches']+=1
            mark=len(obj.game.event_log);result=step(actions)
            for e in obj.game.event_log[mark:]:
                if e['kind']=='report' and e['reporter_id'] in obj.physical_reports:
                    row=obj.physical_reports[e['reporter_id']];row['accepted']+=1;row['accepted_times'].append(dict(victim=e['victim_id'],time=e['time']))
                if e['kind']=='meeting_resolution':obj.segment_done=True
            obj.freeze_order()
            return result
        obj.game.step=observed_step
        obj.freeze_order()
        return obj

    def freeze_order(self):
        game=self.game
        if self.features.speaking_order!='confidence' or game.phase is not Phase.DISCUSSION or game.meeting_id in self.order_frozen:return
        scores={};providers={}
        for pid in game.participants:
            obs=game.observe(pid)
            if pid in self.views:
                view=self.views[pid];view.refresh();raw=view.belief_model.predict(view.memory).by_player
                probs=self.listeners[pid].update(obs,view.memory,raw);providers[pid]='own_frozen_crew_belief'
            else:probs=local_proxy(obs);providers[pid]='own_local_script_proxy'
            scores[pid]=max(probs.values(),default=0.)
        order=confidence_order(scores,game.participants);game.participants=order;game._invalidate();self.order_frozen.add(game.meeting_id)
        self.order_log.append(dict(meeting=game.meeting_id,order=order,private_scores=scores,providers=providers,explicit_confidence_disclosure=True))
        game._record('experimental_discussion_order',channel='public',meeting_id=game.meeting_id,order=order,version=self.features.schema)
        for view in self.views.values():view.refresh()

    def _ready(self,pid):
        if self.stop_after_meeting and self.segment_done:raise SegmentFinished()
        if self.max_roaming_seconds is not None and self.game.phase is Phase.ROAMING and self.game.time-self.start_time>=self.max_roaming_seconds:
            self.segment_reason='ROAMING_SEGMENT_END';raise SegmentFinished()
        if self.features.witness_priority:
            view=self.views[pid];obs=view.observation
            if obs.context.phase is Phase.ROAMING:
                witnessed=[e for e in view.memory.events if type(e.payload) is WitnessedElimination]
                if witnessed:
                    latest=max(e.tick for e in witnessed);victims={e.payload.victim_id for e in witnessed}
                    if any(a.kind is IntentKind.REPORT and a.target in victims for a in obs.actions):return True
                    if latest>self.witness_responses.get(pid,-1) and (obs.tick-latest)*obs.settings.dt<=self.features.witness_fresh_seconds:return True
        return super()._ready(pid)

    def _lease(self,pid,choice):
        super()._lease(pid,choice)
        witnessed=[e.tick for e in self.views[pid].memory.events if type(e.payload) is WitnessedElimination]
        if witnessed:self.witness_responses[pid]=max(witnessed)

    def collect(self,policy,**kwargs):
        self.adapter=DecisionAdapter(policy,self);original=kwargs.pop('decision_observer',None)
        def observe(pid,observation,memory,belief,options,probabilities,packet,selected):
            detail=dict(self.pending_details[pid]);detail['selected']=selected;detail['selected_kind']=options[selected].kind
            detail['selected_target']=options[selected].target_player;detail['direct_witnesses']=[asdict(e) for e in memory.events if type(e.payload) is WitnessedElimination]
            self.decision_log.append(detail)
            if original:original(pid,observation,memory,belief,options,probabilities,packet,selected)
        before=state_hash(policy)
        try:return super().collect(self.adapter,decision_observer=observe,**kwargs)
        except SegmentFinished:
            if state_hash(policy)!=before:raise ValueError('Frozen policy changed during segment')
            if self.adapter.queue:raise ValueError('Unconsumed actor batch')
            return None,dict(seed=self.seed,decisions=self.decisions,reason=self.segment_reason,crew_win=None,
                simulated_seconds=self.game.time,segment=True,complete_match=False,
                actors={pid:dict(actions=dict(self.actions[pid]),reward_components=dict(self.ledgers[pid].components),
                    reward_counts=dict(self.ledgers[pid].counts),reporting=dict(self.reporting[pid])) for pid in self.crew},
                illegal_actions=self.game.metrics['illegal_actions'])
