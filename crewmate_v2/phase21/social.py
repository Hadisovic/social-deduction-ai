"""Private listener state and explicit action interventions, never role truth."""
from collections import defaultdict
import math
import numpy as np
from social_deduction.actor import Phase,Provenance
from social_deduction.phase3_api import ClaimKind,StructuredClaim,WitnessedElimination


def normalize(raw,eligible):
    ids=tuple(sorted(set(eligible)));values={p:max(0.,float(raw.get(p,0))) for p in ids}
    if any(not math.isfinite(v) for v in values.values()):raise ValueError('Nonfinite suspicion')
    total=sum(values.values())
    return {p:v/total for p,v in values.items()} if total>0 else {p:1/len(ids) for p in ids}


def shift_probability(probabilities,target,amount):
    result=dict(probabilities)
    if target not in result or len(result)<2:return result
    old=result[target];new=min(1.,max(0.,old+amount));remaining=1-old
    others=[p for p in result if p!=target]
    for pid in others:result[pid]=(1-new)*(result[pid]/remaining if remaining>1e-15 else 1/len(others))
    result[target]=new
    if not np.isfinite(list(result.values())).all() or abs(sum(result.values())-1)>1e-9:raise ValueError('Invalid social distribution')
    return result


def direct_killers(memory,eligible):
    return {e.payload.killer_id for e in memory.events if type(e.payload) is WitnessedElimination
            and e.provenance is Provenance.DIRECT and e.payload.killer_id in eligible}


class ListenerState:
    def __init__(self,owner,features):
        self.owner=owner;self.features=features;self.meeting=None;self.contributions={};self.locked=set();self.log=[]

    def update(self,obs,memory,raw):
        if obs.own.player_id!=self.owner:raise ValueError('Private listener cannot change owner')
        eligible=tuple(p for p in obs.context.participants if p!=self.owner)
        base=normalize(raw,eligible) if eligible else dict(raw)
        if not obs.own.active:return base
        meeting=obs.context.meeting_id
        if meeting!=self.meeting:self.meeting=meeting;self.contributions={};self.locked=set()
        if meeting is None or not self.features.influence:return base
        # Use only the actor's received evidence and current living participants.
        for event in obs.evidence:
            c=event.payload
            if type(c) is not StructuredClaim or event.provenance is not Provenance.CLAIM:continue
            if c.speaker_id==self.owner or c.speaker_id not in obs.context.participants or c.subject_id not in base:continue
            # Public Meeting envelope establishes the current meeting boundary.
            from social_deduction.actor import Meeting
            starts=[e.tick for e in obs.evidence if type(e.payload) is Meeting and e.payload.meeting_id==meeting]
            if not starts or event.tick<starts[-1]:continue
            sign=1 if c.kind is ClaimKind.SUSPECT_PLAYER else -1 if c.kind is ClaimKind.DEFEND_PLAYER else 0
            if not sign:continue
            key=(c.speaker_id,c.subject_id)
            if key in self.locked:continue
            if key in self.contributions:
                if self.contributions[key]!=sign:
                    self.contributions[key]=0;self.locked.add(key)
                    self.log.append(dict(tick=obs.tick,kind='conflict_cancelled',speaker=c.speaker_id,target=c.subject_id))
                continue
            self.contributions[key]=sign
            self.log.append(dict(tick=obs.tick,kind='received',speaker=c.speaker_id,target=c.subject_id,requested=sign*self.features.influence))
        if self.features.protect_direct and direct_killers(memory,base):return base
        totals=defaultdict(float)
        for (_,target),sign in self.contributions.items():totals[target]+=sign*self.features.influence
        adjusted=dict(base)
        for target,total in totals.items():
            amount=min(self.features.influence_cap,max(-self.features.influence_cap,total))
            adjusted=shift_probability(adjusted,target,amount)
        return adjusted


def decision_rules(obs,memory,options,suspicion,features):
    allowed=np.ones(len(options),dtype=bool);boost=np.zeros(len(options),np.float32);reasons=[]
    direct=direct_killers(memory,suspicion)
    if features.social_guard and obs.context.phase in (Phase.DISCUSSION,Phase.VOTING):
        for i,c in enumerate(options):
            p=suspicion.get(c.target_player);others=[v for pid,v in suspicion.items() if pid!=c.target_player]
            margin=(p-max(others,default=0)) if p is not None else -1
            if c.kind=='claim_defend':allowed[i]=p is not None and p<=features.defense_max and c.target_player not in direct
            elif c.kind=='claim_suspect':allowed[i]=p is not None and p>=features.accusation_min and margin>=features.margin
            elif c.kind=='vote':allowed[i]=p is not None and p>=features.vote_min and margin>=features.margin
        if not allowed.all():reasons.append('suspicion_admissibility')
    if features.witness_priority:
        witnessed=[e for e in memory.events if type(e.payload) is WitnessedElimination and e.provenance is Provenance.DIRECT]
        victims={e.payload.victim_id for e in witnessed};killers={e.payload.killer_id for e in witnessed}
        report=[i for i,c in enumerate(options) if c.kind=='report' and c.target_player in victims]
        if obs.context.phase is Phase.ROAMING and report:
            allowed[:]=False;allowed[report]=True;reasons.append('forced_legal_witness_report')
        elif obs.context.phase is Phase.ROAMING and witnessed:
            fresh={e.payload.victim_id for e in witnessed if (obs.tick-e.tick)*obs.settings.dt<=features.witness_fresh_seconds}
            near={b.victim_id for b in obs.visible_bodies if math.dist((obs.own.position.x,obs.own.position.y),(b.position.x,b.position.y))<=features.witness_approach_max}
            approach=[i for i,c in enumerate(options) if c.kind=='approach_body' and c.target_player in fresh&near]
            if approach:
                allowed[:]=False;allowed[approach]=True;reasons.append('forced_fresh_witness_approach')
        else:
            eligible_killers=killers&set(obs.context.participants)
            for i,c in enumerate(options):
                if c.kind=='claim_defend' and c.target_player in killers:allowed[i]=False
                # Authenticated own direct evidence outranks an erroneous model
                # threshold. This is a labelled intervention, not a new belief.
                if eligible_killers and c.kind in ('vote','claim_suspect') and c.target_player not in eligible_killers:allowed[i]=False
                if (c.kind=='claim_kill' or c.kind=='vote') and c.target_player in eligible_killers:
                    allowed[i]=True
                    boost[i]=features.witness_logit_bonus;reasons.append('direct_evidence_priority')
    if not allowed.any():raise ValueError('Experimental adapter removed all actions')
    return allowed,boost,sorted(set(reasons))


def confidence_order(scores,participants):
    if set(scores)!=set(participants):raise ValueError('Every living participant needs its own confidence provider')
    if any(not math.isfinite(v) or not 0<=v<=1 for v in scores.values()):raise ValueError('Invalid private confidence')
    return tuple(sorted(participants,key=lambda p:(-scores[p],p)))
