"""Scalar observational summaries; actual roles are inspected only after play."""
from collections import Counter
import math
from pathlib import Path
from social_deduction.actor import Phase


class DiagnosticRecorder:
    """Optional read-only file telemetry, not engine publications or actor inputs."""
    def __init__(self,delegate,match):self.delegate=delegate;self.match=match
    def __getattr__(self,name):return getattr(self.delegate,name)
    def record_neural(self,pid,*args,**kwargs):
        detail=self.match.pending_details[pid];labels=kwargs.get('legal_actions',[])
        kwargs['masked_actions']=[label for label,allowed in zip(labels,detail['allowed']) if not allowed]
        self.delegate.record_neural(pid,*args,**kwargs)
        self.delegate.recorder.write_event(dict(kind='experimental_decision_adapter',channel='truth',
            tick=detail['tick'],time=detail['time'],actor_id=pid,version=self.match.features.schema,
            interpretation='Frozen NN plus explicit action intervention; not learned improvement',
            raw_belief=detail['raw_belief'],adjusted_suspicion=detail['adjusted_suspicion'],
            raw_probabilities=detail['raw_probabilities'],effective_probabilities=detail['effective_probabilities'],
            interventions=detail['reasons']))


def summarize(match,summary):
    roles={pid:p.role.value for pid,p in match.game.players.items()};events=match.game.event_log[match.event_start:]
    ballots=[e for e in events if e['kind']=='vote' and e.get('channel')=='public' and roles.get(e['voter_id'])=='crewmate']
    claims=[e for e in events if e['kind']=='claim' and roles.get(e['speaker_id'])=='crewmate']
    counts=Counter()
    counts['correct_votes']=sum(e.get('target_id') is not None and roles.get(e['target_id'])=='impostor' for e in ballots)
    counts['incorrect_votes']=sum(e.get('target_id') is not None and roles.get(e['target_id'])=='crewmate' for e in ballots)
    counts['published_skips']=sum(e.get('target_id') is None for e in ballots)
    counts['correct_accusations']=sum(e.get('claim_kind') in ('suspect_player','saw_elimination') and roles.get(e.get('subject_id'))=='impostor' for e in claims)
    counts['incorrect_accusations']=sum(e.get('claim_kind') in ('suspect_player','saw_elimination') and roles.get(e.get('subject_id'))=='crewmate' for e in claims)
    for c in claims:
        if c.get('claim_kind')=='defend_player':
            for vote in ballots:
                if vote['voter_id']==c['speaker_id'] and vote.get('target_id')==c.get('subject_id') and vote['time']>=c['time']:
                    counts['defense_then_vote_links']+=1
    scored=[];seen=set();interventions=Counter();max_influence=0.;chosen=Counter()
    for row in match.decision_log:
        chosen[row['selected_kind']]+=1;interventions.update(row['reasons'])
        c=row['options'][row['selected']];p=row['adjusted_suspicion'].get(c['target'])
        if c['kind']=='claim_defend' and p is not None and p>match.features.defense_max:counts['defenses_contradicting_suspicion']+=1
        if row['phase']!='voting' or row['actor'] in seen:continue
        seen.add(row['actor']);base=row['eligible_base'];adjusted=row['adjusted_suspicion']
        if not adjusted:continue
        for name,probabilities in (('base',base),('adjusted',adjusted)):
            targets=[pid for pid in probabilities if roles.get(pid)=='impostor']
            if len(targets)!=1:continue
            truth=targets[0];best=max(probabilities.values());ties=[p for p,v in probabilities.items() if abs(v-best)<1e-10]
            scored.append(dict(kind=name,actor=row['actor'],nll=-math.log(max(1e-12,probabilities[truth])),
                brier=sum((v-float(pid==truth))**2 for pid,v in probabilities.items()),
                top1=float(truth in ties)/len(ties),confidence=best,high_confidence_wrong=best>=.85 and truth not in ties))
        max_influence=max(max_influence,sum(abs(base.get(p,0)-v) for p,v in adjusted.items()))
    calibration={}
    for name in ('base','adjusted'):
        rows=[r for r in scored if r['kind']==name];n=len(rows)
        bins=[];ece=0.
        for lo,hi in ((0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.000001)):
            b=[r for r in rows if lo<=r['confidence']<hi]
            if b:
                conf=sum(r['confidence'] for r in b)/len(b);acc=sum(r['top1'] for r in b)/len(b);ece+=len(b)/max(1,n)*abs(conf-acc)
                bins.append(dict(n=len(b),confidence=conf,accuracy=acc))
        calibration[name]=dict(actor_meetings=n,nll=sum(r['nll'] for r in rows)/n if n else None,
            brier=sum(r['brier'] for r in rows)/n if n else None,top1=sum(r['top1'] for r in rows)/n if n else None,
            ece=ece if n else None,high_confidence_wrong=sum(r['high_confidence_wrong'] for r in rows),bins=bins)
    targets=Counter(e['target_id'] for e in ballots if e.get('target_id') is not None)
    result=dict(context=getattr(match,'context','ordinary'),complete_match=match.game.is_terminal,
        team_outcome=match.game.result.winner.value if match.game.is_terminal and match.game.result.winner else None,
        counts=dict(counts),chosen_actions=dict(chosen),interventions=dict(interventions),calibration=calibration,
        max_voting_L1_influence=max_influence,non_skip_vote_consensus=max(targets.values())/sum(targets.values()) if targets else None,
        witness_reports=match.physical_reports,order=match.order_log,
        listener_log_counts={pid:len(s.log) for pid,s in match.listeners.items()},
        experimental_features=match.features.to_dict(),new_events=events)
    return {**summary,**result}
