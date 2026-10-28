"""Offline reconstruction. Role labels are never passed to acting controllers."""
from collections import Counter, defaultdict
import json
from pathlib import Path


def when(row):return float(row.get('time',row.get('sim_time',0)))


def action_probability(neural,kind,target):
    return sum(v for label,v in neural.get('action_probabilities',{}).items()
               if label.split(': ',1)[-1]==kind+((' / '+str(target)) if target is not None else ''))


def event_scope(events):
    """Assign claims lacking a meeting ID to the recorded current meeting."""
    meeting=None;result=[]
    for event in events:
        if event.get('kind')=='meeting_started':meeting=event.get('meeting_id')
        e=dict(event);e['_meeting']=e.get('meeting_id',meeting);result.append(e)
        if event.get('kind') in ('meeting_ended','meeting_resolved','exploration_resumed'):meeting=None
    return result


def audit_replay(path):
    with Path(path).open(encoding='utf-8') as f:rows=[json.loads(line) for line in f]
    header=next(r for r in rows if r['record']=='header');cfg=header.get('config',{})
    events=event_scope([r for r in rows if r['record']=='event'])
    neural=[r for r in rows if r['record']=='neural'];rewards=[r for r in rows if r['record']=='reward']
    roles=header.get('agent_roles',{});cases=[];counter=Counter();belief_action=Counter()
    def decision(actor,event,kind):
        candidates=[n for n in neural if n['player_id']==actor and when(n)<=when(event)+1e-8 and n.get('action_kind')==kind]
        return max(candidates,key=when) if candidates else {}
    for n in neural:
        kind=n.get('action_kind');target=n.get('action_target');scores=n.get('suspicion_probabilities',{})
        if kind=='claim_defend':
            belief_action['selected_defenses']+=1
            if scores.get(target,0)>=.8:belief_action['defend_target_suspicion_at_least_80pct']+=1
        if kind=='claim_suspect':
            belief_action['selected_accusations']+=1
            if target in scores and scores[target]<=.2:belief_action['accuse_target_suspicion_at_most_20pct']+=1
    defenses=[e for e in events if e.get('kind')=='claim' and e.get('claim_kind')=='defend_player']
    votes=[e for e in events if e.get('kind')=='vote_submitted' and e.get('target_id') is not None]
    for defend in defenses:
        actor=defend['speaker_id'];target=defend['subject_id']
        for vote in votes:
            if when(vote)<when(defend) or vote.get('target_id')!=target:continue
            same_actor=vote['voter_id']==actor;same_meeting=vote['_meeting']==defend['_meeting']
            if not same_actor:
                counter['different_agent_later_vote_links']+=1;continue
            if not same_meeting:
                counter['same_agent_later_meeting_links']+=1;continue
            n1=decision(actor,defend,'claim_defend');n2=decision(actor,vote,'vote')
            between=[e for e in events if when(defend)<when(e)<=when(vote) and e.get('channel')=='public'
                     and not (e.get('kind')=='claim' and e.get('speaker_id')==actor)]
            b1=n1.get('suspicion_probabilities',{});b2=n2.get('suspicion_probabilities',{})
            delta=max((abs(b1.get(pid,0)-b2.get(pid,0)) for pid in set(b1)|set(b2)),default=None)
            chosen=n1.get('selected_action_index');ps=list(n1.get('action_probabilities',{}).values())
            selectedp=ps[chosen] if isinstance(chosen,int) and 0<=chosen<len(ps) else None
            reward_after=[r for r in rewards if r.get('player_id')==actor and when(defend)<=when(r)<=when(vote)+1.]
            case=dict(actor=actor,target=target,target_is_impostor=roles.get(target)=='impostor',meeting=defend['_meeting'],
                defend_time=when(defend),vote_time=when(vote),defend_neural=n1,vote_neural=n2,
                target_belief_at_defense=b1.get(target),target_belief_at_vote=b2.get(target),max_belief_change=delta,
                selected_defense_probability=selectedp,defense_was_argmax=(selectedp>=max(ps)-1e-9 if selectedp is not None and ps else None),
                intervening_public_events=between,reward_intervals=reward_after,
                private_observations='Not recorded; no absence-of-private-evidence claim',
                recorded_reward_profile=cfg.get('reward_profile',{}))
            cases.append(case);counter['same_actor_same_meeting_defense_vote']+=1
            counter['defended_impostor_then_voted_impostor']+=case['target_is_impostor']
            counter['defended_innocent_then_voted_innocent']+=not case['target_is_impostor']
            counter['belief_unchanged_within_1e6']+=delta is not None and delta<1e-6
            counter['no_intervening_public_events']+=not between
            counter['no_intervening_public_claims']+=not any(e['kind']=='claim' for e in between)
            counter['defense_argmax']+=case['defense_was_argmax'] is True
            counter['target_suspected_at_defense_80pct']+=(b1.get(target,0)>=.8)
    # Historical opportunity denominators for the impostor are absent. Don't infer them from truth frames.
    reports=[e for e in events if e.get('kind')=='report']
    kills={e.get('victim_id'):e.get('killer_id') for e in events if e.get('kind') in ('elimination','kill')}
    report_counts=Counter();prefix_reports=Counter()
    for e in reports:
        role=roles.get(e.get('reporter_id'),'unknown');report_counts[role]+=1
        if kills.get(e.get('victim_id'))==e.get('reporter_id'):report_counts['self_reports']+=1
        if when(e)<cfg.get('prefix_handoff_seconds',0)-1e-8:
            prefix_reports[role]+=1
    return dict(file=Path(path).name,condition=cfg.get('training_condition'),context=cfg.get('context'),
                training=bool(cfg.get('is_training',False) or 'training' in Path(path).stem),
                seed=header.get('scenario_seed'),cases=cases,counters=dict(counter),belief_action=dict(belief_action),
                reports=dict(report_counts),prefix_reports=dict(prefix_reports),total_kills=len(kills),events=events)


def audit_library(folder):
    groups=defaultdict(lambda:dict(games=0,counters=Counter(),belief_action=Counter(),reports=Counter(),prefix_reports=Counter(),kills=0))
    raw=[]
    for path in sorted(Path(folder).rglob('*.jsonl')):
        game=audit_replay(path);raw.append(game)
        # Both splits use context-seed names; use the registered seed partitions.
        seed=game['seed']
        split=('training_sample' if 2780000<=seed<=2780063 else
               'validation' if 2790000<=seed<=2790007 else 'engineering')
        key=f"{split}/{game['condition']}/{game['context']}";g=groups[key];g['games']+=1
        for field in ('counters','belief_action','reports','prefix_reports'):g[field].update(game[field])
        g['kills']+=game['total_kills']
    return {k:{f:dict(v) if isinstance(v,Counter) else v for f,v in g.items()} for k,g in groups.items()},raw
