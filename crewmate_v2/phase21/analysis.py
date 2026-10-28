"""Read-only historical witnesses/clustering; spectator geometry is offline only."""
from collections import Counter,defaultdict
import json
import math
from pathlib import Path


def analyze_file(path):
    rows=[json.loads(line) for line in Path(path).open(encoding='utf-8')]
    header=next(r for r in rows if r['record']=='header');roles=header['agent_roles'];cfg=header.get('config',{})
    handoff=cfg.get('prefix_handoff_seconds',0);events=[r for r in rows if r['record']=='event'];neural=[r for r in rows if r['record']=='neural']
    ticks=[r for r in rows if r['record']=='tick'];counts=Counter();grouped=Counter();travel=0.;alive_seconds=0.
    for previous,current in zip(ticks,ticks[1:]):
        if previous['phase']!='roaming' or current['phase']!='roaming' or previous['sim_time']<handoff:continue
        dt=current['sim_time']-previous['sim_time'];crew=[pid for pid,p in previous['players'].items() if roles[pid]=='crewmate' and p['status']=='alive']
        for pid in crew:
            alive_seconds+=dt;a=previous['players'][pid]['position'];b=current['players'][pid]['position'];travel+=math.dist(a,b)
            nearest=min((math.dist(a,previous['players'][other]['position']) for other in crew if other!=pid),default=math.inf)
            for threshold in (1.3,2.5,4.5):
                if nearest<=threshold:grouped[str(threshold)]+=dt
    witnesses=[];murders=[e for e in events if e['kind']=='elimination']
    for kill in murders:
        observers={e['observer_id'] for e in events if e['kind']=='elimination_witnessed' and e['tick']==kill['tick'] and e.get('victim_id')==kill['victim_id']
                   and roles.get(e['observer_id'])=='crewmate' and e['observer_id']!=kill['victim_id']}
        counts['kills_total']+=1;counts['kills_with_surviving_crew_witness']+=bool(observers)
        for observer in observers:
            reports=[e for e in events if e['kind']=='report' and e['victim_id']==kill['victim_id'] and e['time']>=kill['time']]
            end=min([e['time'] for e in reports]+[ticks[-1]['sim_time']])
            decisions=[n for n in neural if n['player_id']==observer and kill['time']<=n['sim_time']<=end+1e-8]
            legal=[n for n in decisions if any(label.split(': ',1)[-1]=='report / '+kill['victim_id'] for label in n.get('legal_actions',[]))]
            own=[e for e in reports if e['reporter_id']==observer]
            death=next((e for e in murders if e['victim_id']==observer and e['time']>kill['time']),None)
            witnesses.append(dict(actor=observer,victim=kill['victim_id'],kill_time=kill['time'],prefix=kill['time']<handoff,
                report_time=reports[0]['time'] if reports else None,own_report_time=own[0]['time'] if own else None,
                first_choice=decisions[0]['action_kind'] if decisions else None,
                first_choice_latency=decisions[0]['sim_time']-kill['time'] if decisions else None,
                legal_reporting_choices=len(legal),legal_choices_not_reporting=sum(n['action_kind']!='report' for n in legal),
                report_selected=sum(n['action_kind']=='report' and n.get('action_target')==kill['victim_id'] for n in decisions),
                died_before_body_report=bool(death and (not reports or death['time']<=end))))
    for n in neural:
        if n['sim_time']>=handoff:counts['action_'+n['action_kind']]+=1
    dead=set();task_events=0
    for e in events:
        if e['kind']=='elimination':dead.add(e['victim_id'])
        if e['kind']=='ejection' and e.get('player_id'):dead.add(e['player_id'])
        if e['kind']=='task_completed' and e['time']>=handoff:
            counts['ghost_task_events' if e['player_id'] in dead else 'living_task_events']+=1;task_events+=1
    nav=[e for e in events if e['kind']=='navigate' and roles.get(e.get('player_id'))=='crewmate' and e['time']>=handoff]
    common=sum(a['player_id']!=b['player_id'] and a['target']==b['target'] and abs(a['time']-b['time'])<=2 for i,a in enumerate(nav) for b in nav[i+1:])
    return dict(seed=header['scenario_seed'],condition=cfg.get('training_condition'),context=cfg.get('context'),
        counts=dict(counts),grouped_actor_seconds=dict(grouped),roaming_alive_actor_seconds=alive_seconds,
        roaming_crew_distance=travel,same_target_request_pairs_within_2s=common,witnesses=witnesses)


def analyze_library(folder):
    groups=defaultdict(lambda:dict(games=0,counts=Counter(),grouped_actor_seconds=Counter(),roaming_alive_actor_seconds=0.,roaming_crew_distance=0.,same_target_request_pairs_within_2s=0,witness_counts=Counter()))
    raw=[]
    for path in sorted(Path(folder).rglob('*.jsonl')):
        row=analyze_file(path);raw.append(row);seed=row['seed']
        split='registered_validation' if 2790000<=seed<=2790007 else 'sampled_training' if 2780000<=seed<=2780063 else 'phase2_engineering'
        key=f"{split}/{row['condition']}/{row['context']}";g=groups[key];g['games']+=1
        for name in ('counts','grouped_actor_seconds'):g[name].update(row[name])
        for name in ('roaming_alive_actor_seconds','roaming_crew_distance','same_target_request_pairs_within_2s'):g[name]+=row[name]
        for w in row['witnesses']:
            scope='prefix' if w['prefix'] else 'policy'
            g['witness_counts'][scope+'_actor_body_pairs']+=1
            g['witness_counts'][scope+'_with_legal_reporting_choice']+=w['legal_reporting_choices']>0
            g['witness_counts'][scope+'_legal_choices_not_reporting']+=w['legal_choices_not_reporting']
            g['witness_counts'][scope+'_own_reports']+=w['own_report_time'] is not None
            g['witness_counts'][scope+'_died_before_report']+=w['died_before_body_report']
    out={}
    for key,g in groups.items():
        g={k:dict(v) if isinstance(v,Counter) else v for k,v in g.items()};den=g['roaming_alive_actor_seconds']
        g['grouped_fraction_by_distance']={str(k):g['grouped_actor_seconds'].get(str(k),0)/den if den else None for k in (1.3,2.5,4.5)}
        g['living_tasks_per_roaming_actor_minute']=g['counts'].get('living_task_events',0)*60/den if den else None
        g['travel_per_living_task']=g['roaming_crew_distance']/g['counts']['living_task_events'] if g['counts'].get('living_task_events') else None
        out[key]=g
    return out,raw
