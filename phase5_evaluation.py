"""Paired complete-match evaluation; final family is never used for selection."""
from pathlib import Path
import hashlib
import json
from time import perf_counter
import numpy as np
import torch
from collections import Counter
from phase5_env import CrewmateStrategicEnv
from phase5_policy import StrategicPolicy
from social_deduction.actor import Role


def evaluation_case(index,matches,*,validation=False,quick=False,match_offset=0):
    """Disjoint chunk offsets preserve the same paired ID/patient seed manifest."""
    heldout=not validation and index>=matches//2
    seed=620000+match_offset+index if validation else (
        (970000 if quick else 640000 if heldout else 630000)
        +match_offset+(index%(matches//2))+(1000 if quick and heldout else 0))
    return seed,heldout


def evaluate(output,matches=1000,checkpoints=(),quick=False,validation=False,include_baselines=True,
             random_protocol='strategic-decisions-v1',include_idle=False,match_offset=0,baseline_methods=None):
    if matches<2 or matches%2:raise ValueError('Use an even match count for equal ID/patient partitions')
    if validation and (quick or matches>100):raise ValueError('Validation uses at most 100 reserved ID matches')
    if not quick and not validation and matches>1000:raise ValueError('Final seed partitions contain 500 matches each')
    if type(match_offset) is not int or match_offset<0:raise ValueError('Invalid match offset')
    if quick and match_offset:raise ValueError('Quick evaluations cannot offset the seed manifest')
    if validation and match_offset+matches>100:raise ValueError('Validation seed partition exhausted')
    if not quick and not validation and match_offset+matches//2>500:raise ValueError('Final seed partition exhausted')
    torch.set_num_threads(1);torch.manual_seed(99)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any((output/name).exists() for name in ('metrics.json','matches.jsonl')):
        raise FileExistsError('Choose a new evaluation directory; existing results are preserved')
    models=[]
    for path in checkpoints:
        p,m=StrategicPolicy.load(path)
        if m.get('quick') and not quick:raise ValueError('Smoke checkpoints cannot populate final metrics')
        models.append((str(path),p,m))
    methods=([('random',None,{'execution_protocol':random_protocol}),('scripted',None,{})]
             +([('idle',None,{})] if include_idle else []) if include_baselines else [])+models
    if baseline_methods is not None:
        if not set(baseline_methods)<= {'random','scripted','idle'}:raise ValueError('Unknown baseline method')
        methods=[item for item in methods if item[1] is not None or item[0] in baseline_methods]
    if not methods:raise ValueError('At least one evaluation method is required')
    root=Path(__file__).resolve().parent
    sources={name:hashlib.sha256((root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
             for name in ('phase5_features.py','phase5_env.py','phase5_options.py','phase5_runtime.py','phase5_policy.py','phase5_runner.py','phase5_evaluation.py')}
    rows=[];began=perf_counter();planner=None;belief=None
    for method,policy,metadata in methods:
        from phase5_runtime import checked_environment
        env=checked_environment(metadata,planner=planner,belief_model=belief,ablation=metadata.get('ablation',False))
        planner,belief=env.planner,env.belief_model
        for i in range(matches):
            seed,heldout=evaluation_case(i,matches,validation=validation,quick=quick,match_offset=match_offset)
            env.heldout=heldout;packet,_=env.reset(seed=seed);rng=np.random.default_rng(seed^999)
            action_counts=Counter();entropies=[];decisions=0;simulated_seconds=0.
            while True:
                if method=='scripted':
                    intent=env.match.controllers[env.focal].decide(env.observation)
                    action_counts[intent.kind.value]+=1
                    packet,_,done,_,info=env.step_scripted(intent)
                else:
                    action=0 if method=='idle' else int(rng.choice(np.flatnonzero(packet['mask']))) if policy is None else policy.act(packet,True)[0]
                    action_counts[env.options[action].kind]+=1
                    if policy is not None:
                        from phase5_policy import tensors
                        with torch.no_grad():entropies.append(float(policy(tensors(packet))[0].entropy().item()))
                    packet,_,done,_,info=env.step(action)
                decisions+=1;simulated_seconds+=info['elapsed_seconds']
                if done:break
            game=env.match.game
            votes=[e for e in game.event_log if e['kind']=='vote' and e.get('voter_id')==env.focal]
            cast=[v for v in votes if v.get('target_id') is not None]
            correct=sum(game.players[v['target_id']].role is Role.IMPOSTOR for v in cast)
            ejections=[e for e in game.event_log if e['kind']=='ejection' and e.get('player_id') is not None]
            correct_ejections=sum(game.players[e['player_id']].role is Role.IMPOSTOR for e in ejections if e.get('player_id'))
            rows.append({'method':method,'seed':seed,'family':'patient' if heldout else 'ID',**info['episode'],
                'votes_cast':len(cast),'votes_correct':correct,'skips':len(votes)-len(cast),
                'ejections':len(ejections),'correct_ejections':correct_ejections,
                'innocent_ejections':len(ejections)-correct_ejections,
                'action_counts':dict(action_counts),'mean_action_entropy':float(np.mean(entropies)) if entropies else None,
                'decisions':decisions,'simulated_seconds':simulated_seconds,
                'execution_protocol':metadata.get('execution_protocol','strategic-decisions-v1'),
                'checkpoint_sha256':hashlib.sha256(Path(method).read_bytes()).hexdigest() if policy else None})
            if (i+1)%10==0:print(f'{method}: {i+1}/{matches}',flush=True)
            with (output/'matches.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(rows[-1])+'\n')
    metrics={}
    for method,_,_ in methods:
        metrics[method]={}
        for family in (('ID',) if validation else ('ID','patient')):
            subset=[r for r in rows if r['method']==method and r['family']==family]
            n=len(subset);wins=sum(r['crew_win'] for r in subset);p=wins/n
            z=1.96;denom=1+z*z/n;center=(p+z*z/(2*n))/denom;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
            votes=sum(r['votes_cast'] for r in subset)
            metrics[method][family]={'matches':n,'crew_win_rate':p,'win_rate_wilson95':[center-half,center+half],
                'mean_own_tasks':float(np.mean([r['own_tasks'] for r in subset])),
                'mean_survival':float(np.mean([r['survival'] for r in subset])),
                'voting_accuracy':sum(r['votes_correct'] for r in subset)/votes if votes else None,
                'votes_cast':votes,'skips':sum(r['skips'] for r in subset),
                'ejection_rate':sum(r['ejections']>0 for r in subset)/n,
                'correct_ejections':sum(r['correct_ejections'] for r in subset),
                'innocent_ejections':sum(r['innocent_ejections'] for r in subset),
                'timeouts':sum(r['reason']=='TIMEOUT' for r in subset),
                'illegal_actions':sum(r['illegal_actions'] for r in subset),
                'navigation_failures':sum(r['navigation_failures'] for r in subset)}
    paired={};rng=np.random.default_rng(505)
    for method,policy,_ in methods:
        if policy is None:continue
        paired[method]={}
        for family in (('ID',) if validation else ('ID','patient')):
            baseline={r['seed']:r for r in rows if r['method']=='scripted' and r['family']==family}
            if not baseline:continue
            subset=[r for r in rows if r['method']==method and r['family']==family]
            delta=np.asarray([float(r['crew_win'])-float(baseline[r['seed']]['crew_win']) for r in subset])
            bootstrap=rng.choice(delta,size=(1000,len(delta)),replace=True).mean(1)
            paired[method][family]={'win_rate_difference_vs_scripted':float(delta.mean()),
                'paired_match_bootstrap95':np.quantile(bootstrap,[.025,.975]).tolist()}
    result={'status':'smoke_only' if quick else 'validation' if validation else 'evaluation',
            'sources_sha256':sources,'matches_per_method':matches,'match_offset':match_offset,'metrics':metrics,'paired':paired,
            'seconds':perf_counter()-began,'notice':'No strategic-gain claim from smoke results; match seeds paired across methods.'}
    (output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');return result
