"""Bounded frozen-policy engineering assay. No optimizer, gradients or training."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
from time import perf_counter

import torch
from crewmate_v2.config import LobbySettings, MatchMixture
from crewmate_v2.fourcrew.config import Rewards
from crewmate_v2.fourcrew.policy import FourCrewPolicy, source_lock
from crewmate_v2.fourcrew.scenarios import ScenarioMatch
from crewmate_v2.learning.provenance import digest
from crewmate_v2.learning.train import state_hash
from crewmate_v2.live.protocol import spectator_import
from phase3_runner import behavior_seed
from . import VERSION
from .audit import OpportunityAudit, AuditedController, install_observer, neural_observer, ratio
from .opponents import MixtureController


def schedule():
    return [dict(seed=2760080+i,context='ordinary' if i<6 else 'body',
                 deduction_frequency=float(i>=4),case_index=i) for i in range(8)]


def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def execute(policy,case,arm,*,deadline,recorder_module=None,replay_path=None,instrument=True):
    started=perf_counter();case=dict(case);frequency=case.pop('deduction_frequency')
    match=ScenarioMatch(**case,lobby=LobbySettings(),mixture=MatchMixture(frequency),
        rewards=Rewards.original(),deadline=deadline)
    recorder=None
    if recorder_module:
        recorder=recorder_module.MatchRecorder(match.match,replay_path,
            training_session=VERSION,source_version='runtime:'+digest(source_lock())+':'+VERSION,
            controller_types={pid:'ppo' if pid in match.crew else 'scripted' for pid in match.game.players},
            model_architecture='Frozen FourCrewPolicy; no updates',
            metadata=dict(training_condition='shaped_v2 (frozen)',context=match.context,policy_version=23,
                opponent=arm,engineering_only=True,environment_schema=match.game.lobby.schema,
                deduction_frequency=frequency,reward_profile=Rewards.original().to_dict(),
                prefix_control='scripted legal setup only' if match.context=='body' else 'none',
                prefix_handoff_seconds=75. if match.context=='body' else 0.))
    audit=OpportunityAudit()
    try:
        match.prepare_prefix(recorder)
        if arm=='mixture':match.match.controllers[match.impostor]=MixtureController(behavior_seed(match.seed,match.impostor))
        elif arm!='baseline':raise ValueError('Unknown opponent arm')
        controller=match.match.controllers[match.impostor]
        if instrument:
            match.match.controllers[match.impostor]=AuditedController(controller,audit);install_observer(match,audit)
        rollout,summary=match.collect(policy,initialization=17,policy_version=23,deterministic=True,
            recorder=recorder,decision_observer=neural_observer(audit) if instrument else None)
    finally:
        if recorder and not recorder.recorder._closed:
            recorder.recorder._stream.flush();recorder.recorder._stream.close();recorder.recorder._closed=True
    summary.update(arm=arm,deduction_frequency=frequency,wall_seconds=perf_counter()-started,
        event_sha256=digest(match.game.event_log),mode_history=getattr(controller,'mode_history',[]),
        audit=audit.summary(match.game) if instrument else None)
    return summary,audit


def aggregate(games):
    out={}
    for arm in ('baseline','mixture'):
        out[arm]={}
        for context in ('ordinary','deduction_natural','body'):
            subset=[g for g in games if g['arm']==arm and
                ('body' if g['context']=='body' else 'deduction_natural' if g['deduction_frequency'] else 'ordinary')==context]
            roles={}
            for role in ('crewmate','impostor'):
                totals=Counter();counts=Counter();delays=[]
                for g in subset:
                    r=g['audit']['by_role'][role];counts.update(r['counts']);delays.extend(r['first_legal_to_accepted_seconds'])
                    for key in ('visible_actor_body_pairs','range_actor_body_pairs','legal_actor_body_pairs','decision_actor_body_pairs',
                        'visible_ticks','range_ticks','legal_ticks','accepted_reports','self_reports','accepted_kills'):totals[key]+=r[key]
                roles[role]=dict(totals,counts=dict(counts),
                    accepted_report_pair_conversion=ratio(totals['accepted_reports'],totals['legal_actor_body_pairs']),
                    submitted_report_window_conversion=ratio(counts['action_report'],counts['report_legal_decision_windows']),
                    kill_window_conversion=ratio(totals['accepted_kills'],counts['kill_legal_decision_windows']),
                    median_report_delay=statistics.median(delays) if delays else None)
            outcomes=Counter(g['reason'] for g in subset)
            scores=[];role_macro=defaultdict(list);uniform=[]
            for g in subset:
                meetings=defaultdict(list)
                for row in g['audit']['reporter_only']:meetings[(row['meeting'],row['role'])].append(row)
                for (meeting,role),rows in meetings.items():
                    score=statistics.mean(r['score'] for r in rows);scores.append(score);role_macro[role].append(score)
                    uniform.append(statistics.mean(r['uniform'] for r in rows))
            nav=Counter();claims=Counter();modes=Counter();victims=Counter()
            for g in subset:
                claims.update(g['audit']['claim_types']);modes.update(mode for _,mode in g['mode_history']);victims.update(g['audit']['impostor_victims'])
                # Impostor ID is scored offline from the private local replay header, stored in summary.
                nav.update(g['impostor_navigation'])
            out[arm][context]=dict(games=len(subset),crew_wins=sum(g['crew_win'] for g in subset),endings=dict(outcomes),by_role=roles,
                meetings=sum(g['audit']['meetings'] for g in subset),discovered_bodies=sum(g['audit']['discovered_bodies'] for g in subset),
                unreported_kills=sum(g['audit']['unreported_kills'] for g in subset),
                correct_crew_ballots=sum(g['audit']['correct_crew_ballots'] for g in subset),wrong_crew_ballots=sum(g['audit']['wrong_crew_ballots'] for g in subset),
                skips=sum(g['audit']['published_crew_skips'] for g in subset),impostor_claim_types=dict(claims),mode_visits=dict(modes),
                navigation_choices=dict(nav),victim_ids_offline=dict(victims),
                reporter_only=dict(actor_excluded_meetings=len(scores),macro_accuracy=statistics.mean(scores) if scores else None,
                    uniform_macro=statistics.mean(uniform) if uniform else None,
                    by_reporter_role={r:dict(meetings=len(s),macro_accuracy=statistics.mean(s)) for r,s in role_macro.items()}))
    return out


def run(args):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True,exist_ok=False);start=perf_counter();deadline=start+900
    module=spectator_import(args.spectator_source)
    selected=json.loads((args.study/'study.json').read_text())['selected_checkpoints']
    paths={name:args.study/path for name,path in selected.items()};paths['zero']=args.study/'reference-zero.pt'
    models={};hashes={}
    for label,path in paths.items():
        model,metadata=FourCrewPolicy.load(path);model.requires_grad_(False);models[label]=model
        hashes[label]=dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),tensors=state_hash(model))
    local_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}
    write_json(args.output/'protocol.json',dict(version=VERSION,cases=schedule(),arms=['baseline','mixture'],
        initialization=17,greedy=True,models=hashes,actor_runtime=source_lock(),phase2_sources=local_sources,
        rewards=Rewards.original().to_dict(),lobby=LobbySettings().to_dict(),hard_wall_seconds=900,
        inference_only=True,final_test_opened=False))
    games=[];failure=None;repeat=None
    try:
        for case in schedule():
            for arm in ('baseline','mixture'):
                if perf_counter()>=deadline:raise TimeoutError('Phase 2 engineering wall ceiling')
                path=args.output/'replays'/arm/(('body' if case['context']=='body' else 'deduction' if case['deduction_frequency'] else 'ordinary')+'-'+str(case['seed'])+'.jsonl')
                summary,audit=execute(models['shaped_v2'],case,arm,deadline=deadline,recorder_module=module,replay_path=path)
                # Own opponent ID is not an actor feature; use it only to aggregate recorded navigation.
                imp=next(pid for pid in audit.counts if pid not in summary['actors'])
                summary['impostor_navigation']=dict(audit.moves[imp]);games.append(summary)
                write_json(args.output/f"game-{case['seed']}-{arm}.json",summary)
                write_json(args.output/f"private-observations-{case['seed']}-{arm}.json",dict(decisions=audit.decisions,pairs=list(audit.pairs.values())))
                print(json.dumps(dict(completed=len(games),seed=case['seed'],arm=arm,seconds=round(summary['wall_seconds'],2),reason=summary['reason'])),flush=True)
        # Reproduce one complete natural mixture match with all extra auditing/recording disabled.
        check,_=execute(models['shaped_v2'],schedule()[0],'mixture',deadline=deadline,instrument=False)
        first=next(g for g in games if g['arm']=='mixture' and g['seed']==schedule()[0]['seed'])
        repeat=dict(event_hash_equal=check['event_sha256']==first['event_sha256'],
            actions_equal=check['actors']==first['actors'],decisions_equal=check['decisions']==first['decisions'],seconds=check['wall_seconds'])
        if not all(repeat[k] for k in ('event_hash_equal','actions_equal','decisions_equal')):raise AssertionError('Observer noninterference/reproducibility failed')
    except (TimeoutError,ValueError,AssertionError) as error:
        failure=type(error).__name__+': '+str(error)
    unchanged=all(hashlib.sha256(paths[n].read_bytes()).hexdigest()==h['sha256'] and state_hash(models[n])==h['tensors'] for n,h in hashes.items())
    report=dict(status='engineering_complete' if len(games)==16 and failure is None else 'partial_inconclusive',
        seconds=perf_counter()-start,games=len(games),distinct_scenarios=len({g['seed'] for g in games}),
        groups=aggregate(games),failure=failure,reproducibility=repeat,checkpoints_unchanged=unchanged,
        final_test_opened=False,training_updates=0,acceptance='pending',
        limits=['Eight paired engineering scenarios and one frozen shaped policy; no learning/generalization claim.',
                'Natural ordinary/deduction and legal body prefixes remain separate. Tiny event denominators.',
                'Actor/body pairs, decision windows and reporter-only actor forecasts are correlated.',
                'Reporter identity alone may remain exploitable; strategy diversity is not evidence of robustness.',
                'Mode history is internal script metadata, never a neural feature. No sabotage support.'])
    write_json(args.output/'aggregate.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='groups'},indent=2))
    if failure or not unchanged:raise SystemExit(2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--spectator-source',type=Path,required=True)
    run(p.parse_args())


if __name__=='__main__':main()
