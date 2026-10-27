"""Owner-approved spawn correction, sharing the ORIGINAL study's four-hour deadline."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import time,perf_counter
import torch
from .learning import Session as BaseSession,record_for,verify,load
from .env import DiagnosticEnv
from impostor_options_v4.diagnostics import LegacyEnv
from impostor_options_v4.config import Rewards,Options
from . import SCHEMA
from impostor_research.learning_v2 import load as load_v2
from impostor_research.learning import write_json,exclusive_worker
from impostor_research.frozen import FrozenModels,weight_hash
from impostor_research.protocol import ROOT
from impostor_research.pilot import social_metrics

SEEDS=(71,83)
SCREEN=tuple(range(3200140,3200160))
CONFIRM=tuple(range(3200160,3200180))
STEPS=8192
BUDGET=14400
OLD='artifacts/impostor-ai/release/policy.pt'
PROTOCOL='docs/impostor-v2/options-v41/TRAINING_PROTOCOL.md'
ORIGINAL='artifacts/impostor_research/options-v4-20261027/study.json'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def progress(output,**state):
    data=dict(updated_epoch=time(),**state)
    write_json(output/'progress.json',data)
    if state['stage']=='training':
        detail=f"Initialization {state['seed']}: {state['decisions']}/8192 decisions; {state['episodes']} completed games; {state['wins']} training wins; {state['kills']} kills."
    elif state['stage']=='parallel_training':
        detail='\n\n'.join(f"Initialization{seed}: {row.get('decisions',0)}/8192 decisions; {row.get('episodes',0)} games; {row.get('wins',0)} wins; {row.get('kills',0)} kills." for seed,row in state['runs'].items())
    elif state['stage'] in ('selection','diagnostic','confirmation'):
        detail=f"{state['stage']}: {state['completed']}/{state['total']} executions. Method: {state['method']}."
    else:detail=json.dumps(state,indent=2)
    (output/'LIVE_PROGRESS.md').write_text('# Impostor V4.1 — corrected study\n\n**'+state['stage']+'**\n\n'+detail+
        '\n\nORIGINAL deadline2026-10-28 01:03:50 +03, not four extra hours; no reserved finals/merge/deployment.\n',encoding='utf-8')
    print(json.dumps(data,allow_nan=False),flush=True)


TrainingSession=BaseSession


def registered_record(models):
    record=record_for(models)
    paths=[ROOT/PROTOCOL,ROOT/'tools/run_impostor_options_v41.py']
    for path in paths:
        record['sources'][path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    original=json.loads((ROOT/ORIGINAL).read_text())
    record.update(study_status='Owner approved stop/fix/restart within ORIGINAL deadline after inactive pursuit was discovered',
        approval=dict(date='2026-10-27',timezone='Asia/Amman',message='Fix and restart within original deadline (Recommended)',budget_seconds=BUDGET),
        original_study_sha256=digest(ROOT/ORIGINAL),original_deadline_epoch=original['deadline_epoch'],
        initialization='scratch; no parent or crew weights',
        study_partitions=dict(train=[3111000,3199999],selection=list(SCREEN),confirmation=list(CONFIRM),diagnostic=list(SCREEN[:5])),
        candidate_steps=[4096,8192],neural_decisions_per_run=STEPS)
    record['compute']=dict(training_workers=2,evaluation_workers=2,torch_threads_per_worker=1)
    return record


def check_study(study,models,deadline=True):
    verify(study['record'],models,study['deadline_epoch'] if deadline else None)
    if (study['budget_seconds']!=BUDGET or study['initializations']!=list(SEEDS) or study['steps']!=STEPS
        or study['screen']!=list(SCREEN) or study['confirmation']!=list(CONFIRM)
        or study['diagnostic']!=list(SCREEN[:5]) or study['final_tests_opened']):
        raise ValueError('Registered V4 protocol changed')
    if digest(ROOT/OLD)!=study['old_checkpoint_sha256']:raise ValueError('V2 control bytes changed')
    original=json.loads((ROOT/ORIGINAL).read_text())
    if (digest(ROOT/ORIGINAL)!=study['record']['original_study_sha256']
        or study['deadline_epoch']!=original['deadline_epoch']
        or study['deadline_epoch']!=study['record']['original_deadline_epoch']):
        raise ValueError('Original study deadline/reference changed')


def evaluate(output,study,models,method,partition,checkpoint=None,stochastic=False,progress_output=None):
    output=Path(output)
    if progress_output is not None:progress_output=Path(progress_output);progress_output.mkdir(parents=True,exist_ok=True)
    deadline=study['deadline_epoch'];folder=output/partition;folder.mkdir(exist_ok=True)
    scenarios=SCREEN if partition=='selection' else SCREEN[:5] if partition=='diagnostic' else CONFIRM
    total=80 if partition=='selection' else 5 if partition=='diagnostic' else 40
    if method=='unchanged_v2':
        policy,_=load_v2(ROOT/OLD,models,deadline);env=LegacyEnv(models=models);checkpoint=ROOT/OLD
    else:policy,_=load(checkpoint,models,deadline);env=DiagnosticEnv(models=models)
    rows=[]
    for scenario in scenarios:
        target=folder/f'{method}-{scenario}.json'
        if target.exists():rows.append(json.loads(target.read_text()));continue
        check_study(study,models);started=perf_counter();packet,_=env.reset(seed=scenario)
        generator=torch.Generator().manual_seed(scenario^0xD401)
        decisions=0;counts=Counter()
        while not env.finished:
            if time()>deadline:raise TimeoutError('V4 four-hour study ceiling reached')
            if (output/'STOP').exists():raise RuntimeError('Study stopped by another worker failure')
            action,_,_=policy.act(packet,generator=generator,deterministic=not stochastic)
            counts[env.options[action].kind]+=1;packet,_,_,_,_=env.step(action);decisions+=1
            if env.match.game.metrics['illegal_actions'] or env.match.game.metrics['navigation_failures']:
                raise ValueError('Illegal action/navigation failure in V4 evaluation')
        row=dict(**env.summary(),**social_metrics(env),method=method,partition=partition,
            stochastic=stochastic,seconds=perf_counter()-started,external_decisions=decisions,
            actions=dict(counts),checkpoint_sha256=digest(checkpoint))
        write_json(target,row,exclusive=True);rows.append(row)
        progress(progress_output or output,stage=partition,method=method,completed=len(list(folder.glob('*.json'))),total=total,
            wins=sum(r['impostor_win'] for r in rows),kills=sum(r['kills'] for r in rows))
    return rows


def evaluate_worker(output,study,method,partition,checkpoint):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    models=FrozenModels();output=Path(output)
    return evaluate(output,study,models,method,partition,checkpoint,
                    progress_output=output/'method-progress'/method)


def evaluate_parallel(output,study,methods,partition):
    """Independent fixed method/case sets; rank only after every selection finishes."""
    results={};last=-1;total=80 if partition=='selection' else 40
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(evaluate_worker,str(output),study,method,partition,str(path) if path else None):method
                 for method,path in methods}
        try:
            while futures:
                completed,_=wait(futures,timeout=5,return_when=FIRST_COMPLETED)
                for future in completed:
                    method=futures[future];results[method]=future.result();del futures[future]
                count=len(list((output/partition).glob('*.json'))) if (output/partition).exists() else 0
                if count!=last:
                    progress(output,stage=partition,method='two isolated CPU workers',completed=count,total=total);last=count
        except BaseException:
            (output/'STOP').write_text('Stop after evaluation worker/parent failure; preserve completed cases.\n')
            raise
    return results


def choose(candidates):
    return max(candidates,key=lambda c:(c['wins'],c['kills'],-c['timeouts'],-c['step'],-c['seed']))


def train_seed(output,record,deadline,seed,resume=False,development=False,steps=STEPS):
    """One isolated initialization. Small step counts are development preflight only."""
    if steps!=STEPS and not development:raise ValueError('Production target is fixed8192')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    output=Path(output);models=FrozenModels();folder=output/f'seed-{seed}'
    if (folder/'training.json').exists():return json.loads((folder/'training.json').read_text())
    folder.mkdir(exist_ok=resume)
    if not (folder/'start.json').exists():write_json(folder/'start.json',dict(epoch=time()),exclusive=True)
    snapshots=sorted(folder.glob('resume-*.pt'))
    if snapshots:session=TrainingSession.restore(snapshots[-1],models,deadline,training_approved=True)
    else:
        session=TrainingSession(seed,models=models,record=record,development=development,training_approved=True)
        if not (folder/'checkpoint-00000.pt').exists():session.save(folder/'checkpoint-00000.pt',False)
        session.save(folder/'resume-00000.pt')
    progress(folder,stage='starting_run',seed=seed,decisions=session.decisions)
    while session.decisions<steps:
        if (output/'STOP').exists():raise RuntimeError('Study stopped by another worker failure')
        row=session.update(min(256,steps-session.decisions),deadline)
        session.save(folder/f'resume-{session.decisions:05d}.pt')
        if session.decisions in (2048,4096,8192):session.save(folder/f'checkpoint-{session.decisions:05d}.pt',False)
        write_json(folder/'updates.json',session.updates);write_json(folder/'episodes.json',session.episodes)
        progress(folder,stage='training',seed=seed,decisions=session.decisions,episodes=len(session.episodes),
            wins=sum(r['impostor_win'] for r in session.episodes),kills=sum(r['kills'] for r in session.episodes),
            seconds=session.elapsed_seconds,entropy=row['entropy'],kill_probability=row['kill_probability_when_available'])
    if weight_hash(session.policy)==session.initial_hash:raise ValueError('No neural learning occurred')
    result=dict(seed=seed,decisions=steps,episodes=len(session.episodes),
        simulated_seconds=session.simulated_seconds,wall_seconds=time()-json.loads((folder/'start.json').read_text())['epoch'],
        timed_seconds=session.elapsed_seconds,initial_hash=session.initial_hash,final_hash=weight_hash(session.policy),
        policy_changed=True,frozen_weights_unchanged=True,actions=session.action_counts,development=development)
    write_json(folder/'training.json',result,exclusive=True)
    return result


def train_parallel(output,record,deadline,resume):
    pending=[s for s in SEEDS if not (output/f'seed-{s}/training.json').exists()]
    if not pending:return
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(train_seed,str(output),record,deadline,s,resume):s for s in pending}
        last=None
        try:
            while futures:
                completed,_=wait(futures,timeout=5,return_when=FIRST_COMPLETED)
                for future in completed:future.result();del futures[future]
                states={}
                for s in SEEDS:
                    path=output/f'seed-{s}/progress.json'
                    if path.exists():states[str(s)]=json.loads(path.read_text())
                signature=tuple((s,row.get('decisions',0)) for s,row in states.items())
                if signature!=last:
                    progress(output,stage='parallel_training',runs=states);last=signature
        except BaseException:
            (output/'STOP').write_text('Stop requested after worker/parent failure; preserve snapshots.\n')
            raise


def run(output,resume=False):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT):raise ValueError('Study output must remain inside project')
    if (output/'results.json').exists():raise FileExistsError('Study is already complete; never rerun it')
    if not resume:output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    with exclusive_worker(output):
        models=FrozenModels()
        if resume:study=json.loads((output/'study.json').read_text())
        else:
            record=registered_record(models)
            if record['core']['dirty']:raise ValueError('Commit a clean V4 baseline before training')
            original=json.loads((ROOT/ORIGINAL).read_text());start=time()
            if time()>=original['deadline_epoch']:raise TimeoutError('No original budget remains')
            if not (ROOT/ORIGINAL).with_name('interruption.json').exists():raise ValueError('Original study interruption must be documented')
            study=dict(schema=SCHEMA,record=record,started_epoch=start,deadline_epoch=original['deadline_epoch'],
                original_started_epoch=original['started_epoch'],remaining_at_start_seconds=original['deadline_epoch']-start,
                budget_seconds=BUDGET,initializations=list(SEEDS),steps=STEPS,screen=list(SCREEN),
                confirmation=list(CONFIRM),diagnostic=list(SCREEN[:5]),old_checkpoint=OLD,
                old_checkpoint_sha256=digest(ROOT/OLD),final_tests_opened=False,
                selection_rule='wins, kills, fewer timeouts, earlier checkpoint, lower initialization')
            write_json(output/'study.json',study,exclusive=True)
        check_study(study,models);record=study['record'];deadline=study['deadline_epoch']
        try:
            training_started=time();train_parallel(output,record,deadline,resume)
            training_wall=time()-training_started
            methods=[(f'seed{seed}-step{step}',output/f'seed-{seed}'/f'checkpoint-{step:05d}.pt')
                     for seed in SEEDS for step in (4096,8192)]
            evaluated=evaluate_parallel(output,study,methods,'selection');candidates=[]
            for seed in SEEDS:
                for step in (4096,8192):
                    checkpoint=output/f'seed-{seed}'/f'checkpoint-{step:05d}.pt';method=f'seed{seed}-step{step}'
                    rows=evaluated[method]
                    candidates.append(dict(seed=seed,step=step,path=checkpoint.relative_to(ROOT).as_posix(),
                        method=method,sha256=digest(checkpoint),wins=sum(r['impostor_win'] for r in rows),
                        kills=sum(r['kills'] for r in rows),timeouts=sum(r['reason']=='TIMEOUT' for r in rows)))
            best=choose(candidates);lock=output/'selection-lock.json'
            if lock.exists():
                if json.loads(lock.read_text())['selected']!=best:raise ValueError('Immutable selection differs')
            else:write_json(lock,dict(selected=best,candidates=candidates,locked_epoch=time(),confirmation_unopened=True),exclusive=True)
            diagnostic=evaluate(output,study,models,'selected_sampled','diagnostic',ROOT/best['path'],True)
            confirmation_methods=[('selected_v41',ROOT/best['path']),('unchanged_v2',None)]
            confirmed=evaluate_parallel(output,study,confirmation_methods,'confirmation')
            confirmation=[row for method,_ in confirmation_methods for row in confirmed[method]]
            check_study(study,models)
            result=dict(status='complete',selected=best,selection=candidates,diagnostic=diagnostic,
                confirmation=confirmation,seconds=time()-study['started_epoch'],frozen_weights_unchanged=True,
                training_wall_seconds=training_wall,final_tests_opened=False)
            write_json(output/'results.json',result,exclusive=True)
            progress(output,stage='complete',selected=best,seconds=result['seconds'])
            return result
        except BaseException as exc:
            progress(output,stage='stopped',exception=type(exc).__name__,reason=str(exc),automatic_retry=False)
            raise
