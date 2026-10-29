"""Manual, checkpointed lights-only continuation. No automatic million-stage launch."""
from dataclasses import replace
from pathlib import Path
from time import perf_counter,time
import json,math,os,traceback,platform
import numpy as np
import torch
from .network import PatrickJane
from .lights_setup import LightsExperiment
from .config import DEV_TRAIN,DEV_VALIDATION,ENGINEERING
from .checkpoint import save,load,source_lock,freeze_sources,select_champion
from .continuation_import import import_pilot,digest
from .collector import Pool,collect
from .ppo import update
from .health import audit_rollout,severe_speed_regression
from .progress import Progress,atomic_json
from .telemetry import Resources
from .learning_report import read,generate
from .run_files import run_lock

ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'local-runs/patrick-jane/pilot-250k-seed17-v3'


def validate_pointer(folder,pointer,policy,optimizer,rng,experiment):
    path=(folder/pointer['checkpoint']).resolve()
    if not path.is_relative_to(folder.resolve()) or digest(path)!=pointer['sha256']:
        raise ValueError('Checkpoint path/hash is not valid for this run')
    record=load(path,policy,optimizer,rng=rng)
    if record['experiment']!=experiment.to_dict():raise ValueError('Cannot change a resumed training configuration')
    for state in optimizer.state.values():
        if any(torch.is_tensor(x) and not torch.isfinite(x).all() for x in state.values()):raise ValueError('Nonfinite optimizer')
    return record


def recover(folder,history,policy,optimizer,rng,experiment):
    errors=[]
    try:latest=read(folder/'latest.json')
    except (OSError,json.JSONDecodeError) as exc:
        latest=None;errors.append(dict(checkpoint='latest.json',error=str(exc)))
    candidates=[p for p in [latest,*reversed(history),read(folder/'origin.json')] if p]
    candidates.sort(key=lambda p:p['counters']['update'],reverse=True);seen=set()
    for pointer in candidates:
        identity=(pointer['checkpoint'],pointer['sha256'])
        if identity in seen:continue
        seen.add(identity)
        try:
            record=validate_pointer(folder,pointer,policy,optimizer,rng,experiment)
            kept=[h for h in history if h['counters']['update']<=record['counters']['update']]
            return record['counters'].copy(),pointer,kept,errors
        except (OSError,ValueError,RuntimeError,EOFError) as exc:errors.append(dict(checkpoint=pointer['checkpoint'],error=str(exc)))
    raise RuntimeError('No hash/source-verified checkpoint remains; refusing fresh initialization: '+str(errors))


def audit_lights(rollout):
    audit_rollout(rollout)
    for i,m in enumerate(rollout['matches']):
        if set(m['sabotages_by_kind'])-{'lights'}:raise RuntimeError('Forbidden sabotage in lights-only run')
        actual=sum(v.get('lights_unresolved_time',0) for owner,v in rollout['reward_components'].items() if owner.startswith(str(i)+'/'))
        expected=-.005*m['learning_diagnostics']['lights_actor_seconds']
        if not math.isclose(actual,expected,abs_tol=1e-6):raise RuntimeError('Lights time penalty attribution mismatch')


def select_uncensored_champion(checkpoint,matches,folder):
    # Keep the existing matched-scenario selection rule only when every
    # outcome is legitimate; a technical cut is not a non-winning result.
    if any(m.get('truncated') or m.get('reason')=='TRUNCATED' for m in matches):return False
    return select_champion(checkpoint,matches,folder)


def inspect():
    torch.set_num_threads(1);p=PatrickJane();o=torch.optim.Adam(p.parameters(),lr=.0003);rng=np.random.default_rng(17)
    if not torch.cuda.is_available():raise RuntimeError('Registered CUDA training device unavailable; no silent CPU fallback')
    c,provenance=import_pilot(PILOT/'checkpoints/update-000006.pt',PILOT/'source-snapshot.zip',p,o,rng)
    return dict(workers=32,additional_target=1000000,base_decisions=c['individual_decisions'],
        device=torch.cuda.get_device_name(0),policy_parameters=sum(x.numel() for x in p.parameters()),profile=LightsExperiment().to_dict(),import_provenance=provenance)


def train(folder,*,additional_target=1000000,workers=32,wall_seconds=43200,device='cuda',
          engineering=False,stop_after_updates=None):
    if not 1<=workers<=32 or type(additional_target) is not int or additional_target<=0 or wall_seconds<=0:
        raise ValueError('Positive budget/wall limit and 1..32 workers required')
    if not engineering and stop_after_updates is not None:raise ValueError('Test interruption flag is engineering-only')
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=True)
    with run_lock(folder):
        if read(folder/'completion-status.json',{}).get('state')=='complete':
            print('PATRICK JANE — TRAINING COMPLETE (100%) — existing run; no additional training launched',flush=True)
            print('REPORT: '+str(folder/'TRAINING_REPORT.md'),flush=True);return 0
        return _train(folder,additional_target,workers,wall_seconds,device,engineering,stop_after_updates)


def _train(folder,target,workers,wall_seconds,device,engineering,stop_after_updates):
    began=perf_counter();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    experiment=LightsExperiment()
    if engineering:experiment=replace(experiment,technical_max_seconds=12.,evaluation_matches=2)
    policy=PatrickJane().to(device);optimizer=torch.optim.Adam(policy.parameters(),lr=experiment.learning_rate)
    rng=np.random.default_rng(17);history=read(folder/'history.json',[])
    reg=read(folder/'registration.json');attempts=folder/'attempts';attempts.mkdir(exist_ok=True)
    old_attempts=sorted(p for p in attempts.iterdir() if p.is_dir())
    def previous_duration(p):
        info=read(p/'attempt.json',{});status=read(p/'status.json',{})
        return info.get('elapsed_seconds',max(0,status.get('elapsed',0)-status.get('active_seconds_before_attempt',0))+60)
    previous_seconds=sum(previous_duration(p) for p in old_attempts)
    attempt=attempts/f'attempt-{len(old_attempts)+1:04d}';attempt.mkdir(exist_ok=False)
    atomic_json(attempt/'attempt.json',dict(started_unix=time(),pid=os.getpid(),engineering=engineering))
    state='failed';error=None;current=None;progress=None;import_provenance=None
    try:
        if reg:
            if reg['experiment']!=json.loads(json.dumps(experiment.to_dict())) or reg['workers']!=workers or reg['additional_target']!=target or reg['wall_seconds']!=wall_seconds:
                raise ValueError('Run already registered with different settings; use a separate run folder')
            if reg['sources']!=source_lock():raise ValueError('Frozen run source changed; restore the registered source before resuming')
            counters,pointer,history,rejected=recover(folder,history,policy,optimizer,rng,experiment)
            current=folder/pointer['checkpoint'];atomic_json(attempt/'recovery.json',dict(selected=pointer['checkpoint'],rejected=rejected))
            prior_history=read(folder/'history.json',[])
            if history!=prior_history:atomic_json(attempt/'previous-history.json',prior_history)
            atomic_json(folder/'history.json',history);atomic_json(folder/'latest.json',pointer)
        else:
            counters,import_provenance=import_pilot(PILOT/'checkpoints/update-000006.pt',PILOT/'source-snapshot.zip',policy,optimizer,rng)
            counters.update(run_decisions=0,run_matches=0,run_updates=0)
            if engineering:counters['next_seed']=0
            reg=dict(schema='patrick-jane-lights-run-v1',engineering=engineering,experiment=json.loads(json.dumps(experiment.to_dict())),
                workers=workers,additional_target=target,absolute_additional_ceiling=math.ceil(target*1.10),wall_seconds=wall_seconds,
                base_decisions=255083,base_update=6,base_matches=192,training_seeds=list(ENGINEERING[:80] if engineering else DEV_TRAIN),
                validation_seeds=list((ENGINEERING[80:80+experiment.evaluation_matches]) if engineering else DEV_VALIDATION[:experiment.evaluation_matches]),
                source_snapshot=freeze_sources(folder),sources=source_lock(),import_provenance=import_provenance,
                evaluation='paired entry and final only; no large per-checkpoint evaluations',
                budget_unit='additional individual living crew decisions; complete matches; up to 10% guard',
                runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__),
                    cuda=torch.version.cuda,device=torch.cuda.get_device_name(0) if device=='cuda' else device,worker_hash_seed=0),
                final_partitions='unopened',acceptance='pending')
            launchers={}
            for name in ('Start-PatrickJaneLights.ps1','tools/run_pj_learning.py','tools/report_patrick_jane_pilot.py'):
                source=ROOT/name;dest=folder/'launchers'/Path(name).name;dest.parent.mkdir(exist_ok=True)
                dest.write_bytes(source.read_bytes());launchers[name]=digest(dest)
            reg['launcher_sha256']=launchers
            atomic_json(folder/'registration.json',reg)
            current=folder/f'checkpoints/{attempt.name}-origin.pt'
            sha=save(current,policy,optimizer,experiment,counters.copy(),rng=rng,metadata=dict(import_provenance=import_provenance))
            pointer=dict(checkpoint=current.relative_to(folder).as_posix(),sha256=sha,counters=counters.copy())
            atomic_json(folder/'origin.json',pointer);atomic_json(folder/'latest.json',pointer)
        deadline=perf_counter()+wall_seconds-previous_seconds-(perf_counter()-began)
        atomic_json(folder/'completion-status.json',dict(state='running',success=False,
            additional_decisions=counters['run_decisions'],checkpoint=current.relative_to(folder).as_posix(),
            report=str(folder/'TRAINING_REPORT.md'),attempt=attempt.name))
        def remaining():
            if source_lock()!=reg['sources']:raise RuntimeError('Frozen training source changed')
            if any(digest(ROOT/name)!=sha for name,sha in reg['launcher_sha256'].items()):raise RuntimeError('Frozen launcher/report helper changed')
            seconds=deadline-perf_counter()
            if seconds<=0:raise TimeoutError('Cumulative active wall ceiling reached; preserve checkpoints')
            return seconds
        with Resources() as resources:
            progress=Progress(attempt,workers,resources,target=target,mirror=folder/'status.json')
            progress.started=began-previous_seconds
            progress.console=True
            progress.matches=[m for row in history for m in row['rollout']['matches']]
            for row in history:
                for v in row['rollout']['reward_components'].values():progress.components.update(v)
            def publish(active=None,**extra):
                progress.counters=dict(individual_decisions=counters['run_decisions'],complete_matches=counters['run_matches'],update=counters['run_updates'])
                progress.checkpoint=current
                progress.publish(active,lifetime_decisions=counters['individual_decisions'],base_decisions=255083,
                    trainer_pid=os.getpid(),active_seconds_before_attempt=previous_seconds,report_path=str(folder/'TRAINING_REPORT.md'),**extra)
            def evaluate(pool,label):
                saved=read(folder/f'evaluations/{label}.json')
                expected_sha=read(folder/'origin.json')['sha256'] if label=='entry' else digest(current)
                if saved:
                    if saved.get('checkpoint_sha256')==expected_sha:
                        if label=='entry':
                            progress.history.append(dict(decisions=0,validation_wins=sum(m['winner']=='crewmate' for m in saved['matches']),validation_games=sum(not m.get('truncated') for m in saved['matches'])))
                        return saved
                    atomic_json(attempt/f'previous-{label}-evaluation.json',saved)
                    if label=='entry':raise ValueError('Entry evaluation does not match the frozen origin policy')
                if label=='entry' and counters['run_updates']:
                    raise ValueError('Entry reference is missing; refusing to evaluate a trained policy as baseline')
                progress.stage='validation-'+label;matches=[];start=perf_counter()
                seeds=reg['validation_seeds']
                for offset in range(0,len(seeds),workers):
                    result=collect(policy,pool,experiment,seeds[offset:offset+workers],device=device,held_out=True,
                        update=counters['update'],training_step=counters['individual_decisions'],checkpoint=str(current),
                        replay_dir=attempt/'replays',record_every=8,monitor_dir=progress.monitor_dir,wall_limit=remaining(),
                        status_callback=lambda active:publish(active,validation_complete=len(matches),validation_target=len(seeds)))
                    result.pop('trajectories');atomic_json(attempt/f'batches/{label}-{offset}.json',result);audit_lights(result);matches.extend(result['matches'])
                record=dict(label=label,matches=matches,elapsed=perf_counter()-start,update=counters['update'],
                    checkpoint=str(current),checkpoint_sha256=digest(current))
                atomic_json(folder/f'evaluations/{label}.json',record);select_uncensored_champion(current,matches,folder/'champions')
                progress.history.append(dict(decisions=counters['run_decisions'],validation_wins=sum(m['winner']=='crewmate' for m in matches),validation_games=sum(not m.get('truncated') for m in matches)))
                publish({},validation_complete=len(matches),validation_target=len(seeds));return record
            generate(folder,'running',active_seconds=previous_seconds+perf_counter()-began,plots=False)
            with Pool(workers) as pool:
                evaluate(pool,'entry')
                while counters['run_decisions']<target:
                    active=workers
                    if counters['run_matches']:
                        active=min(workers,max(1,math.ceil((target-counters['run_decisions'])/(counters['run_decisions']/counters['run_matches']))))
                    first=counters['next_seed'];seeds=reg['training_seeds'][first:first+active]
                    if len(seeds)!=active:raise RuntimeError('Registered development training partition exhausted')
                    progress.stage='collecting';publish({})
                    rollout=collect(policy,pool,experiment,seeds,device=device,update=counters['update'],
                        training_step=counters['individual_decisions'],checkpoint=str(current),replay_dir=attempt/'replays',
                        record_every=8,monitor_dir=progress.monitor_dir,wall_limit=remaining(),status_callback=publish,
                        decision_limit=reg['absolute_additional_ceiling']-counters['run_decisions'])
                    batch=attempt/f'batches/training-u{counters["update"]:06d}.json'
                    atomic_json(batch,{k:v for k,v in rollout.items() if k!='trajectories'});audit_lights(rollout)
                    progress.stage='PPO';publish(dict(decisions=rollout['individual_decisions']))
                    learning=update(policy,optimizer,rollout.pop('trajectories'),experiment,rng=rng,device=device,deadline=deadline)
                    remaining();n=rollout['individual_decisions'];games=len(seeds)
                    for k,value in [('individual_decisions',n),('run_decisions',n),('complete_matches',games),('run_matches',games),('next_seed',games),('update',1),('run_updates',1)]:counters[k]+=value
                    current=folder/f'checkpoints/{attempt.name}-update-{counters["update"]:06d}.pt'
                    sha=save(current,policy,optimizer,experiment,counters.copy(),rng=rng,metadata=dict(batch_journal=batch.relative_to(folder).as_posix()))
                    pointer=dict(checkpoint=current.relative_to(folder).as_posix(),sha256=sha,counters=counters.copy())
                    row=dict(**pointer,rollout=rollout,ppo=learning);history.append(row)
                    atomic_json(folder/'history.json',history);atomic_json(folder/'latest.json',pointer)
                    progress.settle(rollout);progress.ppo=learning;progress.stage='checkpoint'
                    publish(progress.final_streams(rollout['matches'],counters['update']-1,8))
                    generate(folder,'running',active_seconds=previous_seconds+perf_counter()-began)
                    print(f"CHECKPOINT · additional {counters['run_decisions']:,}/{target:,} · matches {counters['run_matches']} · PPO {counters['run_updates']} · {current.name}",flush=True)
                    full=[h for h in history if len(h['rollout']['matches'])==workers]
                    if active==workers and severe_speed_regression(full):raise RuntimeError('Sustained severe throughput regression')
                    if stop_after_updates and counters['run_updates']>=stop_after_updates:raise KeyboardInterrupt('Bounded engineering interruption test')
                evaluate(pool,'final');remaining()
            state='complete'
    except KeyboardInterrupt as exc:
        state='interrupted';error=str(exc) or 'KeyboardInterrupt'
    except BaseException as exc:
        error=repr(exc);(attempt/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
    finally:
        elapsed=perf_counter()-began;total=previous_seconds+elapsed
        if state!='complete' and progress:
            counters=read(folder/'latest.json',{}).get('counters',counters)
        atomic_json(attempt/'attempt.json',dict(started_unix=time()-elapsed,elapsed_seconds=elapsed,state=state,error=error))
        try:result=generate(folder,state,active_seconds=total,error=error)
        except BaseException as report_error:
            state='failed';error=f'{error}; report generation: {report_error!r}'
            result=dict(additional_decisions=read(folder/'latest.json',{}).get('counters',{}).get('run_decisions',0))
            (folder/'TRAINING_REPORT.md').write_text(f'# TRAINING FAILED\n\n{error}\n\nCheck latest.json, history.json and attempt logs.\n',encoding='utf-8')
            atomic_json(folder/'training_results.json',dict(state=state,error=error,latest=read(folder/'latest.json')))
        success=state=='complete'
        completion=dict(state=state,success=success,additional_decisions=result['additional_decisions'],
            active_seconds=total,checkpoint=read(folder/'latest.json',{}).get('checkpoint'),
            report=str(folder/'TRAINING_REPORT.md'),error=error)
        completion['progress_percent']=min(100.,100*result['additional_decisions']/target)
        completion['technical_truncations']=result.get('training',{}).get('totals',{}).get('truncations',0)
        atomic_json(folder/'completion-status.json',completion)
        if progress:
            progress.stage='complete' if success else state
            progress.active={};publish({},complete=success,completion_status=completion)
        else:
            saved=read(folder/'latest.json',{}).get('counters',{})
            atomic_json(folder/'status.json',dict(schema='patrick-jane-progress-v1',stage=state,workers=workers,
                counters=dict(individual_decisions=saved.get('run_decisions',0),complete_matches=saved.get('run_matches',0),update=saved.get('run_updates',0)),
                overall_decisions=saved.get('run_decisions',0),target_decisions=target,complete=success,
                completion_status=completion,error=error))
        print('\n'+'='*72,flush=True)
        print('PATRICK JANE — TRAINING COMPLETE (100%)' if success else 'TRAINING INTERRUPTED' if state=='interrupted' else 'TRAINING FAILED',flush=True)
        print(f"Additional decisions: {result['additional_decisions']:,}; active time: {total/3600:.3f} h",flush=True)
        print(f"Progress: {completion['progress_percent']:.1f}%"+(' — stopped before 100% target' if not success and completion['progress_percent']<100 else ''),flush=True)
        print(f"Matches: {result.get('completed_training_matches',0)}; PPO updates: {result.get('ppo_updates',0)}",flush=True)
        print(f"Technical TRUNCATED: {result.get('technical_truncations',0)}; collected segments: {result.get('collected_training_segments',0)}",flush=True)
        print('Final/latest checkpoint: '+str(completion['checkpoint']),flush=True)
        print('REPORT: '+completion['report'],flush=True)
        if error:print('Reason: '+error,flush=True)
        print('='*72,flush=True)
    return 0 if state=='complete' else 130 if state=='interrupted' else 1
