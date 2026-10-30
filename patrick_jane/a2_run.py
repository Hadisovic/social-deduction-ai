"""Owner-launched duration curriculum; six complete matches then unchanged PPO."""
from pathlib import Path
from time import perf_counter,time
import json,math,signal,traceback
import numpy as np
import torch
from .checkpoint import save,source_lock,freeze_sources
from .continuation_import import digest
from .duration_budget import ActiveBudget
from .a2_report import generate
from .learning_report import read
from .learning_run import recover,audit_lights
from .lights_setup import LightsExperiment
from .network import PatrickJane
from .collector import Pool,collect
from .ppo import update
from .progress import Progress,atomic_json
from .run_files import run_lock
from .telemetry import Resources
from .a2_transition import import_update102
from .a2_clock import clock as active_clock,SOURCE as CLOCK_SOURCE

ROOT=Path(__file__).resolve().parents[1]


def experiment():
    from .a2_config import A2Experiment
    return A2Experiment()


def train(folder,origin,*,hours=12.,workers=6,device='cuda',additional_decisions=None,prepare_only=False,profile=None):
    if not 1<=workers<=6 or not math.isfinite(hours) or hours<=0:raise ValueError('Need 1..6 workers and positive finite hours')
    if additional_decisions is not None and (type(additional_decisions) is not int or additional_decisions<=0):raise ValueError('Positive optional decision budget required')
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=True)
    if device!='cuda' or not torch.cuda.is_available():raise RuntimeError('CUDA is required: install a compatible NVIDIA driver; no silent CPU training fallback')
    from portable_support import verify,export_results
    verify(ROOT)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    with run_lock(folder):
        budget=ActiveBudget(folder,hours,clock=active_clock);exp=profile or experiment();policy=PatrickJane().to(device)
        optimizer=torch.optim.Adam(policy.parameters(),lr=exp.learning_rate);rng=np.random.default_rng(17)
        reg=read(folder/'registration.json');history=read(folder/'history.json',[])
        attempts=folder/'attempts';attempts.mkdir(exist_ok=True);attempt=attempts/f'attempt-{len(list(attempts.iterdir()))+1:04d}';attempt.mkdir()
        state='failed';error=None;progress=None;current=None;export=None;stop=[False,0]
        old_signal=signal.getsignal(signal.SIGINT)
        def interrupted(*_):
            stop[1]+=1;stop[0]=True
            print('STOP REQUESTED - finishing this complete wave and PPO checkpoint. Ctrl+C again aborts uncheckpointed work.',flush=True)
            if stop[1]>1:raise KeyboardInterrupt('Immediate operator interruption; uncheckpointed work withheld')
        signal.signal(signal.SIGINT,interrupted)
        def stopped():return stop[0] or (folder/'STOP_REQUESTED').exists()
        (folder/'STOP_REQUESTED').unlink(missing_ok=True)
        atomic_json(attempt/'attempt.json',dict(state='starting',started_unix=time()))
        try:
            profile=json.loads(json.dumps(exp.to_dict()))
            if reg:
                if reg['schema']!='patrick-jane-a2-duration-v1' or reg['experiment']!=profile or reg['workers']!=workers or reg.get('additional_decisions')!=additional_decisions:raise ValueError('Frozen run profile changed')
                if reg['sources']!=source_lock():raise ValueError('Training sources changed; use the exact returned source package')
                counters,pointer,history,rejected=recover(folder,history,policy,optimizer,rng,exp)
                atomic_json(attempt/'recovery.json',dict(selected=pointer,rejected=rejected))
                atomic_json(folder/'history.json',history);atomic_json(folder/'latest.json',pointer)
                current=folder/pointer['checkpoint']
            else:
                counters,provenance=import_update102(origin,policy,optimizer,rng)
                current=folder/f'checkpoints/{attempt.name}-origin.pt'
                sha=save(current,policy,optimizer,exp,counters.copy(),rng=rng,metadata=dict(import_provenance=provenance))
                pointer=dict(checkpoint=current.relative_to(folder).as_posix(),sha256=sha,counters=counters.copy())
                atomic_json(folder/'origin.json',pointer);atomic_json(folder/'latest.json',pointer)
                reg=dict(schema='patrick-jane-a2-duration-v1',reward_regime=exp.reward_regime,experiment=profile,workers=workers,
                    initial_duration_hours=hours,additional_decisions=additional_decisions,budget_unit='collection_and_ppo_active_seconds',base_decisions=2813879,
                    source_version=read(ROOT/'SOURCE_VERSION.json',{'revision':'unpublished-working-tree','source_lock':source_lock()}),
                    clock_source=CLOCK_SOURCE,
                    reward_coefficients=__import__('patrick_jane.a2_config',fromlist=['coefficients']).coefficients(exp),
                    training_seed_range=[3200000,3299999],validation_seeds=[],final_partitions='unopened',
                    source_snapshot=freeze_sources(folder),sources=source_lock(),import_provenance=provenance,
                    checkpoint_selection='latest complete PPO boundary; best not evaluated',acceptance='pending')
                atomic_json(folder/'registration.json',reg)
            if prepare_only:
                state='prepared';return 0
            if budget.reached or additional_decisions is not None and counters['run_decisions']>=additional_decisions:
                if not history:raise RuntimeError('Active budget exhausted without a completed PPO update; refusing fabricated completion')
                state='complete'
            else:
                with Resources() as resources,Pool(workers) as pool:
                    progress=Progress(attempt,workers,resources,mirror=folder/'status.json');progress.console=True
                    progress.matches=[m for row in history for m in row['rollout']['matches']]
                    for row in history:
                        for values in row['rollout']['reward_components'].values():progress.components.update(values)
                    def publish(active=None,**extra):
                        budget.persist();progress.checkpoint=current
                        progress.counters=dict(individual_decisions=counters['run_decisions'],complete_matches=counters['run_matches'],update=counters['run_updates'])
                        in_flight=(active or progress.active).get('decisions',0) if progress.stage in ('collecting','PPO') else 0
                        progress.publish(active,budget_unit='active_training_seconds',**budget.telemetry(),
                            elapsed=budget.value(),throughput=(counters['run_decisions']+in_flight)/max(1,budget.value()),
                            trainer_pid=__import__('os').getpid(),lifetime_decisions=counters['individual_decisions'],
                            lifetime_update=counters['update'],
                            base_decisions=2813879,report_path='TRAINING_REPORT.md',**extra)
                    generate(folder,'running',active_seconds=budget.value())
                    atomic_json(folder/'completion-status.json',dict(state='running',success=False))
                    while not budget.reached and not stopped() and (additional_decisions is None or counters['run_decisions']<additional_decisions):
                        if source_lock()!=reg['sources']:raise RuntimeError('Frozen training source changed')
                        verify(ROOT,controls_only=True)
                        allocations=read(folder/'seed-allocation.json',[])
                        first=max(counters['next_seed'],max((a['end_index'] for a in allocations),default=0))
                        seeds=list(range(3200000+first,3200000+first+workers))
                        if seeds[-1]>3299999:raise RuntimeError('Development partition exhausted; no seed recycling')
                        allocation=dict(start_index=first,end_index=first+workers,seeds=seeds,attempt=attempt.name,policy_update=counters['update'],state='allocated')
                        allocations.append(allocation);atomic_json(folder/'seed-allocation.json',allocations)
                        progress.stage='collecting';budget.begin();publish({})
                        try:
                            rollout=collect(policy,pool,exp,seeds,device=device,update=counters['update'],
                                training_step=counters['individual_decisions'],checkpoint=current.relative_to(ROOT).as_posix(),
                                replay_dir=attempt/'replays',record_every=workers,monitor_dir=progress.monitor_dir,
                                wall_limit=float('inf'),status_callback=publish)
                        finally:budget.end()
                        batch=attempt/f'batches/training-u{counters["update"]:06d}.json'
                        atomic_json(batch,{k:v for k,v in rollout.items() if k!='trajectories'});audit_lights(rollout)
                        progress.stage='PPO';budget.begin();publish(dict(decisions=rollout['individual_decisions']))
                        try:learning=update(policy,optimizer,rollout.pop('trajectories'),exp,rng=rng,device=device)
                        finally:budget.end()
                        if source_lock()!=reg['sources']:raise RuntimeError('Frozen source changed during wave; update withheld')
                        for k,v in [('individual_decisions',rollout['individual_decisions']),('run_decisions',rollout['individual_decisions']),('complete_matches',workers),('run_matches',workers),('update',1),('run_updates',1)]:counters[k]+=v
                        counters['next_seed']=first+workers
                        current=folder/f'checkpoints/{attempt.name}-update-{counters["update"]:06d}.pt'
                        sha=save(current,policy,optimizer,exp,counters.copy(),rng=rng,metadata=dict(batch_journal=batch.relative_to(folder).as_posix(),active_training_seconds=budget.value()))
                        pointer=dict(checkpoint=current.relative_to(folder).as_posix(),sha256=sha,counters=counters.copy())
                        row=dict(**pointer,rollout=rollout,ppo=learning);history.append(row)
                        atomic_json(folder/'history.json',history);atomic_json(folder/'latest.json',pointer)
                        allocation.update(state='checkpointed',checkpoint=pointer['checkpoint']);atomic_json(folder/'seed-allocation.json',allocations)
                        progress.settle(rollout);progress.ppo=learning;progress.stage='checkpoint'
                        publish(progress.final_streams(rollout['matches'],counters['update']-1,workers))
                        generate(folder,'running',active_seconds=budget.value(),plots=counters['run_updates']%10==0)
                    state='complete' if budget.reached or additional_decisions is not None and counters['run_decisions']>=additional_decisions else 'interrupted'
        except KeyboardInterrupt as exc:state='interrupted';error=str(exc)
        except BaseException as exc:
            error=repr(exc);(attempt/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        finally:
            budget.end();signal.signal(signal.SIGINT,old_signal)
            latest=read(folder/'latest.json',{});counters=latest.get('counters',{})
            if latest:
                try:generate(folder,state,active_seconds=budget.value(),error=error,plots=True)
                except Exception as exc:
                    state='failed';error=f'{error or ""}; final report: {exc!r}'
                    (folder/'TRAINING_REPORT.md').write_text('# Reporting failed\n\n'+error+'\n\nLatest verified checkpoint and batch journals remain preserved.\n',encoding='utf-8')
                    atomic_json(folder/'training_results.json',dict(state=state,error=error,latest=latest,active_training_seconds=budget.value()))
                if state=='complete':
                    weights=folder/f'weights/{attempt.name}-update-{counters["update"]:06d}.pt';weights.parent.mkdir(exist_ok=True)
                    with weights.open('xb') as f:torch.save(dict(architecture=policy.architecture(),state_dict={k:v.detach().cpu() for k,v in policy.state_dict().items()},regime=exp.reward_regime,lifetime_decisions=counters['individual_decisions']),f)
                    atomic_json(folder/'final-model.json',dict(weights=weights.relative_to(folder).as_posix(),sha256=digest(weights),resumable=latest))
                atomic_json(folder/'completion-status.json',dict(state=state,success=state=='complete',error=error,**budget.telemetry(),latest=latest))
                atomic_json(attempt/'attempt.json',dict(state=state,error=error,active_seconds=budget.value(),finished_unix=time()))
                if progress:
                    progress.stage=state;progress.counters=dict(individual_decisions=counters.get('run_decisions',0),complete_matches=counters.get('run_matches',0),update=counters.get('run_updates',0));publish({})
                try:export=export_results(ROOT,folder)
                except Exception as exc:
                    atomic_json(folder/'export-error.json',dict(error=repr(exc)));print('EXPORT FAILED; checkpoints/reports preserved:',repr(exc),flush=True)
                    state='failed';error=f'{error or ""}; required final export failed: {exc!r}'
                    atomic_json(folder/'completion-status.json',dict(state=state,success=False,training_boundary_saved=True,error=error,latest=latest,**budget.telemetry()))
                    generate(folder,state,active_seconds=budget.value(),error=error,plots=False)
                    atomic_json(attempt/'attempt.json',dict(state=state,error=error,active_seconds=budget.value(),finished_unix=time()))
                    if progress:progress.stage=state;publish({})
            print('TRAINING COMPLETED' if state=='complete' and export else f'A2 TRAINING {state.upper()}',flush=True)
            if latest:
                print(f"Final update: {counters['update']} | new decisions: {counters['run_decisions']:,} | lifetime decisions: {counters['individual_decisions']:,}",flush=True)
                print(f"Completed match segments: {counters['run_matches']} | active hours: {budget.value()/3600:.3f}",flush=True)
                from .progress import recorded_metrics
                matches=[m for row in history for m in row['rollout']['matches']]
                print('Recorded outcomes:',dict(wins=sum(m['winner']=='crewmate' for m in matches),losses=sum(m['winner']=='impostor' for m in matches),technical_cuts=sum(m.get('truncated',False) for m in matches),**recorded_metrics(matches)),flush=True)
                print('FINAL CHECKPOINT:',folder/latest['checkpoint'],flush=True)
                final_model=read(folder/'final-model.json',{})
                print('FINAL WEIGHTS:',folder/final_model['weights'] if final_model else 'not produced for this state',flush=True)
            print('FINAL REPORT:',folder/'TRAINING_REPORT.md',flush=True)
            if export:print('RESULTS EXPORT:',export,flush=True)
            if error:print('ERROR:',error,flush=True)
        return 0 if state in ('complete','prepared') else 130 if state=='interrupted' else 1
