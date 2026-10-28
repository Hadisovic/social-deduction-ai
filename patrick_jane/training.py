"""Registered complete-match training; explicit approval and a wall ceiling required."""
import json
from pathlib import Path
from time import perf_counter
import random
import os
import math
import platform
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from .network import PatrickJane
from .config import Experiment,DEV_TRAIN,DEV_VALIDATION
from .collector import Pool,collect
from .ppo import update
from .checkpoint import save,load,select_champion,source_lock,freeze_sources
from .telemetry import Resources
from .progress import Progress,atomic_json
from .health import audit_rollout,severe_paired_regression,severe_speed_regression


def train(output,*,target_decisions,owner_approved=False,environments=16,device='cuda',
          experiment=None,resume=None,hard_wall_seconds=None,record_every=8,status_mirror=None):
    if owner_approved is not True:raise PermissionError('Full Patrick Jane training requires separate owner authorization')
    if not isinstance(target_decisions,int) or target_decisions<=0 or not 1<=environments<=32:
        raise ValueError('Register a positive individual-actor decision budget and valid worker count')
    if hard_wall_seconds is None or hard_wall_seconds<=0:raise ValueError('Register a measured wall-clock ceiling')
    folder=Path(output).resolve();folder.mkdir(parents=True,exist_ok=False)
    experiment=experiment or Experiment();torch.set_num_threads(1);torch.manual_seed(17);random.seed(17)
    torch.use_deterministic_algorithms(True);rng=np.random.default_rng(17)
    policy=PatrickJane().to(device);optimizer=torch.optim.Adam(policy.parameters(),lr=experiment.learning_rate)
    counters=dict(individual_decisions=0,complete_matches=0,update=0,next_seed=0,next_evaluation=experiment.evaluation_interval)
    if resume:
        record=load(resume,policy,optimizer,rng=rng)
        if record['experiment']!=experiment.to_dict():raise ValueError('Cannot resume into a changed protocol')
        counters=dict(record['counters'])
    configuration=dict(experiment=experiment.to_dict(),sources=source_lock(),source_snapshot=freeze_sources(folder),target_decisions=target_decisions,
        training_seeds=list(DEV_TRAIN),validation_seeds=list(DEV_VALIDATION),initialization_seed=17,
        workers=environments,device=device,hard_wall_seconds=hard_wall_seconds,resume=str(resume) if resume else None,
        evaluation_schedule='frozen zero-update, each 100000 decisions, final selected state; all 100 development validation scenarios',
        budget_unit='individual living learner decisions; complete-match batches may overshoot',
        tail_batch_rule='ceil(remaining target / empirical mean decisions per completed training match), clamped to 1..workers',
        absolute_decision_ceiling=math.ceil(target_decisions*1.10),
        runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__),cuda=torch.version.cuda,worker_python_hash_seed=0),
        stop_rules=dict(nonfinite=True,source_mismatch=True,legality=True,reward_attribution=True,
            novelty_task_death_report_duplication=True,wall_ceiling=True,
            speed='three consecutive collection rates below 40% of first-three median',
            validation='paired crew-win decline at least 25 percentage points and one-sided exact p<0.01 versus zero'),
        checkpoint_selection='held-out crew wins, then task wins, then correct minus innocent ejections',
        final_test_partitions='unopened',acceptance='pending')
    atomic_json(folder/'registration.json',configuration)
    began=perf_counter();deadline=began+hard_wall_seconds;history=[];evaluated=set();reference_matches=None
    if resume:
        baseline=Path(resume).resolve().parent.parent/'evaluations/update-000000-zero.json'
        if not baseline.is_file():raise ValueError('Resume requires its preserved zero-update validation reference')
        reference_matches=json.loads(baseline.read_text())['matches']
    current_checkpoint=Path(resume) if resume else folder/'checkpoints/update-000000.pt'
    if not resume:save(current_checkpoint,policy,optimizer,experiment,counters.copy(),rng=rng,metadata=dict(zero_update_reference=True))

    def remaining():
        seconds=deadline-perf_counter()
        if seconds<=0:raise TimeoutError('Registered wall ceiling reached; retain complete checkpoints and exact coverage')
        if source_lock()!=configuration['sources']:raise RuntimeError('Source changed during frozen experiment')
        return seconds

    with Resources() as resources:
        progress=Progress(folder,environments,resources,target=target_decisions,mirror=status_mirror)
        progress.counters=counters;progress.checkpoint=current_checkpoint
        def evaluate(pool,label):
            nonlocal current_checkpoint,reference_matches
            evaluation_began=perf_counter()
            progress.stage='validation-'+label;matches=[];eval_decisions=0
            for offset in range(0,experiment.evaluation_matches,environments):
                seeds=DEV_VALIDATION[offset:offset+min(environments,experiment.evaluation_matches-offset)]
                if not seeds:raise ValueError('Held-out coverage exhausted')
                result=collect(policy,pool,experiment,seeds,device=device,held_out=True,wall_limit=remaining(),
                    replay_dir=folder/'replays',record_every=record_every,monitor_dir=progress.monitor_dir,
                    update=counters['update'],training_step=counters['individual_decisions'],checkpoint=str(current_checkpoint),
                    status_callback=lambda active:progress.publish(active,validation_complete=len(matches),validation_target=experiment.evaluation_matches))
                result.pop('trajectories')
                atomic_json(folder/f'batches/validation-u{counters["update"]:06d}-{label}-{offset:03d}.json',result)
                audit_rollout(result)
                matches.extend(result['matches']);eval_decisions+=result['individual_decisions']
            remaining()
            record=dict(label=label,update=counters['update'],checkpoint=str(current_checkpoint),matches=matches,
                        elapsed=perf_counter()-evaluation_began,individual_decisions=eval_decisions,
                        new_champion=select_champion(current_checkpoint,matches,folder/'champions'))
            if label=='zero':reference_matches=matches
            elif reference_matches:record['regression_check']=severe_paired_regression(reference_matches,matches)
            atomic_json(folder/f'evaluations/update-{counters["update"]:06d}-{label}.json',record)
            evaluated.add(counters['update']);progress.active={}
            progress.history.append(dict(decisions=counters['individual_decisions'],validation_wins=sum(m['winner']=='crewmate' for m in matches),
                validation_games=len(matches),tasks=sum(m['metrics']['task_completions'] for m in matches),
                correct_ejections=sum(m['correct_ejections'] for m in matches),innocent_ejections=sum(m['innocent_ejections'] for m in matches)))
            progress.publish(validation_complete=len(matches),validation_target=experiment.evaluation_matches)
            if any(m['metrics']['illegal_actions'] or m['monitor_error'] for m in matches):
                raise RuntimeError('Validation legality/observer regression')
            if record.get('regression_check',{}).get('stop'):raise RuntimeError('Severe paired validation win regression; owner review required')
            return record
        try:
            with Pool(environments) as pool:
                if not resume:evaluate(pool,'zero')
                while counters['individual_decisions']<target_decisions:
                    if counters['individual_decisions']>=counters['next_evaluation']:
                        evaluate(pool,'interval')
                        while counters['next_evaluation']<=counters['individual_decisions']:counters['next_evaluation']+=experiment.evaluation_interval
                        current_checkpoint=folder/f'checkpoints/update-{counters["update"]:06d}-evaluated.pt'
                        save(current_checkpoint,policy,optimizer,experiment,counters.copy(),rng=rng)
                        progress.checkpoint=current_checkpoint
                    active=environments
                    if counters['complete_matches']:
                        mean=counters['individual_decisions']/counters['complete_matches']
                        active=min(environments,max(1,math.ceil((target_decisions-counters['individual_decisions'])/mean)))
                    first=counters['next_seed'];seeds=DEV_TRAIN[first:first+active]
                    if len(seeds)!=active:raise RuntimeError('Registered training seeds exhausted')
                    progress.stage='collecting';progress.publish({})
                    rollout=collect(policy,pool,experiment,seeds,device=device,update=counters['update'],
                        training_step=counters['individual_decisions'],replay_dir=folder/'replays',record_every=record_every,
                        wall_limit=remaining(),status_callback=progress.publish,checkpoint=str(current_checkpoint),monitor_dir=progress.monitor_dir,
                        decision_limit=configuration['absolute_decision_ceiling']-counters['individual_decisions'])
                    progress.stage='PPO';progress.publish(dict(progress.final_streams(rollout['matches'],counters['update'],record_every),decisions=rollout['individual_decisions']))
                    atomic_json(folder/f'batches/training-u{counters["update"]:06d}.json',
                        {key:value for key,value in rollout.items() if key!='trajectories'})
                    audit_rollout(rollout)
                    remaining()
                    learning=update(policy,optimizer,rollout.pop('trajectories'),experiment,rng=rng,device=device,deadline=deadline)
                    remaining()
                    if any(m['metrics']['illegal_actions'] or m['monitor_error'] for m in rollout['matches']):
                        raise RuntimeError('Illegal actions or observer error: readiness regression; withhold checkpoint')
                    counters['individual_decisions']+=rollout['individual_decisions'];counters['complete_matches']+=len(seeds)
                    counters['next_seed']+=len(seeds);counters['update']+=1
                    current_checkpoint=folder/f'checkpoints/update-{counters["update"]:06d}.pt'
                    digest=save(current_checkpoint,policy,optimizer,experiment,counters.copy(),rng=rng)
                    record=dict(counters=counters.copy(),rollout=rollout,ppo=learning,sha256=digest,wall_seconds=perf_counter()-began)
                    history.append(record);atomic_json(folder/'history.json',history)
                    atomic_json(folder/'latest.json',dict(checkpoint=str(current_checkpoint),sha256=digest,counters=counters.copy()))
                    progress.settle(rollout);progress.ppo=learning;progress.checkpoint=current_checkpoint
                    progress.stage='checkpoint';progress.publish(progress.final_streams(rollout['matches'],counters['update']-1,record_every))
                    print(json.dumps(dict(event='checkpoint',**counters,ppo_seconds=learning['elapsed'],wall_seconds=perf_counter()-began)),flush=True)
                    full_batches=[row for row in history if len(row['rollout']['matches'])==environments]
                    if active==environments and severe_speed_regression(full_batches):
                        raise RuntimeError('Three sustained throughput regressions below 40 percent of initial median')
                if counters['update'] not in evaluated:evaluate(pool,'final')
                remaining();progress.stage='complete';progress.publish(complete=True,overshoot_decisions=counters['individual_decisions']-target_decisions)
                atomic_json(folder/'completion.json',dict(counters=counters,elapsed=perf_counter()-began,sources=configuration['sources'],acceptance='pending'))
        except BaseException as exc:
            progress.stage='stopped';progress.publish(error=repr(exc),complete=False)
            atomic_json(folder/'failure.json',dict(error=repr(exc),elapsed=perf_counter()-began,last_complete_checkpoint=str(current_checkpoint),counters=counters))
            raise
    return counters


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--decisions',type=int,default=250_000)
    parser.add_argument('--workers',type=int,required=True);parser.add_argument('--wall-seconds',type=float,required=True)
    parser.add_argument('--status-mirror');parser.add_argument('--resume')
    parser.add_argument('--owner-approved',action='store_true')
    args=parser.parse_args()
    if not args.owner_approved:parser.error('Explicit owner authorization required')
    result=train(args.output,target_decisions=args.decisions,environments=args.workers,hard_wall_seconds=args.wall_seconds,
        status_mirror=args.status_mirror,resume=args.resume,owner_approved=True)
    print(json.dumps(dict(event='complete',**result)),flush=True)


if __name__=='__main__':main()
