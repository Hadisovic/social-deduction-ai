"""Registered full-run infrastructure. Invocation requires explicit owner approval."""
import json
from pathlib import Path
from time import perf_counter
import random
import numpy as np
import torch
from .network import PatrickJane
from .config import Experiment,DEV_TRAIN,DEV_VALIDATION
from .collector import Pool,collect
from .ppo import update
from .checkpoint import save,load,select_champion,source_lock


def train(output,*,target_decisions,owner_approved=False,environments=16,device='cuda',
          experiment=None,resume=None,hard_wall_seconds=None,record_every=16):
    if owner_approved is not True:raise PermissionError('Full Patrick Jane training requires separate owner authorization')
    if not isinstance(target_decisions,int) or target_decisions<=0 or not 1<=environments<=32:
        raise ValueError('Register a positive individual-actor decision budget and valid worker count')
    if hard_wall_seconds is None or hard_wall_seconds<=0:raise ValueError('Register a measured wall-clock ceiling')
    folder=Path(output);folder.mkdir(parents=True,exist_ok=False)
    experiment=experiment or Experiment();torch.set_num_threads(1);torch.manual_seed(17);random.seed(17)
    torch.use_deterministic_algorithms(True);rng=np.random.default_rng(17)
    policy=PatrickJane().to(device);optimizer=torch.optim.Adam(policy.parameters(),lr=experiment.learning_rate)
    counters=dict(individual_decisions=0,complete_matches=0,update=0,next_seed=0,next_evaluation=experiment.evaluation_interval)
    if resume:
        record=load(resume,policy,optimizer,rng=rng)
        if record['experiment']!=experiment.to_dict():raise ValueError('Cannot resume into a changed protocol')
        counters=record['counters']
    configuration=dict(experiment=experiment.to_dict(),sources=source_lock(),target_decisions=target_decisions,
        workers=environments,device=device,hard_wall_seconds=hard_wall_seconds,resume=str(resume) if resume else None)
    (folder/'registration.json').write_text(json.dumps(configuration,indent=2)+'\n')
    began=perf_counter();history=[]
    current_checkpoint=resume
    def status(extra):
        record=dict(schema='patrick-jane-progress-v1',counters=counters,workers=environments,
                    elapsed=perf_counter()-began,target_decisions=target_decisions,**extra)
        temporary=folder/'status.json.tmp';temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(folder/'status.json')
    with Pool(environments) as pool:
        while counters['individual_decisions']<target_decisions:
            remaining=hard_wall_seconds-(perf_counter()-began)
            if remaining<=0:raise TimeoutError('Registered wall ceiling reached; no protocol reduction')
            first=counters['next_seed'];seeds=DEV_TRAIN[first:first+environments]
            if len(seeds)!=environments:raise RuntimeError('Registered training seeds exhausted; request protocol extension')
            rollout=collect(policy,pool,experiment,seeds,device=device,update=counters['update'],
                training_step=counters['individual_decisions'],replay_dir=folder/'replays',record_every=record_every,
                wall_limit=remaining,status_callback=status,checkpoint=str(current_checkpoint) if current_checkpoint else None)
            if perf_counter()-began>=hard_wall_seconds:raise TimeoutError('No PPO update after wall ceiling')
            learning=update(policy,optimizer,rollout.pop('trajectories'),experiment,rng=rng,device=device)
            counters['individual_decisions']+=rollout['individual_decisions'];counters['complete_matches']+=len(seeds)
            counters['next_seed']+=len(seeds);counters['update']+=1
            checkpoint=folder/f'checkpoints/update-{counters["update"]:06d}.pt'
            should_evaluate=counters['individual_decisions']>=counters['next_evaluation']
            if should_evaluate:counters['next_evaluation']+=experiment.evaluation_interval
            save(checkpoint,policy,optimizer,experiment,counters.copy(),rng=rng)
            current_checkpoint=checkpoint
            record=dict(counters=counters.copy(),rollout=rollout,ppo=learning)
            if should_evaluate:
                # Preserve exact coverage: serial batches of held-out matches, no silent reduction.
                matches=[]
                for offset in range(0,experiment.evaluation_matches,environments):
                    seeds=DEV_VALIDATION[offset:offset+min(environments,experiment.evaluation_matches-offset)]
                    if not seeds:raise ValueError('Held-out evaluation registration exhausted')
                    remaining=hard_wall_seconds-(perf_counter()-began)
                    if remaining<=0:raise TimeoutError('Wall ceiling before complete registered evaluation')
                    with Pool(len(seeds)) as evaluation_pool:
                        evaluation=collect(policy,evaluation_pool,experiment,seeds,device=device,held_out=True,wall_limit=remaining)
                    matches.extend(evaluation['matches'])
                record['evaluation']=matches;record['new_champion']=select_champion(checkpoint,matches,folder/'champions')
            history.append(record);(folder/'history.json').write_text(json.dumps(history,indent=2)+'\n')
            status(dict(last_complete_batch=record))
    status(dict(complete=True,overshoot_decisions=counters['individual_decisions']-target_decisions))
    return counters
