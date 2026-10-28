"""Bounded engineering CLI. Full research training is intentionally not exposed."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('smoke','benchmark','evaluate'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--environments',type=int,default=1)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--inline',action='store_true')
    parser.add_argument('--wall-seconds',type=float,default=240.)
    parser.add_argument('--checkpoint')
    args=parser.parse_args()
    if not 1<=args.environments<=32 or not 0<args.wall_seconds<=600:
        parser.error('Engineering checks allow 1..32 workers and at most ten minutes')
    if args.mode=='smoke' and args.environments>2:
        parser.error('Bounded smoke permits at most two complete matches')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    # Imports here keep Windows spawn workers free of CUDA/Torch initialization.
    import random
    import numpy as np
    import torch
    from .network import PatrickJane
    from .config import Experiment,ENGINEERING,DEV_VALIDATION
    from .collector import Pool,collect
    from .ppo import update
    from .checkpoint import save,load,source_lock
    from .telemetry import Resources
    torch.set_num_threads(1);torch.manual_seed(17);random.seed(17)
    torch.use_deterministic_algorithms(True)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    policy=PatrickJane().to(args.device);experiment=Experiment();rng=np.random.default_rng(17)
    optimizer=torch.optim.Adam(policy.parameters(),lr=experiment.learning_rate)
    if args.checkpoint:load(args.checkpoint,policy,optimizer,rng=rng)
    (output/'config.json').write_text(json.dumps(dict(experiment=experiment.to_dict(),architecture=policy.architecture(),
        sources=source_lock(),engineering_only=True,initialization_seed=17,device=args.device,environments=args.environments),indent=2)+'\n')
    print(json.dumps(dict(event='start',mode=args.mode,output=str(output),environments=args.environments)),flush=True)
    def status(x):
        record=dict(schema='patrick-jane-progress-v1',stage=args.mode,counters=dict(individual_decisions=x['decisions'],update=x['update']),
                    workers=args.environments,elapsed=x['elapsed'],environment_progress=x,resources=resources.summary())
        temporary=output/'status.json.tmp';temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(output/'status.json')
        print(json.dumps(dict(event='progress',**{k:v for k,v in x.items() if k!='streams'})),flush=True)
    before={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
    initial_checkpoint=output/'checkpoints/update-000000.pt'
    if args.mode=='smoke':save(initial_checkpoint,policy,optimizer,experiment,
        dict(individual_decisions=0,complete_matches=0,update=0),rng=rng,metadata=dict(engineering_only=True))
    began=perf_counter();result=None
    try:
        with Resources() as resources,Pool(args.environments,not args.inline) as pool:
            result=collect(policy,pool,experiment,(DEV_VALIDATION if args.mode=='evaluate' else ENGINEERING)[:args.environments],
                device=args.device,replay_dir=output/'replays',record_every=8,wall_limit=args.wall_seconds,
                status_callback=status,held_out=args.mode=='evaluate',checkpoint=str(initial_checkpoint) if args.mode=='smoke' else args.checkpoint)
        trajectories=result.pop('trajectories')
        if args.mode=='smoke':
            if perf_counter()-began>=args.wall_seconds:raise TimeoutError('No smoke update after wall budget expired')
            result['ppo']=update(policy,optimizer,trajectories,experiment,rng=rng,device=args.device)
            result['changed_tensors']=sum(not torch.equal(before[k],v.detach().cpu()) for k,v in policy.state_dict().items())
            checkpoint=output/'checkpoints/update-000001.pt'
            result['checkpoint_sha256']=save(checkpoint,policy,optimizer,experiment,
                dict(individual_decisions=result['individual_decisions'],complete_matches=len(result['matches']),update=1),rng=rng,
                metadata=dict(engineering_only=True,acceptance_pending=True))
            fresh=PatrickJane().to(args.device);load(checkpoint,fresh,restore_rng=False)
            result['checkpoint_reload_equal']=all(torch.equal(v,fresh.state_dict()[k]) for k,v in policy.state_dict().items())
        result['total_wall_seconds']=perf_counter()-began
        result['gpu_peak_allocated_bytes']=torch.cuda.max_memory_allocated() if args.device=='cuda' else 0
        result['resources']=resources.summary()
        result['status']='complete'
        final=dict(schema='patrick-jane-progress-v1',stage='complete',counters=dict(individual_decisions=result['individual_decisions'],
            update=1 if args.mode=='smoke' else 0),workers=args.environments,elapsed=result['total_wall_seconds'],
            resources=result['resources'],environment_progress=dict(streams=[dict(environment=i,seed=m['seed'],complete=True,
                simulation_seconds=m['simulation_seconds'],replay=str(output/'replays'/f'env-{i:02d}-seed-{m["seed"]}-update-0.jsonl')
                if i%8==0 else None) for i,m in enumerate(result['matches'])]))
        (output/'status.json').write_text(json.dumps(final,indent=2)+'\n')
    except BaseException as exc:
        failure=dict(status='failed',error=repr(exc),elapsed=perf_counter()-began)
        (output/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
        print(json.dumps(failure),flush=True);raise
    finally:
        if result:(output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(event='finished',decisions=result['individual_decisions'],games=len(result['matches']),
        seconds=result['total_wall_seconds'],ppo=result.get('ppo'))),flush=True)


if __name__=='__main__':main()
