"""Complete paired GPU games, private-stream invariance, PPO and strict reload."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
from pathlib import Path
from time import perf_counter


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    import numpy as np
    import torch
    from patrick_jane.config import Experiment,ENGINEERING
    from patrick_jane.network import PatrickJane
    from patrick_jane.collector import Pool,collect
    from patrick_jane.ppo import update
    from patrick_jane.checkpoint import save,load,freeze_sources,source_lock
    from patrick_jane.health import audit_rollout
    from patrick_jane.progress import atomic_json
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.manual_seed(17)
    torch.use_deterministic_algorithms(True);policy=PatrickJane().cuda();exp=Experiment()
    rng=np.random.default_rng(17);optimizer=torch.optim.Adam(policy.parameters(),lr=exp.learning_rate)
    sources=source_lock();began=perf_counter();deadline=began+600
    atomic_json(args.output/'registration.json',dict(seeds=list(ENGINEERING[:2]),source_snapshot=freeze_sources(args.output),sources=sources))
    initial=args.output/'checkpoints/update-000000.pt';save(initial,policy,optimizer,exp,dict(update=0),rng=rng)
    try:
        reference=[]
        with Pool(1) as pool:
            for seed in ENGINEERING[:2]:
                r=collect(policy,pool,exp,[seed],device='cuda',wall_limit=deadline-perf_counter(),
                    replay_dir=args.output/'serial-replays',record_every=1,checkpoint=str(initial))
                audit_rollout(r);r.pop('trajectories');reference.extend(r['matches'])
        with Pool(2) as pool:
            together=collect(policy,pool,exp,ENGINEERING[:2],device='cuda',wall_limit=deadline-perf_counter(),
                replay_dir=args.output/'replays',record_every=1,checkpoint=str(initial))
        audit_rollout(together)
        exact=all(a['trace']==b['trace'] for a,b in zip(reference,together['matches']))
        atomic_json(args.output/'paired.json',dict(reference=reference,parallel=together['matches'],exact_traces=exact))
        if not exact:raise RuntimeError('Complete-game private streams differ across worker counts')
        learning=update(policy,optimizer,together.pop('trajectories'),exp,rng=rng,device='cuda',deadline=deadline)
        final=args.output/'checkpoints/update-000001.pt'
        digest=save(final,policy,optimizer,exp,dict(update=1,individual_decisions=together['individual_decisions']),rng=rng)
        fresh=PatrickJane().cuda();fresh_optimizer=torch.optim.Adam(fresh.parameters(),lr=exp.learning_rate)
        fresh_rng=np.random.default_rng(123);record=load(final,fresh,fresh_optimizer,rng=fresh_rng)
        equal=all(torch.equal(v,fresh.state_dict()[k]) for k,v in policy.state_dict().items())
        if not equal:raise RuntimeError('Checkpoint reload differs')
        if fresh_rng.bit_generator.state!=record['rng']['numpy']:raise RuntimeError('Optimizer RNG restoration differs')
        for a,b in zip(optimizer.state.values(),fresh_optimizer.state.values()):
            for key in a:
                if torch.is_tensor(a[key]) and not torch.equal(a[key],b[key]):raise RuntimeError('Optimizer state reload differs')
        if sources!=source_lock():raise RuntimeError('Sources changed during qualification')
        atomic_json(args.output/'result.json',dict(complete=True,exact_traces=exact,checkpoint_equal=equal,
            checkpoint_sha256=digest,wall_seconds=perf_counter()-began,ppo=learning,parallel=together,sources=sources))
        print(json.dumps(dict(complete=True,exact_traces=exact,checkpoint_equal=equal,wall_seconds=perf_counter()-began,ppo=learning)),flush=True)
    except BaseException as exc:
        atomic_json(args.output/'failure.json',dict(error=repr(exc),wall_seconds=perf_counter()-began));raise


if __name__=='__main__':main()
