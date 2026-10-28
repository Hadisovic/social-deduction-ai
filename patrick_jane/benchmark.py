"""Comparable fixed-scenario full-game scheduling and recurrent PPO measurements."""
import argparse
import copy
import json
import os
from pathlib import Path
from time import perf_counter


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--counts',nargs='+',type=int,default=[8,16,32])
    parser.add_argument('--games',type=int,default=32);parser.add_argument('--wall-seconds',type=float,default=2400)
    parser.add_argument('--status-mirror');args=parser.parse_args()
    if any(n not in (8,16,32) for n in args.counts) or args.games<32 or args.games>64 or args.games%32:
        parser.error('Use 8/16/32 workers and the same 32 or 64 engineering scenarios')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import random
    import numpy as np
    import torch
    from .config import Experiment,ENGINEERING
    from .network import PatrickJane
    from .collector import Pool,collect
    from .ppo import update
    from .checkpoint import source_lock,freeze_sources
    from .telemetry import Resources
    from .progress import Progress,atomic_json
    folder=Path(args.output).resolve();folder.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(17);random.seed(17);torch.use_deterministic_algorithms(True)
    policy=PatrickJane().cuda();experiment=Experiment();sources=source_lock();began=perf_counter();results={}
    atomic_json(folder/'registration.json',dict(workers=args.counts,seeds=list(ENGINEERING[:args.games]),
        sources=sources,source_snapshot=freeze_sources(folder),experiment=experiment.to_dict(),initialization_seed=17,device='cuda',
        engineering_only=True,wall_seconds=args.wall_seconds,record_every=8,worker_python_hash_seed=0,
        note='Same paired scenarios; policy sampling streams are independent per scenario/actor. Floating-point batching may alter rare sampled ties; report actual trace agreement.'))
    try:
        for count in args.counts:
            if source_lock()!=sources:raise RuntimeError('Source changed during comparable benchmark')
            run=folder/f'workers-{count}';run.mkdir();matches=[];batches=[];ppo_measurement=None
            started=perf_counter()
            with Resources() as resources,Pool(count) as pool:
                progress=Progress(run,count,resources,mirror=args.status_mirror);progress.stage='benchmark'
                for offset in range(0,args.games,count):
                    remaining=args.wall_seconds-(perf_counter()-began)
                    if remaining<=0:raise TimeoutError('Comparable benchmark ceiling')
                    rollout=collect(policy,pool,experiment,ENGINEERING[offset:offset+count],device='cuda',
                        replay_dir=run/'replays',record_every=8,monitor_dir=progress.monitor_dir,
                        wall_limit=remaining,status_callback=progress.publish,update=0)
                    trajectories=rollout.pop('trajectories')
                    if ppo_measurement is None and count==args.counts[0]:
                        # Same real complete trajectories and fresh weights, optimizer and RNG.
                        measured={};states={}
                        for batched,label in ((False,'serial_sequences'),(True,'batched_sequences')):
                            candidate=copy.deepcopy(policy);optimizer=torch.optim.Adam(candidate.parameters(),lr=experiment.learning_rate)
                            progress.stage='PPO benchmark';progress.publish(dict(progress.final_streams(rollout['matches'],0,8),decisions=rollout['individual_decisions']))
                            measured[label]=update(candidate,optimizer,trajectories,experiment,rng=np.random.default_rng(17),device='cuda',batched=batched,deadline=began+args.wall_seconds)
                            states[label]={k:v.detach().cpu().clone() for k,v in candidate.state_dict().items()}
                            del optimizer,candidate
                        measured['maximum_weight_difference']=max(float((states['serial_sequences'][k]-states['batched_sequences'][k]).abs().max()) for k in states['serial_sequences'])
                        measured['speedup']=measured['serial_sequences']['elapsed']/measured['batched_sequences']['elapsed']
                        ppo_measurement=measured;progress.stage='benchmark'
                        rollout['ppo']=measured['batched_sequences']
                    else:
                        candidate=copy.deepcopy(policy);optimizer=torch.optim.Adam(candidate.parameters(),lr=experiment.learning_rate)
                        progress.stage='PPO benchmark';progress.publish(dict(progress.final_streams(rollout['matches'],0,8),decisions=rollout['individual_decisions']))
                        rollout['ppo']=update(candidate,optimizer,trajectories,experiment,rng=np.random.default_rng(17),device='cuda',deadline=began+args.wall_seconds)
                        del optimizer,candidate;progress.stage='benchmark'
                    del trajectories
                    matches.extend(rollout['matches']);batches.append(rollout);progress.settle(rollout)
                    progress.counters.update(individual_decisions=sum(b['individual_decisions'] for b in batches),complete_matches=len(matches))
                    progress.publish(progress.final_streams(rollout['matches'],0,8))
                elapsed=perf_counter()-started
                record=dict(workers=count,matches=matches,batches=batches,wall_seconds=elapsed,
                    collection_seconds=sum(b['elapsed'] for b in batches),
                    individual_decisions=sum(b['individual_decisions'] for b in batches),resources=resources.summary(),
                    ppo_comparison=ppo_measurement,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
                record['decisions_per_collection_second']=record['individual_decisions']/record['collection_seconds']
                record['games_per_collection_minute']=60*len(matches)/record['collection_seconds']
                record['ppo_seconds']=sum(b['ppo']['elapsed'] for b in batches)
                record['training_seconds']=record['collection_seconds']+record['ppo_seconds']
                record['decisions_per_training_second']=record['individual_decisions']/record['training_seconds']
                results[str(count)]=record;atomic_json(run/'metrics.json',record)
                progress.stage='complete';progress.publish(complete=True)
                print(json.dumps(dict(event='benchmark_complete',workers=count,seconds=elapsed,decisions=record['individual_decisions'],rate=record['decisions_per_collection_second'])),flush=True)
        if source_lock()!=sources:raise RuntimeError('Source changed during comparable benchmark')
        reference={m['seed']:m for m in results[str(args.counts[0])]['matches']}
        agreement={n:dict(scenarios=len(r['matches']),identical_traces=sum(m['trace']==reference[m['seed']]['trace'] for m in r['matches']),
            identical_results=sum((m['winner'],m['reason'])==(reference[m['seed']]['winner'],reference[m['seed']]['reason']) for m in r['matches'])) for n,r in results.items()}
        atomic_json(folder/'comparison.json',dict(results=results,agreement=agreement,sources=sources,elapsed=perf_counter()-began))
    except BaseException as exc:
        atomic_json(folder/'failure.json',dict(error=repr(exc),elapsed=perf_counter()-began,completed_counts=list(results)));raise


if __name__=='__main__':main()
