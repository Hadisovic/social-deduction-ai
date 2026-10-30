"""Manual paired frozen development comparisons. Never starts on package launch."""
import argparse,json,os
from pathlib import Path


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    parser=argparse.ArgumentParser();parser.add_argument('--origin',default='origin');parser.add_argument('--checkpoint')
    parser.add_argument('--output',required=True);parser.add_argument('--games',type=int,default=20);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if not 1<=args.games<=32 or not 1<=args.workers<=6:raise ValueError('Development comparison bound: 1..32 games and 1..6 workers')
    import torch,numpy as np
    from patrick_jane.network import PatrickJane
    from patrick_jane.a2_transition import import_update102
    from patrick_jane.checkpoint import load
    from patrick_jane.collector import Pool,collect
    from patrick_jane.a2_config import A2Experiment
    from patrick_jane.lights_setup import LightsExperiment
    from patrick_jane.a2_report import metrics
    from patrick_jane.progress import atomic_json
    from patrick_jane.continuation_import import digest
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    device='cuda';policy=PatrickJane().to(device);rng=np.random.default_rng(17)
    if args.checkpoint:
        record=load(args.checkpoint,policy,restore_rng=False);counters=record['counters'];sha=digest(args.checkpoint)
    else:
        optimizer=torch.optim.Adam(policy.parameters(),lr=.0003)
        counters,provenance=import_update102(args.origin,policy,optimizer,rng);sha=provenance['checkpoint_sha256']
    before={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
    conditions=dict(legacy=LightsExperiment(task_commitment_mode=True,task_pursuit_mode=True,living_task_reward=2.,observer_errors_fatal=False),
        a2_historical=A2Experiment(opponent_regime='historical'),a2_upgraded=A2Experiment(opponent_regime='upgraded-a2-v1'))
    seeds=list(range(3300000,3300000+args.games));results={}
    atomic_json(out/'protocol.json',dict(policy_sha256=sha,seeds=seeds,conditions={k:v.to_dict() for k,v in conditions.items()},training_updates=0,final_partitions='unopened'))
    with Pool(args.workers) as pool:
        for name,exp in conditions.items():
            matches=[]
            for offset in range(0,len(seeds),args.workers):
                batch=collect(policy,pool,exp,seeds[offset:offset+args.workers],device=device,held_out=False,
                    update=counters['update'],training_step=counters['individual_decisions'],checkpoint=args.checkpoint or 'origin/update-000102.pt',
                    replay_dir=out/name/'replays',record_every=args.workers,wall_limit=1800.)
                batch.pop('trajectories');atomic_json(out/name/f'batch-{offset:03d}.json',batch);matches+=batch['matches']
            results[name]=metrics(matches);atomic_json(out/name/'metrics.json',results[name])
    assert all(torch.equal(v.cpu(),before[k]) for k,v in policy.state_dict().items())
    atomic_json(out/'comparison.json',dict(policy_sha256=sha,weights_unchanged=True,conditions=results,
        limitation='Paired development conditions isolate mechanics/opponent difficulty within a fixed policy. Compare a trained A2 policy to update102 on identical conditions. Reward-only causal claims require separate matched training controls.'))
    print(out/'comparison.json');return 0

if __name__=='__main__':raise SystemExit(main())
