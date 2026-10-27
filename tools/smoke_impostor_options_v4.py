"""Bounded development matches only. No optimizer, selection or reserved finals."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--matches',type=int,default=2,choices=(1,2))
    parser.add_argument('--wall-seconds',type=float,default=180.)
    args=parser.parse_args(argv)
    if args.output.exists():parser.error('Never overwrite an existing smoke report')
    if not 0<args.wall_seconds<=300:parser.error('Bounded smoke wall budget must be <=300s')
    import torch
    from impostor_research.frozen import FrozenModels,weight_hash
    from impostor_research.learning_v2 import load
    from impostor_options_v4.env import OptionsEnv
    from impostor_options_v4.learning import record_for
    torch.set_num_threads(1)
    started=perf_counter();models=FrozenModels()
    path=ROOT/'artifacts/impostor-ai/release/policy.pt'
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    policy,metadata=load(path,models);tensor_hash=weight_hash(policy)
    report=dict(purpose='Engineering smoke; not validation accuracy or evidence of learning improvement',
        optimizer_steps=0,checkpoint_sha256=before,checkpoint_schema=metadata['schema'],
        transfer_demo=True,manifest=record_for(models),matches=[],completed=False)
    for seed in range(3500000,3500000+args.matches):
        env=OptionsEnv(models=models);env.reset(seed=seed);decisions=0
        while not env.finished and decisions<3000 and perf_counter()-started<args.wall_seconds:
            action,_,_=policy.act(env.packet);env.step(action);decisions+=1
        row=env.summary();row.update(completed=env.finished,decisions=decisions)
        report['matches'].append(row)
        print(json.dumps(dict(seed=seed,completed=env.finished,reason=row['reason'],decisions=decisions,
            kills=row['kills'],fake_completions=row['fake_completions'],ready_seconds=row['cooldown_ready_seconds'])),flush=True)
        if not env.finished:break
    models.assert_unchanged()
    if before!=hashlib.sha256(path.read_bytes()).hexdigest() or tensor_hash!=weight_hash(policy):
        raise RuntimeError('Smoke altered a checkpoint/policy')
    report['completed']=len(report['matches'])==args.matches and all(r['completed'] for r in report['matches'])
    report['wall_seconds']=perf_counter()-started
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    return 0 if report['completed'] else 2


if __name__=='__main__':raise SystemExit(main())
