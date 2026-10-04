"""Run the preregistered three-seed full/current PPO experiment and paired evaluation."""
import argparse
import json
from pathlib import Path
from phase5_training import train
from phase5_policy import StrategicPolicy
from phase5_evaluation import evaluate


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true',help='One seed, 512 steps and four evaluation matches per method')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    config=json.loads(Path('artifacts/phase5/config.json').read_text())
    seeds=[17] if args.quick else config['training_seeds']
    steps=512 if args.quick else config['steps_per_seed']
    checkpoints=[]
    for ablation in (False,True):
        for seed in seeds:
            tag='current' if ablation else 'full'
            directory=Path('artifacts/phase5')/('quick' if args.quick else 'runs')/f'{tag}-{seed}'
            path=directory/'policy.pt'
            if path.exists():
                _,metadata=StrategicPolicy.load(path)
                if any(metadata.get(k)!=v for k,v in {'seed':seed,'ablation':ablation,'steps':steps,'quick':args.quick}.items()):
                    raise ValueError(f'Existing checkpoint differs from preregistration: {path}')
                print(f'Using completed checkpoint: {path}',flush=True)
            else:train(directory,steps,seed,ablation,args.quick)
            checkpoints.append(path)
    output=args.output or Path('docs/benchmarks/phase5')/('smoke' if args.quick else 'final')
    if not args.quick:
        validation=evaluate(output/'validation',100,checkpoints,validation=True)
        full=checkpoints[:len(seeds)]
        def rank(path):
            row=validation['metrics'][str(path)]['ID']
            return (row['crew_win_rate'],row['mean_own_tasks'],-row['illegal_actions'],-row['timeouts'])
        selected=max(full,key=rank)
        (output/'selection.json').write_text(json.dumps({'selected_full_checkpoint':str(selected),
            'selection_split':'validation seeds 620000..620099 only',
            'ranking':['crew_win_rate','mean_own_tasks','negative_illegal_actions','negative_timeouts'],
            'notice':'Selected before final ID/patient evaluation; three seeds are reported separately.'},indent=2)+'\n')
    return evaluate(output,4 if args.quick else 1000,checkpoints,args.quick)


if __name__=='__main__':main()
