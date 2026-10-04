"""Train the focal crewmate policy. No automatic training on UI launch."""
import argparse
from pathlib import Path
from phase5_training import train


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true',help='Isolated 512-transition smoke run')
    parser.add_argument('--steps',type=int)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--ablation',action='store_true',help='Train a current-observation-only policy without belief outputs')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    tag='current' if args.ablation else 'full'
    output=args.output or Path('artifacts/phase5')/('quick' if args.quick else 'runs')/f'{tag}-{args.seed}'
    train(output,args.steps or (512 if args.quick else 32768),args.seed,args.ablation,args.quick)


if __name__=='__main__':main()
