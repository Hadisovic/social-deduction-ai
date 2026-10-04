"""Evaluate random, original scripted and supplied strategic policies."""
import argparse
from pathlib import Path
from phase5_evaluation import evaluate


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,action='append',default=[])
    parser.add_argument('--matches',type=int,default=1000)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('docs/benchmarks/phase5/evaluation'))
    args=parser.parse_args();evaluate(args.output,args.matches,args.checkpoint,args.quick)
