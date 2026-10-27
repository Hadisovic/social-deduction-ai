"""Run the conditional, deadline-limited learned pursuit correction."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from impostor_pursuit.study import run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--parent-study',required=True,type=Path)
    p.add_argument('--parent-checkpoint',required=True,type=Path);p.add_argument('--resume',action='store_true')
    args=p.parse_args();run(args.output,args.parent_study,args.parent_checkpoint,args.resume)
