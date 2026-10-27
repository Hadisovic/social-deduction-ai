"""Run only the approved 71/83 ordinary-mode pilot; no merge/deploy/final tests."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from impostor_research.pilot import run

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true',help='Explicit recovery of this immutable pilot; original deadline retained')
    args=parser.parse_args()
    run(args.output,resume=args.resume)
