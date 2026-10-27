"""Train two improved impostor policies and run registered selection/confirmation."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from impostor_research.study_v2 import run

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();run(args.output,args.resume)
