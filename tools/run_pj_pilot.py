"""Lightweight Windows spawn entry point for the unchanged frozen training runner.

Keep Torch and all training imports inside the guarded parent entry point so CPU
simulation workers do not load CUDA DLLs while importing __mp_main__.
"""
import argparse
import os
import sys


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--decisions',type=int,default=250000)
    parser.add_argument('--workers',type=int,default=32);parser.add_argument('--wall-seconds',type=float,required=True)
    parser.add_argument('--status-mirror',required=True);parser.add_argument('--owner-approved',action='store_true')
    args=parser.parse_args()
    if not args.owner_approved:parser.error('Owner authorization required')
    from patrick_jane.training import train
    train(args.output,target_decisions=args.decisions,environments=args.workers,hard_wall_seconds=args.wall_seconds,
          status_mirror=args.status_mirror,owner_approved=True)


if __name__=='__mp_main__' and 'torch' in sys.modules:
    raise RuntimeError('CPU worker bootstrap unexpectedly imported Torch')

if __name__=='__main__':main()
