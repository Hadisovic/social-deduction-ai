"""Create a new Classic profile; never starts gameplay, evaluation or training."""
import argparse
from dataclasses import replace
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from jane.classic_v2.settings import load_profile,save_profile,HostProfile
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=ROOT/'jane/configs/classic-v2/default_12p.json')
    for name in ('players','impostors','common','long','short'):parser.add_argument('--'+name,type=int)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();profile=load_profile(args.base);s=profile.settings
    lobby=replace(s.lobby,**{k:getattr(args,k) for k in ('players','impostors') if getattr(args,k) is not None})
    tasks=replace(s.tasks,**{k:getattr(args,k) for k in ('common','long','short') if getattr(args,k) is not None})
    updated=HostProfile(replace(s,lobby=lobby,tasks=tasks,profile_id=args.output.stem))
    save_profile(args.output,updated)
    print(json.dumps(dict(profile=str(args.output),players=lobby.players,crew=lobby.players-lobby.impostors,
        impostors=lobby.impostors,configured_counts=[tasks.common,tasks.long,tasks.short],
        effective_counts=updated.effective_counts,sha256=updated.fingerprint(),mode='configuration only')))


if __name__=='__main__':main()
