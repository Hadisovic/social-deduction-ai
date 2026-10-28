"""Read-only publication checks: artifacts, exact sources, runtime and original art."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import torch
from belief.model import BeliefModel
from crewmate_v2.config import LobbySettings
from phase3_assets import manifest as art_manifest,LOCAL,blob_hash
from .models import ROOT,MODELS,load_model,sha256

MANIFEST=ROOT/'docs/collaboration/impostor-training-handoff/manifest.json'


def publication_hash(path):
    data=path.read_bytes()
    if path.suffix in ('.py','.json','.txt','.md','.yml','.yaml') or path.name in ('.gitattributes','.gitignore'):
        data=data.replace(b'\r\n',b'\n')
    return hashlib.sha256(data).hexdigest()


def verify():
    data=json.loads(MANIFEST.read_text());failures=[]
    for relative,expected in data['files'].items():
        p=ROOT/relative
        if not p.is_file() or publication_hash(p)!=expected:failures.append(relative)
    if failures:raise ValueError('Publication source/asset hash mismatch: '+', '.join(failures))
    models={}
    for name in MODELS:
        policy,metadata=load_model(name)
        models[name]=dict(strict_load=True,sha256=sha256(ROOT/MODELS[name][0]),
            parameters=sum(p.numel() for p in policy.parameters()),arm=metadata['arm'],
            actor_decisions=metadata['actor_decisions'],gradients_enabled=any(p.requires_grad for p in policy.parameters()))
    belief=BeliefModel.load()
    for entry in art_manifest()['files'].values():
        path=LOCAL/entry['path']
        if not path.is_file() or blob_hash(path.read_bytes())!=entry['git_blob_sha1']:
            raise ValueError('Missing/original character artwork mismatch: '+entry['path'])
    from spectator.v2.preferences import DEFAULTS
    if DEFAULTS['omniscient'] or DEFAULTS['reveal_ballots'] or DEFAULTS['favorites'] or DEFAULTS['recent']:
        raise ValueError('Unsafe/personal viewer defaults')
    result=dict(status='verified',models=models,hashed_files=len(data['files']),
        unique_artwork_files=len({v['path'] for v in art_manifest()['files'].values()}),
        frozen_belief_sha256=sha256(ROOT/'artifacts/phase4/belief.pt'),
        python=platform.python_version(),torch=str(torch.__version__),lobby=LobbySettings().to_dict(),
        safe_spectator_defaults=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);args=p.parse_args()
    result=verify();text=json.dumps(result,indent=2)+'\n'
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    print(text)


if __name__=='__main__':main()
