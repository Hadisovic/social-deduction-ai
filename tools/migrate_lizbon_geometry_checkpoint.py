"""Explicit weights-only transfer of registered pre-fix Lizbon checkpoints."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
CATALOG=ROOT/'docs/jane-impostor-v2/study-20261030/checkpoint-manifest.json'
PROFILE=ROOT/'jane/configs/classic-v2/default_12p.json'


def migrate(parent,output,profile_path=PROFILE):
    parent=Path(parent);output=Path(output)
    if output.exists():raise FileExistsError(output)
    verified_bytes=parent.read_bytes()
    digest=hashlib.sha256(verified_bytes).hexdigest()
    catalog=json.loads(CATALOG.read_text(encoding='utf-8'))
    approved=next((r for r in catalog['checkpoints'] if r['sha256']==digest),None)
    if approved is None:raise ValueError('Parent is not a registered pre-fix development checkpoint')
    # Exact recorded bytes are verified BEFORE torch.load. No arbitrary old
    # source/settings contract is accepted by the normal checkpoint loader.
    import torch
    torch.set_num_threads(1)
    from jane.classic_v2.settings import load_profile
    from jane.impostor.v1.model import ImpostorPolicy
    from jane.impostor.v1.run import tensor_hash
    from jane.impostor.v2 import SCHEMA
    from jane.impostor.v2.checkpoint import save,load
    from jane.impostor.v2.run import known_passes
    profile=load_profile(profile_path)
    old=torch.load(io.BytesIO(verified_bytes),map_location='cpu',weights_only=True)
    model=ImpostorPolicy(old['initialization_seed'])
    if (old['schema']!=SCHEMA or old['architecture']!=model.architecture()
        or old['runtime']!=dict(python=platform.python_version(),torch=str(torch.__version__))
        or old['host_profile']!=profile.to_dict() or old['counters']!=approved['counters']
        or old['receipt']['profile_sha256']!=catalog['profile_sha256']
        or old['receipt'].get('geometry_revision') is not None
        or old['crew_update']!=102 or old['boundary']!='complete-matches'
        or old['reward_regime']!='jane-impostor-team-v1'):
        raise ValueError('Historical checkpoint migration contract mismatch')
    model.load_state_dict(old['state_dict'],strict=True);before=tensor_hash(model)
    ancestry=dict(transfer=True,parent_sha256=digest,parent_schema=old['schema'],
        parent_counters=old['counters'],parent_total_steps=known_passes(old),
        parent_source_commit=catalog['source_commit'],parent_receipt=old['receipt'],
        migration='geometry-pair-v1',migration_tool_sha256=hashlib.sha256(
            Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        optimizer='not transferred; fresh Adam required for any future approved study',
        reason='Explicit cross-geometry-revision weights transfer, not exact study continuation')
    counters=dict(complete_matches=0,technical_cuts=0,optimizer_steps=0,training_observations=0)
    output.parent.mkdir(parents=True,exist_ok=True)
    checksum=save(output,model,None,profile,counters,old['initialization_seed'],ancestry)
    restored=ImpostorPolicy(old['initialization_seed']);load(output,restored,profile)
    assert tensor_hash(restored)==before
    return dict(output=str(output),sha256=checksum,tensor_sha256=before,tensors_identical=True,
        inherited_optimizer_passes=ancestry['parent_total_steps'],new_optimizer_passes=0,
        parent_sha256=digest,mode='offline weights-only migration; no inference/PPO/evaluation')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',type=Path,default=PROFILE)
    args=p.parse_args();print(json.dumps(migrate(args.parent,args.output,args.profile)))


if __name__=='__main__':main()
