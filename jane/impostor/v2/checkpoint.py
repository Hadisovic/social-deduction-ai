"""Strict V2 receipts and explicit weights-only transfer from preserved V1."""
import hashlib
import json
from pathlib import Path
import platform


def sources():
    from jane.impostor.v1.bootstrap import source_lock
    folder=Path(__file__).resolve().parent
    return {**{'v1/'+k:v for k,v in source_lock().items()},
        **{p.name:hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sorted(folder.glob('*.py'))}}


def save(path,model,optimizer,profile,counters,seed,ancestry=None):
    import torch
    from . import SCHEMA
    from jane.classic_v2.settings import receipt
    record=dict(schema=SCHEMA,architecture=model.architecture(),sources=sources(),
        runtime=dict(python=platform.python_version(),torch=str(torch.__version__)),
        host_profile=profile.to_dict(),receipt=receipt(profile),state_dict=model.state_dict(),
        optimizer=optimizer.state_dict() if optimizer else None,counters=dict(counters),
        initialization_seed=seed,ancestry=ancestry,crew_update=102,
        boundary='complete-matches',reward_regime='jane-impostor-team-v1')
    with Path(path).open('xb') as handle:torch.save(record,handle)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path,model,profile):
    import torch
    from . import SCHEMA
    from jane.classic_v2.settings import receipt
    r=torch.load(path,map_location='cpu',weights_only=True)
    if (r['schema']!=SCHEMA or r['architecture']!=model.architecture() or r['sources']!=sources()
        or r['runtime']!=dict(python=platform.python_version(),torch=str(torch.__version__))
        or r['host_profile']!=profile.to_dict() or r['receipt']!=receipt(profile)
        or r['reward_regime']!='jane-impostor-team-v1'):
        raise ValueError('Lizbon V2 source/runtime/profile/reward mismatch')
    model.load_state_dict(r['state_dict'],strict=True)
    return r


def warm_start(path,model,historical_profile):
    from jane.impostor.v1.run import load_checkpoint
    from jane.shared.settings import load_settings
    r=load_checkpoint(path,model,load_settings(historical_profile))
    return dict(transfer=True,parent_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        parent_schema=r['schema'],parent_counters=r['counters'],optimizer='fresh Adam; weights-only transfer',
        reason='New Classic engine/task allocation; not exact V1 continuation')
