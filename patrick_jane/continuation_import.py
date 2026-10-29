"""Narrow audited bridge from the preserved pilot; strict ordinary load is unchanged."""
import hashlib,json,platform,random,zipfile
from pathlib import Path

PILOT_SHA='d8b81ef06788a0139822a5677801f34069aef116286cc079cb12de8c39538ab8'
ARCHIVE_SHA='645c7a8d3cf78eedc1315479fd722470298a60974edd6641e69bc7f9ba467460'
UNCHANGED=('network.py','ppo.py','features.py','actions.py','memory.py','navigation.py','decoder.py','rewards.py','config.py')
REGIME_FILES={'patrick_jane/environment.py','patrick_jane/worker.py','patrick_jane/opponents.py','patrick_jane/collector.py','patrick_jane/ppo.py','patrick_jane/progress.py'}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_pilot(checkpoint,archive,policy,optimizer,rng):
    import numpy as np
    import torch
    from .checkpoint import ROOT
    if digest(checkpoint)!=PILOT_SHA or digest(archive)!=ARCHIVE_SHA:
        raise ValueError('Historical checkpoint/source archive hash mismatch; no fallback initialization')
    record=torch.load(checkpoint,map_location='cpu',weights_only=True)
    with zipfile.ZipFile(archive) as z:
        sources=json.loads(z.read('source-lock.json'))
        if sources!=record['sources']:raise ValueError('Historical checkpoint and archive disagree')
        for name,expected in sources.items():
            if hashlib.sha256(z.read(name).replace(b'\r\n',b'\n')).hexdigest()!=expected:
                raise ValueError('Corrupt historical source archive')
            if name not in REGIME_FILES and hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=expected:
                raise ValueError('Unapproved historical dependency change: '+name)
        for file in UNCHANGED:
            name='patrick_jane/'+file
            expected=z.read(name).replace(b'\r\n',b'\n')
            if file=='ppo.py':
                expected=expected.replace(b"if not trajectory[-1].done:raise ValueError('PPO requires complete match settlement')",
                    b"if not (trajectory[-1].done or trajectory[-1].truncated):raise ValueError('PPO requires a terminal or bootstrapped technical boundary')")
            if (ROOT/name).read_bytes().replace(b'\r\n',b'\n')!=expected:
                raise ValueError('Warm-start architecture/PPO/action/reward compatibility changed: '+name)
    runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__))
    if record['runtime']!=runtime or record['architecture']!=policy.architecture():
        raise ValueError('Warm-start runtime/architecture mismatch')
    if record['counters']['individual_decisions']!=255083:raise ValueError('Unexpected pilot budget')
    policy.load_state_dict(record['state_dict'],strict=True);optimizer.load_state_dict(record['optimizer'])
    if any(not torch.isfinite(p).all() for p in policy.parameters()):raise ValueError('Nonfinite imported policy')
    torch.set_rng_state(record['rng']['torch'])
    if torch.cuda.is_available() and record['rng']['cuda']:torch.cuda.set_rng_state_all(record['rng']['cuda'])
    random.setstate(record['rng']['python']);rng.bit_generator.state=record['rng']['numpy']
    return dict(record['counters']),dict(checkpoint_sha256=PILOT_SHA,source_archive_sha256=ARCHIVE_SHA,
        source_files_verified=len(sources),unchanged_modules=[n for n in UNCHANGED if n!='ppo.py'],
        ppo_only_change_verified='accept explicit technical bootstrap boundary; original GAE kernel and hyperparameters unchanged',policy_optimizer_rng_imported=True,
        regime_change='lights-only +0.005/s cost, lightweight evaluation; not a same-protocol resume')
