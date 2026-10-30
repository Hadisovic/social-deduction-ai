"""Explicit update102 warm-start. Historical strict loading is never weakened."""
import hashlib,json,random,platform,zipfile
from pathlib import Path

SHA='6cf7048310e889967b01dd785317d313ddc71f310cd9faedcf346b950b8fd01b'
ARCHIVE='2ee74da35096493975e8bfb8ef2d536704623c0fca588be14d524b8738cb42e3'
CHANGED={'patrick_jane/api.py','patrick_jane/features.py','patrick_jane/health.py','patrick_jane/worker.py',
         'patrick_jane/progress.py'}  # Publication adds observer-only recorded terminal metrics.


def import_update102(origin,policy,optimizer,rng):
    import torch,numpy as np
    from .checkpoint import ROOT
    from .continuation_import import digest
    origin=Path(origin);checkpoint=origin/'update-000102.pt';archive=origin/'source-snapshot.zip'
    if digest(checkpoint)!=SHA or digest(archive)!=ARCHIVE:raise ValueError('Update102 checkpoint/archive hash failure')
    r=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if r['architecture']!=policy.architecture():raise ValueError('Architecture mismatch')
    if r['runtime']!=dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__)):raise ValueError('Use pinned Python/NumPy/PyTorch runtime')
    if (r['counters']['update'],r['counters']['individual_decisions'])!=(102,2813879):raise ValueError('Unexpected update102 counters')
    if r['experiment']['living_task_reward']!=2. or not r['experiment']['task_commitment_mode'] or not r['experiment']['task_pursuit_mode']:raise ValueError('Parent regime mismatch')
    with zipfile.ZipFile(archive) as z:
        if json.loads(z.read('source-lock.json'))!=r['sources']:raise ValueError('Historical source manifest mismatch')
        for name,sha in r['sources'].items():
            data=z.read(name).replace(b'\r\n',b'\n')
            if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Corrupt archive: '+name)
            if name not in CHANGED and (ROOT/name).read_bytes().replace(b'\r\n',b'\n')!=data:raise ValueError('Undeclared source bridge change: '+name)
    def finite(v):
        if torch.is_tensor(v):return bool(torch.isfinite(v).all())
        if isinstance(v,dict):return all(finite(x) for x in v.values())
        if isinstance(v,(list,tuple)):return all(finite(x) for x in v)
        return True
    if not finite(r['state_dict']) or not finite(r['optimizer']):raise ValueError('Nonfinite origin')
    policy.load_state_dict(r['state_dict'],strict=True);optimizer.load_state_dict(r['optimizer'])
    if not all(torch.equal(v.cpu(),r['state_dict'][k]) for k,v in policy.state_dict().items()):raise ValueError('Weights changed during import')
    torch.set_rng_state(r['rng']['torch'])
    if torch.cuda.is_available() and r['rng']['cuda']:torch.cuda.set_rng_state_all(r['rng']['cuda'])
    random.setstate(r['rng']['python']);rng.bit_generator.state=r['rng']['numpy']
    c=dict(r['counters']);c.update(next_seed=0,run_decisions=0,run_matches=0,run_updates=0)
    return c,dict(checkpoint_sha256=SHA,source_archive_sha256=ARCHIVE,source_files_verified=len(r['sources']),
        policy_tensors_verified=len(r['state_dict']),optimizer='Retain all Adam moments and step counts; explicit new reward/mechanics/feature regime.',
        feature_migration='Unchanged shapes/old normalized codes; append vent enter/exit and door-obstruction event codes. All 52 tensors reused.',
        parent_lifetime_decisions=2813879,source_changes=sorted(CHANGED))
