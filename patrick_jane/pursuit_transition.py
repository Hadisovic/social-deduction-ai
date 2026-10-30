"""Explicit one-time update-18 bridge; normal resumes retain strict source checks."""
import hashlib,json,platform,random,zipfile
from pathlib import Path

SHA='933ba18f86c868b7ecee1e380f0caff9ff0ce309fc046db5122148fb1476225e'
ARCHIVE='bccc7b0a6fdbaf5ba982484a2394a513d08e2b79b474f414668f4091226f7258'
CHANGED={'patrick_jane/lights_setup.py','patrick_jane/features.py','patrick_jane/decoder.py',
         'patrick_jane/commitment.py','patrick_jane/rewards.py','patrick_jane/worker.py','patrick_jane/health.py','patrick_jane/progress.py'}


def import_update18(origin,policy,optimizer,rng):
    import torch,numpy as np
    from .checkpoint import ROOT
    from .continuation_import import digest
    origin=Path(origin)
    checkpoint=origin/'update-000018.pt';archive=origin/'source-snapshot.zip'
    if digest(checkpoint)!=SHA or digest(archive)!=ARCHIVE:raise ValueError('Update18 checkpoint/archive integrity failure')
    r=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if r['runtime']!=dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__)):raise ValueError('Install the bundled exact Python/NumPy/PyTorch runtime')
    if r['architecture']!=policy.architecture():raise ValueError('Update18 neural architecture mismatch')
    if r['counters']['individual_decisions']!=1518786 or r['counters']['update']!=18:raise ValueError('Unexpected update18 counters')
    if not r['experiment'].get('task_commitment_mode') or r['experiment']['allowed_sabotages'] not in (('lights',),['lights']):raise ValueError('Unexpected parent gameplay regime')
    with zipfile.ZipFile(archive) as z:
        if json.loads(z.read('source-lock.json'))!=r['sources']:raise ValueError('Historical source manifest mismatch')
        for name,sha in r['sources'].items():
            data=z.read(name).replace(b'\r\n',b'\n')
            if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Historical archive corrupted: '+name)
            if name not in CHANGED and (ROOT/name).read_bytes().replace(b'\r\n',b'\n')!=data:raise ValueError('Unregistered parent source change: '+name)
    def finite(x):
        if torch.is_tensor(x):return bool(torch.isfinite(x).all())
        if isinstance(x,dict):return all(finite(v) for v in x.values())
        if isinstance(x,(list,tuple)):return all(finite(v) for v in x)
        return True
    if not finite(r['state_dict']) or not finite(r['optimizer']):raise ValueError('Nonfinite parent policy/Adam tensors')
    policy.load_state_dict(r['state_dict'],strict=True);optimizer.load_state_dict(r['optimizer'])
    if not all(torch.equal(v.cpu(),r['state_dict'][k]) for k,v in policy.state_dict().items()):raise ValueError('Update18 weight transfer mismatch')
    torch.set_rng_state(r['rng']['torch'])
    if torch.cuda.is_available() and r['rng']['cuda']:torch.cuda.set_rng_state_all(r['rng']['cuda'])
    random.setstate(r['rng']['python']);rng.bit_generator.state=r['rng']['numpy']
    c=dict(r['counters']);c.update(run_decisions=0,run_matches=0,run_updates=0,next_seed=0)
    return c,dict(checkpoint_sha256=SHA,archive_sha256=ARCHIVE,parent_lifetime_decisions=1518786,
        tensors_verified=len(r['state_dict']),optimizer='Retain Adam moments and step counters; retain saved RNG. New reward/action regime, not an exact historical replay.',
        source_changes=sorted(CHANGED),regime='task-pursuit-v1; full living task +2; commitment enabled')
