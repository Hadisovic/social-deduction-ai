"""Versioned curriculum bridge from completed Rewards V2; strict resume is unchanged."""
import hashlib,json,platform,random,zipfile

SHA='0696f9662e986bd8d10eb93c409e05b8122e99b5c59640df7337946b415c6387'
ARCHIVE='7285f20ef60dfde7d765ffc0c8d0a6201643ce08ffa46f5404a15148fafb8eea'
CHANGED={'patrick_jane/replay.py','patrick_jane/features.py','patrick_jane/navigation.py','patrick_jane/worker.py',
         'patrick_jane/lights_setup.py','patrick_jane/learning_run.py','patrick_jane/learning_report.py'}

def import_update16(policy,optimizer,rng):
    import torch,numpy as np
    from .checkpoint import ROOT
    from .continuation_import import digest
    from .reward_transition import APPROVED
    from .rewards import COEFFICIENTS
    from .lights_setup import LightsExperiment
    old=ROOT/'local-runs/patrick-jane/lights-rewards-v2';q=old/'checkpoints/attempt-0002-update-000016.pt'
    if digest(q)!=SHA or digest(old/'source-snapshot.zip')!=ARCHIVE:raise ValueError('Update16 checkpoint/archive hash mismatch')
    r=torch.load(q,map_location='cpu',weights_only=True);reg=json.loads((old/'registration.json').read_text(encoding='utf-8'))
    if r['sources']!=reg['sources']:raise ValueError('Historical source registration mismatch')
    with zipfile.ZipFile(old/'source-snapshot.zip') as z:
        if json.loads(z.read('source-lock.json'))!=r['sources']:raise ValueError('Historical archive mismatch')
        for name,sha in r['sources'].items():
            data=z.read(name).replace(b'\r\n',b'\n')
            if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Corrupt archive: '+name)
            if name not in CHANGED and (ROOT/name).read_bytes().replace(b'\r\n',b'\n')!=data:raise ValueError('Unapproved source change: '+name)
    expected=LightsExperiment().to_dict();expected.pop('task_commitment_mode')
    if json.loads(json.dumps(expected))!=json.loads(json.dumps(r['experiment'])) or COEFFICIENTS!=APPROVED:raise ValueError('Unapproved settings/reward change')
    if r['runtime']!=dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__)) or r['architecture']!=policy.architecture():raise ValueError('Architecture/runtime mismatch')
    if r['counters']['individual_decisions']!=1266883 or r['counters']['update']!=16:raise ValueError('Unexpected historical budget')
    def finite(x):
        if torch.is_tensor(x):return bool(torch.isfinite(x).all())
        if isinstance(x,dict):return all(finite(v) for v in x.values())
        return True
    if not finite(r['state_dict']) or not finite(r['optimizer']):raise ValueError('Nonfinite historical tensors')
    policy.load_state_dict(r['state_dict'],strict=True);optimizer.load_state_dict(r['optimizer'])
    if not all(torch.equal(v.cpu(),r['state_dict'][k]) for k,v in policy.state_dict().items()):raise ValueError('Weight transfer mismatch')
    torch.set_rng_state(r['rng']['torch'])
    if torch.cuda.is_available() and r['rng']['cuda']:torch.cuda.set_rng_state_all(r['rng']['cuda'])
    random.setstate(r['rng']['python']);rng.bit_generator.state=r['rng']['numpy']
    return dict(r['counters']),dict(checkpoint_sha256=SHA,source_archive_sha256=ARCHIVE,source_files_verified=len(r['sources']),policy_tensors_verified=len(r['state_dict']),optimizer='retain Adam and RNG; rewards/PPO unchanged; explicit constrained-action curriculum',previous_regime_additional_decisions=1011800,regime_change='task commitment v1, legal report/repair pauses, no abandonment penalty')
