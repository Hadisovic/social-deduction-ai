"""Pinned, audited reward-regime transition; never weakens ordinary checkpoint load."""
import ast,hashlib,json,platform,random,zipfile
from pathlib import Path

CHECKPOINT_SHA='86eb2b968fd6453cf842c0c3fffd304f6224b76ea88ba0e3675f0e91f6e8e753'
ARCHIVE_SHA='713ed713d4f9951bf712a59ff87375851a908a2c6cbc44f2f3d7661c36050563'
APPROVED=dict(win=10.,loss=-10.,task=1.,ghost_task=0.,repair=3.,impostor_ejection=4.,innocent_ejection=-2.,report=1.25,exploration=.02,new_player=.05,death=-2.,idle=-.01,invalid=0.,vote=0.,accuse=0.,defend=0.)

def import_update11(policy,optimizer,rng):
    import numpy as np
    import torch
    from .checkpoint import ROOT
    from .rewards import COEFFICIENTS
    from .continuation_import import digest
    old=ROOT/'local-runs/patrick-jane/lights-learning-v1'
    checkpoint=old/'checkpoints/attempt-0001-update-000011.pt';archive=old/'source-snapshot.zip'
    if digest(checkpoint)!=CHECKPOINT_SHA or digest(archive)!=ARCHIVE_SHA:raise ValueError('Update-11 checkpoint/archive hash mismatch')
    reg=json.loads((old/'registration.json').read_text(encoding='utf-8'))
    pointer=json.loads((old/'latest.json').read_text(encoding='utf-8'))
    record=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if pointer['sha256']!=CHECKPOINT_SHA or pointer['counters']!=record['counters']:raise ValueError('Historical pointer disagreement')
    if record['sources']!=reg['sources'] or json.loads(json.dumps(record['experiment']))!=reg['experiment']:raise ValueError('Historical registration disagreement')
    with zipfile.ZipFile(archive) as z:
        sources=json.loads(z.read('source-lock.json'))
        if sources!=record['sources']:raise ValueError('Historical source disagreement')
        for name,sha in sources.items():
            data=z.read(name).replace(b'\r\n',b'\n')
            if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Corrupt archived source: '+name)
            if name not in ('patrick_jane/rewards.py','patrick_jane/health.py','patrick_jane/learning_run.py','patrick_jane/learning_report.py'):
                if (ROOT/name).read_bytes().replace(b'\r\n',b'\n')!=data:raise ValueError('Unapproved source difference: '+name)
        def without_coefficients(text):
            tree=ast.parse(text)
            tree.body=[n for n in tree.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='COEFFICIENTS' for t in n.targets))]
            return ast.dump(tree)
        if without_coefficients(z.read('patrick_jane/rewards.py').decode())!=without_coefficients((ROOT/'patrick_jane/rewards.py').read_text(encoding='utf-8')):raise ValueError('Reward implementation changed beyond coefficients')
    if COEFFICIENTS!=APPROVED:raise ValueError('Unapproved reward coefficients')
    if record['runtime']!=dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__)) or record['architecture']!=policy.architecture():raise ValueError('Runtime/architecture mismatch')
    c=record['counters']
    if c['individual_decisions']!=764660 or c['run_decisions']!=509577 or c['update']!=11:raise ValueError('Unexpected update-11 counters')
    for values in (record['state_dict'],record['optimizer']['state']):
        def finite(x):
            if torch.is_tensor(x):return bool(torch.isfinite(x).all())
            if isinstance(x,dict):return all(finite(v) for v in x.values())
            return True
        if not finite(values):raise ValueError('Nonfinite historical policy/optimizer')
    policy.load_state_dict(record['state_dict'],strict=True)
    if optimizer.state:raise ValueError('Reward transition requires a fresh Adam optimizer')
    if not all(torch.equal(v.cpu(),record['state_dict'][k]) for k,v in policy.state_dict().items()):raise ValueError('Imported weights differ')
    torch.set_rng_state(record['rng']['torch'])
    if torch.cuda.is_available() and record['rng']['cuda']:torch.cuda.set_rng_state_all(record['rng']['cuda'])
    random.setstate(record['rng']['python']);rng.bit_generator.state=record['rng']['numpy']
    return dict(c),dict(checkpoint_sha256=CHECKPOINT_SHA,source_archive_sha256=ARCHIVE_SHA,source_files_verified=len(sources),policy_tensors_verified=len(record['state_dict']),optimizer='reset Adam moments; unchanged PPO settings; retain all learned policy/critic weights',previous_regime_additional_decisions=509577,previous_regime_path=str(old),reward_coefficients=APPROVED,regime_change='owner-approved rewards-v2; separate experiment, not homogeneous resume')
