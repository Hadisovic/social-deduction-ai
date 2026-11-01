"""Explicit, hash-pinned source bridge for the preserved failed A2 continuation.

Prepares a fresh run directory only. Never collects experience or invokes PPO.
"""
import argparse,hashlib,json,platform,random,shutil,sys,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
PARENT_SHA='2839a413e3b4d537a8ad8170b763642c1f07550a7370981ccee99890f331bc68'
ARCHIVE_SHA='9899e08042b42e008d4352af14c3a4448fa95c08a78259e3601540a102755827'
CHANGED={'jane/classic_v2/engine.py','patrick_jane/a2_report.py'}
PARENT_ACTIVE_SECONDS=578.8447452


def import_parent(origin,policy,optimizer,rng):
    import torch,numpy as np
    from patrick_jane.checkpoint import ROOT as source_root
    from patrick_jane import SCHEMA
    from patrick_jane.a2_config import A2Experiment
    from jane.shared.runtime import PATRICK_PROFILE
    from portable_support import digest
    origin=Path(origin);path=origin/'update-000103.pt';archive=origin/'source-snapshot.zip'
    if digest(path)!=PARENT_SHA or digest(archive)!=ARCHIVE_SHA:raise ValueError('Update103 checkpoint/archive hash failure')
    r=torch.load(path,map_location='cpu',weights_only=True)
    if r.get('schema')!=SCHEMA or r.get('boundary')!='complete-matches':raise ValueError('Parent schema/boundary mismatch')
    if r['architecture']!=policy.architecture():raise ValueError('Architecture mismatch')
    if r['runtime']!=dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__)):raise ValueError('Pinned runtime required')
    if r['experiment']!=A2Experiment(shared_engine_profile=PATRICK_PROFILE).to_dict():raise ValueError('Parent experiment changed')
    expected=dict(update=103,individual_decisions=2829668,run_decisions=15789,run_matches=6,run_updates=1,next_seed=12)
    if any(r['counters'][k]!=v for k,v in expected.items()):raise ValueError('Unexpected update103 counters')
    with zipfile.ZipFile(archive) as z:
        if json.loads(z.read('source-lock.json'))!=r['sources']:raise ValueError('Parent source manifest mismatch')
        for name,sha in r['sources'].items():
            data=z.read(name).replace(b'\r\n',b'\n')
            if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Corrupt parent source: '+name)
            if name not in CHANGED and (source_root/name).read_bytes().replace(b'\r\n',b'\n')!=data:raise ValueError('Undeclared source bridge change: '+name)
    def finite(value):
        if torch.is_tensor(value):return bool(torch.isfinite(value).all())
        if isinstance(value,dict):return all(finite(v) for v in value.values())
        if isinstance(value,(list,tuple)):return all(finite(v) for v in value)
        return True
    if not finite(r['state_dict']) or not finite(r['optimizer']):raise ValueError('Nonfinite parent state')
    policy.load_state_dict(r['state_dict'],strict=True);optimizer.load_state_dict(r['optimizer'])
    weights=sum(torch.equal(v.cpu(),r['state_dict'][k]) for k,v in policy.state_dict().items())
    adam=[torch.equal(v.cpu(),r['optimizer']['state'][k][name].cpu()) for k,values in optimizer.state_dict()['state'].items() for name,v in values.items()]
    if weights!=52 or len(adam)!=156 or not all(adam):raise ValueError('Policy/Adam transfer mismatch')
    torch.set_rng_state(r['rng']['torch'])
    if torch.cuda.is_available() and r['rng']['cuda']:torch.cuda.set_rng_state_all(r['rng']['cuda'])
    random.setstate(r['rng']['python']);rng.bit_generator.state=r['rng']['numpy']
    counters=dict(r['counters']);counters['next_seed']=18
    return counters,dict(checkpoint_sha256=PARENT_SHA,source_archive_sha256=ARCHIVE_SHA,
        policy_tensors_verified=weights,adam_tensors_verified=len(adam),optimizer='Retained all moments and steps; no updates during bridge.',
        parent_update=103,parent_lifetime_decisions=2829668,parent_checkpointed_new_decisions=15789,
        parent_active_seconds=PARENT_ACTIVE_SECONDS,skipped_rejected_wave_seeds=list(range(3200012,3200018)),source_changes=sorted(CHANGED))


def prepare(package,parent):
    import torch,numpy as np
    from patrick_jane.network import PatrickJane
    from patrick_jane.a2_config import A2Experiment
    from patrick_jane.checkpoint import save,source_lock,freeze_sources
    from patrick_jane.progress import atomic_json
    from patrick_jane.a2_report import generate
    from jane.shared.runtime import PATRICK_PROFILE,fingerprint
    from portable_support import digest,safe_path
    package=Path(package).resolve();parent=Path(parent).resolve();folder=package/'results/investigation-a2-v1'
    if folder.exists():raise FileExistsError('Use a fresh run directory; historical runs are never rewritten')
    reg=json.loads((parent/'registration.json').read_text());status=json.loads((parent/'completion-status.json').read_text())
    allocations=json.loads((parent/'seed-allocation.json').read_text());history=json.loads((parent/'history.json').read_text())
    if status['state']!='failed' or abs(status['active_training_seconds']-PARENT_ACTIVE_SECONDS)>1e-6:raise ValueError('Unexpected parent duration/state')
    if [a['seeds'] for a in allocations]!=[list(range(3200006,3200012)),list(range(3200012,3200018))]:raise ValueError('Unexpected seed journal')
    if len(history)!=1 or history[0]['sha256']!=PARENT_SHA:raise ValueError('Unexpected parent training history')
    expected_profile=json.loads(json.dumps(A2Experiment(shared_engine_profile=PATRICK_PROFILE).to_dict()))
    if reg['workers']!=6 or reg['initial_duration_hours']!=3 or reg['experiment']!=expected_profile:raise ValueError('Parent run profile changed')
    torch.set_num_threads(1);policy=PatrickJane();optimizer=torch.optim.Adam(policy.parameters(),lr=.0003);rng=np.random.default_rng(17)
    counters,bridge=import_parent(package/'continuation',policy,optimizer,rng)
    if reg['sources']!=torch.load(package/'continuation/update-000103.pt',map_location='cpu',weights_only=True)['sources']:raise ValueError('Parent registration/source mismatch')
    folder.mkdir(parents=True);attempt=folder/'attempts/attempt-0001';attempt.mkdir(parents=True)
    old_checkpoint=safe_path(folder,history[0]['checkpoint']);old_checkpoint.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(package/'continuation/update-000103.pt',old_checkpoint)
    history[0]['source_segment']='shared-repair-v1';history[0]['source_revision']=reg['source_version']['revision']
    checkpoint=folder/'checkpoints/attempt-0001-origin.pt'
    sha=save(checkpoint,policy,optimizer,A2Experiment(shared_engine_profile=PATRICK_PROFILE),counters,rng=rng,metadata=dict(explicit_source_bridge=bridge))
    pointer=dict(checkpoint=checkpoint.relative_to(folder).as_posix(),sha256=sha,counters=counters)
    transition=dict(from_revision=reg['source_version']['revision'],to_revision=json.loads((package/'SOURCE_VERSION.json').read_text())['revision'],
        at_update=103,checkpointed_decisions=15789,parent_active_seconds=PARENT_ACTIVE_SECONDS,bridge=bridge)
    reg.update(sources=source_lock(),source_snapshot=freeze_sources(folder),source_version=json.loads((package/'SOURCE_VERSION.json').read_text()),
        first_training_seed=3200018,shared_engine=fingerprint(),source_transitions=[transition],continuation_provenance=bridge,
        original_import_provenance=reg['import_provenance'])
    reg['import_provenance']={**reg['import_provenance'],'continuation_checkpoint':bridge}
    for name,value in [('registration.json',reg),('history.json',history),('seed-allocation.json',allocations),('latest.json',pointer),('origin.json',pointer),
        ('duration-budget.json',json.loads((parent/'duration-budget.json').read_text())),
        ('active-time.json',dict(seconds=PARENT_ACTIVE_SECONDS,target_seconds=10800.,in_flight=False)),
        ('completion-status.json',dict(state='prepared',success=False)),('attempts/attempt-0001/attempt.json',dict(state='prepared-source-bridge',neural_updates=0))]:atomic_json(folder/name,value)
    generate(folder,'prepared',active_seconds=PARENT_ACTIVE_SECONDS,plots=True)
    print(json.dumps(dict(prepared=str(folder),bridge=bridge,active_seconds_carried=PARENT_ACTIVE_SECONDS,target_total_seconds=10800.,next_training_seed=3200018),indent=2))
    return folder


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--package',required=True);parser.add_argument('--parent-run',required=True);a=parser.parse_args()
    prepare(a.package,a.parent_run)
