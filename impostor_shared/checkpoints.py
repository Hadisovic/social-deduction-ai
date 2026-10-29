"""Source-locked NEW training snapshots at complete-episode boundaries only."""
import hashlib,json,platform
from pathlib import Path
import numpy as np
import torch
from frozen_handoff.verify import verify,MANIFEST
from .model import FamilyPolicy
from .weights import tensor_hash

SCHEMA='impostor-shared-training-v1'

def contract():
    # Provenance checks must not consume the learner's global Torch RNG stream.
    with torch.random.fork_rng(devices=[]):verified=verify()
    return dict(publication_manifest_sha256=hashlib.sha256(MANIFEST.read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__)),
        adapter={p.name:hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))},
        frozen_checkpoints={k:v['sha256'] for k,v in verified['models'].items()},
        collection='complete-episodes-only',gamma_per_second=1.,lambda_per_second=.995,crew_accounting='eligible-v1')

def optimizer_for(policy,lr=3e-4):
    return torch.optim.Adam([dict(params=policy.actor_parameters(),name='actor'),
                             dict(params=policy.critic_parameters(),name='critic')],lr=lr,eps=1e-5)

def save(path,policy,optimizer,generator,*,record,next_seed,episode_complete):
    if not episode_complete:raise ValueError('Mid-episode snapshots are unsupported; do not invent resumable world state')
    if type(next_seed) is not int or not 3600000<=next_seed<=3609999:raise ValueError('Next seed must be in the new handoff TRAIN partition')
    parameters=[p for group in optimizer.param_groups for p in group['params']]
    if len(parameters)!=len(list(policy.parameters())) or {id(p) for p in parameters}!={id(p) for p in policy.parameters()}:
        raise ValueError('Optimizer must contain exactly the impostor parameters, never frozen crew/belief')
    if [g.get('name') for g in optimizer.param_groups]!=['actor','critic']:raise ValueError('Use the declared actor/critic optimizer groups')
    plain=json.loads(json.dumps(record,allow_nan=False))
    payload=dict(schema=SCHEMA,contract=contract(),record=plain,next_seed=next_seed,episode_complete=True,
        policy=policy.state_dict(),policy_hash=tensor_hash(policy),optimizer=optimizer.state_dict(),generator=generator.get_state())
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:torch.save(payload,stream)

def load(path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('schema')!=SCHEMA or payload.get('contract')!=contract() or not payload.get('episode_complete'):
        raise ValueError('Shared checkpoint schema/source/runtime mismatch; no implicit migration')
    policy=FamilyPolicy();policy.load_state_dict(payload['policy'],strict=True)
    if tensor_hash(policy)!=payload['policy_hash'] or any(not torch.isfinite(p).all() for p in policy.parameters()):
        raise ValueError('Checkpoint tensors changed or are nonfinite')
    optimizer=optimizer_for(policy);optimizer.load_state_dict(payload['optimizer'])
    generator=torch.Generator();generator.set_state(payload['generator'])
    return policy,optimizer,generator,payload
