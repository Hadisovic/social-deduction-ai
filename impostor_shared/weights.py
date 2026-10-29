"""Fail-closed explicit transfer loader, separate from historical strict loaders."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from .features import KINDS
from .model import FamilyPolicy,FAMILIES

ASSETS=Path(__file__).parent/'assets-v1'
SOURCE_SHA='69965afd8fab28e6ca5bdda46d9f8efdd8d6036ce57a05b86ff89fe22cf17ca9'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def tensor_hash(policy):
    h=hashlib.sha256()
    for key,value in sorted(policy.state_dict().items()):
        h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def load_transfer(*,trainable=False,assets=ASSETS):
    assets=Path(assets);receipt=json.loads((assets/'migration.json').read_text(encoding='utf-8'))
    path=assets/'policy-transfer.pt'
    if (receipt['source_checkpoint_sha256']!=SOURCE_SHA or digest(path)!=receipt['export_sha256']
        or not receipt['original_strict_loader_passed'] or receipt['optimizer_exported']):
        raise ValueError('Unverified transfer identity')
    payload=torch.load(path,map_location='cpu',weights_only=True);record=payload['record']
    if (record['schema']!='impostor-shared-inference-transfer-v1' or record['source_checkpoint_sha256']!=SOURCE_SHA
        or record['kinds']!=list(KINDS) or record['families']!=list(FAMILIES)):
        raise ValueError('Incompatible transfer schema/actions')
    policy=FamilyPolicy(record['seed']);policy.load_state_dict(payload['policy'],strict=True)
    if sum(p.numel() for p in policy.parameters())!=75075 or tensor_hash(policy)!=record['source_policy_hash']:
        raise ValueError('Transferred tensors differ')
    if any(not torch.isfinite(p).all() for p in policy.parameters()):raise ValueError('Nonfinite model')
    policy.eval().requires_grad_(trainable)
    return policy,receipt

def verify_parity(policy,receipt,assets=ASSETS):
    assets=Path(assets);path=assets/'synthetic-parity.npz'
    if digest(path)!=receipt['synthetic_fixture_sha256']:raise ValueError('Fixture bytes changed')
    maximum=0.
    with np.load(path,allow_pickle=False) as fixtures:
        for n,reference in enumerate(receipt['synthetic_reference']):
            packet={k:fixtures[f'{n}_{k}'] for k in ('global_','extra','players','actions','targets','mask')}
            with torch.no_grad():
                distribution,_,families,_,_=policy.evaluate({k:torch.as_tensor(v).unsqueeze(0) for k,v in packet.items()})
            index,logp,value=policy.act(packet)
            actual=distribution.probs[0,:len(reference['probabilities'])].numpy()
            maximum=max(maximum,float(np.max(np.abs(actual-np.asarray(reference['probabilities'])))))
            np.testing.assert_allclose(actual,reference['probabilities'],rtol=1e-5,atol=1e-6)
            np.testing.assert_allclose(families.probs[0].numpy(),reference['families'],rtol=1e-5,atol=1e-6)
            np.testing.assert_allclose([logp,value],[reference['logp'],reference['value']],rtol=1e-5,atol=1e-6)
            if index!=reference['index']:raise AssertionError('Greedy choice changed across runtime')
    return dict(cases=3,synthetic_only=True,maximum_candidate_probability_error=maximum,
                original_tensor_hash=tensor_hash(policy),no_optimizer_updates=True)
