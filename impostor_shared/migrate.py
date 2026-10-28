"""Explicit inference-only export: run with the ORIGINAL runtime and checkout.

The unchanged V4.1 strict loader must succeed before any tensor is exported.
This does not claim that the original checkpoint passed the new runtime's lock.
"""
import argparse,hashlib,json,platform,subprocess,sys
from pathlib import Path

ORIGINAL_SHA='69965afd8fab28e6ca5bdda46d9f8efdd8d6036ce57a05b86ff89fe22cf17ca9'
SCHEMA='impostor-shared-inference-transfer-v1'

def export(source_root,output):
    source_root=Path(source_root).resolve();output=Path(output).resolve()
    sys.path.insert(0,str(source_root))
    import numpy as np
    import torch
    from impostor_options_v41.learning import load
    from impostor_research.frozen import FrozenModels,weight_hash
    from impostor_research.features import KINDS,GLOBAL_COUNT,PLAYER_COUNT,MAX_ACTIONS
    from impostor_research.strategic_v2 import FAMILIES,EXTRA_COUNT,ACTION_COUNT_V2
    checkpoint=source_root/'artifacts/impostor-options-v41/release/policy.pt'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=ORIGINAL_SHA:raise ValueError('Original checkpoint changed')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    models=FrozenModels();policy,metadata=load(checkpoint,models);models.assert_unchanged()
    if sum(p.numel() for p in policy.parameters())!=75075:raise ValueError('Unexpected architecture')
    output.mkdir(parents=True,exist_ok=True)
    target=output/'policy-transfer.pt'
    record=dict(schema=SCHEMA,source_checkpoint_sha256=ORIGINAL_SHA,source_policy_hash=weight_hash(policy),
        source_schema=metadata['schema'],seed=metadata['seed'],decisions=metadata['decisions'],
        source_runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__)),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source_root,text=True).strip(),
        target_handoff='c3320583f7d562bc69ea527580a91ddc5db54f18',parameters=75075,
        kinds=list(KINDS),families=list(FAMILIES),optimizer_exported=False,training=False,
        original_strict_loader_passed=True,original_weights_unchanged=True,
        caveat='Explicit runtime/interface transfer; requires synthetic inference parity and shared-environment tests.')
    state=dict(record=record,policy={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()})
    with target.open('xb') as stream:torch.save(state,stream)
    rng=np.random.default_rng(20261028);packets={};reference=[]
    for n in range(3):
        count=5+n*4
        actions=np.zeros((MAX_ACTIONS,ACTION_COUNT_V2),np.float32)
        for i in range(count):actions[i,i%len(KINDS)]=1.;actions[i,len(KINDS):]=rng.uniform(-1,1,ACTION_COUNT_V2-len(KINDS))
        packet=dict(global_=rng.uniform(-1,1,GLOBAL_COUNT).astype(np.float32),
                    extra=rng.uniform(-1,1,EXTRA_COUNT).astype(np.float32),
                    players=rng.uniform(0,1,(4,PLAYER_COUNT)).astype(np.float32),actions=actions,
                    targets=np.asarray([i%5-1 for i in range(MAX_ACTIONS)],np.int64),
                    mask=np.asarray([i<count for i in range(MAX_ACTIONS)],np.int8))
        packets.update({f'{n}_{k}':v for k,v in packet.items()})
        data={k:torch.as_tensor(v).unsqueeze(0) for k,v in packet.items()}
        with torch.no_grad():distribution,value,families,_,_=policy.evaluate(data)
        index,logp,estimate=policy.act(packet)
        reference.append(dict(index=index,logp=logp,value=estimate,
                              probabilities=distribution.probs[0,:count].tolist(),families=families.probs[0].tolist()))
    fixture=output/'synthetic-parity.npz'
    with fixture.open('xb') as stream:np.savez_compressed(stream,**packets)
    record.update(export_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  synthetic_fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                  synthetic_reference=reference,synthetic_data_only=True)
    receipt=output/'migration.json'
    with receipt.open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
    print(json.dumps({k:v for k,v in record.items() if k!='synthetic_reference'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();export(a.source_root,a.output)
