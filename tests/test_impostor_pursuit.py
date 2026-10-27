"""New family balance, own-model warm start, exact resume and viewer compatibility."""
from copy import deepcopy
import pytest
import torch
from impostor_research.frozen import FrozenModels,weight_hash
from impostor_research.learning_v2 import Session as V2Session
from impostor_research.strategic_v2 import StrategicEnv,FAMILIES as V2_FAMILIES
from impostor_research.features import KINDS
from impostor_pursuit.policy import PursuitPolicy,FAMILIES
from impostor_pursuit.learning import Session,load,verify


@pytest.fixture(scope='module')
def models():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    return FrozenModels()


def parent_checkpoint(models,tmp_path):
    parent=V2Session(71,models=models,development=True);path=tmp_path/'v2.pt';parent.save(path,False)
    return parent,path


def test_pursuit_is_balanced_and_warm_start_preserves_learned_layers(models,tmp_path):
    parent,path=parent_checkpoint(models,tmp_path)
    policy=PursuitPolicy(71);policy.warm_start(parent.policy)
    for name,p in parent.policy.state_dict().items():
        if name not in ('family_embedding.weight','kind_family'):torch.testing.assert_close(p,policy.state_dict()[name],rtol=0,atol=0)
    torch.testing.assert_close(policy.family_embedding.weight[:12],parent.policy.family_embedding.weight,rtol=0,atol=0)
    torch.testing.assert_close(policy.family_embedding.weight[-1],parent.policy.family_embedding.weight[V2_FAMILIES.index('travel')],rtol=0,atol=0)
    packet=parent.packet
    with torch.no_grad():
        policy.candidate_score.weight.zero_();policy.candidate_score.bias.zero_()
        policy.family_score[-1].weight.zero_();policy.family_score[-1].bias.zero_()
    d,_,families,ids,_=policy.evaluate({k:torch.as_tensor(v).unsqueeze(0) for k,v in packet.items()})
    active=families.probs[0]>0
    torch.testing.assert_close(families.probs[0,active],torch.full_like(families.probs[0,active],1/int(active.sum())))
    for i,c in enumerate(parent.env.options):
        if c.kind=='go_to_location':assert ids[0,i]==FAMILIES.index('pursuit') and c.target is not None
        if c.kind in ('go_to_room','go_to_task'):assert ids[0,i]==FAMILIES.index('travel')
    assert not d.probs[0,~torch.as_tensor(packet['mask'],dtype=torch.bool)].any()
    assert sum(p.numel() for p in policy.parameters())==75091


def test_continuation_gradient_frozen_parent_and_exact_resume(models,tmp_path):
    parent,path=parent_checkpoint(models,tmp_path)
    frozen=models.hashes();parent_hash=weight_hash(parent.policy)
    a=Session(71,models=models,parent=path,development=True);before=weight_hash(a.policy)
    row=a.update(8)
    assert row['actor_gradient_norm']>0 and weight_hash(a.policy)!=before
    checkpoint=tmp_path/'resume.pt';a.save(checkpoint)
    b=Session.restore(checkpoint,models);a.update(8);b.update(8)
    assert weight_hash(a.policy)==weight_hash(b.policy) and a.episode_actions==b.episode_actions
    assert models.hashes()==frozen and weight_hash(parent.policy)==parent_hash
    release=tmp_path/'pursuit.pt';a.save(release,False);policy,_=load(release,models)
    assert policy.act(a.packet)==a.policy.act(a.packet)
    legacy,_=load(path,models);assert legacy.act(parent.packet)==parent.policy.act(parent.packet)
    changed=deepcopy(a.study_record);changed['network']['parameters']=1
    with pytest.raises(ValueError,match='architecture'):verify(changed)
    changed=deepcopy(a.study_record);changed['parent']['sha256']='wrong'
    with pytest.raises(ValueError,match='parent'):verify(changed)
    models.assert_unchanged()


def test_viewer_loads_pursuit_and_v2_checkpoints(models,tmp_path,monkeypatch):
    from run_impostor_ai import main
    parent,path=parent_checkpoint(models,tmp_path);a=Session(71,models=models,parent=path,development=True)
    pursuit=tmp_path/'pursuit.pt';a.save(pursuit,False)
    monkeypatch.setenv('SDL_VIDEODRIVER','dummy');monkeypatch.setenv('SDL_AUDIODRIVER','dummy')
    for index,checkpoint in enumerate((path,pursuit)):
        assert main(['--checkpoint',str(checkpoint),'--max-frames','2','--capture',str(tmp_path/f'capture-{index}')])==0
        assert (tmp_path/f'capture-{index}'/'impostor.png').is_file()
    models.assert_unchanged()
