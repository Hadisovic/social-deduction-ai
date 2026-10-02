import json
import numpy as np
import pytest
import torch
from belief.features import FEATURE_NAMES, probabilities
from belief.metrics import fit_temperature, metrics, match_weights
from belief.model import CandidateNet, BeliefModel, batch_logits
from phase4_data import SPLITS, QUICK_SPLITS, generate_match, study_match


@pytest.mark.parametrize('architecture',['linear','set'])
def test_model_order_equivariance_and_equal_evidence(architecture):
    torch.manual_seed(19)
    model = CandidateNet(architecture).eval()
    x = torch.randn(8,4,len(FEATURE_NAMES))
    perm = [2,0,3,1]
    assert torch.allclose(model(x[:,perm]),model(x)[:,perm],atol=1e-6)
    same = torch.zeros(1,4,len(FEATURE_NAMES))
    assert torch.equal(torch.softmax(model(same),-1),torch.full((1,4),.25))


def test_checkpoint_roundtrip_and_schema_mismatch(tmp_path):
    model = BeliefModel(CandidateNet(),1.5)
    path = tmp_path/'model.pt'; model.save(path)
    restored = BeliefModel.load(path)
    x = np.zeros((3,4,len(FEATURE_NAMES)),np.float32)
    assert np.array_equal(batch_logits(model,x),batch_logits(restored,x))
    assert restored.temperature == 1.5
    state = torch.load(path,weights_only=True); state['feature_names'].reverse(); torch.save(state,path)
    with pytest.raises(ValueError,match='schema'):
        BeliefModel.load(path)
    with pytest.raises(FileNotFoundError,match='python train_phase4.py'):
        BeliefModel.load(tmp_path/'missing.pt')


def test_scores_and_group_weighting():
    p = np.full((7,4),.25); y = np.array([0,1,2,3,0,1,2]); groups = np.array([1,1,1,1,1,1,2])
    result = metrics(p,y,groups)
    assert result['accuracy'] == pytest.approx(.25)
    assert result['nll'] == pytest.approx(np.log(4))
    assert result['brier'] == pytest.approx(.75)
    assert result['ece'] == pytest.approx(0)
    assert match_weights(groups)[-1] == pytest.approx(.5)
    logits = np.array([[10.,0,0,0]]*8); labels = np.array([0,0,1,1,2,2,3,3])
    temperature = fit_temperature(logits,labels,np.arange(8))
    assert temperature > 1
    assert metrics(probabilities(logits,temperature),labels,np.arange(8))['nll'] < metrics(probabilities(logits),labels,np.arange(8))['nll']


def test_registered_split_disjoint_and_heldout_family():
    sets = [set(range(a,a+n)) for a,n in [*SPLITS.values(),*QUICK_SPLITS.values()]]
    for i,a in enumerate(sets):
        assert all(not a&b for b in sets[i+1:])
    for heldout in (False,True):
        match = study_match(5,heldout)
        allowed = {'patient'} if heldout else {'hunter','self_report'}
        assert {c.style.impostor for c in match.controllers.values()} <= allowed


def test_generation_deterministic_causal_and_aligned():
    a,am = generate_match(920001); b,bm = generate_match(920001)
    assert a.keys() == b.keys()
    for key in a:
        assert np.array_equal(a[key],b[key]),key
    assert am['trajectory_hash'] == bm['trajectory_hash']
    assert len(a['y']) > 4 and a['tick'].min() == 0
    assert 'finished' not in a['stage']
    assert (a['tick']*.2 < am['result']['simulation_time']).all()
    assert np.all(a['candidates'][np.arange(len(a['y'])),a['y']] == am['impostor_id'])
    assert am['impostor_id'] not in a['actor']
