"""Engineering-only process isolation, trajectory credit and exact continuation."""
from copy import deepcopy
from time import perf_counter
import numpy as np
import pytest
import torch
from crewmate_v2.config import LobbySettings
from crewmate_v2.learning.policy import V2Policy
from crewmate_v2.learning.train import state_hash
from crewmate_v2.collection.bank import EnvironmentBank,signature
from crewmate_v2.collection.collector import Collector
from crewmate_v2.collection.config import context,validate_approval,lock,proposal
from crewmate_v2.collection.ppo import advantages,update,save,resume
from crewmate_v2.learning.provenance import digest


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def policy():
    torch.manual_seed(17)
    return V2Policy()


def collector(workers=1):
    return Collector(workers=workers,streams=2,partition='fixtures',deadline=perf_counter()+90,
                     lobby=LobbySettings(timeout=1.2))


def test_disjoint_seed_streams_and_reserved_partitions_refused():
    values = [context('engineering',s,e,8) for e in range(10) for s in range(8)]
    assert len({v['seed'] for v in values}) == 80
    assert [v['rotation'] for v in values[:3]] == [0,1,2]
    with pytest.raises(ValueError):context('final_id',0,0,8)
    with pytest.raises(ValueError):context('engineering',0,10000,8)


def test_private_state_and_invalid_action_mask():
    bank = EnvironmentBank([0,1],2,'shaped','fixtures',perf_counter()+60,LobbySettings(timeout=1.2))
    a,b = bank.envs.values()
    assert a.memory is not b.memory and a.public_memory is not b.public_memory
    assert a.match.ledger is not b.match.ledger and a.match.controllers is not b.match.controllers
    before = bank.snapshot()[1]
    with pytest.raises(ValueError):bank.step({0:96})
    action = int(np.flatnonzero(bank.packets[0]['mask'])[0])
    bank.step({0:action})
    assert bank.snapshot()[1] == before


def test_gae_does_not_cross_streams_and_bootstraps_partial_episodes():
    from phase5_training import time_aware_gae
    r = dict(actions=[0]*4,rewards=[1.,2.,100.,200.],values=[0.]*4,next_values=[3.,4.,5.,6.],
             discounts=[.9]*4,dones=[False]*4,durations=[.6]*4,segments=[(0,2),(2,4)])
    a,returns = advantages(r)
    expected = time_aware_gae(*[np.asarray(r[k][:2]) for k in ('rewards','values','next_values','discounts','dones','durations')])[0]
    assert np.array_equal(a[:2],expected)
    assert a[1] == pytest.approx(2+.9*4)
    changed = deepcopy(r);changed['rewards'][2:] = [-999.,-999.]
    assert np.array_equal(advantages(changed)[0][:2],a[:2])
    r['segments'] = [(1,4)]
    with pytest.raises(ValueError):advantages(r)


def test_single_trajectory_update_matches_frozen_ppo():
    from phase5_options_training import ppo_update
    with collector() as c:
        a = policy();r=c.collect(a,8)
    r['segments'] = [(0,8)]
    b=deepcopy(a)
    update(a,torch.optim.Adam(a.parameters(),lr=.0003),r,np.random.default_rng(17))
    ppo_update(b,torch.optim.Adam(b.parameters(),lr=.0003),r,.02,np.random.default_rng(17))
    assert state_hash(a)==state_hash(b)


def test_two_spawn_workers_match_inline_packets_rewards_and_terminal_flags():
    records=[]
    for workers in (1,2):
        with collector(workers) as c:
            p=policy();r=c.collect(p,8)
            assert len(r['episodes'])>=2
            assert len(r['actions'])==8 and len(r['segments'])==2
            assert all(packet['mask'][action] for packet,action in zip(r['packets'],r['actions']))
            assert all(row['info']['episode']['reason']=='TIMEOUT' for row in r['transition_audits'] if row['done'])
            records.append(([t.hexdigest() for t in c.traces],r['rewards'],r['dones'],c.snapshot()['environments']))
    assert records[0]==records[1]


def test_checkpoint_resume_exact_next_rollout_and_optimizer(tmp_path):
    with collector() as c:
        p=policy();o=torch.optim.Adam(p.parameters(),lr=.0003);rng=np.random.default_rng(17)
        first=c.collect(p,8);update(p,o,first,rng)
        path=tmp_path/'checkpoint.pt';save(path,p,o,rng,c,8,evidence={'note':'preserved'})
        with pytest.raises(FileExistsError):save(path,p,o,rng,c,8)
        next_a=c.collect(p,8);update(p,o,next_a,rng);final_a=state_hash(p)
    with collector() as c:
        p,o,rng,n,evidence=resume(path,lambda weights:torch.optim.Adam(weights,lr=.0003),c)
        assert n==8 and evidence=={'note':'preserved'}
        next_b=c.collect(p,8);update(p,o,next_b,rng)
        assert next_a['actions']==next_b['actions'] and next_a['rewards']==next_b['rewards']
        assert state_hash(p)==final_a


def test_live_prefix_resume_reconstructs_private_memory():
    with Collector(workers=1,streams=2,partition='fixtures',deadline=perf_counter()+90) as c:
        p=policy();c.collect(p,2);snapshot=c.snapshot();expected={s:signature(v) for s,v in c.packets.items()}
        c.collect(p,2);c.restore(snapshot)
        assert {s:signature(v) for s,v in c.packets.items()}==expected
        a=c.collect(p,2);c.restore(snapshot);b=c.collect(p,2)
        assert a['actions']==b['actions'] and a['rewards']==b['rewards']


def test_original_reward_accounting_and_optimized_bootstrap_cache_equivalence():
    records=[]
    for original in (True,False):
        with collector() as c:
            r=c.collect(policy(),8,batched=not original,original_inference=original)
            records.append(r)
    assert records[0]['actions']==records[1]['actions']
    assert records[0]['rewards']==records[1]['rewards']
    assert np.allclose(records[0]['next_values'],records[1]['next_values'],atol=1e-7)
    for row in records[0]['transition_audits']:
        assert row['info']['undiscounted_reward']==pytest.approx(sum(row['info']['reward_components'].values()))


def test_collection_pilot_needs_new_explicit_approval_before_outputs(tmp_path):
    from crewmate_v2.collection.train import pilot,run_one
    with pytest.raises(PermissionError):pilot(tmp_path/'pilot',1,{}, {}, {})
    with pytest.raises(PermissionError):run_one(tmp_path/'run','original',1,perf_counter()+10,{})
    assert not (tmp_path/'pilot').exists() and not (tmp_path/'run').exists()
    record=dict(approved=True,execution_sha256=digest(proposal(1)),runtime_sha256=digest(lock()))
    validate_approval(record,1)
    with pytest.raises(PermissionError):validate_approval(record,2)
