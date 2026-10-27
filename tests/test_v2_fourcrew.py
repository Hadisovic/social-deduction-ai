"""Four-policy-actor engineering tests. No full experiment or final partitions."""
from dataclasses import replace
from time import perf_counter
import numpy as np
import pytest
import torch
from social_deduction.actor import Phase,Role
from social_deduction.phase3_api import Intent,IntentKind
from crewmate_v2.config import LobbySettings,MatchMixture
from crewmate_v2.fourcrew.match import FourCrewMatch
from crewmate_v2.fourcrew.policy import FourCrewPolicy
from crewmate_v2.fourcrew.collector import collect_matches
from crewmate_v2.fourcrew.config import Rewards
from crewmate_v2.collection.ppo import advantages,update
from crewmate_v2.learning.train import state_hash


@pytest.fixture(autouse=True)
def threads():torch.set_num_threads(1);torch.use_deterministic_algorithms(True)


def match(seconds=1.2):return FourCrewMatch(2760000,lobby=LobbySettings(timeout=seconds),deadline=perf_counter()+60)


def test_four_distinct_actor_memories_and_same_input_shape():
    m=match();assert len(m.views)==4 and len(m.match.controllers)==1
    assert len({id(v.memory) for v in m.views.values()})==4
    assert len({id(v.public_memory) for v in m.views.values()})==4
    assert len({id(v.belief_model) for v in m.views.values()})==1
    for pid,v in m.views.items():
        packet=v._packet();assert packet['players'].shape==(4,29)
        assert v.memory.player_id==pid and v.observation.own.player_id==pid
        assert set(v.memory.candidates)==set(m.game.players)-{pid}
        with pytest.raises(ValueError):v.memory.update(m.game.observe(next(p for p in m.crew if p!=pid)))


def test_deterministic_sampling_is_per_actor_and_gae_segments_are_isolated():
    torch.manual_seed(17);p=FourCrewPolicy();runs=[]
    for _ in range(2):
        m=match();r,s=m.collect(p);runs.append((r,s))
        assert len(r['segments'])==4 and len(set(r['trajectory_keys']))==4
        for lo,hi in r['segments']:
            assert r['dones'][hi-1] and not any(r['dones'][lo:hi-1])
        a,_=advantages(r);r['rewards'][-1]+=100
        b,_=advantages(r);first_end=r['segments'][0][1]
        assert np.array_equal(a[:first_end],b[:first_end])
    assert runs[0][0]['actions']==runs[1][0]['actions']
    assert runs[0][1]==runs[1][1]


def test_no_dead_actor_policy_calls_or_memory_updates_and_ghost_credit_zero():
    m=match();pid=m.crew[0];view=m.views[pid];memory=view.memory.to_json()
    m.game.players[pid].status='dead';m.game._make_ghost(m.game.players[pid]);m.game._invalidate()
    torch.manual_seed(17);r,s=m.collect(FourCrewPolicy())
    assert view.memory.to_json()==memory and s['actors'][pid]['actions']=={}
    assert s['actors'][pid]['reward_components'].get('ghost_tasks',0)==0
    assert len(r['segments'])==3


def test_four_actor_ppo_log_probs_checkpoint_and_no_overwrite(tmp_path):
    torch.manual_seed(17);p=FourCrewPolicy();r=collect_matches(p,[2760000],lobby=LobbySettings(timeout=1.2))
    packets={k:torch.as_tensor(np.stack([row[k] for row in r['packets']])) for k in r['packets'][0]}
    with torch.no_grad():d,_=p(packets)
    assert np.allclose(d.log_prob(torch.tensor(r['actions'])).numpy(),r['logp'],atol=1e-6)
    old=state_hash(p);update(p,torch.optim.Adam(p.parameters(),lr=.0003),r,np.random.default_rng(17))
    assert state_hash(p)!=old
    path=tmp_path/'new.pt';p.save(path,{'engineering':True});q,_=FourCrewPolicy.load(path)
    assert state_hash(p)==state_hash(q)
    with pytest.raises(FileExistsError):p.save(path,{})
    torch.save({'schema':'crewmate-v2-learning-v1'},tmp_path/'old.pt')
    with pytest.raises(ValueError):FourCrewPolicy.load(tmp_path/'old.pt')


def test_legitimate_evidence_persists_across_meetings_and_new_match_resets():
    m=match(40.);pid=m.crew[0];v=m.views[pid];g=m.game
    before=set(v.memory.events)
    g._start_meeting(g.players[pid]);v.refresh();first=set(v.memory.events)
    assert before<=first
    g.phase=Phase.VOTING;g._invalidate()
    for p in g.participants:g._vote(p,None)
    g.step();v.refresh();assert first<=set(v.memory.events)
    g._start_meeting(g.players[m.crew[1]]);v.refresh();assert first<=set(v.memory.events)
    other=FourCrewMatch(2760001,lobby=LobbySettings(timeout=1.2))
    assert not any(type(e.payload).__name__=='Meeting' for x in other.views.values() for e in x.memory.events)


def test_only_development_seeds_and_zero_ghost_rewards():
    with pytest.raises(ValueError):FourCrewMatch(1250000)
    with pytest.raises(ValueError):Rewards(ghost_task=1.)


def test_two_workers_equal_inline_and_no_cross_match_advantages():
    torch.manual_seed(17);p=FourCrewPolicy();records=[]
    for workers in (1,2):
        r=collect_matches(p,[2760000,2760001],workers=workers,lobby=LobbySettings(timeout=1.2))
        records.append(r)
    for key in ('actions','logp','rewards','dones','durations','next_values','segments','trajectory_keys','episodes'):
        assert records[0][key]==records[1][key],key
    for a,b in zip(records[0]['packets'],records[1]['packets']):
        assert all(np.array_equal(a[k],b[k]) for k in a)


def test_real_four_agent_replay_capture_has_no_effect(tmp_path):
    from pathlib import Path
    import os
    from crewmate_v2.live.protocol import spectator_import
    source=Path(os.environ.get('SPECTATOR_SOURCE',Path.cwd().parent/'_spectator_fourcrew'))
    if not source.exists():pytest.skip('Separate new spectator source required')
    module=spectator_import(source)
    torch.manual_seed(17);p=FourCrewPolicy();records=[]
    for enabled in (False,True):
        m=match();rec=None
        if enabled:
            rec=module.MatchRecorder(m.match,tmp_path/'fourcrew.jsonl',controller_types={pid:'ppo' if pid in m.crew else 'scripted' for pid in m.game.players},
                metadata={'training_condition':'Four-crew engineering only'})
        records.append(m.collect(p,recorder=rec))
    a,b=records
    for key in ('actions','logp','values','rewards','next_values','discounts','dones','durations'):
        assert a[0][key]==b[0][key]
    assert a[1]==b[1]
    from spectator.replay_format import parse_replay
    replay=parse_replay(tmp_path/'fourcrew.jsonl');assert replay['complete']
    assert len({r['player_id'] for r in replay['neural']})==4
    assert all(r['suspicion_probabilities'] for r in replay['neural'])
    assert sum(v=='ppo' for v in replay['header']['controller_types'].values())==4
