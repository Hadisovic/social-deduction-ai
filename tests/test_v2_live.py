"""Local spectator connection tests; no final partitions or full pilot."""
from pathlib import Path
from time import perf_counter
import hashlib
import os
import numpy as np
import pytest
import torch
from crewmate_v2.config import LobbySettings
from crewmate_v2.learning.policy import V2Policy
from crewmate_v2.learning.train import state_hash
from crewmate_v2.learning.provenance import digest,runtime_lock
from crewmate_v2.collection.config import lock as base_lock
from crewmate_v2.collection.ppo import update
from crewmate_v2.live.collector import LiveCollector
from crewmate_v2.live.protocol import lock,validate,spectator_import


@pytest.fixture
def spectator():
    source=Path(os.environ.get('SPECTATOR_SOURCE',Path.cwd().parent/'_spectator_live')).resolve()
    if not (source/'spectator/recorder.py').exists():pytest.skip('Separate recovered spectator source required')
    spectator_import(source);return source


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)


def test_recording_is_pure_and_probabilities_are_actual_policy_outputs(spectator,tmp_path):
    rows=[]
    for enabled in (False,True):
        config=dict(spectator_source=str(spectator),replay_dir=str(tmp_path/'replays'),streams=[0,1],
                    session='unit-engineering',workers=1,source_version='test') if enabled else None
        with LiveCollector(workers=1,streams=2,partition='fixtures',lobby=LobbySettings(timeout=1.2),
                           recording=config,deadline=perf_counter()+60) as c:
            torch.manual_seed(17);p=V2Policy();o=torch.optim.Adam(p.parameters(),lr=.0003)
            r=c.collect(p,8);update(p,o,r,np.random.default_rng(17))
            rows.append((r['actions'],r['rewards'],r['dones'],[t.hexdigest() for t in c.traces],state_hash(p),c.snapshot()['environments']))
    assert rows[0]==rows[1]
    from spectator.replay_format import parse_replay
    files=list((tmp_path/'replays').rglob('*.jsonl'));assert files
    for f in files:
        data=parse_replay(f);assert data['complete'] and len(data['ticks'][0]['players'])==5 and data['neural']
        assert data['result']['reason']=='TIMEOUT'
        n=data['neural'][0];assert abs(sum(n['action_probabilities'].values())-1)<1e-6
        assert n['action_kind'] in n['legal_actions'] and n['selected_action_index']>=0


def test_observer_preserves_original_source_compatibility(spectator):
    assert digest(runtime_lock(LobbySettings()))=='e729b2ffb79710ce904b6943972fc669626606a63ec4ca2be77b28eeff81e95d'
    assert digest(base_lock())=='ecd49a56cb51863d8feac271786b76dfbd6ce3d2980f5afa67925f7dd7f2cf4a'


def test_no_pilot_before_explicit_verified_gui_and_proof(spectator,tmp_path):
    from crewmate_v2.live.train import pilot
    with pytest.raises(PermissionError):pilot(tmp_path/'pilot',spectator,tmp_path/'replays',tmp_path/'status',{}, {},tmp_path/'missing',{})
    assert not (tmp_path/'pilot').exists()


def test_extra_validation_audit_preserves_frozen_policy_behavior():
    from crewmate_v2.learning.evaluate import evaluate_game as old
    from crewmate_v2.live.evaluate import evaluate_game as observed
    torch.manual_seed(17);p=V2Policy()
    args=dict(seed=2750090,mode=1.,lobby=LobbySettings(timeout=1.2),deadline=perf_counter()+60)
    a=old(p,**args);b=observed(p,**args)
    assert {k:b[k] for k in a}==a
    assert b['action_counts'] and b['common_legacy_return']==pytest.approx(b['counterfactual_shaped_return'])
