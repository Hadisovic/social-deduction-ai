"""Approved trainer preflight using disposable development episodes, never holdouts."""
from copy import deepcopy
import json
from time import time
import numpy as np
import pytest
import torch
from impostor_research.frozen import FrozenModels,weight_hash
from impostor_research.learning import (LearnerSession,PilotConfig,gae,load_inference,
    pilot_manifest,verify_record,exclusive_worker,packet_hash,ppo_update)
from impostor_research.env import ImpostorEnv


@pytest.fixture(scope='module')
def models():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    return FrozenModels()


def test_seconds_gae_discount_and_terminal_boundaries():
    a,returns=gae([1.,2.],[0.,0.],[0.,99.],[1.,1.],[False,True],[.6,120.])
    assert a[1]==2 and a[0]==pytest.approx(1+.99**.6*2)
    b,_=gae([1.,2.],[0.,0.],[0.,99.],[1.,1.],[False,True],[120.,.6])
    assert b[0]==pytest.approx(1+.99**120*2)
    np.testing.assert_allclose(a,returns)
    c,_=gae([1.,2.],[0.,0.],[99.,99.],[1.,1.],[True,True],[.6,.6])
    np.testing.assert_array_equal(c,[1.,2.])
    with pytest.raises(ValueError):gae([1.],[0.],[0.],[1.],[False],[0.])


def test_pilot_config_is_fixed():
    for kwargs in (dict(decisions=8192),dict(rollout=512),dict(learning_rate=.001),dict(lambda_per_second=.95)):
        with pytest.raises(ValueError):PilotConfig(**kwargs)


def test_actual_update_only_changes_impostor_and_resume_is_exact(models,tmp_path):
    session=LearnerSession(71,models=models,development=True)
    frozen=models.hashes();before=weight_hash(session.policy)
    row=session.update(8)
    assert row['gradient_norm']>0 and np.isfinite(row['loss'])
    assert weight_hash(session.policy)!=before and models.hashes()==frozen
    checkpoint=tmp_path/'resume.pt';session.save(checkpoint)
    with pytest.raises(FileExistsError):session.save(checkpoint)
    restored=LearnerSession.restore(checkpoint,models=models)
    assert packet_hash(session.packet)==packet_hash(restored.packet)
    assert session.env.memory.owner==restored.env.memory.owner
    assert session.env.memory.events==restored.env.memory.events
    assert session.env.public_memory.events==restored.env.public_memory.events
    assert session.env.match.trace.hexdigest()==restored.env.match.trace.hexdigest()
    # Next private RNG action must match without consuming either live stream.
    ga=torch.Generator();ga.set_state(session.generator.get_state())
    gb=torch.Generator();gb.set_state(restored.generator.get_state())
    assert session.policy.act(session.packet,generator=ga,deterministic=False)==restored.policy.act(restored.packet,generator=gb,deterministic=False)
    a=session.update(8);b=restored.update(8)
    assert weight_hash(session.policy)==weight_hash(restored.policy)
    assert session.episode_actions==restored.episode_actions
    assert session.action_counts==restored.action_counts
    for key in ('loss','entropy','value_loss','approximate_kl'):assert a[key]==b[key]
    assert models.hashes()==frozen;models.assert_unchanged()
    inference=tmp_path/'inference.pt';session.save(inference,resume=False)
    policy,payload=load_inference(inference,models=models)
    assert policy.act(session.packet)==session.policy.act(session.packet)
    assert payload['development'] is True


def test_optimizer_rejects_frozen_or_foreign_parameters(models):
    session=LearnerSession(83,models=models,development=True)
    rollout=session.collect(2)
    bad=torch.optim.Adam([*session.policy.parameters(),*models.policy.parameters()])
    with pytest.raises(ValueError,match='exactly impostor'):ppo_update(session.policy,bad,rollout,session.rng,.02)
    models.assert_unchanged()


def test_provenance_deadline_and_schema_fail_closed(models,tmp_path):
    record=pilot_manifest(ImpostorEnv(models=models))
    verify_record(record,models=models)
    with pytest.raises(TimeoutError):verify_record(record,deadline=time()-1)
    altered=deepcopy(record);altered['sources']['impostor_research/features.py']='wrong'
    with pytest.raises(ValueError,match='source'):verify_record(altered)
    altered=deepcopy(record);altered['config']['decisions']=8192
    with pytest.raises(ValueError,match='config'):verify_record(altered)
    altered=deepcopy(record);altered['lobby']['kill_cooldown']=12
    with pytest.raises(ValueError,match='condition'):verify_record(altered)
    altered=deepcopy(record);altered['frozen_checkpoints']={}
    with pytest.raises(ValueError,match='registration'):verify_record(altered)
    path=tmp_path/'wrong.pt';torch.save({'schema':'old'},path)
    with pytest.raises(ValueError,match='schema'):load_inference(path,models=models)


def test_checkpoint_replay_tampering_rejected(models,tmp_path):
    session=LearnerSession(71,models=models,development=True);session.collect(2)
    path=tmp_path/'resume.pt';session.save(path)
    data=torch.load(path,weights_only=True);data['packet_hash']='tampered'
    altered=tmp_path/'tampered.pt';torch.save(data,altered)
    with pytest.raises(ValueError,match='replay'):LearnerSession.restore(altered,models=models)


def test_os_lock_prevents_duplicate_worker(tmp_path):
    with exclusive_worker(tmp_path):
        with pytest.raises((OSError,BlockingIOError)):
            with exclusive_worker(tmp_path):pass
