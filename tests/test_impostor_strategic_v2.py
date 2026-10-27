"""Family probabilities, actor-only features, dense rewards and exact resume."""
from copy import deepcopy
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from impostor_research.frozen import FrozenModels,weight_hash
from impostor_research.strategic_v2 import StrategicEnv,FamilyPolicy,FAMILIES,family
from impostor_research.learning_v2 import Session,load,verify
from impostor_research.learning import packet_hash
from social_deduction.actor import Role
from social_deduction.phase3_api import IntentKind


@pytest.fixture(scope='module')
def models():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    return FrozenModels()


def packet_tensor(p):return {k:torch.as_tensor(v).unsqueeze(0) for k,v in p.items()}


def test_family_mass_not_destination_count_and_target_masks(models):
    env=StrategicEnv(models=models);packet,_=env.reset(seed=3500010);policy=FamilyPolicy()
    assert env.observation_space.contains(packet)
    # Zero logits should allocate equal probability to active families, regardless
    # of how many public travel destinations each contains.
    with torch.no_grad():
        policy.candidate_score.weight.zero_();policy.candidate_score.bias.zero_()
        policy.family_score[-1].weight.zero_();policy.family_score[-1].bias.zero_()
    d,_,f,ids,_=policy.evaluate(packet_tensor(packet))
    active=f.probs[0]>0
    torch.testing.assert_close(f.probs[0,active],torch.full_like(f.probs[0,active],1/int(active.sum())))
    for index in range(len(FAMILIES)):
        torch.testing.assert_close(d.probs[0,ids[0]==index].sum(),f.probs[0,index])
    assert not d.probs[0,~torch.as_tensor(packet['mask'],dtype=torch.bool)].any()
    assert policy.act(packet)==policy.act(packet)


def test_successful_kill_reward_is_dense_without_forced_kill(models):
    env=StrategicEnv(models=models);env.reset(seed=3500010);game=env.match.game
    own=game.players[env.match.focal];crew=[p for p in game.players.values() if p.role is Role.CREWMATE]
    for p,room in zip(crew[1:],('Reactor','Navigation','Communications')):
        p.motion.spawn(next(d.standing for d in game.map.destinations if d.room==room))
    own.motion.spawn((-.7,-2.8));crew[0].motion.spawn((-1.4,-2.8));own.cooldown=0.;game._invalidate();env._observe()
    index=next(i for i,c in enumerate(env.options) if c.kind=='kill')
    _,reward,done,_,info=env.step(index)
    assert not done and info['kill_bonus']==2. and reward>1.9
    assert env.match.game.metrics['eliminations']==1
    models.assert_unchanged()


def test_travel_option_preserves_ticks_and_visual_trace(models):
    results=[]
    for trace in (False,True):
        env=StrategicEnv(models=models,trace=trace);env.reset(seed=3500011)
        index=next(i for i,c in enumerate(env.options) if c.kind=='go_to_room' and c.intent.target=='Reactor')
        _,_,_,_,info=env.step(index)
        assert .6-1e-8<=info['elapsed_seconds']<=3.+1e-8
        results.append((packet_hash(env.packet),env.match.trace.hexdigest(),env.summary()))
        if trace:assert len(env.visual_frames)>=4
    assert results[0]==results[1]


def test_actor_features_ignore_unseen_positions(models):
    env=StrategicEnv(models=models);env.reset(seed=3500012);g=env.match.game
    own=g.players[env.match.focal];own.motion.spawn((-.7,-2.8))
    crew=[p for p in g.players.values() if p.role is Role.CREWMATE]
    for p in crew:p.motion.spawn(next(d.standing for d in g.map.destinations if d.room=='Reactor'))
    g._invalidate();env._observe();before={k:v.copy() for k,v in env.packet.items()}
    assert not env.envelope.actor.visible_players
    crew[0].motion.spawn(next(d.standing for d in g.map.destinations if d.room=='Navigation'));g._invalidate();env._observe()
    for k in before:np.testing.assert_array_equal(before[k],env.packet[k])


def test_real_split_gradient_update_and_exact_resume(models,tmp_path):
    a=Session(71,models=models,development=True);before=weight_hash(a.policy);frozen=models.hashes()
    row=a.update(8)
    assert row['actor_gradient_norm']>0 and row['critic_gradient_norm']>0
    assert not set(map(id,a.policy.actor_parameters()))&set(map(id,a.policy.critic_parameters()))
    assert weight_hash(a.policy)!=before and frozen==models.hashes()
    path=tmp_path/'resume.pt';a.save(path)
    with pytest.raises(FileExistsError):a.save(path)
    b=Session.restore(path,models)
    assert packet_hash(a.packet)==packet_hash(b.packet)
    a.update(8);b.update(8)
    assert weight_hash(a.policy)==weight_hash(b.policy)
    assert a.episode_actions==b.episode_actions
    assert frozen==models.hashes();models.assert_unchanged()
    inference=tmp_path/'policy.pt';a.save(inference,False);loaded,_=load(inference,models)
    assert loaded.act(a.packet)==a.policy.act(a.packet)
    bad=deepcopy(a.record);bad['shaping']['successful_kill']=100
    with pytest.raises(ValueError,match='Shaping'):verify(bad)


def test_viewer_cli_headless(models,tmp_path,monkeypatch):
    monkeypatch.setenv('SDL_VIDEODRIVER','dummy');monkeypatch.setenv('SDL_AUDIODRIVER','dummy')
    a=Session(71,models=models,development=True);path=tmp_path/'policy.pt';a.save(path,False)
    from run_impostor import main
    assert main(['--checkpoint',str(path),'--max-frames','2','--capture',str(tmp_path/'capture')])==0
    assert (tmp_path/'capture'/'impostor.png').is_file()
