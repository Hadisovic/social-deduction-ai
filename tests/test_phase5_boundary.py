"""Counterfactual actor inputs through strategic features, choices and inference."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from test_phase3_boundary import planner,scene,project,change_others
from social_deduction.actor import Point,Role
from social_deduction.truth import BodyTruth,TaskTruth
from belief.memory import ActorMemory
from belief.model import BeliefModel
from phase5_features import choices,encode
from phase5_policy import StrategicPolicy,tensors


@pytest.mark.parametrize('hidden',['roles','position','death','tasks','cooldown'])
def test_hidden_truth_cannot_change_policy_features_masks_or_distribution(scene,hidden):
    torch.set_num_threads(1)
    game,world,actor=scene
    own=next(p for p in world.players if p.identity.player_id==actor)
    others=[p for p in world.players if p.identity.player_id!=actor]
    if hidden=='roles':
        roles=[p.role for p in others];mapping=dict(zip([p.identity.player_id for p in others],roles[1:]+roles[:1]))
        changed=change_others(world,actor,lambda p:replace(p,role=mapping[p.identity.player_id]))
    elif hidden=='position':
        far=next(d.standing for d in game.map.destinations if d.room=='Navigation')
        changed=change_others(world,actor,lambda p:replace(p,position=Point(*map(float,far)),room='Navigation'))
    elif hidden=='death':
        victim=next(p for p in others if p.role is Role.CREWMATE)
        changed=change_others(world,actor,lambda p:replace(p,active=False) if p is victim else p)
        changed=replace(changed,bodies=(BodyTruth(victim.identity.player_id,victim.position,victim.room,19,'hidden'),))
    elif hidden=='tasks':
        changed=change_others(world,actor,lambda p:replace(p,tasks=(TaskTruth('private','hidden',.8),)))
    else:changed=change_others(world,actor,lambda p:replace(p,private_cooldown=300))
    a,b=project(scene,world),project(scene,changed)
    assert a==b
    model=BeliefModel.load();packets=[];options=[]
    for obs in (a,b):
        memory=ActorMemory();memory.update(obs)
        belief=model.predict(memory);options.append(choices(obs,memory));packets.append(encode(obs,memory,belief,options[-1]))
    assert options[0]==options[1]
    assert all(np.array_equal(packets[0][k],packets[1][k]) for k in packets[0])
    torch.manual_seed(17);policy=StrategicPolicy()
    da,va=policy(tensors(packets[0]));db,vb=policy(tensors(packets[1]))
    assert torch.equal(da.probs,db.probs) and torch.equal(va,vb)


def test_task_macro_physically_arrives_completes_and_rewards_only_once():
    from phase5_env import CrewmateStrategicEnv
    from phase3_engine import GameConfig
    torch.set_num_threads(1)
    env=CrewmateStrategicEnv(config=GameConfig(initial_kill_cooldown=1000,kill_cooldown=1000,timeout=120))
    env.reset(seed=600000)
    index=next(i for i,c in enumerate(env.options) if c.kind=='task')
    task=env.options[index].task_id
    env.step(index)
    for _ in range(200):
        if task in env._credited or env.finished:break
        env.step(0)
    assert task in env._credited and env.completed_tasks==1
    assert env.match.game.metrics['illegal_actions']==0
    for _ in range(4):
        if not env.finished:env.step(0)
    assert env.completed_tasks==1
