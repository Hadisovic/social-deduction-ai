"""Whole-history paired-world audit through memory, features and actual inference."""
from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from belief.memory import ActorMemory
from belief.features import encode
from belief.model import BeliefModel, CandidateNet, DEFAULT_CHECKPOINT
from social_deduction.actor import Identity, Point, Provenance, Role
from social_deduction.phase3_api import EventView, WitnessedElimination
from social_deduction.truth import BodyTruth, InternalEvent, TaskTruth
from test_phase3_boundary import planner, scene, project, change_others


@pytest.fixture
def predictor():
    torch.set_num_threads(1)
    if DEFAULT_CHECKPOINT.exists():
        return BeliefModel.load()
    torch.manual_seed(17)
    return BeliefModel(CandidateNet())


@pytest.mark.parametrize('hidden', ['roles','positions','death','tasks','cooldowns','metadata','ordering','body_metadata'])
def test_matched_histories_cannot_change_memory_features_or_belief(scene,predictor,hidden):
    game,world,actor = scene
    own = next(p for p in world.players if p.identity.player_id == actor)
    others = [p for p in world.players if p.identity.player_id != actor]
    if hidden == 'roles':
        roles = [p.role for p in others]
        mapping = dict(zip([p.identity.player_id for p in others],roles[1:]+roles[:1]))
        changed = change_others(world,actor,lambda p:replace(p,role=mapping[p.identity.player_id]))
    elif hidden == 'positions':
        far = next(d.standing for d in game.map.destinations if d.room == 'Navigation')
        changed = change_others(world,actor,lambda p:replace(p,position=Point(*map(float,far)),room='Navigation'))
    elif hidden == 'death':
        victim = next(p for p in others if p.role is Role.CREWMATE)
        changed = change_others(world,actor,lambda p:replace(p,active=False) if p is victim else p)
        changed = replace(changed,bodies=(BodyTruth(victim.identity.player_id,victim.position,victim.room,19,'secret'),))
    elif hidden == 'tasks':
        changed = change_others(world,actor,lambda p:replace(p,tasks=(TaskTruth('private','hidden',.81),)))
    elif hidden == 'cooldowns':
        changed = change_others(world,actor,lambda p:replace(p,private_cooldown=888.))
    elif hidden == 'ordering':
        changed = replace(world,players=tuple(reversed(world.players)))
    elif hidden == 'body_metadata':
        body = BodyTruth(others[0].identity.player_id,own.position,own.room,4,others[1].identity.player_id)
        world = replace(world,bodies=(body,))
        changed = replace(world,bodies=(replace(body,death_tick=17,killer_id=others[2].identity.player_id),))
    else:
        changed = replace(world,internal_events=(InternalEvent(0,'secret','p0'),InternalEvent(999,'future','p1')))
    memories = [ActorMemory(),ActorMemory()]
    # Shared observed prefix, then unseen counterfactual changes across two updates.
    prefix = change_others(replace(world,tick=10,bodies=()),actor,lambda p:replace(
        p,position=own.position,room=own.room,interacting=p.identity.player_id==others[0].identity.player_id))
    for memory in memories:
        memory.update(project(scene,prefix))
    for tick in (20,30):
        a,b = project(scene,replace(world,tick=tick)),project(scene,replace(changed,tick=tick))
        assert a == b
        memories[0].update(a); memories[1].update(b)
        assert memories[0].to_json() == memories[1].to_json()
        assert np.array_equal(encode(memories[0]),encode(memories[1]))
        assert predictor.predict(memories[0]) == predictor.predict(memories[1])


def test_public_id_color_and_roster_permutation_equivariance(scene,predictor):
    _,world,actor = scene
    candidates = [p.identity.player_id for p in world.players if p.identity.player_id!=actor]
    witness = EventView(9,Provenance.DIRECT,WitnessedElimination(candidates[0],candidates[1],Point(0.,0.),'Reactor'))
    memory = ActorMemory(); memory.update(project(scene,direct_events=(witness,)))
    original = predictor.predict(memory)
    assert max(original.probabilities)-min(original.probabilities) > 1e-6
    data = memory.to_dict()
    mapping = {pid:f'identity-{7-i}' for i,pid in enumerate(i.player_id for i in memory.roster)}
    def rename(value):
        if type(value) is str:
            return mapping.get(value,value)
        if type(value) is list:
            return [rename(v) for v in value]
        if type(value) is dict:
            return {mapping.get(k,k):rename(v) for k,v in value.items()}
        return value
    renamed = rename(data)
    renamed['roster'].reverse()
    for i,row in enumerate(renamed['roster']):
        row['display_name'] = f'color-{100+i}'
    other = ActorMemory.from_json(json.dumps(renamed))
    prediction = predictor.predict(other)
    for pid,value in original.by_player.items():
        assert prediction.by_player[mapping[pid]] == pytest.approx(value,abs=1e-7)


def test_actor_modules_have_no_privileged_imports():
    import ast
    from pathlib import Path
    forbidden = ('truth','training','engine','runner','phase4_data','phase4_scripts','phase3_bots')
    for path in (Path(__file__).parents[1]/'belief').glob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = [node.module or ''] if isinstance(node,ast.ImportFrom) else (
                    [n.name for n in node.names] if isinstance(node,ast.Import) else [])
            assert not any(bad in name for name in names for bad in forbidden),path
