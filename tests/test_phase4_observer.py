from dataclasses import replace
import numpy as np
import torch
from belief.model import CandidateNet, BeliefModel
from belief.features import encode
from belief.memory import ActorMemory
from phase3_runner import ScriptedMatch
from phase4_observer import MatchObserver
from social_deduction.actor import Ejection, MatchEnded, Phase, Provenance, Role
from social_deduction.phase3_api import EventView


def test_observer_does_not_change_scripted_trajectory_or_receive_outcome():
    torch.set_num_threads(1)
    first = ScriptedMatch(15)
    observer = MatchObserver(first,BeliefModel(CandidateNet()))
    while not first.game.is_terminal:
        observer.step(first)
    second = ScriptedMatch(15,planner=first.game.planner); second.run()
    assert first.trajectory_hash == second.trajectory_hash
    for memory in observer.memories.values():
        assert memory.phase is not Phase.FINISHED
        assert not any(type(e.payload) is MatchEnded for e in memory.events)
        assert memory.tick*.2 < first.game.time
    original = observer.focal; observer.cycle(); assert observer.focal != original


def test_ejection_does_not_mask_historical_role_candidate():
    match = ScriptedMatch(15)
    obs = match.game.observe('p0')
    memory = ActorMemory(); memory.update(obs)
    pid = memory.candidates[0]
    memory.update(replace(obs,tick=3,time_seconds=.6,
        evidence=(EventView(3,Provenance.PUBLIC,Ejection('m0',pid)),)))
    assert len(memory.candidates)==4 and pid in memory.candidates
    model = BeliefModel(CandidateNet())
    assert model.predict(memory).by_player[pid] > 0


def test_private_own_task_order_action_order_and_winner_do_not_enter_features():
    match = ScriptedMatch(15)
    obs = match.game.observe('p0')
    a,b = ActorMemory(),ActorMemory()
    a.update(obs)
    changed = replace(obs,own=replace(obs.own,tasks=tuple(reversed(obs.own.tasks))),
                      actions=tuple(reversed(obs.actions)),
                      evidence=obs.evidence+(EventView(0,Provenance.PUBLIC,MatchEnded(Role.IMPOSTOR)),))
    b.update(changed)
    assert np.array_equal(encode(a),encode(b))
    assert a.to_json()==b.to_json()
