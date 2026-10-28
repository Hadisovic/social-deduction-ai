from dataclasses import replace
from pathlib import Path
from time import perf_counter
import json
import pytest
import torch
from crewmate_v2.fourcrew.scenarios import ScenarioMatch
from crewmate_v2.fourcrew.policy import FourCrewPolicy
from crewmate_v2.fourcrew.config import Rewards
from crewmate_v2.config import LobbySettings
from crewmate_v2.learning.train import state_hash
from crewmate_v2.phase21.scenarios import clone_prefix,social_fixture,witness_fixture,WITNESS_CASES
from crewmate_v2.phase21.runtime import ExperimentalMatch
from crewmate_v2.phase21.config import Features
from crewmate_v2.phase21.opponents import CoherentMixture
from crewmate_v2.phase2.opponents import MixtureController
from social_deduction.actor import Phase
from social_deduction.phase3_api import IntentKind,WitnessedElimination,intent_allowed


@pytest.fixture(autouse=True)
def threads():torch.set_num_threads(1)


def test_disabled_bridge_preserves_actual_actions_events_and_private_clones():
    torch.manual_seed(17);policy=FourCrewPolicy().eval().requires_grad_(False);before=state_hash(policy)
    base=ScenarioMatch(2760090,lobby=LobbySettings(timeout=10),rewards=Rewards(),deadline=perf_counter()+60);base.prepare_prefix()
    other=clone_prefix(base);bridge=ExperimentalMatch.adopt(other,Features(),deadline=perf_counter()+60)
    assert base.views[base.crew[0]].memory is not bridge.views[bridge.crew[0]].memory
    assert bridge.views[bridge.crew[0]].match.game is bridge.game
    _,a=base.collect(policy,deterministic=True);_,b=bridge.collect(policy,deterministic=True)
    assert base.game.event_log==bridge.game.event_log and a['actors']==b['actors']
    assert state_hash(policy)==before and all(not p.requires_grad for p in policy.parameters())
    assert all(set(row['packets'])=={'global','players','own_tasks_v2','actions','targets','task_targets','mask'} for rows in bridge.rows.values() for row in rows)


@pytest.mark.parametrize('case',WITNESS_CASES)
def test_legal_visibility_cases_use_actual_actor_witness_evidence(case):
    m,ids=witness_fixture(2760091,case);g=m.game;obs=g.observe(ids['witness'])
    witnessed=[e for e in obs.evidence if type(e.payload) is WitnessedElimination and e.payload.victim_id==ids['victim']]
    if case in ('immediate','outside_range','multiple_race'):assert witnessed
    if case in ('wall_obstructed','spectator_only','unknown_killer'):assert not witnessed
    reports=[a for a in obs.actions if a.kind is IntentKind.REPORT and a.target==ids['victim']]
    if case=='immediate':assert reports
    if case=='outside_range':assert not reports and any(a.kind is IntentKind.GO_TO_BODY for a in obs.actions)
    if case=='dead_witness':assert not obs.own.active and not obs.actions
    if case=='multiple_race':
        witnesses=[pid for pid in m.crew if g.players[pid].active and any(type(e.payload) is WitnessedElimination for e in g.observe(pid).evidence)]
        assert len(witnesses)==3
    assert g.metrics['illegal_actions']==0


def test_confidence_order_is_frozen_actual_turn_order_and_no_scores_enter_packets():
    base,_,_=social_fixture(2760092,'direct');m=ExperimentalMatch.adopt(base,Features(speaking_order='confidence'),stop_after_meeting=True)
    order=m.game.participants;entry=m.order_log[0]
    assert order==tuple(sorted(order,key=lambda pid:(-entry['private_scores'][pid],pid)))
    assert m.game.context.speaker_id in (order[0],None)
    m.freeze_order();assert len(m.order_log)==1 and m.game.participants==order
    for pid in m.crew:
        if m.game.players[pid].active:
            packet=m.views[pid]._packet();assert 'private_scores' not in packet
    assert entry['providers'][m.impostor]=='own_local_script_proxy'


def test_goal_retention_does_not_randomly_replace_a_moving_destination():
    base=ScenarioMatch(2760093,deadline=perf_counter()+60);obs=base.game.observe(base.impostor)
    old=MixtureController(7);new=CoherentMixture(7);old_moves=new_moves=0
    for i in range(50):
        current=replace(obs,tick=i*3,time_seconds=i*.6,visible_players=(),navigation_status='moving')
        a=old.decide(current);b=new.decide(current)
        old_moves+=a.kind in (IntentKind.GO_TO_ROOM,IntentKind.GO_TO_TASK)
        new_moves+=b.kind in (IntentKind.GO_TO_ROOM,IntentKind.GO_TO_TASK)
        assert intent_allowed(current,b)
    assert old_moves>1 and new_moves==0 and new.retained_goals>0


def test_live_guard_influence_order_and_priority_are_explicit_frozen_mechanisms():
    base,_,_=social_fixture(2760093,'direct');torch.manual_seed(17)
    policy=FourCrewPolicy().eval().requires_grad_(False);before=state_hash(policy)
    m=ExperimentalMatch.adopt(clone_prefix(base),Features(social_guard=True,influence=.1,speaking_order='confidence',witness_priority=True),
        deadline=perf_counter()+60,stop_after_meeting=True)
    _,summary=m.collect(policy,deterministic=True)
    assert state_hash(policy)==before and m.game.metrics['illegal_actions']==0
    assert m.decision_log and m.order_log and not m.adapter.queue
    assert all('raw_probabilities' in r and 'effective_probabilities' in r for r in m.decision_log)
    for row in m.decision_log:
        choice=row['options'][row['selected']]
        if choice['kind']=='claim_defend':assert row['adjusted_suspicion'][choice['target']]<=.30
    assert summary.get('complete_match',m.game.is_terminal) is m.game.is_terminal
