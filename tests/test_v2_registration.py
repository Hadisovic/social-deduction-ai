"""Readiness uses engineering seeds only; study and final partitions remain unopened."""
import inspect
from time import perf_counter
import pytest
import torch
from crewmate_v2.fourcrew.scenarios import ScenarioMatch
from crewmate_v2.fourcrew.registration import plan,authorize,stop_state,validation_contexts,training_context
from crewmate_v2.fourcrew.policy import FourCrewPolicy
from crewmate_v2.learning.train import state_hash
from crewmate_v2.learning.policy import check_packet
from crewmate_v2.fourcrew.config import Rewards
from social_deduction.phase3_api import IntentKind
from social_deduction.actor import Phase,Role


@pytest.fixture(autouse=True)
def threads():torch.set_num_threads(1)


def test_registration_disjoint_partitions_and_coverage_without_opening_study_seeds():
    p=plan();contexts=list(validation_contexts())
    assert len(contexts)==16 and len({c['seed'] for c in contexts})==8
    assert sum(c['context']=='meeting_crew' for c in contexts)==2
    assert sum(c['context']=='meeting_impostor' for c in contexts)==2
    assert sum(c['context']=='body' for c in contexts)==4
    assert {training_context(i)['seed'] for i in range(32)}.isdisjoint(c['seed'] for c in contexts)
    assert p['arms']['shaped_v2']==Rewards().to_dict()
    with pytest.raises(PermissionError):authorize({'approved':False})


@pytest.mark.parametrize('case',range(8))
def test_all_controlled_prefix_families_are_legal_and_create_private_history(case):
    context='body' if case<4 else 'meeting_crew' if case%2==0 else 'meeting_impostor'
    m=ScenarioMatch(2760050+case,context=context,case_index=case,deadline=perf_counter()+60)
    m.prepare_prefix()
    assert m.game.metrics['illegal_actions']==0 and m.decisions==0
    assert m.game.tick==(376 if case>=4 else 375)
    assert all(not x.frames and not x.components for x in m.ledgers.values())
    live=[pid for pid in m.crew if m.game.players[pid].active]
    assert len(live)==3 and len({id(m.views[pid].memory) for pid in live})==3
    for pid in live:check_packet(m.views[pid]._packet())
    if case<4:
        assert all(any(a.kind is IntentKind.REPORT for a in m.views[pid].observation.actions) for pid in live)
    else:
        report=next(e for e in m.game.event_log if e['kind']=='report')
        assert m.game.players[report['reporter_id']].role is (Role.CREWMATE if case%2==0 else Role.IMPOSTOR)
        assert m.game.phase is Phase.DISCUSSION and m.prefix['prefix_reports']==1
    assert 'spawn(' not in inspect.getsource(type(m))


def test_fresh_paired_initialization_and_frozen_reference():
    models=[]
    for _ in range(3):torch.manual_seed(17);models.append(FourCrewPolicy())
    assert len({state_hash(p) for p in models})==1
    models[2].requires_grad_(False)
    with torch.no_grad():next(models[0].parameters()).add_(.1)
    assert state_hash(models[1])==state_hash(models[2])!=state_hash(models[0])


def test_exact_common_stopping_and_overshoot_rules():
    assert stop_state({'A':2048,'B':2300},16)=='complete'
    assert stop_state({'A':2048,'B':2047},16)=='continue'
    assert stop_state({'A':2048,'B':2300},14)=='continue'
    assert stop_state({'A':3073,'B':2048},16)=='inadmissible_overshoot'
    assert stop_state({'A':2000,'B':2000},32)=='inconclusive_schedule_exhausted'


def test_full_runner_cannot_create_output_or_models_without_final_authorization(tmp_path):
    from crewmate_v2.fourcrew.experiment import run
    with pytest.raises(PermissionError):run(tmp_path/'study',tmp_path/'spectator',tmp_path/'replays',tmp_path/'status.json',{'approved':False})
    assert list(tmp_path.iterdir())==[]


def test_complete_match_rollout_combination_keeps_actor_segments():
    from crewmate_v2.fourcrew.experiment import combine
    from crewmate_v2.fourcrew.match import FIELDS
    x={k:[0,1] for k in FIELDS};x['segments']=[(0,1),(1,2)]
    merged=combine([(x,{}),(x,{})])
    assert merged['segments']==[(0,1),(1,2),(2,3),(3,4)]
