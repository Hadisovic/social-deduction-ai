"""Diagnostic-only reporter legality, attribution and shortcut tests."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import inspect
from pathlib import Path
from time import perf_counter
import pytest
from phase3_bots import BotStyle
from phase3_runner import canonical_bytes
from social_deduction.actor import Phase,Role
from social_deduction.phase3_api import Intent,IntentKind,intent_allowed
from crewmate_v2.config import LobbySettings
from tools.diagnose_crewmate_v2 import run_game as original_run
from tools.reporting_diversity_v2 import (
    DiverseReporter,ExposureTracker,STYLES,baseline_audit,finish_exposures,heuristic_summary,
    loso_summary,reporter_weights,run_game,
)
from tests.test_crewmate_v2_diagnostic import body_fixture
from among_us_map import get_map
from navigation_service import NavigationPlanner


@pytest.fixture(scope='module')
def planner():return NavigationPlanner(get_map(),cache_size=96)


@pytest.mark.parametrize('style',STYLES)
@pytest.mark.parametrize('role',('crewmate','impostor'))
def test_role_symmetric_reporting_variants_emit_only_legal_actions(planner,style,role):
    g,killer,crew=body_fixture(planner);p=crew if role=='crewmate' else killer
    obs=g.observe(p.player_id);bot=DiverseReporter(17,BotStyle('nearest','hunter','evidence'))
    bot.reporting_style=style
    action=bot.decide(obs)
    assert intent_allowed(obs,action)
    if style=='prompt':assert action.kind is IntentKind.REPORT
    elif style=='inspect':assert action.kind is IntentKind.WAIT
    else:assert action.kind not in (IntentKind.REPORT,IntentKind.GO_TO_BODY)


def test_inspection_delay_is_sight_based_and_eventually_reports(planner):
    g,_,crew=body_fixture(planner);obs=g.observe(crew.player_id)
    bot=DiverseReporter(17,BotStyle());bot.reporting_style='inspect'
    assert bot.decide(obs).kind is IntentKind.WAIT
    assert bot.decide(replace(obs,time_seconds=obs.time_seconds+6.,tick=obs.tick+30)).kind is IntentKind.REPORT


def test_reporting_style_draw_independent_of_role_and_base_style():
    for seed in range(20):
        a=DiverseReporter(seed,BotStyle('nearest','hunter','evidence'))
        b=DiverseReporter(seed,BotStyle('social','self_report','skeptical'))
        assert a.reporting_style==b.reporting_style
    source=inspect.getsource(DiverseReporter)
    for forbidden in ('Role.','event_log','killer_id','world','game.players','models'):
        assert forbidden not in source


def test_hidden_killer_metadata_cannot_change_reporting_from_identical_packet(planner):
    g,_,crew=body_fixture(planner);obs=g.observe(crew.player_id)
    for style in STYLES:
        a=DiverseReporter(43,BotStyle());b=DiverseReporter(43,BotStyle())
        a.reporting_style=b.reporting_style=style
        # Change hidden body attribution, without changing legitimate direct history.
        for body in g.bodies.values():body.killer_id='private-counterfactual'
        g.private_ghost_notes={'killer':'never passed to actor'};g._invalidate()
        other=g.observe(crew.player_id)
        assert canonical_bytes(obs)==canonical_bytes(other)
        assert a.decide(obs)==b.decide(other)
        assert a is not b and a.first_body_sight is not b.first_body_sight


def test_kill_priority_and_meeting_decisions_remain_original(planner):
    from phase4_scripts import StudyController
    g,killer,crew=body_fixture(planner)
    killer.cooldown=0.;g._invalidate();obs=g.observe(killer.player_id)
    bot=DiverseReporter(43,BotStyle());bot.reporting_style='bypass'
    assert bot.decide(obs).kind is IntentKind.KILL
    context=replace(obs.context,phase=Phase.VOTING,meeting_id='fixture',participants=tuple(g.players))
    vote=replace(obs,context=context,actions=(Intent(IntentKind.VOTE,crew.player_id),Intent(IntentKind.SKIP)))
    assert DiverseReporter(17,BotStyle()).decide(vote)==StudyController(17,BotStyle()).decide(vote)


def test_exposure_range_latency_censoring_and_window_denominators(planner):
    g,_,crew=body_fixture(planner);obs=g.observe(crew.player_id)
    report=next(a for a in obs.actions if a.kind is IntentKind.REPORT)
    tracker=ExposureTracker()
    far=replace(obs,actions=tuple(a for a in obs.actions if a.kind is not IntentKind.REPORT))
    tracker.observe(far,Intent(IntentKind.WAIT))
    legal=replace(obs,time_seconds=obs.time_seconds+1.2,tick=obs.tick+6)
    tracker.observe(legal,Intent(IntentKind.WAIT))
    later=replace(obs,time_seconds=obs.time_seconds+2.4,tick=obs.tick+12)
    tracker.observe(later,report)
    pair=tracker.pairs[(crew.player_id,report.target)]
    assert pair['visible_windows']==3 and pair['legal_windows']==2
    assert len(tracker.pairs)==1 and tracker.legal_actor_windows[crew.player_id]==2
    assert pair['first_legal']-pair['first_visible']==pytest.approx(1.2)
    assert pair['submissions'][0]['time']-pair['first_legal']==pytest.approx(1.2)


def test_public_reporter_heuristics_known_self_exclusion_and_no_role_lookup():
    w=reporter_weights(('b','c','d'),'b')
    assert w['accuse']==dict(b=1.,c=0.,d=0.)
    assert w['avoid']==dict(b=0.,c=.5,d=.5)
    assert reporter_weights(('b','c','d'),'self')['accuse']==w['uniform']
    assert reporter_weights(('b',),'b')['avoid']==dict(b=1.)
    source=inspect.getsource(reporter_weights)
    assert 'roles' not in source and 'Role' not in source


def test_scenario_grouped_shortcut_crossvalidation():
    records=[]
    for seed in (1,2,3):
        for condition in ('A','B'):
            records.append(dict(seed=seed,condition=condition,scores=dict(accuse=1.,avoid=0.,uniform=.5),
                direct_witness=False,body_info=False,heard_claim=False))
    result=loso_summary(records)
    assert result['folds']==3 and result['scenario_balanced']==1.
    assert result['event_weighted']==1. and result['uniform_event_weighted']==.5
    assert set(result['held_out_direction'].values())=={'accuse'}


def test_new_meter_does_not_change_old_baseline_replay():
    lobby=LobbySettings(timeout=1.8)
    old,op,_=original_run(2630102,'A','baseline',perf_counter()+30,lobby)
    new,np,_,_,_=run_game(2630102,'A','baseline',perf_counter()+30,lobby)
    assert op==np
    for key in ('actor_sha256','public_extra_sha256','truth_sha256','counts'):
        assert old[key]==new[key]


def test_archived_opportunity_choice_audit_reconciles_private_evidence():
    row,ex=baseline_audit(Path('temp_v2_pretraining_run/private/baseline-2630005-A.json'))
    crew=ex['by_role']['crewmate']
    assert crew['legal_unique']==crew['submissions']==1 and crew['accepted']==0
    assert crew['unaccepted_submissions']==1 and crew['legal_to_submit']==[0.]
    assert len(ex['ballots'])==row['counts']['crewmate_vote_unique_opportunities']


def test_censoring_uses_earliest_death_not_a_later_body_report(planner):
    g,killer,crew=body_fixture(planner)
    obs=g.observe(crew.player_id);tracker=ExposureTracker();tracker.observe(obs,Intent(IntentKind.WAIT))
    victim=obs.visible_bodies[0].victim_id
    g._eliminate(killer,crew)
    other=next(p for p in g.players.values() if p.active and p.role is Role.CREWMATE)
    g.tick+=3;g._start_meeting(other,g.bodies[victim]);g._finish(None,'TIMEOUT')
    result=finish_exposures(g,tracker,None,'diverse')
    assert result['pairs'][0]['censor_reason']=='actor_absent'
    assert result['pairs'][0]['censor_time']==obs.time_seconds
    assert result['by_role']['crewmate']['censored_legal_followup']==[0.]
