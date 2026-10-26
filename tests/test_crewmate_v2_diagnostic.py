"""Measurement/actor-boundary fixtures, not research performance evaluations."""
from dataclasses import replace
from collections import Counter
import inspect
from time import perf_counter
import pytest
from among_us_map import get_map
from navigation_service import NavigationPlanner
from phase3_bots import BotStyle
from phase3_runner import canonical_bytes
from crewmate_v2.config import LobbySettings, MatchMixture
from crewmate_v2.engine import CrewmateV2Game
from social_deduction.actor import Phase, Role
from social_deduction.phase3_api import Intent, IntentKind, intent_allowed
from tools.diagnose_crewmate_v2 import (
    FIXTURE_SEEDS, OpportunityMeter, StaggeredReporter, offline_score, ratio, run_game, seed_guard, summarize,
)


@pytest.fixture(scope='module')
def planner():
    return NavigationPlanner(get_map(),cache_size=96)


def body_fixture(planner):
    g=CrewmateV2Game(FIXTURE_SEEDS[0],planner=planner)
    killer=next(p for p in g.players.values() if p.role is Role.IMPOSTOR)
    crew=[p for p in g.players.values() if p.role is Role.CREWMATE]
    killer.motion.spawn((-.7,-2.8));crew[0].motion.spawn((-1.4,-2.8))
    crew[1].motion.spawn((-.7,-3.0));killer.cooldown=0.;g._invalidate()
    g.step({killer.player_id:Intent(IntentKind.KILL,crew[0].player_id)})
    return g,killer,crew[1]


def test_registry_disjoint_and_undefined_ratio_not_zero():
    seed_guard()
    assert ratio(0,0) is None and ratio(1,2)==.5


def test_match_mode_no_initial_marker_and_independent_sampling(planner):
    a=CrewmateV2Game(FIXTURE_SEEDS[1],planner=planner,mixture=MatchMixture(0.))
    b=CrewmateV2Game(FIXTURE_SEEDS[1],planner=planner,mixture=MatchMixture(1.))
    assert not a.deduction_only and b.deduction_only
    for pid in a.players:
        assert canonical_bytes(a.observe_v2(pid))==canonical_bytes(b.observe_v2(pid))
    assert MatchMixture(.37).select(FIXTURE_SEEDS[1])==MatchMixture(.37).select(FIXTURE_SEEDS[1])


@pytest.mark.parametrize('role',['crewmate','impostor'])
def test_probe_can_report_legally_for_either_role_without_world_input(planner,role):
    g,killer,crew=body_fixture(planner)
    p=crew if role=='crewmate' else killer
    obs=g.observe(p.player_id)
    assert any(a.kind is IntentKind.REPORT for a in obs.actions)
    bot=StaggeredReporter(41,BotStyle('nearest','hunter','evidence'))
    bot.report_delay=0.
    action=bot.decide(obs)
    assert action.kind is IntentKind.REPORT and intent_allowed(obs,action)
    assert not any(name in bot.__dict__ for name in ('world','game','roles','meter','controllers'))
    g.step({p.player_id:action})
    assert g.metrics['reports']==1 and g.phase is Phase.DISCUSSION


def test_probe_delay_uses_only_personal_body_sight_and_role_independent_stream(planner):
    g,killer,crew=body_fixture(planner)
    obs=g.observe(crew.player_id)
    bot=StaggeredReporter(41,BotStyle())
    bot.report_delay=18.
    bot._observe(obs)
    assert bot._report(obs) is None
    later=replace(obs,time_seconds=obs.time_seconds+18.,tick=obs.tick+90)
    bot._observe(later)
    assert bot._report(later).kind is IntentKind.REPORT
    assert StaggeredReporter(41,BotStyle()).report_delay==StaggeredReporter(41,BotStyle('social','self_report','cautious')).report_delay
    source=inspect.getsource(StaggeredReporter)
    assert 'Role.' not in source and 'event_log' not in source


def test_opportunity_windows_do_not_inflate_unique_actor_body_denominator(planner):
    g,killer,crew=body_fixture(planner)
    obs=g.observe(crew.player_id)
    meter=OpportunityMeter()
    for _ in range(5):meter.observe(obs,Intent(IntentKind.WAIT))
    assert meter.counts['crewmate_report_legal_windows']==5
    assert len(meter.report_pairs['crewmate'])==1
    assert len(meter.body_pairs)==1
    assert meter.counts['actionable_windows']==5
    empty=replace(obs,actions=())
    meter.observe(empty,None)
    assert meter.counts['forced_no_actions']==1
    assert meter.counts['living_windows']==6


def test_vote_opportunity_is_one_actor_meeting_not_repeated_windows(planner):
    g,killer,crew=body_fixture(planner)
    obs=g.observe(crew.player_id)
    context=replace(obs.context,phase=Phase.VOTING,meeting_id='fixture',participants=tuple(g.players))
    ballot=replace(obs,context=context,actions=(Intent(IntentKind.VOTE,killer.player_id),Intent(IntentKind.SKIP)))
    meter=OpportunityMeter()
    for _ in range(4):meter.observe(ballot,Intent(IntentKind.SKIP))
    assert meter.vote_windows['crewmate']==4
    assert len(meter.vote_opportunities['crewmate'])==1


def test_short_fixture_replays_and_a_b_prefixes_exactly():
    lobby=LobbySettings(timeout=1.8)
    a,ap,_=run_game(FIXTURE_SEEDS[2],'A','baseline',perf_counter()+30,lobby)
    b,bp,_=run_game(FIXTURE_SEEDS[2],'B','baseline',perf_counter()+30,lobby)
    repeat,rp,_=run_game(FIXTURE_SEEDS[2],'A','baseline',perf_counter()+30,lobby)
    assert ap==bp==rp
    # Private configuration truth correctly records different mode selections.
    ignore={'wall_seconds','phase_wall_seconds','condition','truth_sha256'}
    assert {k:v for k,v in a.items() if k not in ignore}=={k:v for k,v in b.items() if k not in ignore}
    assert a['actor_sha256']==repeat['actor_sha256'] and a['truth_sha256']==repeat['truth_sha256']
    aggregate=summarize([a,b])
    assert aggregate['first_event_times']['first_kill']['censored']==2
    assert aggregate['ratios']['crew_cast_accuracy'] is None


def test_tick_deadline_stops_instead_of_silently_extending_budget():
    with pytest.raises(TimeoutError):
        run_game(FIXTURE_SEEDS[3],'A','baseline',perf_counter()-1)


def test_offline_scorer_tracks_accepted_report_and_ghost_completion(planner):
    g,killer,crew=body_fixture(planner)
    victim=next(p for p in g.players.values() if not p.active)
    # Controlled event-attribution fixture; never a policy input.
    for owner in (victim,crew):
        task=owner.tasks[0];task.stage=len(task.plan.stages)
        g._completed(owner,task,ghost=not owner.active)
    meter=OpportunityMeter();obs=g.observe(crew.player_id)
    report=next(a for a in obs.actions if a.kind is IntentKind.REPORT)
    meter.observe(obs,report);g.step({crew.player_id:report})
    g._finish(None,'TIMEOUT')
    scored=offline_score(g,meter,0.,Counter(),Counter(),0.,[])
    c=scored['counts']
    assert c['tasks_ghost']==c['tasks_living']==1
    assert c['crewmate_reports']==c['crewmate_report_submissions']==c['crewmate_report_unique_pairs']==1
    assert c['report_submissions_not_accepted']==0


def test_offline_scorer_distinguishes_explicit_and_deadline_skips(planner):
    g,killer,crew=body_fixture(planner)
    report=next(a for a in g.observe(crew.player_id).actions if a.kind is IntentKind.REPORT)
    g.step({crew.player_id:report});g.phase=Phase.VOTING;g._invalidate()
    meter=OpportunityMeter();meter.observe(g.observe(crew.player_id),Intent(IntentKind.SKIP))
    g._vote(crew.player_id,None);g._vote(killer.player_id,None)
    g._finish(None,'TIMEOUT')
    c=offline_score(g,meter,0.,Counter(),Counter(),0.,[])['counts']
    assert c['crewmate_explicit_skips']==1 and c['impostor_automatic_skips']==1
    assert c['crewmate_vote_unique_opportunities']==1
