"""Legal actor-only opponent diversity, not learned competence tests."""
from dataclasses import replace
import inspect
from random import Random
from time import perf_counter

import pytest
import torch
from crewmate_v2.fourcrew.match import FourCrewMatch
from crewmate_v2.phase2.opponents import MixtureController, PROFILES
from crewmate_v2.phase2.scenarios import evidence_fixture, discussion_messages, CASES
from social_deduction.actor import Phase, Provenance
from social_deduction.phase3_api import IntentKind, ClaimKind, intent_allowed, WitnessedElimination, StructuredClaim


@pytest.fixture(autouse=True)
def threads():torch.set_num_threads(1)


@pytest.fixture(scope='module')
def local_observation():
    m=FourCrewMatch(2760070,deadline=perf_counter()+60)
    return m.game.observe(m.impostor)


def test_local_only_surface_and_separate_rng_streams(local_observation):
    a=MixtureController(10);b=MixtureController(10)
    # Extra movement draws cannot change reporting, social or mode streams.
    for _ in range(15):a.streams['movement'].random()
    for name in ('report','social','kill','mode'):assert a.streams[name].getstate()==b.streams[name].getstate()
    source=inspect.getsource(__import__('crewmate_v2.phase2.opponents',fromlist=['']))
    for forbidden in ('game.players','game.observe','ActorMemory','belief','torch','engine','get_map'):
        assert forbidden not in source
    for weights in ({'unknown':1},{'social':float('nan')},{'social':0}):
        with pytest.raises(ValueError):MixtureController(10,weights)
    a=MixtureController(10);a.decide(local_observation)
    with pytest.raises(ValueError):a.decide(replace(local_observation,own=replace(local_observation.own,player_id='other')))


@pytest.mark.parametrize('profile',list(PROFILES))
def test_profiles_stay_legal_reproducible_and_switch_locally(local_observation,profile):
    a=MixtureController(12,{profile:1});b=MixtureController(12,{profile:1})
    for i in range(60):
        obs=replace(local_observation,tick=i*3,time_seconds=i*.6)
        left=a.decide(obs);right=b.decide(obs)
        assert left==right and intent_allowed(obs,left)
    assert len(a.mode_history)>=2


def test_reporting_has_contextual_variation_without_always_never_profiles():
    m,_,ids=evidence_fixture(2760071,'profile_variation')
    # Legitimately return to a previously observed body; no teleports.
    body=m.game.bodies[ids['victim']]
    m._drive({m.impostor:tuple(body.position)},500)
    obs=m.game.observe(m.impostor)
    assert any(a.kind is IntentKind.REPORT for a in obs.actions)
    choices=[]
    for seed in range(60):
        c=MixtureController(seed,{'self_report':1})
        choices.append(c.decide(obs).kind)
    assert IntentKind.REPORT in choices and any(k is not IntentKind.REPORT for k in choices)


@pytest.mark.parametrize('case',CASES)
def test_twelve_legal_engineering_situations(case):
    m,_,ids=evidence_fixture(2760072,case)
    g=m.game
    assert g.metrics['illegal_actions']==0
    if case=='weak_evidence':
        assert not g.bodies and all(p.active for p in g.players.values())
        assert any(e['kind']=='emergency_called' for e in g.event_log)
        assert not any(e['kind'] in ('report','elimination','claim') for e in g.event_log)
        return
    assert not g.players[ids['victim']].active
    obs=g.observe(ids['observer'])
    witnesses=[e for e in g.observe(ids['witness']).evidence if type(e.payload) is WitnessedElimination]
    if case in ('direct_witness','strong_evidence','new_evidence_after_defense'):
        assert witnesses and all(e.provenance is Provenance.DIRECT for e in witnesses)
    if case in ('unwitnessed','weak_evidence','multiple_suspects'):
        assert not [e for e in obs.evidence if type(e.payload) is WitnessedElimination]
    if case in ('leave_unreported','unwitnessed','profile_variation'):
        assert not any(e['kind']=='report' for e in g.event_log)
    if case=='self_report':assert next(e for e in g.event_log if e['kind']=='report')['reporter_id']==ids['impostor']
    if case=='innocent_report':assert next(e for e in g.event_log if e['kind']=='report')['reporter_id']==ids['observer']
    if case in ('frame_innocent','contradictory_claims','new_evidence_after_defense'):
        messages={ids['impostor']:(ClaimKind.SUSPECT_PLAYER,ids['innocent'])}
        if case=='contradictory_claims':messages[ids['observer']]=(ClaimKind.DEFEND_PLAYER,ids['innocent'])
        if case=='new_evidence_after_defense':
            messages={ids['observer']:(ClaimKind.DEFEND_PLAYER,ids['impostor']),
                      ids['witness']:(ClaimKind.SAW_ELIMINATION,ids['impostor'])}
        accepted=discussion_messages(m,messages)
        assert len(accepted)==len(messages) and g.metrics['illegal_actions']==0
        if case=='new_evidence_after_defense':
            defense=next(e for e in accepted if e['claim_kind']=='defend_player')
            witness=next(e for e in accepted if e['claim_kind']=='saw_elimination')
            assert defense['tick']<witness['tick']
            observer=g.observe(ids['observer'])
            claim=next(e for e in observer.evidence if type(e.payload) is StructuredClaim and e.payload.kind is ClaimKind.SAW_ELIMINATION)
            assert claim.provenance is Provenance.CLAIM
            assert not any(type(e.payload) is WitnessedElimination for e in observer.evidence)
    if case=='multiple_suspects':
        accepted=discussion_messages(m,{ids['observer']:(ClaimKind.SUSPECT_PLAYER,ids['impostor']),
            ids['innocent']:(ClaimKind.SUSPECT_PLAYER,ids['observer'])})
        assert len({e['subject_id'] for e in accepted})==2
    if g.phase is not Phase.ROAMING:
        assert ids['victim'] not in obs.context.participants
        assert all(a.kind not in (IntentKind.VOTE,IntentKind.SKIP) for a in g.observe(ids['victim']).actions)
