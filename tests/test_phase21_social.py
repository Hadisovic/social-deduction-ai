from dataclasses import replace
from types import SimpleNamespace
import pytest
from crewmate_v2.phase21.config import Features
from crewmate_v2.phase21.social import ListenerState,shift_probability,confidence_order,decision_rules
from social_deduction.actor import Meeting,Phase,Provenance
from social_deduction.phase3_api import ClaimKind,StructuredClaim,EventView,WitnessedElimination
from social_deduction.actor import Point


def observation(owner='a',claims=(),active=True):
    start=EventView(1,Provenance.PUBLIC,Meeting('m1',('a','b','c','d')))
    return SimpleNamespace(own=SimpleNamespace(player_id=owner,active=active),tick=10,
        context=SimpleNamespace(meeting_id='m1',participants=('a','b','c','d'),phase=Phase.DISCUSSION),evidence=(start,*claims))


def claim(speaker='b',target='c',kind=ClaimKind.SUSPECT_PLAYER,tick=2):
    return EventView(tick,Provenance.CLAIM,StructuredClaim(kind,speaker,target,'Cafeteria',tick,tick))


def test_single_claim_moves_exactly_ten_percentage_points_and_normalizes():
    state=ListenerState('a',Features(influence=.1));mem=SimpleNamespace(events=[])
    raw={'b':.25,'c':.55,'d':.20}
    p=state.update(observation(claims=[claim()]),mem,raw)
    assert p['c']==pytest.approx(.65) and sum(p.values())==pytest.approx(1)
    assert raw['c']==.55
    q=ListenerState('a',Features(influence=.1)).update(observation(claims=[claim(kind=ClaimKind.DEFEND_PLAYER)]),mem,raw)
    assert q['c']==pytest.approx(.45)


def test_duplicates_cannot_farm_influence_and_conflicting_repeat_cancels():
    mem=SimpleNamespace(events=[]);state=ListenerState('a',Features(influence=.1));raw={'b':.25,'c':.55,'d':.20}
    first=state.update(observation(claims=[claim()]),mem,raw)
    repeated=state.update(observation(claims=[claim(tick=i) for i in range(2,8)]),mem,raw)
    assert repeated==first and len(state.contributions)==1
    conflict=state.update(observation(claims=[claim(),claim(kind=ClaimKind.DEFEND_PLAYER,tick=3),claim(tick=4)]),mem,raw)
    assert conflict==raw and ('b','c') in state.locked


def test_own_unreceived_and_ghost_claims_do_not_adjust_private_state():
    mem=SimpleNamespace(events=[]);raw={'b':.25,'c':.55,'d':.20};state=ListenerState('a',Features(influence=.1))
    assert state.update(observation(claims=[claim(speaker='a'),claim(speaker='ghost')]),mem,raw)==raw
    assert not state.contributions
    assert state.update(observation(claims=[claim()],active=False),mem,raw)==raw and not state.contributions
    with pytest.raises(ValueError):state.update(observation(owner='b'),mem,raw)


def test_direct_evidence_protection_and_zero_influence_baseline():
    event=EventView(2,Provenance.DIRECT,WitnessedElimination('b','v',Point(0,0),'Cafeteria'))
    raw={'b':.99,'c':.005,'d':.005};mem=SimpleNamespace(events=[event]);obs=observation(claims=[claim()])
    assert ListenerState('a',Features(influence=.1)).update(obs,mem,raw)==raw
    assert ListenerState('a',Features(influence=0)).update(obs,mem,raw)==raw
    assert ListenerState('a',Features(influence=.1,protect_direct=False)).update(obs,mem,raw)['c']>.005


def test_simplex_extremes_multiple_listener_independence_and_config_bounds():
    for raw in ({'b':1.,'c':0.},{'b':0.,'c':1.}):
        for delta in (-1,.1,1):
            p=shift_probability(raw,'b',delta);assert all(0<=v<=1 for v in p.values()) and sum(p.values())==pytest.approx(1)
    a=ListenerState('a',Features(influence=.1));b=ListenerState('b',Features(influence=.1));a.update(observation(claims=[claim()]),SimpleNamespace(events=[]),{'b':.25,'c':.5,'d':.25})
    assert not b.contributions
    for kwargs in ({'influence':float('nan')},{'defense_max':.8},{'speaking_order':'hidden_truth'}):
        with pytest.raises(ValueError):Features(**kwargs)


def test_order_uses_provided_private_confidence_tie_breaks_and_never_role_truth():
    assert confidence_order({'a':.9,'b':.9,'c':.4},('c','b','a'))==('a','b','c')
    with pytest.raises(ValueError):confidence_order({'a':.9},('a','b'))


def test_guard_prevents_defense_of_high_suspicion_without_forcing_accusation():
    options=[SimpleNamespace(kind=k,target_player=p) for k,p in [('continue',None),('claim_defend','b'),('claim_suspect','b'),('claim_defend','c'),('skip',None)]]
    allow,boost,reasons=decision_rules(observation(),SimpleNamespace(events=[]),options,{'b':.9,'c':.1},Features(social_guard=True))
    assert allow.tolist()==[True,False,True,True,True]
    # Legitimate later belief reversal changes which targets can be defended.
    later,_,_=decision_rules(observation(),SimpleNamespace(events=[]),options,{'b':.1,'c':.9},Features(social_guard=True))
    assert later[1] and not later[3]


def test_ten_point_hearsay_can_cross_a_vote_threshold_while_three_points_do_not():
    # Synthetic boundary stress test, not a fabricated frozen-policy prediction.
    raw={'b':.20,'c':.65,'d':.15};mem=SimpleNamespace(events=[]);obs=observation(claims=[claim()]);obs.context.phase=Phase.VOTING
    options=[SimpleNamespace(kind='skip',target_player=None),SimpleNamespace(kind='vote',target_player='c')]
    outcomes=[]
    for magnitude in (0,.03,.10):
        config=Features(social_guard=True,influence=magnitude)
        suspicion=ListenerState('a',config).update(obs,mem,raw)
        allowed,_,_=decision_rules(obs,mem,options,suspicion,config)
        outcomes.append((suspicion['c'],bool(allowed[1])))
        assert allowed[0]  # threshold crossing never forces a vote
    assert outcomes[0][0]==pytest.approx(.65) and outcomes[1][0]==pytest.approx(.68) and outcomes[2][0]==pytest.approx(.75)
    assert [x[1] for x in outcomes]==[False,False,True]
