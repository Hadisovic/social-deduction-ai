from types import SimpleNamespace
from social_deduction.actor import Point,Phase,Provenance
from social_deduction.phase3_api import EventView,WitnessedElimination
from crewmate_v2.phase21.config import Features
from crewmate_v2.phase21.social import decision_rules


def test_authenticated_direct_evidence_overrides_a_wrong_frozen_probability():
    event=EventView(10,Provenance.DIRECT,WitnessedElimination('b','v',Point(0,0),'Cafeteria'))
    memory=SimpleNamespace(events=[event]);obs=SimpleNamespace(context=SimpleNamespace(phase=Phase.VOTING,participants=('a','b','c','d')))
    options=[SimpleNamespace(kind=k,target_player=t) for k,t in [('skip',None),('vote','b'),('vote','c')]]
    allow,boost,_=decision_rules(obs,memory,options,{'b':.01,'c':.98,'d':.01},Features(social_guard=True,witness_priority=True))
    assert allow.tolist()==[True,True,False] and boost[1]>0


def test_unknown_killer_reporting_is_never_gated_by_suspicion():
    obs=SimpleNamespace(context=SimpleNamespace(phase=Phase.ROAMING,participants=()))
    options=[SimpleNamespace(kind='continue',target_player=None),SimpleNamespace(kind='report',target_player='body')]
    allow,boost,_=decision_rules(obs,SimpleNamespace(events=[]),options,{'b':.01,'c':.49,'d':.50},Features(social_guard=True,witness_priority=True))
    assert allow.all() and not boost.any()
