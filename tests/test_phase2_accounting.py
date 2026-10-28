"""Record a historical ledger limitation without changing it or its coefficients."""
from crewmate_v2.fourcrew.rewards import ActorLedger
from crewmate_v2.fourcrew.config import Rewards
from social_deduction.actor import Role
from social_deduction.phase3_api import Intent, IntentKind


def test_historical_same_step_ejection_suppresses_an_otherwise_valid_vote():
    roles={'a':Role.CREWMATE,'b':Role.CREWMATE,'i':Role.IMPOSTOR}
    event=dict(kind='vote_submitted',voter_id='a',target_id='b',meeting_id='m1')
    living=ActorLedger('a',roles,Rewards());ejected=ActorLedger('a',roles,Rewards())
    intent=Intent(IntentKind.VOTE,'b')
    living.consume([event],active=True,intent=intent,legal=True)
    ejected.consume([event,dict(kind='ejection',player_id='a')],active=False,intent=intent,legal=True)
    assert living.counts['incorrect_votes']==1 and living.components['incorrect_votes']==-1
    assert ejected.counts['incorrect_votes']==0 and ejected.components['incorrect_votes']==0
    assert ejected.counts['death']==1


def test_ghost_task_rewards_and_frozen_coefficients_are_preserved():
    assert Rewards().report==1.5 and Rewards().correct_vote==2 and Rewards().incorrect_vote==-1
    assert Rewards().ghost_task==0 and Rewards.original().report==0


def test_current_emergency_does_not_inherit_a_previous_body_reporter():
    from types import SimpleNamespace
    from social_deduction.actor import Meeting, Report, Provenance
    from social_deduction.phase3_api import EventView
    from crewmate_v2.phase2.audit import OpportunityAudit
    events=(EventView(10,Provenance.PUBLIC,Report('i','v','Admin')),
            EventView(10,Provenance.PUBLIC,Meeting('m1',('a','i'))),
            EventView(20,Provenance.PUBLIC,Meeting('m2',('a','i'))))
    obs=SimpleNamespace(context=SimpleNamespace(meeting_id='m2'),evidence=events)
    assert OpportunityAudit._reporter(obs) is None
    obs.context.meeting_id='m1'
    assert OpportunityAudit._reporter(obs)=='i'

