import pytest
from crewmate_v2.phase21.rewards import EligibleLedger,SCHEMA
from crewmate_v2.fourcrew.rewards import ActorLedger
from crewmate_v2.fourcrew.config import Rewards
from social_deduction.actor import Role
from social_deduction.phase3_api import Intent,IntentKind


ROLES={'a':Role.CREWMATE,'b':Role.CREWMATE,'i':Role.IMPOSTOR}


@pytest.mark.parametrize('target,kind,category,value',[('i',IntentKind.VOTE,'correct_votes',2),('b',IntentKind.VOTE,'incorrect_votes',-1),(None,IntentKind.SKIP,'skips',0)])
def test_eligible_ballots_count_exactly_once_even_after_same_step_ejection(target,kind,category,value):
    ledger=EligibleLedger('a',ROLES,Rewards());event=dict(kind='vote_submitted',voter_id='a',target_id=target,meeting_id='m1')
    intent=Intent(kind,target)
    ledger.consume([event,dict(kind='ejection',player_id='a')],active=False,intent=intent,legal=True)
    ledger.consume([event],active=False,intent=intent,legal=True)
    assert ledger.counts[category]==1 and ledger.components[category]==value
    assert ledger.counts['death']==1 and ledger.schema==SCHEMA


def test_historical_bug_stays_reproducible_while_new_ledger_repairs_it():
    e=dict(kind='vote_submitted',voter_id='a',target_id='b',meeting_id='m1');intent=Intent(IntentKind.VOTE,'b')
    old=ActorLedger('a',ROLES,Rewards());new=EligibleLedger('a',ROLES,Rewards())
    for l in (old,new):l.consume([e,dict(kind='ejection',player_id='a')],active=False,intent=intent,legal=True)
    assert old.counts['incorrect_votes']==0 and new.counts['incorrect_votes']==1
    assert old.config==new.config and old.components['death']==new.components['death']==-5


@pytest.mark.parametrize('kind,target',[(IntentKind.VOTE,'i'),(IntentKind.SKIP,None),(IntentKind.REPORT,'v')])
def test_dead_actor_never_gets_vote_or_report_credit(kind,target):
    ledger=EligibleLedger('a',ROLES,Rewards());ledger.alive=False;ledger.death_paid=True
    e=dict(kind='report',reporter_id='a',victim_id='v') if kind is IntentKind.REPORT else dict(kind='vote_submitted',voter_id='a',target_id=target,meeting_id='m1')
    ledger.consume([e],active=False,intent=Intent(kind,target),legal=True)
    assert not any(ledger.counts[k] for k in ('reports','correct_votes','incorrect_votes','skips','automatic_skips'))


def test_invalid_submission_missing_ballot_and_automatic_skip_are_distinct():
    ledger=EligibleLedger('a',ROLES,Rewards())
    ledger.consume([dict(kind='illegal_action',player_id='a')],active=True,intent=Intent(IntentKind.VOTE,'unknown'),legal=False)
    assert ledger.counts['correct_votes']==ledger.counts['incorrect_votes']==ledger.counts['skips']==0
    ledger.consume([dict(kind='vote_submitted',voter_id='a',target_id=None,meeting_id='m1')],active=True)
    assert ledger.counts['automatic_skips']==1 and ledger.counts['skips']==0


def test_terminal_credit_and_correct_ballot_survive_ejection_event_order():
    ledger=EligibleLedger('a',ROLES,Rewards())
    e=dict(kind='vote_submitted',voter_id='a',target_id='i',meeting_id='m1')
    ledger.consume([dict(kind='ejection',player_id='a'),e],active=False,terminal=True,winner=Role.CREWMATE,intent=Intent(IntentKind.VOTE,'i'),legal=True)
    assert ledger.counts['correct_votes']==ledger.counts['death']==ledger.counts['win']==1
