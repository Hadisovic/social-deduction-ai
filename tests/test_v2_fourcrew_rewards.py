"""Versioned reward change and real engineering report/ejection attribution."""
import pytest
from crewmate_v2.learning.config import RewardConfig
from crewmate_v2.fourcrew.config import Rewards
from crewmate_v2.fourcrew.rewards import ActorLedger
from social_deduction.actor import Role
from social_deduction.phase3_api import Intent,IntentKind

ROLES={'a':Role.CREWMATE,'b':Role.CREWMATE,'i':Role.IMPOSTOR}


def test_new_report_ratio_preserves_original_focal_config():
    assert Rewards().report/Rewards().alive_task==1.5
    assert RewardConfig().report==1.25 and RewardConfig().schema=='crewmate-v2-rewards-v1'
    assert Rewards().schema=='crewmate-fourcrew-rewards-v2'
    assert Rewards.original().ghost_task==Rewards().ghost_task==0


def test_owner_only_successful_report_once_and_no_dead_credit():
    a=ActorLedger('a',ROLES,Rewards());b=ActorLedger('b',ROLES,Rewards())
    intent=Intent(IntentKind.REPORT,'body');events=[dict(kind='report',reporter_id='a',victim_id='body')]
    for _ in range(2):
        a.consume(events,active=True,intent=intent,legal=True)
        b.consume(events,active=True,intent=intent,legal=True)
    assert a.components['reports']==1.5 and b.components['reports']==0
    dead=ActorLedger('a',ROLES,Rewards());dead.consume(events,active=False,intent=intent,legal=True)
    assert dead.components['reports']==0
    invalid=ActorLedger('a',ROLES,Rewards());invalid.consume(events,active=True,intent=intent,legal=False)
    assert invalid.components['reports']==0


def test_team_broadcast_once_per_actor_and_task_owner_credits_not_broadcast():
    ledgers={pid:ActorLedger(pid,ROLES,Rewards()) for pid in ('a','b')}
    for pid,x in ledgers.items():
        x.consume([dict(kind='task_completed',player_id='a',task_id='task')],active=True)
        x.consume([],active=True,terminal=True,winner=Role.CREWMATE)
        x.consume([],active=True,terminal=True,winner=Role.CREWMATE)
        assert x.components['win']==10 and x.counts['win']==1
    assert ledgers['a'].components['alive_tasks']==1 and ledgers['b'].components['alive_tasks']==0


@pytest.mark.parametrize('reporter_role',[Role.CREWMATE,Role.IMPOSTOR])
def test_real_legal_reports_can_come_from_both_roles_without_role_inputs(reporter_role):
    from crewmate_v2.engine import CrewmateV2Game
    from social_deduction.actor import Phase
    g=CrewmateV2Game(2760002)
    imp=next(p for p in g.players.values() if p.role is Role.IMPOSTOR)
    crew=[p for p in g.players.values() if p.role is Role.CREWMATE]
    victim=crew[0];caller=imp if reporter_role is Role.IMPOSTOR else crew[1]
    for p in g.players.values():p.motion.spawn(next(d.standing for d in g.map.destinations if d.room=='Navigation'))
    imp.motion.spawn((-.7,-2.8));victim.motion.spawn((-1.4,-2.8));imp.cooldown=0
    g._invalidate();g.step({imp.player_id:Intent(IntentKind.KILL,victim.player_id)})
    caller.motion.spawn(victim.position);g._invalidate();obs=g.observe(caller.player_id)
    report=next(a for a in obs.actions if a.kind is IntentKind.REPORT)
    roles={pid:p.role for pid,p in g.players.items()};ledger=ActorLedger(caller.player_id,roles,Rewards())
    mark=len(g.event_log);g.step({caller.player_id:report})
    assert g.phase is Phase.DISCUSSION
    ledger.consume(g.event_log[mark:],active=True,intent=report,legal=True)
    assert ledger.components['reports']==1.5
    # This fixture submits a legal action deliberately, not a learned reporting result.
