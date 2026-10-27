"""Post-pilot V2-only repairs. Deterministic engineering states, not learned play."""
import copy
import pytest
from crewmate_v2.engine import CrewmateV2Game
from crewmate_v2.config import LobbySettings
from social_deduction.actor import Phase,Role,Vote
from social_deduction.phase3_api import Intent,IntentKind


def game():return CrewmateV2Game(2760000,lobby=LobbySettings(timeout=600.))
def voting(g):
    g.phase=Phase.VOTING;g.meeting_id='repair-fixture';g.participants=tuple(g.players)
    g.phase_started_tick=g.tick;g._invalidate()


def test_all_living_ballots_resolve_same_tick_private_until_then():
    g=game();voting(g);ids=list(g.players)
    for pid in ids[:-1]:assert g._vote(pid,None)
    assert not any(type(e.payload) is Vote for e in g.observe(ids[-1]).evidence)
    assert not g._vote(ids[0],ids[1])
    g.step({ids[-1]:Intent(IntentKind.SKIP)})
    assert g.phase is Phase.ROAMING and g.time==pytest.approx(.2)
    assert len([e for e in g.event_log if e['kind']=='vote'])==5


@pytest.mark.parametrize('status',['dead','ejected'])
def test_dead_direct_vote_and_claim_rejected_and_do_not_block(status):
    g=game();voting(g);dead=next(p for p in g.players.values() if p.role is Role.CREWMATE)
    dead.status=status;g._make_ghost(dead);g._invalidate()
    assert not g._vote(dead.player_id,None)
    for pid in g._eligible_voters():assert g._vote(pid,None)
    g.step({dead.player_id:Intent(IntentKind.SKIP)})
    assert g.phase is Phase.ROAMING
    assert not any(e.get('voter_id')==dead.player_id for e in g.event_log)
    assert not g.observe(dead.player_id).actions


def test_timeout_fills_only_eligible_and_simultaneous_ejection_order():
    g=game();voting(g);dead=next(p for p in g.players.values() if p.role is Role.CREWMATE)
    dead.status='dead';g._make_ghost(dead)
    g.tick=599;g._invalidate();g.step()
    assert g.phase is Phase.ROAMING
    assert len([e for e in g.event_log if e['kind']=='vote_submitted'])==4
    g=game();voting(g);imp=next(pid for pid,p in g.players.items() if p.role is Role.IMPOSTOR)
    g.step({pid:Intent(IntentKind.VOTE,imp) for pid in g.players})
    assert g.result.reason=='IMPOSTOR_EJECTED'


def test_ghost_movement_work_and_incubation_frozen_through_meeting():
    g=game();p=next(p for p in g.players.values() if p.role is Role.CREWMATE)
    p.status='dead';g._make_ghost(p)
    # Real own task work plus an unfinished incubation timer.
    p.tasks[0].work=.5;p.tasks[1].ready_at=50.
    before=copy.deepcopy(g.ghosts);progress=[t.progress for t in p.tasks]
    voting(g);g.phase=Phase.DISCUSSION
    for _ in range(10):g.step()
    assert g.ghosts==before and [t.progress for t in p.tasks]==progress
    assert p.tasks[1].ready_at-g.time==pytest.approx(50.)
    assert not any(e['kind']=='task_completed' for e in g.event_log)
    g.phase=Phase.VOTING;g.phase_started_tick=g.tick
    for _ in range(5):g.step()
    assert g.ghosts==before and [t.progress for t in p.tasks]==progress
    for pid in g._eligible_voters():g._vote(pid,None)
    g.step();assert g.phase is Phase.ROAMING
    g.step();assert g.ghosts!=before or [t.progress for t in p.tasks]!=progress


def test_living_partial_interaction_cannot_progress_during_meeting():
    g=game();p=next(p for p in g.players.values() if p.role is Role.CREWMATE)
    t=p.tasks[0];p.motion.spawn(g.planner.destinations[t.console_id].standing)
    g._invalidate();g.step({p.player_id:Intent(IntentKind.INTERACT,t.task_id)})
    work=t.work;g._start_meeting(p);g.phase=Phase.VOTING;g._invalidate()
    for _ in range(5):g.step()
    assert t.work==work
    for pid in g.players:g._vote(pid,None)
    g.step();g.step({p.player_id:Intent(IntentKind.INTERACT,t.task_id)})
    assert t.work>work
