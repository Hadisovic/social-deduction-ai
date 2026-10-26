"""Ghost-secret noninterference and immutable actor/private-memory contracts."""
from dataclasses import replace
import inspect
import numpy as np
import pytest
from among_us_map import get_map
from navigation_service import NavigationPlanner
from phase3_runner import canonical_bytes
from social_deduction.actor import Phase, Point, Provenance, Role
from social_deduction.phase3_api import (ClaimDraft,ClaimKind,EventView,Intent,IntentKind,WitnessedElimination)
from crewmate_v2.api import V2Event, TaskBar, validate
from crewmate_v2.config import LobbySettings, MatchMixture
from crewmate_v2.engine import CrewmateV2Game
from crewmate_v2.ghosts import advance_ghost
from crewmate_v2.runtime import CrewmateV2Env, PublicMemory
from test_crewmate_v2 import crew, impostor, kill, setup_kill, position, call_meeting


@pytest.fixture(scope='module')
def planner():return NavigationPlanner(get_map(),cache_size=96)


def test_ghost_secret_attribution_history_and_timing_cannot_signal(planner):
    a=CrewmateV2Game(2610000,planner=planner);b=CrewmateV2Game(2610000,planner=planner)
    _,victim,living=kill(a);_,other,_=kill(b)
    # Only secret attribution/history differs. Physical state and public consequences match.
    b.bodies[other.player_id].killer_id=living[0].player_id
    b.direct_events[other.player_id].append(EventView(b.tick,Provenance.DIRECT,
        WitnessedElimination(living[1].player_id,other.player_id,Point(*other.position),'Cafeteria')))
    b.ghost_only_notes={'killer':'different','suspicions':[1.,0.,0.,0.], 'deliberate_delay':1000}
    owners=sorted(pid for pid,p in a.players.items() if p.active)
    completed=False
    for _ in range(1800):
        for pid in owners:assert canonical_bytes(a.observe_v2(pid))==canonical_bytes(b.observe_v2(pid))
        a.step();b.step()
        assert a.ghosts==b.ghosts
        assert [t.progress for t in victim.tasks]==[t.progress for t in other.tasks]
        if all(t.progress==1 for t in victim.tasks):completed=True;break
    assert completed
    assert a.observe_v2(owners[0]).task_bar==b.observe_v2(owners[0]).task_bar
    assert 'world' not in inspect.signature(advance_ghost).parameters
    source=inspect.getsource(__import__('crewmate_v2.ghosts',fromlist=['advance_ghost']))
    for forbidden in ('event_log','killer_id','controllers','observations','self.memory','Random','Role'):
        assert forbidden not in source


def test_actual_different_hidden_killers_preserve_uninformed_living_view(planner):
    from types import MappingProxyType
    from crewmate_v2.tasks import assignments
    a=CrewmateV2Game(2610000,planner=planner);b=CrewmateV2Game(2610000,planner=planner)
    killer=impostor(a).player_id;victim=crew(a)[0].player_id;alternative=crew(a)[1].player_id
    observers=[p.player_id for p in crew(a)[2:]]
    # A controlled different-role world, with identical observer-owned tasks, geometry,
    # victim state, death time and private ghost task plan. Own roles of the two
    # alternative killers legitimately differ; their views are not compared.
    b.players[killer].role=Role.CREWMATE;b.players[alternative].role=Role.IMPOSTOR
    plans=assignments(b.map,b.seed,b.players);b.tasks={}
    for pid,p in b.players.items():
        p.tasks=list(plans[pid]) if p.role is Role.CREWMATE else []
        b.tasks.update({t.task_id:t for t in p.tasks})
    b._owners=MappingProxyType({t.task_id:t.owner_id for t in b.tasks.values()})
    for g in (a,b):
        for pid in (killer,alternative):position(g,g.players[pid],(-.7,-2.8));g.players[pid].cooldown=0.
        position(g,g.players[victim],(-1.4,-2.8))
        for pid,room in zip(observers,('Reactor','Navigation')):
            position(g,g.players[pid],next(d.standing for d in g.map.destinations if d.room==room))
    a.step({killer:Intent(IntentKind.KILL,victim)});b.step({alternative:Intent(IntentKind.KILL,victim)})
    assert a.bodies[victim].killer_id!=b.bodies[victim].killer_id
    for _ in range(1200):
        for pid in observers:assert a.observe_v2(pid)==b.observe_v2(pid)
        assert a.ghosts==b.ghosts
        a.step();b.step()
        if all(t.progress==1 for t in a.players[victim].tasks):break
    assert all(t.progress==1 for t in a.players[victim].tasks)


@pytest.mark.parametrize('kind',[IntentKind.CLAIM,IntentKind.REPORT,IntentKind.VOTE,IntentKind.SKIP,
    IntentKind.CALL_MEETING,IntentKind.GO_TO_LOCATION,IntentKind.GO_TO_TASK,IntentKind.INTERACT,IntentKind.KILL])
def test_ghost_submissions_cannot_control_movement_progress_or_living_information(planner,kind):
    a=CrewmateV2Game(2610000,planner=planner);b=CrewmateV2Game(2610000,planner=planner)
    _,victim,living=kill(a);kill(b)
    args={}
    if kind is IntentKind.GO_TO_LOCATION:args['position']=Point(999.,999.)
    elif kind is IntentKind.CLAIM:args['claim']=ClaimDraft(ClaimKind.SUSPECT_PLAYER,living[0].player_id,'Cafeteria',a.tick)
    else:args['target']=living[0].player_id
    a.step({victim.player_id:Intent(kind,**args)});b.step()
    assert a.metrics['illegal_actions']==1
    assert a.ghosts==b.ghosts
    for p in living:assert a.observe_v2(p.player_id)==b.observe_v2(p.player_id)
    assert len(a.publications)==len(b.publications)


def test_ghost_never_becomes_meeting_participant_or_visible_player(planner):
    g=CrewmateV2Game(2610000,planner=planner);_,victim,living=kill(g)
    call_meeting(g,living[0]);assert victim.player_id not in g.participants
    assert g.observe(victim.player_id).actions==()
    assert victim.player_id not in g.context.voted_ids
    assert all(victim.player_id not in [q.player_id for q in g.observe(p.player_id).visible_players] for p in living)


def test_v2_api_rejects_privileged_payloads_mutable_collections_and_future_events(planner):
    g=CrewmateV2Game(2610000,planner=planner);obs=g.observe_v2(crew(g)[0].player_id)
    with pytest.raises(TypeError):validate(g.snapshot())
    with pytest.raises(TypeError):validate(replace(obs,events=[]))
    with pytest.raises(TypeError):validate(replace(obs,events=(V2Event(0,Provenance.DIRECT,g.snapshot()),)))
    with pytest.raises(ValueError):validate(replace(obs,task_bar=TaskBar(17,16)))
    with pytest.raises(ValueError):validate(replace(obs,own_tasks=obs.own_tasks[:-1]))
    with pytest.raises(ValueError):validate(replace(obs,own_tasks=(replace(obs.own_tasks[0],task_type=g),)+obs.own_tasks[1:]))
    forged=replace(obs)
    object.__setattr__(forged,'ghost_backref',g)
    with pytest.raises(TypeError,match='extra attributes'):validate(forged)
    from crewmate_v2.api import ConfirmedEjection
    event=V2Event(1,Provenance.PUBLIC,ConfirmedEjection('m','p0',Role.CREWMATE))
    with pytest.raises(ValueError,match='future'):validate(replace(obs,events=(event,)))


def test_independent_actor_memory_and_frozen_on_death(planner):
    g=CrewmateV2Game(2610000,planner=planner);first,second=crew(g)[:2]
    a,b=PublicMemory(),PublicMemory()
    a.update(g.observe_v2(first.player_id));b.update(g.observe_v2(second.player_id))
    assert a.events is not b.events and a._known is not b._known
    with pytest.raises(ValueError,match='owners'):a.update(g.observe_v2(second.player_id))
    original=(a.tick,list(a.events))
    obs=g.observe_v2(first.player_id)
    dead=replace(obs,actor=replace(obs.actor,own=replace(obs.actor.own,active=False)))
    a.update(dead);assert original==(a.tick,a.events)


def test_environment_new_schema_no_mode_marker_original_death_reward_and_no_override(planner):
    from phase6.env import Phase6Env
    assert CrewmateV2Env._advance is Phase6Env._advance
    a=CrewmateV2Env(lobby=LobbySettings(timeout=1.),planner=planner)
    b=CrewmateV2Env(lobby=LobbySettings(timeout=1.),mixture=MatchMixture(1.),planner=planner)
    packet,_=a.reset(seed=2610000);other,_=b.reset(seed=2610000)
    assert a.config==a.match.game.config and a.config.kill_cooldown==45
    assert all(np.array_equal(packet[k],other[k]) for k in packet)
    assert a.observation_space.contains(packet)
    assert packet['players'].shape==(4,29) and packet['own_tasks_v2'].shape==(4,8)
    g=a.match.game;killer=impostor(g);victim=g.players[a.focal]
    for p,room in zip([p for p in g.players.values() if p not in (killer,victim)],('Reactor','Navigation','Communications')):
        position(g,p,next(d.standing for d in g.map.destinations if d.room==room))
    position(g,killer,(-.7,-2.8));position(g,victim,(-1.4,-2.8));killer.cooldown=0.;g._invalidate()
    first=True
    def controlled_tick(intent=None):
        nonlocal first
        actions={killer.player_id:Intent(IntentKind.KILL,victim.player_id)} if first else {}
        first=False
        return g.step(actions)
    a.match.tick=controlled_tick;a._observe();a._packet()
    _,_,done,_,info=a.step(0)
    assert done and a._eliminated and info['episode']['own_tasks']==0
    # One -5 death + -5 draw + original per-simulated-second cost, no new shaping.
    assert info['undiscounted_reward']==pytest.approx(-10.-.005*1./.6)
    with pytest.raises(RuntimeError):a.step(0)


def test_strong_witness_suspicion_does_not_override_selected_skip(planner):
    env=CrewmateV2Env(planner=planner);env.reset(seed=2610000)
    g=env.match.game;killer=impostor(g)
    victim=next(p for p in crew(g) if p.player_id!=env.focal)
    g._eliminate(killer,victim)
    g.tick=1;g.phase=Phase.VOTING;g.phase_started_tick=1;g.meeting_id='choice-fixture'
    g.participants=tuple(pid for pid,p in g.players.items() if p.active);g._invalidate()
    env._observe();env._packet()
    assert env.belief.by_player[killer.player_id]>.9
    selected=next(i for i,c in enumerate(env.options) if c.kind=='skip')
    env.step(selected)
    assert env.focal in g.votes and g.votes[env.focal] is None
