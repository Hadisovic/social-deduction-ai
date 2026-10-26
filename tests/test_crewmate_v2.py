"""Bounded engineering fixtures, outside all Phase 6 research/final partitions."""
from dataclasses import replace
import math
from random import Random
import pytest
from among_us_map import get_map
from navigation_service import NavigationPlanner
from social_deduction.actor import Phase, Role, Vote
from social_deduction.phase3_api import Intent, IntentKind
from crewmate_v2.api import ConfirmedEjection, VisualTaskProof
from crewmate_v2.config import LobbySettings, MatchMixture
from crewmate_v2.engine import CrewmateV2Game
from crewmate_v2.ghosts import GhostState, advance_ghost
from crewmate_v2.tasks import OwnedTask, make_plan, task_catalog


@pytest.fixture(scope='module')
def planner():return NavigationPlanner(get_map(),cache_size=96)


@pytest.fixture
def game(planner):return CrewmateV2Game(2610000,planner=planner)


def crew(g):return [p for p in g.players.values() if p.role is Role.CREWMATE]
def impostor(g):return next(p for p in g.players.values() if p.role is Role.IMPOSTOR)
def position(g,p,xy):p.motion.spawn(xy);g._invalidate()


def install(g,p,kind,category='Long',index=1):
    old=p.tasks[index]
    task=OwnedTask(old.task_id,make_plan(p.player_id,kind,category,task_catalog(g.map),Random(22)))
    p.tasks[index]=task;g.tasks[task.task_id]=task;g._invalidate()
    return task


def setup_kill(g):
    killer=impostor(g);victim=crew(g)[0]
    others=[p for p in g.players.values() if p not in (killer,victim)]
    for p,room in zip(others,('Reactor','Navigation','Communications')):
        position(g,p,next(d.standing for d in g.map.destinations if d.room==room))
    position(g,killer,(-.7,-2.8));position(g,victim,(-1.4,-2.8));killer.cooldown=0.
    g._invalidate()
    return killer,victim,others


def kill(g):
    killer,victim,others=setup_kill(g)
    g.step({killer.player_id:Intent(IntentKind.KILL,victim.player_id)})
    return killer,victim,others


def call_meeting(g,p=None):
    p=p or crew(g)[-1]
    console=next(d for d in g.map.destinations if d.name=='EmergencyConsole')
    position(g,p,console.standing)
    g.step({p.player_id:Intent(IntentKind.CALL_MEETING,console.id)})
    assert g.phase is Phase.DISCUSSION
    return p


def complete_stage(g,p,task):
    position(g,p,g.planner.destinations[task.console_id].standing)
    initial=task.stage
    g.step({p.player_id:Intent(IntentKind.INTERACT,task.task_id)})
    for _ in range(200):
        if task.stage!=initial:return
        g.step()
    pytest.fail('Task stage did not complete')


def test_lobby_mapping_and_actual_speed_vision_cooldown(game):
    c=game.config;l=game.lobby
    assert c.speed==2.5 and c.kill_range==pytest.approx(4*game.map.radius)==pytest.approx(.88)
    assert c.discussion_seconds==15 and c.voting_seconds==120
    assert c.kill_cooldown==c.initial_kill_cooldown==45
    assert l.emergency_cooldown==0 and l.emergency_meetings==1
    assert l.confirm_ejects and not l.anonymous_votes and l.visual_tasks and l.task_bar_updates=='always'
    assert game.observe(crew(game)[0].player_id).settings.sight_range==4.5
    assert game.observe(impostor(game).player_id).settings.sight_range==6.75
    assert impostor(game).cooldown==45
    p=crew(game)[0];position(game,p,(-.7,-2.8));before=p.position
    game.step({p.player_id:Intent(IntentKind.GO_TO_LOCATION,position=__import__('social_deduction.actor',fromlist=['Point']).Point(-2.,-2.8))})
    assert math.dist(before,p.position)==pytest.approx(.5)
    assert impostor(game).cooldown==pytest.approx(44.8)


@pytest.mark.parametrize('change',[dict(anonymous_votes=True),dict(task_bar_updates='meetings'),dict(common_tasks=2),
    dict(emergency_cooldown=1),dict(dt=.4),dict(kill_cooldown=0),dict(visual_tasks=1)])
def test_unsupported_or_invalid_settings_fail_explicitly(change):
    with pytest.raises(ValueError):LobbySettings(**change)


def test_public_profile_serialization_and_load(tmp_path):
    import json
    path=tmp_path/'lobby.json';path.write_text(json.dumps(LobbySettings().to_dict()))
    assert LobbySettings.from_file(path)==LobbySettings()


def test_ballot_target_changes_are_private_until_results(planner):
    a=CrewmateV2Game(2610000,planner=planner);b=CrewmateV2Game(2610000,planner=planner)
    voter=crew(a)[0].player_id;observer=crew(a)[1].player_id
    for g in (a,b):
        g.phase=Phase.VOTING;g.meeting_id='privacy-fixture';g.participants=tuple(g.players);g.phase_started_tick=0
    a._vote(voter,impostor(a).player_id);b._vote(voter,crew(b)[2].player_id)
    assert a.observe_v2(observer)==b.observe_v2(observer)
    assert not any(type(e.payload) is Vote for e in a.observe(observer).evidence)
    for g in (a,b):g.tick=599;g._invalidate();g.step()
    assert a.observe_v2(observer)!=b.observe_v2(observer)
    assert any(type(e.payload) is Vote for e in a.observe(observer).evidence)


def test_categories_common_type_and_independent_progress(game):
    players=crew(game)
    assert len(game.tasks)==16 and impostor(game).tasks==[]
    assert len({p.tasks[0].plan.task_type for p in players})==1
    assert all([t.plan.category for t in p.tasks]==['Common','Long','Short','Short'] for p in players)
    assert all(len({t.plan.task_type for t in p.tasks[2:]})==2 for p in players)
    assert len({t.task_id for p in players for t in p.tasks})==16
    task=players[0].tasks[0];assert task.plan.task_type=='SwipeCard'
    complete_stage(game,players[0],task)
    assert task.progress==1 and all(p.tasks[0].progress==0 for p in players[1:])
    assert game.observe_v2(players[1].player_id).task_bar.completed==1
    foreign=players[1].tasks[0]
    game.step({players[0].player_id:Intent(IntentKind.INTERACT,foreign.task_id)})
    assert foreign.progress==0 and game.metrics['illegal_actions']==1


def test_all_catalog_types_construct_and_multistage_support(planner):
    g=CrewmateV2Game(2610001,planner=planner)
    categories={t['taskType']:t['length'] for t in g.map.data['tasks'].values() if t['taskType']!='VentCleaning'}
    catalog=task_catalog(g.map)
    for kind,category in categories.items():
        plan=make_plan('p0',kind,category,catalog,Random(8))
        assert all(s.console_id in g.planner.destinations and s.seconds>0 for s in plan.stages)
    p=crew(g)[0];task=install(g,p,'AlignEngineOutput')
    first=task.console_id;complete_stage(g,p,task)
    assert task.stage==1 and task.console_id!=first and task.progress==.5
    complete_stage(g,p,task);assert task.progress==1
    wiring=make_plan(p.player_id,'FixWiring','Common',catalog,Random(3))
    assert len(wiring.stages)==len({s.console_id for s in wiring.stages})==3
    fuel=make_plan(p.player_id,'FuelEngines','Long',catalog,Random(3))
    assert [g.planner.destinations[s.console_id].room for s in fuel.stages]==['Storage','Lower Engine','Storage','Upper Engine']


def test_sample_incubation_is_own_visible_time_and_legal_actions(game):
    p=crew(game)[0];task=install(game,p,'InspectSample')
    complete_stage(game,p,task)
    assert task.stage==1 and task.progress==.5
    assert game.observe_v2(p.player_id).own_tasks[1].ready_in_seconds>=0
    assert task.task_id not in game.observe(p.player_id).interactable_tasks
    start=game.time
    for _ in range(299):game.step()
    assert game.time-start==pytest.approx(59.8)
    assert task.task_id not in game.observe(p.player_id).interactable_tasks
    game.step();assert task.task_id in game.observe(p.player_id).interactable_tasks
    complete_stage(game,p,task);assert task.progress==1


def test_permanent_ownership_after_death_and_public_absence(game):
    before={pid:tuple(t.task_id for t in p.tasks) for pid,p in game.players.items()}
    killer,victim,others=kill(game)
    assert victim.player_id in game.ghosts and not victim.active
    # A public body report cannot move tasks to living crew.
    reporter=others[0];position(game,reporter,game.bodies[victim.player_id].position)
    game.step({reporter.player_id:Intent(IntentKind.REPORT,victim.player_id)})
    assert game.phase is Phase.DISCUSSION
    assert before=={pid:tuple(t.task_id for t in p.tasks) for pid,p in game.players.items()}
    assert game.observe_v2(reporter.player_id).task_bar.total==16


def test_ghost_wall_travel_own_work_and_world_bounds(game):
    _,victim,_=kill(game)
    ghost=game.ghosts[victim.player_id]
    task=sorted(victim.tasks,key=lambda t:t.task_id)[0]
    goal=game._ghost_goals[task.console_id]
    # One direct route crosses native obstacles that living movement must respect.
    crossings=[d.standing for d in game.map.destinations if not game.map.segment_clear(ghost.position,d.standing)]
    assert crossings
    from crewmate_v2.tasks import TaskPlan,Stage
    fixture=OwnedTask('only',TaskPlan(victim.player_id,'Fixture','Short',(Stage(task.console_id,1.),)))
    state=GhostState(victim.player_id,ghost.position,ghost.bounds,2.5)
    goals={task.console_id:crossings[0]}
    assert not game.map.segment_clear(state.position,crossings[0])
    done=None
    for i in range(400):
        old=state;state,done=advance_ghost(state,(fixture,),goals,.2,i*.2)
        assert math.dist(old.position,state.position)<=.5+1e-9
        assert state.bounds[0]<=state.position[0]<=state.bounds[2] and state.bounds[1]<=state.position[1]<=state.bounds[3]
        if done:break
    assert done is fixture and fixture.progress==1
    with pytest.raises(ValueError):advance_ghost(state,(replace(fixture,plan=replace(fixture.plan,owner_id='foreign')),),goals,.2,100.)
    with pytest.raises(ValueError):GhostState(victim.player_id,(999.,999.),ghost.bounds,2.5)


def test_ghost_incubation_never_backdates_work():
    from crewmate_v2.tasks import TaskPlan,Stage
    task=OwnedTask('waiting',TaskPlan('owner','Fixture','Long',(Stage('console',1.),)),ready_at=10.)
    state=GhostState('owner',(0.,0.),(-1.,-1.,1.,1.),2.5)
    state,_=advance_ghost(state,(task,),{'console':(0.,0.)},.2,10.)
    assert task.work==0
    state,_=advance_ghost(state,(task,),{'console':(0.,0.)},.2,10.2)
    assert task.work==pytest.approx(.2)


def test_ghosts_complete_owned_tasks_without_visibility_or_visual_proofs(game):
    _,victim,others=kill(game)
    frozen=victim.position
    for _ in range(2000):
        game.step()
        if all(t.progress==1 for t in victim.tasks):break
    assert all(t.progress==1 for t in victim.tasks) and victim.position==frozen
    for p in others:
        obs=game.observe_v2(p.player_id)
        assert victim.player_id not in [q.player_id for q in obs.actor.visible_players]
        assert not any(type(e.payload) is VisualTaskProof and e.payload.player_id==victim.player_id for e in obs.events)
        assert obs.task_bar.completed==4
    assert game.observe(victim.player_id).actions==()


def test_visual_proofs_only_for_visible_real_living_completion(game):
    p,witness,far=crew(game)[:3]
    task=install(game,p,'SubmitScan')
    destination=game.planner.destinations[task.console_id]
    position(game,witness,destination.standing)
    position(game,far,next(d.standing for d in game.map.destinations if d.room=='Navigation'))
    complete_stage(game,p,task)
    assert any(type(e.payload) is VisualTaskProof and e.payload.player_id==p.player_id for e in game.observe_v2(witness.player_id).events)
    assert not any(type(e.payload) is VisualTaskProof for e in game.observe_v2(far.player_id).events)
    impost=impostor(game);position(game,impost,destination.standing)
    count=len(game.v2_direct[witness.player_id]);bar=game.observe_v2(witness.player_id).task_bar
    game.step({impost.player_id:Intent(IntentKind.FAKE_TASK,destination.id)})
    for _ in range(21):game.step()
    assert len(game.v2_direct[witness.player_id])==count and game.observe_v2(witness.player_id).task_bar==bar


def test_visual_tasks_off_suppresses_proof_without_changing_ownership_progress(planner):
    g=CrewmateV2Game(2610000,planner=planner,lobby=LobbySettings(visual_tasks=False))
    p,witness=crew(g)[:2];task=install(g,p,'SubmitScan')
    position(g,witness,g.planner.destinations[task.console_id].standing)
    complete_stage(g,p,task)
    assert task.progress==1 and g.observe_v2(witness.player_id).task_bar.completed==1
    assert not g.observe_v2(witness.player_id).events


@pytest.mark.parametrize('confirmed',[True,False])
def test_meeting_settings_visible_votes_confirmed_eject_and_ghost(planner,confirmed):
    g=CrewmateV2Game(2610000,planner=planner,lobby=LobbySettings(confirm_ejects=confirmed))
    caller=call_meeting(g);start=g.phase_started_tick;victim=crew(g)[0]
    assert caller.meetings_remaining==0
    while g.tick<start+75:g.step()
    assert g.phase is Phase.VOTING
    assert not g.v2_publications
    g.step({pid:Intent(IntentKind.VOTE,victim.player_id) for pid in g.participants})
    ballot=[e for e in g.observe(caller.player_id).evidence if type(e.payload) is Vote]
    assert ballot==[] and set(g.context.voted_ids)==set(g.participants)
    while g.tick<start+75+600-1:g.step()
    assert victim.active and not g.v2_publications
    g.step()
    assert victim.status=='ejected' and victim.player_id in g.ghosts and len(victim.tasks)==4
    events=[e for e in g.observe_v2(caller.player_id).events if type(e.payload) is ConfirmedEjection]
    assert bool(events)==confirmed
    if confirmed:assert events[-1].payload.role is Role.CREWMATE and events[-1].tick==g.tick
    ballot=[e for e in g.observe(caller.player_id).evidence if type(e.payload) is Vote]
    assert len(ballot)==5 and {e.payload.voter_id for e in ballot}==set(g.players)
    assert all(e.tick==g.tick for e in ballot)
    assert g.time==135.
    # Another living player can call immediately; there is no global emergency delay.
    other=next(p for p in crew(g) if p.active and p.meetings_remaining)
    call_meeting(g,other);assert other.meetings_remaining==0


def test_hidden_mode_tasks_ejection_parity_and_timeout(planner):
    ordinary=CrewmateV2Game(2610000,planner=planner,lobby=LobbySettings(timeout=1.))
    deduction=CrewmateV2Game(2610000,planner=planner,lobby=LobbySettings(timeout=1.),mixture=MatchMixture(1.))
    assert ordinary.observe_v2('p0')==deduction.observe_v2('p0')
    for g in (ordinary,deduction):
        for task in g.tasks.values():task.stage=len(task.plan.stages)
        g.step()
    assert ordinary.result.reason=='TASKS_COMPLETED' and not deduction.is_terminal
    assert deduction.observe_v2('p0').task_bar.completed==16
    while not deduction.is_terminal:deduction.step()
    assert deduction.result.reason=='TIMEOUT' and deduction.time==1.
    assert MatchMixture(.37).select(2610001)==MatchMixture(.37).select(2610001)
    g=CrewmateV2Game(2610000,planner=planner,mixture=MatchMixture(1.))
    for p in crew(g)[:3]:g._interrupt(p,'fixture');p.status='dead';g._make_ghost(p)
    g._check_win();assert g.result.reason=='PARITY'
    g=CrewmateV2Game(2610000,planner=planner,mixture=MatchMixture(1.))
    imp=impostor(g);g.phase=Phase.VOTING;g.meeting_id='fixture';g.participants=tuple(g.players)
    g.votes={pid:imp.player_id for pid in g.players};g._resolve_meeting()
    assert g.result.reason=='IMPOSTOR_EJECTED' and imp.player_id not in g.ghosts


def test_simultaneous_ejection_parity_takes_priority_over_ghost_task_win(game):
    from dataclasses import replace
    victims=crew(game)[:2];target=crew(game)[2]
    for p in victims:game._interrupt(p,'fixture');p.status='dead';game._make_ghost(p)
    for task in game.tasks.values():task.stage=len(task.plan.stages)
    task=victims[0].tasks[0];task.stage=len(task.plan.stages)-1;task.work=task.duration-.2
    ghost=game.ghosts[victims[0].player_id]
    game.ghosts[victims[0].player_id]=replace(ghost,position=game._ghost_goals[task.console_id])
    game.phase=Phase.VOTING;game.meeting_id='fixture';game.phase_started_tick=0;game.tick=599
    game.participants=tuple(pid for pid,p in game.players.items() if p.active)
    game.votes={pid:target.player_id for pid in game.participants};game._invalidate()
    game.step()
    assert task.progress==1 and game.result.reason=='PARITY' and game.result.winner is Role.IMPOSTOR
