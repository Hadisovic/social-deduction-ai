"""Run alone with an explicit pinned source root; no RL training or final sets."""
import os
from pathlib import Path
from dataclasses import replace
import pytest

SOURCE=os.environ.get('JANE_UPSTREAM_REVIEW_ROOT')
if not SOURCE:pytest.skip('Explicit pinned source required',allow_module_level=True)
from jane.impostor.v1.bootstrap import activate
activate(SOURCE)
from jane.classic_v2.settings import HostProfile,load_profile
from jane.classic_v2.engine import experiment,game_class,make_match
from jane.classic_v2.tasks import assignments,family
from social_deduction.actor import Role,Phase,Vote
from social_deduction.phase3_api import Intent,IntentKind

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def profile():return load_profile(ROOT/'jane/configs/classic-v2/default_12p.json')


def custom(profile,players=12,impostors=2,counts=(1,1,2),**meeting):
    s=profile.settings
    return HostProfile(replace(s,lobby=replace(s.lobby,players=players,impostors=impostors),
        tasks=replace(s.tasks,common=counts[0],long=counts[1],short=counts[2]),
        meetings=replace(s.meetings,**meeting)))


@pytest.mark.parametrize('players,impostors',[(p,i) for i,m in ((1,4),(2,7),(3,9)) for p in range(m,16)])
def test_all_legal_lobbies_construct_and_project_privately(profile,players,impostors):
    p=custom(profile,players,impostors);game=game_class()(9410010,experiment=experiment(p))
    assert len(game.players)==players
    assert sum(a.role is Role.IMPOSTOR for a in game.players.values())==impostors
    for pid,actor in game.players.items():
        obs=game.observe_pj(pid)
        assert obs.actor.own.player_id==pid
        assert len(obs.actor.roster)==players
        if actor.role is Role.CREWMATE:assert not obs.teammates
    game.assert_invariants()


def test_every_task_count_combination_is_unique_and_common_matchwide(profile):
    game=game_class()(9410010,experiment=experiment(profile))
    for common in range(3):
        for long in range(4):
            for short in range(6):
                p=custom(profile,counts=(common,long,short))
                plans=assignments(game.base_map,9410000+common*100+long*10+short,game.players,p.effective_counts)
                commons=set();identities=set()
                for pid,tasks in plans.items():
                    assert len(tasks)==sum(p.effective_counts)
                    assert len({family(t.plan.task_type) for t in tasks})==len(tasks)
                    assert all(t.owner_id==pid for t in tasks)
                    commons.add(tuple((t.plan.task_type,t.plan.stages) for t in tasks if t.plan.category=='Common'))
                    for task in tasks:assert task.task_id not in identities;identities.add(task.task_id)
                assert len(commons)==1


def test_fifteen_players_one_impostor_uses_fourteen_crewmates(profile):
    game=game_class()(9410011,experiment=experiment(custom(profile,15,1,(2,3,5))))
    assert len(game.tasks)==140
    match=make_match(9410011,experiment(custom(profile,15,1,(2,3,5))))
    assert len(match.crew)==14
    from patrick_jane.a2_environment import A2Game
    assert type(match.game) is not A2Game  # Scoped constructor binding was restored.


def test_zero_all_tasks_falls_back_to_one_short_task(profile):
    p=custom(profile,counts=(0,0,0));game=game_class()(9410010,experiment=experiment(p))
    assert p.effective_counts==(0,0,1)
    assert all(len(p.tasks)==1 for p in game.players.values() if p.role is Role.CREWMATE)
    assert not game.is_terminal


def test_reported_dead_position_under_door_does_not_invalidate_living_world(profile):
    game=game_class()(9410010,experiment=experiment(profile))
    dead=next(p for p in game.players.values() if p.role is Role.CREWMATE)
    actor=next(p for p in game.players.values() if p.role is Role.IMPOSTOR)
    dead.motion.spawn((-.20165175055810952,-8.95852724275098));dead.status='dead'
    game.tick=200
    assert game.close_doors(actor.player_id,'Storage')
    assert not game.map.contains(dead.position) and game.base_map.contains(dead.position)
    game.assert_invariants()
    # A living surface actor at the same stored point must still fail.
    dead.status='alive'
    with pytest.raises(AssertionError):game.assert_invariants()


def test_meetings_preserve_task_objects_progress_and_unique_ownership(profile):
    game=game_class()(9410010,experiment=experiment(profile));before={p:tuple(a.tasks) for p,a in game.players.items()}
    owner=next(p for p,a in game.players.items() if a.role is Role.CREWMATE)
    task=game.players[owner].tasks[0];task.work=.25
    game._start_meeting(game.players[owner]);game.phase=Phase.VOTING;game.votes={pid:None for pid in game.participants}
    game._resolve_meeting()
    assert task.work==.25 and not game.is_terminal
    assert all(tuple(a.tasks)==before[pid] for pid,a in game.players.items())
    game.assert_invariants()


def test_taskbar_modes_hide_global_progress(profile):
    s=profile.settings
    for mode in ('never','meetings','always'):
        p=HostProfile(replace(s,tasks=replace(s.tasks,task_bar_updates=mode)))
        game=game_class()(9410010,experiment=experiment(p));pid=next(iter(game.players))
        task=next(iter(game.tasks.values()));task.stage=len(task.plan.stages)
        bar=game.observe_pj(pid).task_bar
        assert bar==((0,0) if mode=='never' else (0,40) if mode=='meetings' else (1,40))
        if mode=='meetings':
            game._start_meeting(game.players[pid]);assert game.observe_pj(pid).task_bar==(1,40)


def test_anonymous_ballots_never_publish_voter_target_pairs(profile):
    p=custom(profile,anonymous_votes=True);game=game_class()(9410010,experiment=experiment(p))
    pid=next(iter(game.players));game._start_meeting(game.players[pid]);game.phase=Phase.VOTING
    game.votes={who:None for who in game.participants};game._resolve_meeting()
    assert not any(type(e.payload) is Vote for e in game.publications)
    assert any(e['kind']=='anonymous_vote' for e in game.event_log)


def test_zero_discussion_goes_directly_to_voting(profile):
    p=custom(profile,discussion_seconds=0)
    game=game_class()(9410010,experiment=experiment(p))
    game._start_meeting(next(iter(game.players.values())))
    assert game.phase is Phase.VOTING


def test_task_allocation_is_deterministic_and_ghost_progress_stays_owned(profile):
    config=experiment(custom(profile,4,1,(2,3,5)))
    first=game_class()(9410010,experiment=config);second=game_class()(9410010,experiment=config)
    assert [(p,[t.plan for t in a.tasks]) for p,a in first.players.items()]==[(p,[t.plan for t in a.tasks]) for p,a in second.players.items()]
    owner=next(p for p,a in first.players.items() if a.role is Role.CREWMATE)
    first.players[owner].status='dead';first._make_ghost(first.players[owner])
    assert owner in first.ghosts
    assert all(t.owner_id==owner for t in first.players[owner].tasks)
    first.assert_invariants()
