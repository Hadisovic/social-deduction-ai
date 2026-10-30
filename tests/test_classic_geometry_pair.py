"""Real native geometry regression; inference/training/final partitions unused."""
import os
from pathlib import Path
from itertools import combinations,islice,chain
import pytest

SOURCE=os.environ.get('JANE_UPSTREAM_REVIEW_ROOT')
if not SOURCE:pytest.skip('Explicit pinned source required',allow_module_level=True)
from jane.impostor.v1.bootstrap import activate
activate(SOURCE)
from jane.classic_v2.engine import game_class,experiment
from jane.classic_v2.settings import load_profile
from patrick_jane.a2_geometry import geometry,planner
from social_deduction.actor import Phase,Role

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def game():
    geometry.cache_clear();planner.cache_clear()
    profile=load_profile(ROOT/'jane/configs/classic-v2/default_12p.json')
    result=game_class()(9700031,experiment=experiment(profile));result.tick=200
    yield result
    geometry.cache_clear();planner.cache_clear()


def close_storage(game):
    owner=next(pid for pid,p in game.players.items() if p.role is Role.IMPOSTOR)
    assert game.close_doors(owner,'Storage')
    assert game.map.closed_doors and game.map is game.planner.map
    return owner


def coherent(game):
    assert game.map is game.planner.map
    assert all(p.motion.map is game.map for p in game.players.values())
    for pid in game.players:game.observe_pj(pid)
    game.assert_invariants()


def test_geometry_only_eviction_keeps_projection_and_physics_on_planner_map(game):
    close_storage(game);old=game.planner
    # Populate cached observations before the change; repair must invalidate them.
    for p in game.players:game.observe_pj(p)
    assert game._observation_cache
    geometry.cache_clear();game._update_doors()
    assert game.planner is old
    assert not game._observation_cache
    coherent(game)


def test_actual_lru_pressure_cannot_separate_collision_and_projection(game):
    close_storage(game);old=game.planner
    keys=tuple(sorted(game.door_until));ids=tuple(d['id'] for d in game.base_map.data['doors'])
    states=(tuple(sorted(s)) for s in chain(combinations(ids,1),combinations(ids,2)))
    others=list(islice((s for s in states if s!=keys),17))
    assert len(others)==17
    for closed in others:geometry(closed)
    assert geometry.cache_info().currsize==16
    # Explicitly demonstrate eviction of the live planner's geometry instance.
    assert geometry(keys) is not old.map
    game._update_doors()
    coherent(game)
    assert game.planner is old


@pytest.mark.parametrize('evict',('planner','both'))
def test_planner_or_both_caches_can_be_replaced_without_stale_physics(game,evict):
    close_storage(game)
    planner.cache_clear()
    if evict=='both':geometry.cache_clear()
    game._update_doors();coherent(game)


def test_expiry_and_meeting_reset_keep_maps_coherent(game):
    owner=close_storage(game)
    geometry.cache_clear();game._update_doors();coherent(game)
    game.tick+=int(game.experiment.doors_duration/game.config.dt)+1
    game._update_doors();coherent(game);assert not game.map.closed_doors
    game.tick+=int(game.experiment.doors_cooldown/game.config.dt)+1
    assert game.close_doors(owner,'Storage')
    geometry.cache_clear();game._update_doors();coherent(game)
    reporter=next(p for p in game.players.values() if p.role is Role.CREWMATE)
    game._start_meeting(reporter);coherent(game)
    assert not game.map.closed_doors
    game.phase=Phase.VOTING;game.votes={pid:None for pid in game.participants}
    game._resolve_meeting();coherent(game)


def test_eviction_preserves_dead_and_vented_position_handling(game):
    owner=next(pid for pid,p in game.players.items() if p.role is Role.IMPOSTOR)
    dead=next(p for p in game.players.values() if p.role is Role.CREWMATE)
    dead.motion.spawn((-.20165175055810952,-8.95852724275098));dead.status='dead'
    # A legal underground actor retains its own stored position and private view.
    vent=next(iter(game.vents));game.players[owner].motion.spawn(game.vents[vent]['position'])
    assert game.vent_action(owner,'vent_enter',vent)
    other=next(pid for pid,p in game.players.items() if p.role is Role.IMPOSTOR and pid!=owner)
    assert game.close_doors(other,'Storage')
    geometry.cache_clear();game._update_doors();coherent(game)
    assert not game.map.contains(dead.position)
    assert not game.observe_pj(owner).actor.visible_players


def test_equivalent_repair_does_not_disable_real_projection_guard(game):
    close_storage(game)
    game.planner=planner(())
    game._invalidate()
    with pytest.raises(ValueError,match='Visibility and interaction geometry'):
        game.observe_pj(next(iter(game.players)))


def test_deterministic_door_movement_matches_pre_fix_when_caches_are_coherent():
    from patrick_jane.a2_environment import A2Game
    from social_deduction.phase3_api import Intent,IntentKind
    profile=load_profile(ROOT/'jane/configs/classic-v2/default_12p.json')
    Fixed=game_class()
    class PreFix(Fixed):
        _update_doors=A2Game._update_doors
    def trace(cls):
        geometry.cache_clear();planner.cache_clear()
        game=cls(9700031,experiment=experiment(profile));game.tick=200
        owner=next(pid for pid,p in game.players.items() if p.role is Role.IMPOSTOR)
        frames=[]
        for step in range(60):
            if step==3:assert game.close_doors(owner,'Storage')
            actions={owner:Intent(IntentKind.GO_TO_ROOM,'Storage')} if step==0 else {}
            game.step(actions)
            frames.append((game.tick,game.phase.value,tuple(sorted(game.map.closed_doors)),
                tuple((p.player_id,p.position,p.status,p.cooldown) for p in game.players.values())))
        return frames
    assert trace(Fixed)==trace(PreFix)
