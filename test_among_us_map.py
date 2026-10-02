"""Physical fidelity invariants for the separate source-backed Skeld environment."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from shapely.geometry import Point, Polygon

from among_us_map import ASSET_DIR, ROOMS, AmongUsMap, get_map
from among_us_map_simulation import AmongUsMapEnv
from among_us_renderer import MapRenderer


@pytest.fixture(scope='module')
def world(): return get_map()


def test_original_map_preserved_byte_for_byte():
    root=Path(__file__).parent
    assert (root/'skeld_config.py').read_bytes()==(root/'skeld_config_legacy.py').read_bytes()


def test_sources_match_blueprint_and_no_unsupported_solid_paths(world):
    for name,digest in world.data['source_hashes'].items():
        assert hashlib.sha256((ASSET_DIR/'sources'/name).read_bytes()).hexdigest()==digest
    for group in ('walls','obstacles','regions','doors','visibility'):
        assert all(r['geometry_supported'] for r in world.data[group])
    assert len(world.data['doors'])==13
    assert len(world.data['vents'])==14
    assert len(world.data['cameras'])==4
    assert sum('/Table' in r['name'] for r in world.data['obstacles'])==5
    assert sum('medBay_bed' in r['name'] for r in world.data['obstacles'])==4


def test_all_rooms_and_station_nodes_physically_connected(world):
    assert world.connected_grid_count()==len(world.nodes)
    assert set(ROOMS)<={n for n,_ in world.regions}
    for name,region in world.regions:
        if name in ROOMS: assert region.intersection(world.center_domain).area>1,name
    for dest in world.destinations:
        assert world.contains(dest.standing),dest.id
        assert math.dist(dest.position,dest.standing)<=dest.radius,dest.id


def test_vent_surfaces_are_walkable_and_network_has_six_components(world):
    vents=world.data['vents']; seen=set(); count=0
    for key,v in vents.items():
        p=v['position']; position=(p['x'],p['y'])
        assert world.contains(position),key
        # A movement straight across the marker is not blocked by the vent itself.
        a=(p['x']-.04,p['y']); b=(p['x']+.04,p['y'])
        assert world.segment_clear(a,b)
        assert math.dist(world.move(a,(.08,0))[0],b)<1e-8
        if key in seen: continue
        count+=1; queue=[key]
        while queue:
            current=queue.pop()
            if current in seen:continue
            seen.add(current)
            for direction in ('left','center','right'):
                nxt=vents[current][direction]
                if nxt is not None:
                    assert int(current) in [vents[str(nxt)][k] for k in ('left','center','right')]
                    queue.append(str(nxt))
    assert count==6


def test_multistage_tasks_have_every_physical_station(world):
    by_name = {}
    for d in world.destinations:
        by_name.setdefault(d.name, set()).add(d.room)
    assert by_name['DownloadData'] == {'Cafeteria','Weapons','Navigation','Electrical','Communications'}
    assert by_name['UploadData'] == {'Admin'}
    assert by_name['FuelEngines'] == {'Storage','Upper Engine','Lower Engine'}
    assert by_name['AcceptDivertedPower'] == {'Upper Engine','Lower Engine','Weapons','Navigation',
        'Shields','Communications','O2','Security'}
    assert by_name['EmptyGarbage'] == {'Cafeteria'}
    assert by_name['EmptyChute'] == {'O2'}
    assert by_name['EmptyGarbage/EmptyChute'] == {'Storage'}
    assert by_name['ChartCourse'] == {'Navigation'}
    for d in world.destinations:
        if d.room == 'Communications':
            assert world.region(d.standing) == 'Communications',d.id


def test_tables_engines_crates_and_beds_are_solid(world):
    for record in world.data['obstacles']:
        if record['closed']:
            inside=Polygon(record['points']).representative_point()
            assert not world.contains(inside.coords[0]),record['name']
    for point in [(-28,0),(20,5),(0,7.1),(-13,-8),(-4,-6),(3,-12.5)]:
        # Last point is a valid corridor, a positive control for the classifier.
        assert world.contains(point)==(point==(3,-12.5)),point


def test_large_moves_cannot_tunnel_and_rays_see_blocking_surface(world):
    start=(-.7,-2.8)
    end,hit=world.move(start,(0,30))
    assert hit and world.contains(end)
    assert world.segment_clear(start,end)
    assert end[1]<-.2  # blocked by the emergency table
    ray=world.raycast(start,(0,1),30)
    assert 0<ray<3
    assert abs(math.dist(start,end)+world.radius-ray)<.08


def test_no_direct_electrical_medbay_security_shortcuts(world):
    locations={room: next(d.standing for d in world.destinations if d.room==room)
               for room in ('Electrical','MedBay','Security')}
    for a,b in [('Electrical','MedBay'),('MedBay','Security'),('Security','Electrical')]:
        assert not world.segment_clear(locations[a],locations[b])


def test_every_interaction_route_executes_with_clearance(world):
    origin=(-.7,-2.8)
    for dest in world.destinations:
        route=world.astar(origin,dest.standing)
        assert route,dest.id
        pos=origin
        for waypoint in route[1:]:
            delta=np.array(waypoint)-pos
            pos,hit=world.move(pos,delta)
            assert not hit,(dest.id,waypoint)
            assert math.dist(pos,waypoint)<1e-8
        assert math.dist(pos,dest.standing)<1e-8
        assert Point(pos).distance(world.barriers)>=world.radius-1e-6


def test_transforms_are_uniform_and_reversible(world):
    renderer=MapRenderer(world,(1100,700))
    for point in [(-20,0),(0,0),(15,-10),(-.7,-2.8)]:
        assert np.allclose(renderer.screen_to_world(renderer.world_to_screen(point)),point,atol=1e-12)
    origin=np.array(renderer.world_to_screen((0,0)))
    assert math.isclose(np.linalg.norm(np.array(renderer.world_to_screen((1,0)))-origin),
                        np.linalg.norm(np.array(renderer.world_to_screen((0,1)))-origin))


@pytest.mark.parametrize('mode',['task','task_to_task','room_to_room'])
def test_seeded_resets_and_legacy_observation_action_layout(mode):
    a=AmongUsMapEnv(goal_mode=mode); b=AmongUsMapEnv(goal_mode=mode)
    obs,info=a.reset(seed=312); obs2,info2=b.reset(seed=312)
    assert np.array_equal(obs,obs2) and info==info2
    assert a.observation_space.shape==(22,) and a.action_space.shape==(2,)
    if mode == 'room_to_room':
        assert a.map.region(a.position) in ROOMS and a.map.region(a.goal) in ROOMS
        assert a.map.region(a.position) != a.map.region(a.goal)
    for action in [np.zeros(2),np.array([1,0]),np.array([0,1])]:
        oa,ra,ta,ua,ia=a.step(action); ob,rb,tb,ub,ib=b.step(action)
        assert np.array_equal(oa,ob) and (ra,ta,ua,ia)==(rb,tb,ub,ib)
        assert a.observation_space.contains(oa)


def test_positive_action_y_moves_down_and_timeout():
    env=AmongUsMapEnv(max_episode_time=.01)
    env.reset(seed=0,options={'start':(-.7,-2.8),'goal':(-.7,-3.5)})
    _,_,terminated,truncated,_=env.step([0,1])
    assert env.position[1]<-2.8 and truncated and not terminated
    with pytest.raises(RuntimeError):env.step([0,0])
    with pytest.raises(ValueError):env.reset(options={'start':(100,100)})


def test_rgb_renderer_and_gym_contract():
    env=AmongUsMapEnv(render_mode='rgb_array',render_size=(880,600))
    check_env(env,skip_render_check=True)
    image=env.render()
    assert image.shape==(600,880,3) and image.dtype==np.uint8
    assert image.std()>20
    env.close()

