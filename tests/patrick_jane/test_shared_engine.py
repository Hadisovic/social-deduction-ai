"""Shared repair regressions; no optimizer updates or reserved scenarios."""
import pickle
from dataclasses import replace
import numpy as np
import pytest
from patrick_jane.a2_config import A2Experiment
from patrick_jane.worker import Match
from social_deduction.actor import Role

PROFILE={'engine':'paired-map-planner-v1','allocation':'patrick-jane-native-v1'}


@pytest.mark.parametrize('actor_id',('p0','p10'))
def test_navigation_target_closed_by_same_batch_door_action(actor_id):
    from social_deduction.actor import Point
    from social_deduction.phase3_api import Intent,IntentKind
    from patrick_jane.api import Action
    g=Match(3200016,A2Experiment(shared_engine_profile=PROFILE)).game
    g.tick=231;g.players[actor_id].motion.spawn((-9.173630452496454,-12.699789577560162))
    g.players['p1'].motion.spawn((-9.120000000000289,-12.300000000000107))
    target=(-9.720000000000276,-13.680000000000078);g._invalidate()
    assert g.base_map.contains(target) and g.map.contains(target)
    g.step({actor_id:Intent(IntentKind.GO_TO_LOCATION,position=Point(*target)), 'p1':Action('doors','Electrical')})
    assert not g.map.contains(target) and g.base_map.contains(target)
    assert g.metrics['illegal_actions']==0
    assert g.players[actor_id].navigator.target is not None
    assert g.map.contains(g.players[actor_id].position)


def test_permanently_invalid_navigation_target_still_rejected():
    from social_deduction.actor import Point
    from social_deduction.phase3_api import Intent,IntentKind
    from patrick_jane.api import Action
    g=Match(3200016,A2Experiment(shared_engine_profile=PROFILE)).game;g.tick=231
    g.step({'p0':Intent(IntentKind.GO_TO_LOCATION,position=Point(-100.,-100.)), 'p1':Action('doors','Electrical')})
    assert g.metrics['illegal_actions']==1


def test_update103_version_bridge_preserves_weights_adam_and_skips_failed_wave(tmp_path):
    import shutil,torch
    from pathlib import Path
    from tools.prepare_pj_a2_repair import import_parent
    from patrick_jane.network import PatrickJane
    parent=Path(__file__).resolve().parents[2]/'dist/PatrickJane-A2-SharedRepair-V1/results/investigation-a2-v1'
    if not parent.exists():pytest.skip('Private preserved update103 run is unavailable')
    shutil.copyfile(parent/'checkpoints/attempt-0002-update-000103.pt',tmp_path/'update-000103.pt')
    shutil.copyfile(parent/'source-snapshot.zip',tmp_path/'source-snapshot.zip')
    p=PatrickJane();opt=torch.optim.Adam(p.parameters(),lr=.0003)
    c,info=import_parent(tmp_path,p,opt,np.random.default_rng(17))
    original=torch.load(tmp_path/'update-000103.pt',map_location='cpu',weights_only=True)
    assert len(p.state_dict())==52
    assert all(torch.equal(v.cpu(),original['state_dict'][k]) for k,v in p.state_dict().items())
    adam=[torch.equal(v.cpu(),original['optimizer']['state'][k][n].cpu()) for k,d in opt.state_dict()['state'].items() for n,v in d.items()]
    assert len(adam)==156 and all(adam)
    assert c['individual_decisions']==2829668 and c['update']==103 and c['next_seed']==18
    assert (c['run_decisions'],c['run_matches'],c['run_updates'])==(15789,6,1)
    assert info['parent_active_seconds']==pytest.approx(578.8447452)
    assert info['parent_checkpointed_new_decisions']==15789
    with (tmp_path/'update-000103.pt').open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='hash'):import_parent(tmp_path,p,opt,np.random.default_rng(17))

def test_spawn_safe_profile_and_native_task_equivalence():
    old=A2Experiment();new=replace(old,shared_engine_profile=PROFILE)
    new=pickle.loads(pickle.dumps(new))
    a=Match(2920010,old);b=Match(2920010,new)
    x=a.game.experiment.to_dict();y=b.game.experiment.to_dict();y['shared_engine_profile']=None
    assert x==y
    for pid in a.game.players:
        assert a.game.players[pid].tasks==b.game.players[pid].tasks
        assert a.game.players[pid].position==b.game.players[pid].position
    a.prepare();b.prepare()
    for pid in a.crew:
        for field in ('self_features','players','events','targets','interaction_mask','navigation_mask','vote_mask'):
            assert np.array_equal(getattr(a.packets[pid],field),getattr(b.packets[pid],field)),field

def test_dead_position_and_cache_eviction_preserve_living_guard():
    m=Match(2920010,A2Experiment(shared_engine_profile=PROFILE));g=m.game
    dead=g.players[m.crew[0]];dead.motion.spawn((-.20165175055810952,-8.95852724275098));dead.status='dead'
    imp=next(p for p in g.players.values() if p.role is Role.IMPOSTOR);g.tick=200
    assert g.close_doors(imp.player_id,'Storage')
    from patrick_jane.a2_geometry import geometry,planner
    geometry.cache_clear();g._update_doors()
    assert g.map is g.planner.map
    assert all(p.motion.map is g.map for p in g.players.values())
    assert not g.map.contains(dead.position);g.assert_invariants()
    import pytest
    dead.status='alive'
    with pytest.raises(AssertionError):g.assert_invariants()

def test_unknown_shared_profile_rejected():
    import pytest
    with pytest.raises(ValueError):A2Experiment(shared_engine_profile={'engine':'unknown','allocation':'patrick-jane-native-v1'})


def test_native_ghost_rewards_reporting_occlusion_and_meeting_vent_transitions():
    from phase3_engine import Body
    from shapely.geometry import Polygon
    from social_deduction.actor import Phase
    from patrick_jane.a2_rewards import A2Ledger
    m=Match(2920010,A2Experiment(shared_engine_profile=PROFILE));g=m.game
    ledger=A2Ledger(g);pid=m.crew[0]
    g._record('task_completed',player_id=pid,task_id='diagnostic',ghost=True)
    assert ledger.collect(g)[pid]['ghost_task']==0
    door=next(d for d in g.base_map.data['doors'] if d['room']=='Electrical')
    cx,cy=Polygon(door['points']).centroid.coords[0]
    victim=m.crew[1];point=(cx,cy+.8);g.players[victim].motion.spawn(point)
    g.players[victim].status='dead';g.bodies[victim]=Body(victim,point,g.tick,'diagnostic')
    g.players[pid].motion.spawn((cx,cy-.8));g._invalidate()
    assert victim in {b.victim_id for b in g.observe_pj(pid).actor.visible_bodies}
    imp=next(p for p in g.players.values() if p.role is Role.IMPOSTOR);g.tick=50
    assert g.close_doors(imp.player_id,'Electrical')
    assert victim not in {b.victim_id for b in g.observe_pj(pid).actor.visible_bodies}
    assert g.bodies[victim].position==point
    vent=next(iter(g.vents));imp.motion.spawn(g.vents[vent]['position'])
    assert g.vent_action(imp.player_id,'vent_enter',vent)
    g.assert_invariants();g._start_meeting(g.players[pid])
    assert not g.vented and not g.map.closed_doors and g.map is g.planner.map
    g.phase=Phase.VOTING;g.votes={p:None for p in g.participants};g._resolve_meeting();g.assert_invariants()
