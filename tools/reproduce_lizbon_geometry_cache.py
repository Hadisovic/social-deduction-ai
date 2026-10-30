"""Controlled cache-eviction reproduction; no model inference or training."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def reproduce(source):
    from jane.impostor.v1.bootstrap import activate
    activate(source)
    from jane.classic_v2.settings import load_profile
    from jane.classic_v2.engine import game_class,experiment
    from patrick_jane.a2_geometry import geometry,planner
    profile=load_profile(ROOT/'jane/configs/classic-v2/default_12p.json')
    game=game_class()(9700030,experiment=experiment(profile))
    owner=next(pid for pid,p in game.players.items() if p.role.value=='impostor')
    game.tick=200
    assert game.close_doors(owner,'Storage')
    assert game.map is game.planner.map
    # Stand in for geometry-only LRU eviction. The independent planner cache
    # retains its old map. This is a minimal mechanism test, not a full replay
    # of every action/cache entry in the failed training process.
    geometry.cache_clear()
    game._update_doors()
    mismatch=game.map is not game.planner.map
    equivalent=game.map.center_domain.equals(game.planner.map.center_domain)
    error=None
    try:game.observe_pj(owner)
    except ValueError as exc:error=str(exc)
    assert mismatch and equivalent and error=='Visibility and interaction geometry must share the same public map'
    # Proposed coherent-pair repair in memory only; source on disk stays pinned.
    game.map=game.planner.map
    for player in game.players.values():player.motion.map=game.map
    game._invalidate()
    obs=game.observe_pj(owner)
    assert obs.actor.own.player_id==owner
    game.assert_invariants()
    return dict(reproduction='deliberate geometry-only cache eviction, no NN/PPO',
        mismatch_reproduced=mismatch,equivalent_collision_domains=equivalent,error=error,
        proposed_pairing_passed=True,upstream_source_modified=False,
        exact_failed_process_cache_contents_recorded=False,
        geometry_cache=str(geometry.cache_info()),planner_cache=str(planner.cache_info()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output',type=Path);args=p.parse_args();result=reproduce(args.source_root)
    if args.output:
        with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
