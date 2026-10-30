"""Reproduce the failed native door invariant from saved state; never trains."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--partial-replay',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from jane.impostor.v1.bootstrap import activate,experiment
    activate(args.source_root)
    from patrick_jane.a2_environment import A2Game
    from patrick_jane.a2_geometry import geometry,planner
    header=state=snapshot=None;requests=[]
    with args.partial_replay.open(encoding='utf-8') as handle:
        for line in handle:
            try:r=json.loads(line)
            except ValueError:break
            if r['record']=='header':header=r
            elif r['record']=='tick':snapshot=r
            elif r['record']=='event' and r.get('kind')=='a2_world_state':state=r
            elif r['record']=='neural' and r['action_kind']=='doors':requests.append(r)
    if not header or not snapshot or not state:raise ValueError('Saved diagnostic state missing')
    request=next(r for r in reversed(requests) if r['tick']>=snapshot['tick'])
    settings,config=experiment(ROOT/'jane/configs/patrick_jane_a2_12p.json')
    game=A2Game(header['scenario_seed'],experiment=config)
    game.tick=snapshot['tick']
    for pid,record in snapshot['players'].items():
        game.players[pid].motion.spawn(record['position'])
        game.players[pid].status=record['status']
    game.door_until=dict(state['doors']);game.door_pending=set()
    game.map=geometry(tuple(sorted(game.door_until)));game.planner=planner(tuple(sorted(game.door_until)))
    for player in game.players.values():player.motion.map=game.map
    before=[pid for pid,p in game.players.items() if not game.map.contains(p.position)]
    legal=game.can_close(request['player_id'],request['action_target'])
    accepted=game.close_doors(request['player_id'],request['action_target'])
    outside=[dict(player=pid,status=p.status,position=list(p.position),
                  base_map_contains=game.base_map.contains(p.position),
                  current_door_map_contains=game.map.contains(p.position))
             for pid,p in game.players.items() if not game.map.contains(p.position)]
    failed=False
    try:game.assert_invariants()
    except AssertionError:failed=True
    result=dict(source_commit='d459f38674ebc537acde20647cbd07700c494533',
        scenario=header['scenario_seed'],snapshot_tick=snapshot['tick'],snapshot_seconds=snapshot['sim_time'],
        submitted_by=request['player_id'],room=request['action_target'],preexisting_outside=before,
        can_close=legal,accepted=accepted,outside_after_closure=outside,invariant_reproduced=failed,
        all_living_players_inside=all(game.map.contains(p.position) for p in game.players.values() if p.active),
        inference_calls=0,optimizer_steps=0,
        scope='Minimal geometry fixture from saved positions/statuses/doors, not a replayed match outcome',
        proposed_fix='Check inactive stored positions against base floor; living non-vented actors against current collision geometry. Keep physics and invariants enabled.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not (legal and accepted and failed and not before and result['all_living_players_inside']):
        raise RuntimeError('Expected narrow diagnostic was not reproduced')


if __name__=='__main__':main()
