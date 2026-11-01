"""Bounded development-only collection, frozen update102, never PPO.

Optional original-package mode captures the old worker's world on a failure
without altering that preserved package. No final/validation partitions used.
"""
import argparse,json,os,sys,time,traceback
from pathlib import Path


def original_serve(connection,source):
    sys.path.insert(0,source)
    from patrick_jane.worker import Match
    match=None
    while True:
        try:
            command,payload=connection.recv();began=time.perf_counter()
            if command=='close':break
            if command=='reset':match=Match(**payload);value=match.prepare()
            else:value=match.advance(payload)
            value['worker_simulation_seconds']=time.perf_counter()-began
            connection.send(('ok',value))
        except EOFError:break
        except BaseException:
            detail={}
            if match is not None:
                g=match.game
                detail=dict(seed=g.seed,tick=g.tick,phase=g.phase.value,closed_doors=list(g.map.closed_doors),
                    actors=[dict(player=pid,status=p.status,position=p.position,vented=pid in g.vented,
                        dynamic_valid=g.map.contains(p.position),base_valid=g.base_map.contains(p.position)) for pid,p in g.players.items()])
            connection.send(('error',traceback.format_exc()+'\nINTERNAL WORLD DIAGNOSTIC '+json.dumps(detail,default=str)));break
    connection.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--folder',required=True);parser.add_argument('--original-package');parser.add_argument('--seconds',type=float,default=570)
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--seed-start',type=int,default=3200000)
    parser.add_argument('--record-every',type=int,default=1)
    a=parser.parse_args();root=Path(__file__).resolve().parents[1];source=Path(a.original_package).resolve() if a.original_package else root
    sys.path.insert(0,str(source));folder=Path(a.folder).resolve();folder.mkdir(parents=True,exist_ok=False)
    os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['PYTHONHASHSEED']='0'
    os.environ['OMP_NUM_THREADS']='1';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch,numpy as np,multiprocessing as mp
    from patrick_jane.collector import Pool,collect
    from patrick_jane.a2_config import A2Experiment
    from patrick_jane.network import PatrickJane
    from patrick_jane.a2_transition import import_update102
    from patrick_jane.progress import atomic_json
    origin=source/'origin' if a.original_package else root/'artifacts/patrick-jane/task-pursuit-v1-update102'
    checkpoint=origin/'update-000102.pt' if a.original_package else origin/'checkpoint.pt'
    if a.checkpoint:checkpoint=a.checkpoint.resolve()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    policy=PatrickJane();record=torch.load(checkpoint,map_location='cpu',weights_only=True)
    policy.load_state_dict(record['state_dict'],strict=True);policy.to('cuda');policy.eval()
    profile=A2Experiment() if a.original_package else A2Experiment(shared_engine_profile={'engine':'paired-map-planner-v1','allocation':'patrick-jane-native-v1'})
    result=dict(schema='shared-a2-readiness-v1',seeds=list(range(a.seed_start,a.seed_start+6)),development_only=True,training_updates=0,
        source=str(source),checkpoint_sha256=__import__('hashlib').sha256(checkpoint.read_bytes()).hexdigest(),workers=6,wall_ceiling=a.seconds)
    began=time.perf_counter()
    pool=Pool(6,processes=not a.original_package)
    if a.original_package:
        pool.processes=True;ctx=mp.get_context('spawn')
        for _ in range(6):
            parent,child=ctx.Pipe();p=ctx.Process(target=original_serve,args=(child,str(source)),daemon=True);p.start();child.close();pool.connections.append(parent);pool.children.append(p)
    try:
        rollout=collect(policy,pool,profile,result['seeds'],device='cuda',wall_limit=a.seconds,update=record['counters']['update'],
            checkpoint=checkpoint.name+'/frozen-diagnostic',training_step=record['counters']['individual_decisions'],
            replay_dir=folder/'replays',record_every=a.record_every,monitor_dir=folder/'live',
            status_callback=lambda s:(atomic_json(folder/'progress.json',s),print(json.dumps(s),flush=True)))
        result.update(outcome='completed',**{k:v for k,v in rollout.items() if k!='trajectories'})
        if not a.original_package:
            from patrick_jane.learning_run import audit_lights
            audit_lights(rollout)
        if any(m.get('truncated') for m in rollout['matches']):raise RuntimeError('Incomplete natural-game coverage blocks readiness')
    except BaseException as exc:result.update(outcome='failed',error=repr(exc),traceback=traceback.format_exc())
    finally:
        pool.close();result['elapsed_wall_seconds']=time.perf_counter()-began
        result['policy_tensors_unchanged']=sum(torch.equal(v.cpu(),record['state_dict'][k]) for k,v in policy.state_dict().items())
        atomic_json(folder/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('matches','traceback','reward_components')},default=str),flush=True)
    return 0 if result['outcome']=='completed' and result['policy_tensors_unchanged']==52 else 1


if __name__=='__main__':raise SystemExit(main())
