"""Bounded engineering, training or development evaluation of the new impostor."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import math
import subprocess
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = None


def atomic(path, value):
    from patrick_jane.progress import atomic_json
    atomic_json(Path(path), value)


def tensor_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode()); h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def save_checkpoint(path, model, optimizer, settings, counters, seed):
    import torch
    from . import SCHEMA
    from .bootstrap import source_lock
    from jane.shared.settings import bundle_receipt
    record = dict(schema=SCHEMA, architecture=model.architecture(), sources=source_lock(),
                  runtime=dict(python=platform.python_version(),torch=str(torch.__version__)),
                  settings_sha256=settings.fingerprint(), state_dict=model.state_dict(),
                  optimizer=optimizer.state_dict() if optimizer else None,counters=counters,initialization_seed=seed,
                  boundary='complete-matches',crew_update=102,shared_bundle=bundle_receipt(settings)['bundle_sha256'])
    with Path(path).open('xb') as handle: torch.save(record,handle)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_checkpoint(path, model, settings):
    import torch
    from . import SCHEMA
    from .bootstrap import source_lock
    from jane.shared.settings import bundle_receipt
    record = torch.load(path,map_location='cpu',weights_only=True)
    if (record['schema']!=SCHEMA or record['architecture']!=model.architecture() or
            record['sources']!=source_lock() or record['settings_sha256']!=settings.fingerprint() or
            record['runtime']!=dict(python=platform.python_version(),torch=str(torch.__version__)) or
            record['shared_bundle']!=bundle_receipt(settings)['bundle_sha256']):
        raise ValueError('Impostor checkpoint source/schema/runtime/settings mismatch')
    model.load_state_dict(record['state_dict'],strict=True)
    return record


def main(argv=None):
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--profile',type=Path,default=ROOT/'jane/configs/patrick_jane_a2_12p.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('smoke','train','evaluate'),default='smoke')
    parser.add_argument('--seed',type=int,default=731,help='New neural initialization seed')
    parser.add_argument('--episodes',type=int,default=8)
    parser.add_argument('--max-minutes',type=float,default=30)
    parser.add_argument('--match-wall-seconds',type=float,default=300)
    parser.add_argument('--smoke-observations',type=int,default=120)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--deterministic',action='store_true')
    args = parser.parse_args(argv)
    if not 1<=args.episodes<=1000 or not all(math.isfinite(v) and v>0 for v in (args.max_minutes,args.match_wall_seconds)):
        parser.error('Positive budgets required')
    if args.checkpoint and args.mode=='train':parser.error('First study trains from scratch; checkpoint input is evaluation/smoke only')
    from .bootstrap import activate,experiment,frozen_crew,source_lock
    source = activate(args.source_root)
    import torch
    import numpy as np
    torch.set_num_threads(1)
    from .model import ImpostorPolicy
    from .session import run_match
    from .ppo import update
    settings, native = experiment(args.profile)
    if (settings.crewmates,settings.lobby.impostors)!=(10,2):
        raise ValueError('First learning study is registered for 10 crew + 2 impostors only')
    args.output.mkdir(parents=True,exist_ok=False)
    OUTPUT = args.output
    started = perf_counter(); model = ImpostorPolicy(args.seed); model.eval()
    crew = frozen_crew(source); crew_hash = tensor_hash(crew)
    optimizer = torch.optim.Adam(model.parameters(),lr=3e-4,eps=1e-5)
    loaded=load_checkpoint(args.checkpoint,model,settings) if args.checkpoint else None
    initial_tensor_hash = tensor_hash(model)
    counters = dict(complete_matches=0,technical_cuts=0,training_observations=0,optimizer_steps=0)
    initial_hash = save_checkpoint(args.output/'initial.pt',model,None,settings,counters,args.seed)
    registration = dict(schema='jane-impostor-study-v1',mode=args.mode,created_utc=datetime.now(timezone.utc).isoformat(),
        architecture=model.architecture(),sources=source_lock(),settings=settings.to_dict(),settings_sha256=settings.fingerprint(),
        initialization_seed=args.seed,initial_sha256=initial_hash,crew_inference_sha256='da1d43d73c140ac39bf44834746dbae605b9b36d3c6c7af8a4f2be975e535e3c',
        frozen_crew_tensor_hash=crew_hash,source_implementation='d459f38674ebc537acde20647cbd07700c494533',
        runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__),device='cpu'),
        budget=dict(episodes=args.episodes,max_minutes=args.max_minutes,match_wall_seconds=args.match_wall_seconds),
        rewards=dict(team_win=10,team_loss=-10,team_kill=1,cooldown_ready_per_second=-.005,
                     ejection_extra=0,fake_bonus=0,sabotage_bonus=0,pursuit_shaping=0),
        seed_domains=dict(engineering=[9300001,9301000],training=[9310001,9311000],
                          development=[9320001,9321000],reserved_final='9330001..9331000 NEVER OPENED'),
        eligibility='Natural complete games only for PPO; technical cuts preserved and excluded',
        shared_bundle=__import__('jane.shared.settings',fromlist=['bundle_receipt']).bundle_receipt(settings),
        checkpoint_input=str(args.checkpoint) if args.checkpoint else None,
        checkpoint_input_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
        checkpoint_input_counters=loaded['counters'] if loaded else None,
        original_native_script_mixture='Constructed but replaced before first decision; zero script actions',
        interpretation='New initialization and new role interface; not continuation of old 75K models')
    atomic(args.output/'registration.json',registration)
    rows=[]; updates=[]
    status_path=args.output/'status.json'
    def status(value):
        atomic(status_path,{**value,'phase':'collecting','mode':args.mode,'counters':dict(counters),
                            'monitor_dir':str((args.output/'live').resolve()),'worker_count':1,
                            'total_wall_seconds':perf_counter()-started})
    for episode in range(1,args.episodes+1):
        remaining=args.max_minutes*60-(perf_counter()-started)
        if remaining<=1: break
        seed=(9300000 if args.mode=='smoke' else 9310000 if args.mode=='train' else 9320000)+episode
        replay=args.output/f'match-{episode:03d}.jsonl'
        summary,trajectories=run_match(model,crew,settings,native,seed,output=replay,
            max_wall_seconds=min(remaining,args.match_wall_seconds),
            max_observations=args.smoke_observations if args.mode=='smoke' else None,
            deterministic=args.deterministic,status=status,live_dir=args.output/'live',
            policy_update=loaded['counters'].get('optimizer_steps',0) if loaded else counters['optimizer_steps'],mode=args.mode)
        summary.update(episode=episode,mode=args.mode)
        natural=summary['winner'] in ('crewmate','impostor') and not summary['truncated']
        if natural: counters['complete_matches']+=1
        else: counters['technical_cuts']+=1
        if args.mode=='train' and natural:
            atomic(status_path,dict(phase='optimizing',episode=episode,counters=counters,wall_seconds=perf_counter()-started,
                monitor_dir=str((args.output/'live').resolve()),worker_count=1))
            optimization=perf_counter(); metrics=update(model,optimizer,trajectories,summary['simulation_seconds'],
                deadline=started+args.max_minutes*60)
            counters['optimizer_steps']+=len(metrics)
            counters['training_observations']+=summary['impostor_observations']
            updates.extend([dict(episode=episode,**m) for m in metrics])
            summary['optimization_seconds']=perf_counter()-optimization
            summary['optimization_budget_cut']=len(metrics)<4*sum(bool(t) for t in trajectories)
            save_checkpoint(args.output/f'checkpoint-{episode:03d}.pt',model,optimizer,settings,counters,args.seed)
        summary['counters']=dict(counters);rows.append(summary)
        with (args.output/'matches.jsonl').open('a',encoding='utf-8') as handle: handle.write(json.dumps(summary)+'\n')
        atomic(args.output/'ppo.json',updates)
        print(json.dumps(dict(episode=episode,winner=summary['winner'],reason=summary['reason'],
                             simulation_seconds=summary['simulation_seconds'],wall_seconds=summary['wall_seconds'],
                             observations=summary['impostor_observations'],counters=counters)),flush=True)
    if tensor_hash(crew)!=crew_hash: raise ValueError('Frozen crewmate tensors changed')
    final_hash=save_checkpoint(args.output/'final.pt',model,optimizer,settings,counters,args.seed)
    completed=[r for r in rows if not r['truncated'] and r['winner']]
    result=dict(mode=args.mode,counters=counters,matches=rows,wall_seconds=perf_counter()-started,
        wins=sum(r['winner']=='impostor' for r in completed),games=len(completed),
        frozen_crew_unchanged=True,final_sha256=final_hash,initial_sha256=initial_hash,
        interpretation='training outcomes' if args.mode=='train' else 'engineering cut, not a win-rate study' if args.mode=='smoke' else 'development evaluation, not reserved final',
        model_changed=tensor_hash(model)!=initial_tensor_hash)
    atomic(args.output/'results.json',result)
    atomic(status_path,dict(phase='finished',mode=args.mode,counters=counters,wall_seconds=result['wall_seconds'],games=result['games'],wins=result['wins'],
        monitor_dir=str((args.output/'live').resolve()),worker_count=1))
    print(json.dumps(dict(finished=True,output=str(args.output),counters=counters,wall_seconds=result['wall_seconds'])),flush=True)
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        if OUTPUT is not None:
            (OUTPUT/'failure.json').write_text(json.dumps(dict(phase='failed',error=type(error).__name__,
                message=str(error),training_success_claimed=False),indent=2)+'\n',encoding='utf-8')
        raise
