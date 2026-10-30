"""Explicit bounded Lizbon Classic V2 runs; training is never the default."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from time import perf_counter

ROOT=Path(__file__).resolve().parents[3]


def known_passes(record):
    ancestry=record.get('ancestry') or {}
    return record['counters']['optimizer_steps']+ancestry.get('parent_total_steps',
        ancestry.get('parent_counters',{}).get('optimizer_steps',0))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--profile',type=Path,default=ROOT/'jane/configs/classic-v2/default_12p.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('smoke','train','evaluate'),default='smoke')
    parent=p.add_mutually_exclusive_group()
    parent.add_argument('--warm-start-v1',type=Path)
    parent.add_argument('--transfer-v2',type=Path,help='Explicit weights-only transfer to another host profile; not validated there')
    parent.add_argument('--checkpoint',type=Path,help='Strict same-profile V2 weights')
    p.add_argument('--resume-optimizer',action='store_true')
    p.add_argument('--seed',type=int,default=731)
    p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--scenario-offset',type=int,default=0)
    p.add_argument('--max-minutes',type=float,default=2)
    p.add_argument('--match-wall-seconds',type=float,default=90)
    p.add_argument('--smoke-observations',type=int,default=30)
    p.add_argument('--deterministic',action='store_true')
    args=p.parse_args(argv)
    if not 1<=args.episodes<=1000 or not 0<=args.scenario_offset<=1000-args.episodes:
        p.error('Use a bounded fresh domain of 1000 scenarios')
    if not all(math.isfinite(v) and v>0 for v in (args.max_minutes,args.match_wall_seconds)) or args.smoke_observations<1:
        p.error('Positive budgets required')
    if args.resume_optimizer and (not args.checkpoint or args.mode!='train'):
        p.error('Optimizer resume requires strict same-profile V2 checkpoint and train mode')
    from jane.impostor.v1.bootstrap import activate,frozen_crew
    source=activate(args.source_root)
    import torch
    import numpy as np
    torch.set_num_threads(1)
    from jane.classic_v2.settings import load_profile,HostProfile,receipt
    from jane.impostor.v1.model import ImpostorPolicy
    from jane.impostor.v1.ppo import update
    from jane.impostor.v1.run import atomic,tensor_hash
    from .checkpoint import save,load,warm_start,sources
    from .session import run_match
    profile=load_profile(args.profile);model=ImpostorPolicy(args.seed);model.eval()
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4,eps=1e-5)
    ancestry=None;inherited=0
    if args.warm_start_v1:
        ancestry=warm_start(args.warm_start_v1,model,ROOT/'jane/configs/patrick_jane_a2_12p.json')
        inherited=ancestry['parent_counters']['optimizer_steps']
    elif args.transfer_v2:
        old=torch.load(args.transfer_v2,map_location='cpu',weights_only=True)
        old=load(args.transfer_v2,model,HostProfile.from_dict(old['host_profile']))
        inherited=known_passes(old)
        ancestry=dict(transfer=True,parent_sha256=hashlib.sha256(args.transfer_v2.read_bytes()).hexdigest(),
            parent_profile=old['host_profile'],parent_counters=old['counters'],parent_total_steps=inherited,optimizer='fresh Adam',
            reason='Explicit cross-profile weights transfer; target-profile performance unvalidated')
    elif args.checkpoint:
        old=load(args.checkpoint,model,profile);inherited=known_passes(old)
        if args.resume_optimizer:
            if old['optimizer'] is None:raise ValueError('Checkpoint has no optimizer')
            optimizer.load_state_dict(old['optimizer'])
        ancestry=dict(transfer=not args.resume_optimizer,parent_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            parent_counters=old['counters'],parent_total_steps=inherited,optimizer='resumed' if args.resume_optimizer else 'fresh Adam',
            reason='New registered scenario stream; RNG domains recorded below')
    crew=frozen_crew(source);crew_hash=tensor_hash(crew);initial_tensor=tensor_hash(model)
    args.output.mkdir(parents=True,exist_ok=False)
    began=perf_counter();start=datetime.now(timezone.utc)
    counters=dict(complete_matches=0,technical_cuts=0,optimizer_steps=0,training_observations=0)
    base={'smoke':9700000,'train':9710000,'evaluate':9720000}[args.mode]
    seeds=[base+args.scenario_offset+i for i in range(1,args.episodes+1)]
    registration=dict(schema='lizbon-classic-study-v2',created_utc=start.isoformat(),mode=args.mode,
        profile=profile.to_dict(),receipt=receipt(profile),sources=sources(),initialization_seed=args.seed,
        scenarios=seeds,ancestry=ancestry,architecture=model.architecture(),
        reward=dict(team_win=10,team_loss=-10,team_kill=1,cooldown_ready_per_second=-.005),
        eligibility='Complete natural matches only; no technical cut enters PPO or win-rate denominator',
        budget=dict(episodes=args.episodes,max_minutes=args.max_minutes,match_wall_seconds=args.match_wall_seconds),
        runtime=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,
            processor=platform.processor(),device='cpu',torch_threads=1),
        frozen_crew_tensor_hash=crew_hash,crew_update=102,
        frozen_crew_training='task-pursuit-v1; not A2-trained',
        reserved_final='9730001..9731000 and all historical final partitions remain sealed')
    atomic(args.output/'registration.json',registration)
    initial_hash=save(args.output/'initial.pt',model,None,profile,counters,args.seed,ancestry)
    rows=[];updates=[];status_path=args.output/'status.json'
    def status(value):atomic(status_path,dict(phase='collecting',mode=args.mode,counters=dict(counters),
        total_wall_seconds=perf_counter()-began,**value))
    try:
        for episode,seed in enumerate(seeds,1):
            remaining=args.max_minutes*60-(perf_counter()-began)
            if remaining<=1:break
            summary,trajectories=run_match(model,crew,profile,seed,output=args.output/f'match-{episode:03d}.jsonl',
                max_wall_seconds=min(remaining,args.match_wall_seconds),
                max_observations=args.smoke_observations if args.mode=='smoke' else None,
                deterministic=args.deterministic,status=status,live_dir=args.output/'live',
                policy_update=inherited+counters['optimizer_steps'],mode=args.mode)
            natural=summary['winner'] in ('crewmate','impostor') and not summary['truncated']
            counters['complete_matches' if natural else 'technical_cuts']+=1
            summary.update(episode=episode,mode=args.mode)
            if args.mode=='train' and natural:
                atomic(status_path,dict(phase='optimizing',episode=episode,counters=dict(counters)))
                optimization=perf_counter()
                metrics=update(model,optimizer,trajectories,summary['simulation_seconds'],deadline=began+args.max_minutes*60)
                counters['optimizer_steps']+=len(metrics);counters['training_observations']+=summary['impostor_observations']
                updates.extend([dict(episode=episode,**r) for r in metrics])
                summary['optimization_seconds']=perf_counter()-optimization
                summary['optimization_budget_cut']=len(metrics)<4*sum(bool(t) for t in trajectories)
                save(args.output/f'checkpoint-{episode:03d}.pt',model,optimizer,profile,counters,args.seed,ancestry)
            summary['counters']=dict(counters);rows.append(summary)
            with (args.output/'matches.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(summary)+'\n')
            atomic(args.output/'ppo.json',updates)
            print(json.dumps(dict(episode=episode,winner=summary['winner'],reason=summary['reason'],
                simulation_seconds=summary['simulation_seconds'],wall_seconds=summary['wall_seconds'],counters=counters)),flush=True)
        if tensor_hash(crew)!=crew_hash:raise ValueError('Frozen crew weights changed')
        final_hash=save(args.output/'final.pt',model,optimizer,profile,counters,args.seed,ancestry)
        completed=[r for r in rows if not r['truncated'] and r['winner']]
        result=dict(mode=args.mode,matches=rows,counters=counters,wins=sum(r['winner']=='impostor' for r in completed),
            games=len(completed),initial_sha256=initial_hash,final_sha256=final_hash,
            started_utc=start.isoformat(),finished_utc=datetime.now(timezone.utc).isoformat(),
            wall_seconds=perf_counter()-began,model_changed=tensor_hash(model)!=initial_tensor,frozen_crew_unchanged=True,
            interpretation='training outcomes' if args.mode=='train' else 'engineering, not performance evidence' if args.mode=='smoke' else 'development evaluation; not final testing')
        atomic(args.output/'results.json',result)
        atomic(status_path,dict(phase='finished',mode=args.mode,counters=counters,wall_seconds=result['wall_seconds']))
        print(json.dumps(dict(finished=True,output=str(args.output),counters=counters)),flush=True)
        return 0
    except BaseException as error:
        failure=dict(phase='failed',error=type(error).__name__,message=str(error),counters=counters,
            wall_seconds=perf_counter()-began,training_success_claimed=False)
        atomic(args.output/'failure.json',failure);atomic(status_path,failure)
        raise


if __name__=='__main__':raise SystemExit(main())
