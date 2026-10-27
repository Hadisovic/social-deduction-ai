"""Registered study driver. No execution without a separately supplied owner approval."""
import argparse
import json
from pathlib import Path
from time import perf_counter,time
from threading import Timer
from _thread import interrupt_main
import numpy as np
import torch
from phase6.config import verify_frozen
from crewmate_v2.learning.provenance import digest
from crewmate_v2.learning.train import state_hash
from crewmate_v2.collection.ppo import update
from .policy import FourCrewPolicy
from .match import FIELDS
from .config import Rewards
from .registration import plan,runtime,authorize,stop_state,training_context,validation_contexts,EXPERIMENT
from .execution import jobs,run_jobs
from .analysis import review


def write(path,record,exclusive=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if exclusive:
        with path.open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
    else:
        temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(record),encoding='utf-8');temp.replace(path)


def append(path,record):
    with Path(path).open('a',encoding='utf-8') as stream:stream.write(json.dumps(record)+'\n')


def combine(results):
    flat={k:[] for k in FIELDS};flat['segments']=[]
    for rollout,summary in results:
        offset=len(flat['actions'])
        for k in FIELDS:flat[k].extend(rollout[k])
        flat['segments'].extend((lo+offset,hi+offset) for lo,hi in rollout['segments'])
    return flat


def run(output,spectator,replays,status,approval):
    # This gate deliberately precedes output creation, policy initialization and games.
    authorize(approval,spectator)
    p=plan();locked=runtime(spectator);started=perf_counter();deadline=started+p['total_wall_seconds']
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write(output/'registration.json',dict(plan=p,runtime=locked,approval=approval,started_unix=time(),hard_seconds=14400))
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    models={}
    for arm in ('original','shaped_v2','zero'):
        torch.manual_seed(17);models[arm]=FourCrewPolicy()
    initial=state_hash(models['zero']);assert all(state_hash(x)==initial for x in models.values())
    models['zero'].requires_grad_(False);models['zero'].eval()
    models['zero'].save(output/'reference-zero.pt',dict(experiment=EXPERIMENT,initialization=17,updates=0,initial_sha256=initial))
    optimizers={a:torch.optim.Adam(models[a].parameters(),lr=.0003) for a in ('original','shaped_v2')}
    rng={a:np.random.default_rng(17) for a in optimizers};decisions={a:0 for a in optimizers};selected={};accepted=[];state='continue'
    def check():
        if perf_counter()>=deadline:raise TimeoutError('Four-hour global study ceiling')
        if runtime(spectator)!=locked:raise RuntimeError('Registered source/recorder drift')
        if state_hash(models['zero'])!=initial:raise RuntimeError('Zero-update reference changed')
        verify_frozen()
    watchdog=Timer(max(0.,deadline-perf_counter()),interrupt_main);watchdog.daemon=True;watchdog.start()
    try:
        for first in range(0,32,2):
            check();contexts=[training_context(i) for i in (first,first+1)];payloads=[]
            for arm in optimizers:
                rewards=Rewards.original() if arm=='original' else Rewards()
                # Sampling schedule is fixed before outcomes: first two seeds of each block of four.
                recording=replays if first%4==0 else None
                payloads+=jobs(models[arm],contexts,rewards=rewards,label=arm,deadline=deadline,
                    session=EXPERIMENT,version=first//2,replay_dir=recording,spectator_source=spectator)
            rows=run_jobs(payloads,deadline,on_result=lambda i,r:append(output/'collected-training.jsonl',dict(job=i,**r[1])))
            check()
            paired={a:rows[j*2:j*2+2] for j,a in enumerate(optimizers)}
            projected={a:decisions[a]+sum(len(r[0]['actions']) for r in paired[a]) for a in optimizers}
            if any(n>3072 for n in projected.values()):state='inconclusive_rejected_overshoot_batch';break
            for arm in optimizers:
                diagnostics=update(models[arm],optimizers[arm],combine(paired[arm]),rng[arm]);decisions[arm]=projected[arm]
                check()
                checkpoint=output/arm/f'checkpoint-batch-{first//2+1:02d}-choices-{decisions[arm]}.pt'
                models[arm].save(checkpoint,dict(experiment=EXPERIMENT,arm=arm,plan_sha256=digest(p),initialization=17,
                    initial_sha256=initial,actor_decisions=decisions[arm],matched_seed_exposure=list(range(2780000,2780000+first+2)),
                    optimizer=optimizers[arm].state_dict(),update_rng=rng[arm].bit_generator.state,torch_rng=torch.get_rng_state()))
                selected[arm]=str(checkpoint.relative_to(output));append(output/'updates.jsonl',dict(arm=arm,decisions=decisions[arm],**diagnostics))
            accepted.extend(r[1] for r in rows)
            state=stop_state(decisions,first+2)
            write(status,dict(status='training',experiment=EXPERIMENT,decisions=decisions,matched_games_per_arm=first+2,elapsed_seconds=perf_counter()-started),False)
            if state!='continue':break
        if state!='complete':
            write(output/'incomplete.json',dict(status=state,decisions=decisions,seconds=perf_counter()-started,validation_games=0));return
        check();payloads=[]
        for label,model in models.items():
            if label!='zero':
                loaded,_=FourCrewPolicy.load(output/selected[label]);assert state_hash(loaded)==state_hash(model)
            payloads+=jobs(model,list(validation_contexts()),rewards=Rewards.original(),label=label,deadline=deadline,
                session=EXPERIMENT,greedy=True,version=-1 if label=='zero' else len(accepted)//4,replay_dir=replays,spectator_source=spectator)
        results=run_jobs(payloads,deadline,on_result=lambda i,r:append(output/'validation.jsonl',dict(policy=list(models)[i//16],**r[1])))
        check();evaluation={a:[x[1] for x in results[i*16:i*16+16]] for i,a in enumerate(models)}
        coverage={a:dict(train_legal_pairs=sum(sum(v['opportunities']['choice_legal'] for v in r['actors'].values()) for r in accepted[i*2::4]+accepted[i*2+1::4]),
                        train_reports=sum(sum(v['reporting'].get('accepted',0) for v in r['actors'].values()) for r in accepted[i*2::4]+accepted[i*2+1::4])) for i,a in enumerate(optimizers)}
        analysis=review(evaluation,{a:accepted[i*2::4]+accepted[i*2+1::4] for i,a in enumerate(optimizers)})
        write(output/'aggregate.json',analysis)
        write(output/'study.json',dict(status='development_complete_experimental_review_pending',seconds=perf_counter()-started,
            decisions=decisions,selected_checkpoints=selected,training_coverage=coverage,evaluation=evaluation,inconclusive_flags=analysis['inconclusive_flags'],
            zero_reference_unchanged=state_hash(models['zero'])==initial,final_test_opened=False))
        write(status,dict(status='development_complete_acceptance_pending',decisions=decisions,validation_games=48,elapsed_seconds=perf_counter()-started),False)
    except BaseException as error:
        write(output/'failure.json',dict(status='incomplete',error=type(error).__name__,message=str(error),seconds=perf_counter()-started,
            decisions=decisions,selected_checkpoints=selected));raise
    finally:watchdog.cancel()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--spectator-source',type=Path,required=True);p.add_argument('--replays',type=Path,required=True)
    p.add_argument('--status',type=Path,required=True);p.add_argument('--approval',type=Path,required=True);a=p.parse_args()
    run(a.output,a.spectator_source,a.replays,a.status,json.loads(a.approval.read_text()))


if __name__=='__main__':main()
