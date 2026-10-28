"""Owner-approved bounded shared-environment pilot, with immutable selection lock."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import time
import traceback

import torch

from frozen_handoff.models import load_model, sha256
from .checkpoints import contract, save, load, optimizer_for
from .training import PPOConfig, collect_episode, ppo_update, source_signature, check_sources
from .weights import load_transfer, tensor_hash, verify_parity

SCHEMA = 'impostor-shared-pilot-v1'
STREAMS = (101, 202)
TRAIN = tuple(range(3600000, 3600008))
SELECT = tuple(range(3610000, 3610008))
CONFIRM = tuple(range(3610020, 3610036))


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value, *, exclusive=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, allow_nan=False)+'\n'
    if exclusive:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(data)
    else:
        pending = path.with_suffix(path.suffix+'.pending')
        pending.write_text(data, encoding='utf-8'); os.replace(pending, path)


def journal(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())


def setup():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def remaining_deadline(unix_deadline):
    remaining = unix_deadline-time.time()
    if remaining <= 0:
        raise TimeoutError('Registered study wall ceiling reached')
    return time.perf_counter()+min(remaining, 600.)


def registration():
    policy, receipt = load_transfer()
    parity = verify_parity(policy, receipt)
    return dict(schema=SCHEMA, created_utc=utc(), owner_approval='2026-10-29: start training',
        budget_minutes=90, sources=source_signature(), checkpoint_contract=contract(),
        initial_tensor_hash=tensor_hash(policy), original_transfer_sha256=receipt['export_sha256'],
        parity=parity, ppo=PPOConfig().__dict__, streams=list(STREAMS), episodes_per_stream=8,
        initial_weights_shared=True, training_seeds=list(TRAIN), selection_seeds=list(SELECT),
        confirmation_seeds=list(CONFIRM), candidate_episodes=[4,8],
        crew_training='B', crew_confirmation=['A','B'], crew_sampling_initialization=17,
        selection_rule='wins, kills, fewer own ejections, earlier episode, lower stream',
        reference='untouched transferred V4.1', all_partitions='ordinary development; historical final sets sealed',
        complete_episodes_only=True, training_cutoff_minutes=45,
        abort_policy='No automatic extension, no tuning using confirmation, retain incomplete report',
        hardware=dict(cpu=platform.processor(),logical_cpus=os.cpu_count(),workers=2,torch_threads_per_worker=1,device='cpu'),
        metrics_limits='Witness counts are retrospective; claims do not prove deception. No causal escape or deception claim.')


def train_stream(output, stream, record, deadline):
    setup(); folder=Path(output)/f'stream-{stream}'; folder.mkdir(exist_ok=False)
    policy, _=load_transfer(trainable=True); optimizer=optimizer_for(policy)
    generator=torch.Generator().manual_seed(stream)
    crew_policy, _=load_model('B'); decisions=0; updates=0; candidates=[]
    write_json(folder/'status.json', dict(stage='training', stream=stream, pid=os.getpid(), started_utc=utc()))
    for episode, seed in enumerate(TRAIN, 1):
        check_sources(record['sources'])
        rows, summary=collect_episode(policy, crew_policy, seed=seed, generator=generator,
            deadline=remaining_deadline(deadline), replay=folder/f'episode-{episode:02d}.jsonl',
            policy_update=updates, training_decisions=decisions)
        diagnostics=ppo_update(policy, optimizer, rows, generator, completed_episode=episode)
        decisions+=len(rows); updates+=diagnostics['minibatch_updates']
        event=dict(episode=episode, stream=stream, seed=seed, utc=utc(), decisions=decisions,
            optimizer_updates=updates, match=summary, ppo=diagnostics)
        path=folder/f'checkpoint-episode-{episode:02d}.pt'
        save(path, policy, optimizer, generator, record=dict(study=record, stream=stream,
            completed_episodes=episode, decisions=decisions, optimizer_updates=updates),
            next_seed=seed+1, episode_complete=True)
        journal(folder/'training.jsonl', event)
        write_json(folder/'status.json', dict(stage='training', stream=stream, pid=os.getpid(),
            episode=episode, decisions=decisions, last_completed_utc=utc(), checkpoint=path.name))
        if episode in (4,8):
            candidates.append(dict(stream=stream, episode=episode, path=str(path), sha256=sha256(path)))
        print(json.dumps(dict(stage='training', stream=stream, episode=episode,
            win=summary['impostor_win'], kills=summary['behavior']['kills'], reason=summary['reason'],
            decisions=decisions, kl=diagnostics['approximate_kl'])), flush=True)
    write_json(folder/'status.json', dict(stage='training_complete', stream=stream,
        completed_utc=utc(), episodes=8, decisions=decisions, optimizer_updates=updates))
    return candidates


def evaluation_block(output, label, checkpoint, crew, seeds, record, deadline):
    setup(); folder=Path(output)/label; folder.mkdir(parents=True, exist_ok=False)
    if checkpoint:
        policy, _, _, payload=load(checkpoint); policy.eval().requires_grad_(False)
        if payload['record'].get('study') != record:
            raise ValueError('Candidate belongs to a different registered study')
        decisions=payload['record']['decisions']; updates=payload['record']['optimizer_updates']
    else:
        policy, _=load_transfer(); decisions=8192; updates=0
    crew_policy, _=load_model(crew); results=[]
    for index, seed in enumerate(seeds):
        check_sources(record['sources'])
        _, result=collect_episode(policy, crew_policy, seed=seed, crew=crew,
            deadline=remaining_deadline(deadline), replay=folder/f'seed-{seed}.jsonl' if index==0 else None,
            policy_update=updates, training_decisions=decisions)
        result.update(label=label, utc=utc())
        journal(folder/'matches.jsonl', result); results.append(result)
        write_json(folder/'status.json', dict(stage='evaluation', matches=len(results), total=len(seeds), last_completed_utc=utc()))
        print(json.dumps(dict(stage=label, match=len(results), total=len(seeds),
            win=result['impostor_win'], reason=result['reason'])), flush=True)
    summary=dict(label=label, crew=crew, matches=results, wins=sum(r['impostor_win'] for r in results),
        kills=sum(r['behavior']['kills'] for r in results), ejections=sum(r['reason']=='IMPOSTOR_EJECTED' for r in results),
        checkpoint=checkpoint, policy_hash=tensor_hash(policy))
    write_json(folder/'summary.json', summary, exclusive=True)
    return summary


def select_candidate(candidates, summaries):
    if len(candidates)!=4 or len(summaries)!=4:
        raise ValueError('All four registered candidates must complete selection')
    for candidate, summary in zip(candidates, summaries):
        if [m['seed'] for m in summary['matches']] != list(SELECT):
            raise ValueError('Selection seed mismatch')
        if summary['checkpoint'] != candidate['path'] or summary['crew'] != 'B':
            raise ValueError('Candidate/selection mismatch')
    paired=list(zip(candidates,summaries))
    candidate, summary=max(paired,key=lambda pair:(pair[1]['wins'],pair[1]['kills'],-pair[1]['ejections'],
        -pair[0]['episode'],-pair[0]['stream']))
    return dict(candidate=candidate, wins=summary['wins'], kills=summary['kills'],
        ejections=summary['ejections'], all_zero_selection_wins=all(s['wins']==0 for s in summaries))


def preflight(output):
    setup(); output=Path(output); output.mkdir(parents=True,exist_ok=False)
    record=registration(); write_json(output/'registration.json',record,exclusive=True)
    policy,_=load_transfer(trainable=True); crew,_=load_model('B'); gen=torch.Generator().manual_seed(103)
    rows,result=collect_episode(policy,crew,seed=2760010,generator=gen,
        deadline=time.perf_counter()+600,replay=output/'replay.jsonl')
    diag=ppo_update(policy,optimizer:=optimizer_for(policy),rows,gen,completed_episode=1)
    save(output/'preflight.pt',policy,optimizer,gen,record=dict(preflight=True,decisions=len(rows),optimizer_updates=diag['minibatch_updates']),
        next_seed=3600000,episode_complete=True)
    restored,_,rng,_=load(output/'preflight.pt')
    if tensor_hash(restored)!=tensor_hash(policy) or not torch.equal(rng.get_state(),gen.get_state()):
        raise AssertionError('Preflight checkpoint roundtrip failed')
    write_json(output/'result.json',dict(status='passed',utc=utc(),match=result,ppo=diag,
        reused_for_study=False,checkpoint_roundtrip=True),exclusive=True)
    print(json.dumps(dict(preflight='passed',decisions=len(rows),reason=result['reason'],ppo=diag)),flush=True)


def run(output):
    setup(); output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    record=registration(); start=time.time(); deadline=start+90*60
    record.update(started_utc=utc(), deadline_unix=deadline)
    write_json(output/'registration.json',record,exclusive=True)
    status=dict(schema=SCHEMA,status='running',stage='training',started_utc=utc(),pid=os.getpid(),
        deadline_unix=deadline,output=str(output));write_json(output/'status.json',status)
    print(json.dumps(status),flush=True)
    try:
        candidates=[]
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(train_stream,str(output),stream,record,start+45*60) for stream in STREAMS]
            for future in as_completed(futures):candidates.extend(future.result())
        candidates.sort(key=lambda c:(c['stream'],c['episode']))
        write_json(output/'candidates.json',candidates,exclusive=True)
        status.update(stage='selection');write_json(output/'status.json',status)
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(evaluation_block,str(output),f'selection-{c["stream"]}-ep{c["episode"]}',c['path'],'B',SELECT,record,deadline) for c in candidates]
            summaries=[f.result() for f in futures]
        selection=select_candidate(candidates,summaries)
        chosen=selection['candidate']
        if sha256(chosen['path'])!=chosen['sha256']:
            raise ValueError('Selected checkpoint bytes changed')
        lock=dict(schema=SCHEMA,locked_utc=utc(),selection=selection,registration_sha256=sha256(output/'registration.json'),
            sources=record['sources'],confirmation_seeds=list(CONFIRM),final_tests_opened=False)
        write_json(output/'selection-lock.json',lock,exclusive=True)
        status.update(stage='confirmation',selection=selection);write_json(output/'status.json',status)
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(evaluation_block,str(output),f'confirmation-{method}-{crew}',
                chosen['path'] if method=='selected' else None,crew,CONFIRM,record,deadline)
                for method in ('selected','reference') for crew in ('A','B')]
            confirmed=[f.result() for f in futures]
        write_json(output/'results.json',dict(schema=SCHEMA,selection=selection,confirmation=confirmed,
            elapsed_wall_seconds=time.time()-start,completed_utc=utc(),final_tests_opened=False),exclusive=True)
        status.update(status='complete',stage='complete',completed_utc=utc(),elapsed_wall_seconds=time.time()-start)
        from .report import generate
        generate(output)
    except Exception:
        status.update(status='incomplete',stage='stopped',stopped_utc=utc(),elapsed_wall_seconds=time.time()-start,
            error=traceback.format_exc())
        write_json(output/'failure.json',status,exclusive=True)
        raise
    finally:
        write_json(output/'status.json',status)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preflight',action='store_true')
    a=p.parse_args();preflight(a.output) if a.preflight else run(a.output)


if __name__=='__main__':main()
