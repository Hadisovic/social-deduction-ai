"""Curate the stopped shared pilot, preserving missing cases as missing.

This offline tool performs no simulation, optimizer update, selection, or resume.
It lives outside the frozen impostor_shared namespace to preserve checkpoint locks.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from impostor_shared.checkpoints import load,contract
from impostor_shared.training import check_sources
from impostor_shared.weights import tensor_hash
from impostor_shared.report import wilson


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def summarize(rows,planned):
    outcomes=Counter(r['reason'] for r in rows)
    behavior=Counter()
    for row in rows:
        for key in ('kills','accepted_reports','accepted_self_reports','witnessed_kill_events','fake_task_starts','fake_task_completions'):
            behavior[key]+=row['behavior'][key]
    wins=sum(r['impostor_win'] for r in rows)
    return dict(completed=len(rows),planned=planned,wins=wins,outcomes=dict(outcomes),behavior=dict(behavior),
        illegal_actions=sum(r['illegal_actions'] for r in rows),
        simulated_seconds=sum(r['simulated_seconds'] for r in rows),
        collection_wall_seconds_sum=sum(r['collection_seconds'] for r in rows),
        completed_collection_cpu_seconds=sum(r['collection_cpu_seconds'] for r in rows),
        reward_sum=sum(r['credit']['reward_sum'] for r in rows),
        cooldown_ready_seconds=sum(r['credit']['cooldown_ready_seconds'] for r in rows),
        wilson95_completed_cases=wilson(wins,len(rows)),
        warning='Completed-case fractions are conditional on completion; missing cases are not losses or game timeouts.')


def curate(run):
    torch.set_num_threads(1);run=Path(run).resolve()
    registration=read(run/'registration.json');status=read(run/'status.json')
    if status['status']!='incomplete' or status['stage']!='stopped':
        raise ValueError('This curator requires the preserved stopped pilot, not healthy workers')
    if (run/'selection-lock.json').exists() or (run/'results.json').exists():
        raise ValueError('Unexpected locked/completed study; never replace its selection')
    check_sources(registration['sources'])
    if contract()!=registration['checkpoint_contract']:
        raise ValueError('Frozen runtime/source contract changed')
    report=ROOT/'docs/collaboration/impostor-shared-integration/pilot-20261029'
    release=ROOT/'artifacts/impostor-shared-pilot-20261029'
    report.mkdir(exist_ok=False);release.mkdir(exist_ok=False)
    figures=report/'figures';figures.mkdir()
    training={s:lines(run/f'stream-{s}/training.jsonl') for s in (101,202)}
    train_summaries={str(s):summarize([r['match'] for r in rs],8) for s,rs in training.items()}
    for s,rs in training.items():
        if len(rs)!=8 or [r['seed'] for r in rs]!=registration['training_seeds']:
            raise ValueError('Incomplete training stream')
        train_summaries[str(s)].update(decisions=rs[-1]['decisions'],optimizer_updates=rs[-1]['optimizer_updates'],
            optimization_wall_seconds=sum(r['ppo']['optimization_seconds'] for r in rs),
            optimization_cpu_seconds=sum(r['ppo']['optimization_cpu_seconds'] for r in rs),
            completed_utc=read(run/f'stream-{s}/status.json')['completed_utc'])
    candidates=[];validation=[]
    for c in read(run/'candidates.json'):
        source=Path(c['path'])
        if digest(source)!=c['sha256']:raise ValueError('Candidate bytes changed')
        policy,_,_,payload=load(source)
        if payload['record']['study']!=registration or payload['record']['completed_episodes']!=c['episode']:
            raise ValueError('Candidate provenance mismatch')
        destination=release/f'stream-{c["stream"]}-episode-{c["episode"]:02d}.pt'
        shutil.copyfile(source,destination)
        candidates.append(dict(stream=c['stream'],episode=c['episode'],path=destination.relative_to(ROOT).as_posix(),
            sha256=digest(destination),tensor_hash=tensor_hash(policy),strict_load_passed=True))
        folder=run/f'selection-{c["stream"]}-ep{c["episode"]}'
        rows=lines(folder/'matches.jsonl')
        if [r['seed'] for r in rows]!=registration['selection_seeds'][:len(rows)]:
            raise ValueError('Unexpected validation prefix')
        summary=summarize(rows,8)
        summary.update(stream=c['stream'],episode=c['episode'],seeds=[r['seed'] for r in rows],
            missing_seeds=[s for s in registration['selection_seeds'] if s not in {r['seed'] for r in rows}])
        validation.append(summary)
    all_matches=[r['match'] for rs in training.values() for r in rs]
    all_matches += [r for c in candidates for r in lines(run/f'selection-{c["stream"]}-ep{c["episode"]}/matches.jsonl')]
    assert all(r['illegal_actions']==0 and r['private_actor_memories'] for r in all_matches)
    assert len({r['frozen_belief_hash'] for r in all_matches})==1
    assert len({r['frozen_crew_hash'] for r in all_matches})==1
    assert len(list(run.glob('stream-*/checkpoint-episode-*.pt')))==16
    assert sum(v['completed'] for v in validation)==30
    aggregate=dict(status='incomplete_validation',started_utc=status['started_utc'],stopped_utc=status['stopped_utc'],
        wall_seconds=status['elapsed_wall_seconds'],training=train_summaries,validation=validation,
        hardware=registration['hardware'],runtime=registration['checkpoint_contract']['runtime'],
        registration_sha256=digest(run/'registration.json'),candidates=candidates,
        completed_training_episodes=16,completed_validation_matches=30,planned_validation_matches=32,
        confirmation_matches=0,selection_locked=False,reserved_final_tests_opened=False,
        failed_case=dict(stream=101,episode=4,seed=3610006,failure='600-second wall execution ceiling',
            game_outcome=None,next_case_not_started=3610007),
        integrity=dict(strict_candidate_loads=4,all_episode_checkpoints_preserved=16,
            illegal_actions=0,frozen_crew_and_belief_hashes_consistent=True,source_signature_unchanged=True),
        completed_matches_cpu_seconds=sum(r['collection_cpu_seconds'] for r in all_matches),
        optimization_cpu_seconds=sum(r['ppo']['optimization_cpu_seconds'] for rs in training.values() for r in rs),
        cpu_accounting='Lower bound: failed-case, loading, parent and verifier CPU were not fully persisted',
        assessment='Training completed, but validation is incomplete. No selection or paired confirmation. Improvement over the untouched reference is unestablished.')
    write(report/'aggregate.json',aggregate)
    write(release/'manifest.json',dict(schema='shared-pilot-candidates-v1',status='unselected',
        source_commit='7bb6616',registration_sha256=aggregate['registration_sha256'],candidates=candidates,
        checkpoint_loading='impostor_shared.checkpoints.load; exact original frozen source/runtime contract required'))
    for name in ('registration.json','failure.json'):
        shutil.copyfile(run/name,report/name)
    # A compact failure record is public; local absolute paths in the traceback
    # are unnecessary for review, so retain the raw traceback only in local-runs.
    (report/'failure.json').write_text(json.dumps(dict(status='incomplete',elapsed_wall_seconds=aggregate['wall_seconds'],
        failure=aggregate['failed_case'],type='TimeoutError',message='Engineering match wall ceiling',
        traceback_preserved_locally=True),indent=2)+'\n',encoding='utf-8')
    write(report/'training-metrics.json',{str(s):[dict(episode=r['episode'],seed=r['seed'],
        decisions=r['decisions'],optimizer_updates=r['optimizer_updates'],ppo=r['ppo'],
        reason=r['match']['reason'],win=r['match']['impostor_win'],credit=r['match']['credit'],
        behavior=r['match']['behavior']) for r in rs] for s,rs in training.items()})
    def save(fig,name):
        fig.tight_layout();fig.savefig(figures/name,dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for s,rs in training.items():
        xs=[r['episode'] for r in rs];label=f'Stream {s}'
        axes[0,0].plot(xs,[r['match']['credit']['reward_sum'] for r in rs],marker='o',label=label)
        axes[0,1].plot(xs,[r['match']['behavior']['kills'] for r in rs],marker='o',label=label)
        axes[1,0].plot(xs,[sum(x['match']['impostor_win'] for x in rs[:i+1]) for i in range(8)],marker='o',label=label)
        axes[1,1].plot(xs,[r['match']['credit']['cooldown_ready_seconds'] for r in rs],marker='o',label=label)
    for ax,title in zip(axes.flat,['Shaped training return','Kills per training episode','Cumulative training wins','Cooldown-ready roaming seconds']):
        ax.set_title(title);ax.set_xlabel('Complete episode');ax.grid(alpha=.2);ax.legend()
    save(fig,'training.png')
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,metric in zip(axes.flat,['approximate_kl','clip_fraction','family_entropy','value_loss','actor_gradient_norm','critic_gradient_norm']):
        for s,rs in training.items():ax.plot([r['episode'] for r in rs],[r['ppo'][metric] for r in rs],marker='o',label=str(s))
        ax.set_title(metric.replace('_',' '));ax.set_xlabel('Episode update');ax.grid(alpha=.2);ax.legend()
    save(fig,'ppo.png')
    fig,ax=plt.subplots(figsize=(9,5));xs=list(range(4))
    wins=[v['wins'] for v in validation];losses=[v['completed']-v['wins'] for v in validation]
    ax.bar(xs,wins,label='Wins');ax.bar(xs,losses,bottom=wins,label='Completed non-wins')
    ax.bar(xs,[8-v['completed'] for v in validation],bottom=[v['completed'] for v in validation],
        hatch='//',color='lightgray',edgecolor='black',label='Execution aborted / not run')
    ax.set_xticks(xs,[f'{v["stream"]} / episode {v["episode"]}' for v in validation]);ax.set_ylim(0,8.8)
    ax.set_ylabel('Cases / 8 planned');ax.set_title('Incomplete B validation — no selection lock');ax.legend()
    save(fig,'validation.png')
    fig,ax=plt.subplots(figsize=(9,4));ax.barh(['Training episodes','Validation matches','Confirmation matches'],[16,30,0],color='#517c99')
    ax.barh(['Training episodes','Validation matches','Confirmation matches'],[0,2,64],left=[16,30,0],
        color='lightgray',hatch='//',label='Not completed')
    ax.set_xlabel('Completed / planned executions');ax.set_title('Study stopped at validation execution ceiling');ax.legend()
    save(fig,'coverage.png')
    table=['| Stream / checkpoint | Wins | Completed | Planned |','|---|---:|---:|---:|']
    table += [f'| {v["stream"]} / episode {v["episode"]} | {v["wins"]} | {v["completed"]} | 8 |' for v in validation]
    readme=['# Shared impostor pilot: training complete, validation incomplete','',
        '**No model is selected or promoted. Improvement over the untouched reference is not established.**','',
        f'The pilot ran for {aggregate["wall_seconds"]/60:.2f} minutes on two one-thread CPU workers.',
        'Both streams completed eight episodes. Stream 101 won 7/8 training matches with 19 kills; stream 202 won 7/8 with 16 kills.',
        'These are stochastic training outcomes. The streams share starting tensors and are not independent parameter initializations.','',
        '## Validation outcome','']+table+['',
        'Stream 101 / episode 4 hit the 600-second wall execution limit on seed 3610006. Seed 3610007 was not started.',
        'This is a computational abort, not a game TIMEOUT or an impostor loss. The three other validation blocks completed.',
        'Stream 101 / episode 8 won all eight B validation cases, but this small, selection-used result has no paired untouched-reference comparison.',
        'The fixed protocol requires all candidates to finish. No lock was written; all 64 planned A/B confirmation executions remain unrun. Historical final-test partitions stayed sealed.','',
        '![Training curves](figures/training.png)',
        'Returns include kill rewards and ready-cooldown costs. Training wins and returns do not measure held-out accuracy.',
        '', '![PPO diagnostics](figures/ppo.png)',
        'KL and clipping show policy movement; entropy describes exploration. Value loss and pre-clipping gradient norms describe the optimizer. Actor/critic clipping limits were .5/1.0.',
        '', '![Validation](figures/validation.png)',
        'Gray cases are missing, not losses. Comparing fractions with unequal completion can be biased.',
        '', '![Completed coverage](figures/coverage.png)',
        'Confirmation is absent; there is no confirmation win-rate graph to show.','',
        '## Files and verification',
        'All 16 local episode checkpoints remain intact. Four strict-load candidate snapshots are published in `artifacts/impostor-shared-pilot-20261029/`; they are unselected research artifacts.',
        'Source signatures, frozen crew/belief hashes, zero illegal actions and private actor memories passed on all 46 completed study executions.',
        'Seventeen targeted tests passed in 19.87 seconds. A separate 40.43-second preflight match plus .37-second optimizer update passed checkpoint/RNG reload; its learned weights were discarded.',
        'See aggregate.json for runtime, processor, completed-match CPU accounting, exact counts and checkpoint hashes; registration.json preserves settings/seeds. CPU totals exclude some failure/loading overhead.',
        '', '## Watch a completed learned-policy match',
        'Run from the isolated integration checkout:',
        '```powershell',
        '.\\.venv\\Scripts\\python.exe run_impostor_shared.py --replay local-runs/shared-pilot-20261029/selection-101-ep8/seed-3610000.jsonl',
        '```',
        'This is recorded playback of the episode-8 candidate, not live training or a selected release. Private replays stay local.',
        '', '## Recommended next step',
        'Diagnose the slow validation path using per-case progress/timing and profiling before approving a separately documented continuation. Preserve the same checkpoint bytes and comparison cases; do not retrain merely to work around this execution failure.',
        'A future evaluation-only amendment could retain the overall wall cap while budgeting long matches explicitly. Its source/runtime/checkpoint contract must be handled without weakening historical locks.',
        'Only after complete paired A/B evidence should we consider more PPO, a preregistered A/B/mixed-crew pool, separate sabotage mechanics and both-role adaptation, larger lobbies, or self-play.',
        'High training wins can reflect exploitable frozen opponents. Self-reports, witnessed kills and claims remain observed behavior; no causal deception or escape competence is established.','']
    (report/'README.md').write_text('\n'.join(readme),encoding='utf-8')
    print(json.dumps(aggregate,indent=2,allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    curate(parser.parse_args().run)
