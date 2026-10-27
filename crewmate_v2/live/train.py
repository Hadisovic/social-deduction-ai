"""Authorized fixed seed17 pair, gated by real replay and native desktop verification."""
import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter,time
import numpy as np
import torch
from crewmate_v2.learning.policy import V2Policy
from crewmate_v2.learning.train import write,append,state_hash
from crewmate_v2.learning.provenance import digest
from crewmate_v2.collection.config import proposal
from crewmate_v2.collection.ppo import update,save
from .collector import LiveCollector,LABELS
from .protocol import lock,plan,validate


def atomic(path,record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(record,allow_nan=False),encoding='utf-8');temporary.replace(path)


def status(path,arm,stage,decisions,games,started,diagnostics=None):
    elapsed=perf_counter()-started
    atomic(path,dict(condition=LABELS[arm],stage=stage,decisions=decisions,validation_games=games,
                    elapsed_seconds=elapsed,hard_remaining_seconds=max(0.,10800-elapsed),diagnostics=diagnostics or {},updated_unix=time()))

def run_one(output,arm,deadline,approval,spectator_source,replay_dir,status_file,experiment_started):
    validate(approval,spectator_source)
    workers=4
    if arm not in ('original', 'shaped'):
        raise ValueError('Unregistered reward arm')
    source = lock(spectator_source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    episodes, updates = [], []
    simulated, forced, completed = 0., 0, 0
    actions = Counter()
    opportunities = [dict(bodies=[], reports=[], ballots=[]) for _ in range(8)]
    recording=dict(spectator_source=str(Path(spectator_source).resolve()),replay_dir=str(Path(replay_dir).resolve()),
                   streams=[0,1,2,3],session='seed17-live-pilot',workers=4,source_version='live-runtime:'+digest(source))
    with LiveCollector(workers=4,arm=arm,partition='train',deadline=deadline,recording=recording) as collector:
        torch.manual_seed(17)
        policy = V2Policy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=.0003)
        rng = np.random.default_rng(17)
        initial = state_hash(policy)
        write(output/'config.json', dict(runtime=source, execution=proposal(workers), arm=arm,
              initialization_sha256=initial, resume_checkpoint=None))
        while completed < 512:
            if perf_counter() >= deadline or lock(spectator_source) != source:
                raise RuntimeError('Wall ceiling or source/runtime drift')
            rollout = collector.collect(policy,128)
            if perf_counter() >= deadline:
                raise TimeoutError('Compute ceiling before PPO update')
            diagnostics = update(policy,optimizer,rollout,rng)
            collector.policy_version+=1
            if perf_counter()>=deadline:raise TimeoutError("Hard ceiling during optimizer update")
            completed += 128
            simulated += sum(rollout['durations'])
            forced += sum(t['info']['coalesced_forced_windows'] for t in rollout['transition_audits'])
            for row in rollout['transition_audits']:
                slot = row['slot']
                audit, opportunity = row['audit'], opportunities[slot]
                for target, values in [('bodies',audit['body_visible']),('reports',audit['legal_reports']),
                                       ('ballots',[] if audit['ballot'] is None else [audit['ballot']])]:
                    opportunity[target] = sorted(set(opportunity[target]).union(values))
                actions[audit['action_kind']] += 1
                if row['done']:
                    episode = dict(slot=slot,seed=row['seed'],rotation=row['rotation'],
                                   policy_body_visible_unique=len(opportunity['bodies']),
                                   policy_report_legal_unique=len(opportunity['reports']),
                                   policy_ballot_opportunities=len(opportunity['ballots']), **row['info']['episode'])
                    episodes.append(episode)
                    append(output/'episodes.jsonl',episode)
                    opportunities[slot] = dict(bodies=[],reports=[],ballots=[])
            updates.append(dict(decisions=completed,simulated_seconds=simulated,
                                seconds=perf_counter()-began,**diagnostics))
            append(output/'updates.jsonl',updates[-1])
            status(status_file,arm,'training',completed,0,experiment_started,updates[-1])
            print(json.dumps(dict(arm=arm,**updates[-1])),flush=True)
            save(output/f'checkpoint-{completed}.pt',policy,optimizer,rng,collector,completed,
                 evidence=dict(episodes=episodes,updates=updates,simulated=simulated,forced=forced,
                               actions=dict(actions),opportunities=opportunities,initial=initial,live_runtime=source))
        partial = collector.snapshot()['environments']
        recordings=collector._command('recording_summary')
    result = dict(status='trained_pending_experimental_review',arm=arm,initialization_seed=17,
                  actionable_decisions=completed,simulated_seconds=simulated,coalesced_forced_windows=forced,
                  seconds=perf_counter()-began,episodes=episodes,episodes_completed=len(episodes),
                  partial_streams=partial,actions=dict(actions),updates=updates,
                  initial_state_sha256=initial,final_state_sha256=state_hash(policy),
                  final_test_opened=False,recording_summary=recordings)
    write(output/'result.json',result)
    return result


def pilot(output,spectator_source,replay_dir,status_file,approval,proof,gui_state,old_smoke):
    validate(approval,spectator_source)
    source=lock(spectator_source)
    if proof.get('runtime')!=source or proof.get('status')!='passed_engineering_only' or not proof.get('overhead_acceptable'):
        raise ValueError('Matching real PPO recording/noninterference/overhead proof required')
    gui=json.loads(Path(gui_state).read_text())
    if time()-gui['updated_unix']>15 or not gui.get('replay') or not str(gui.get('condition','')).startswith('Model A'):
        raise RuntimeError('Verified native GUI must still be running a real PPO replay before training')
    if old_smoke.get('runtime_sha256')!=digest(source['collection']['learning']):
        raise ValueError('Previous validation-cost smoke runtime mismatch')
    bench=json.loads(Path('docs/crewmate-v2-performance/benchmark-results.json').read_text())
    if bench['runtime_sha256']!=digest(source['collection']):raise ValueError('Preserved benchmark core source mismatch')
    recording_ratio=max(1.,proof['recording_wall_ratio'])
    training=bench['pilot_forecast']['training_seconds']*recording_ratio
    validation=bench['pilot_forecast']['serial_validation_seconds']
    planning=2*(training+validation)
    if planning>10800:raise TimeoutError('Conservative forecast exceeds the three-hour ceiling; do not start')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    began=perf_counter();deadline=began+10800
    write(output/'registration.json',dict(plan=plan(),runtime=source,approval=approval,
          actual_start_unix=time(),hard_ceiling_seconds=10800,training_forecast=training,
          validation_forecast=validation,planning_seconds=planning,recording_ratio=recording_ratio))
    from .evaluate import evaluate_game
    from crewmate_v2.learning.evaluate import validation_contexts
    from crewmate_v2.learning.comparison import compare
    runs=[]
    try:
        for arm in ('original','shaped'):
            status(status_file,arm,'starting training',0,0,began)
            folder=output/f'{arm}-17'
            result=run_one(folder,arm,deadline,approval,spectator_source,replay_dir,status_file,began)
            policy,_=V2Policy.load(folder/'checkpoint-512.pt')
            if state_hash(policy)!=result['final_state_sha256']:raise ValueError('Selected checkpoint roundtrip failed')
            if arm=='shaped' and result['initial_state_sha256']!=runs[0]['result']['initial_state_sha256']:
                raise ValueError('Paired model initializations differ')
            games=[]
            for context in validation_contexts():
                if perf_counter()>=deadline or lock(spectator_source)!=source:raise RuntimeError('Hard ceiling or source drift')
                game=evaluate_game(policy,deadline=deadline,**context)
                append(folder/'validation.jsonl',game);games.append(game)
                status(status_file,arm,'validation',512,len(games),began)
                print(json.dumps(dict(arm=arm,validation_games=len(games),reason=game['reason'],win=game['crew_win'])),flush=True)
            write(folder/'validation.json',dict(status='development_only',games=games))
            runs.append(dict(arm=arm,initialization_seed=17,result=result,games=games))
        write(output/'comparison.json',compare(runs))
        result=dict(status='development_pilot_complete_acceptance_pending',seconds=perf_counter()-began,runs=runs,
                    final_test_opened=False,seed29_started=False,gui_state_file=Path(gui_state).name)
        write(output/'pilot.json',result)
        status(status_file,'shaped','pilot complete — review pending',512,20,began)
        return result
    except BaseException as error:
        write(output/'failure.json',dict(status='incomplete_preserve_outputs',reason=repr(error),completed_arms=len(runs),seconds=perf_counter()-began))
        atomic(status_file,dict(condition='Pilot stopped',stage=repr(error),elapsed_seconds=perf_counter()-began,updated_unix=time()))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','spectator-source','replay-dir','status-file','approval','proof','gui-state','old-smoke'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    pilot(a.output,a.spectator_source,a.replay_dir,a.status_file,json.loads(a.approval.read_text()),
          json.loads(a.proof.read_text()),a.gui_state,json.loads(a.old_smoke.read_text()))


if __name__=='__main__':main()
