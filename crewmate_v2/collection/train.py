"""Approval-gated vector PPO proposal; benchmarks cannot authorize real training."""
import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from crewmate_v2.learning.policy import V2Policy
from crewmate_v2.learning.train import write, append, state_hash
from crewmate_v2.learning.provenance import digest
from .collector import Collector
from .config import proposal, lock, validate_approval
from .ppo import update, save, resume


def run_one(output, arm, workers, deadline, approval, resume_from=None):
    validate_approval(approval, workers)
    if arm not in ('original', 'shaped'):
        raise ValueError('Unregistered reward arm')
    source = lock()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    episodes, updates = [], []
    simulated, forced, completed = 0., 0, 0
    actions = Counter()
    opportunities = [dict(bodies=[], reports=[], ballots=[]) for _ in range(8)]
    with Collector(workers=workers, arm=arm, partition='train', deadline=deadline) as collector:
        torch.manual_seed(17)
        policy = V2Policy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=.0003)
        rng = np.random.default_rng(17)
        initial = state_hash(policy)
        if resume_from is not None:
            policy, optimizer, rng, completed, evidence = resume(
                resume_from, lambda p: torch.optim.Adam(p, lr=.0003), collector)
            if completed not in (128,256,384):
                raise ValueError('Resume only from an unfinished registered update boundary')
            episodes, updates = evidence['episodes'], evidence['updates']
            simulated, forced = evidence['simulated'], evidence['forced']
            actions = Counter(evidence['actions'])
            opportunities = evidence['opportunities']
            initial = evidence['initial']
        write(output/'config.json', dict(runtime=source, execution=proposal(workers), arm=arm,
              initialization_sha256=initial, resume_checkpoint=Path(resume_from).name if resume_from else None))
        while completed < 512:
            if perf_counter() >= deadline or lock() != source:
                raise RuntimeError('Wall ceiling or source/runtime drift')
            rollout = collector.collect(policy,128)
            if perf_counter() >= deadline:
                raise TimeoutError('Compute ceiling before PPO update')
            diagnostics = update(policy,optimizer,rollout,rng)
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
            save(output/f'checkpoint-{completed}.pt',policy,optimizer,rng,collector,completed,
                 evidence=dict(episodes=episodes,updates=updates,simulated=simulated,forced=forced,
                               actions=dict(actions),opportunities=opportunities,initial=initial))
        partial = collector.snapshot()['environments']
    result = dict(status='trained_pending_experimental_review',arm=arm,initialization_seed=17,
                  actionable_decisions=completed,simulated_seconds=simulated,coalesced_forced_windows=forced,
                  seconds=perf_counter()-began,episodes=episodes,episodes_completed=len(episodes),
                  partial_streams=partial,actions=dict(actions),updates=updates,
                  initial_state_sha256=initial,final_state_sha256=state_hash(policy),
                  final_test_opened=False)
    write(output/'result.json',result)
    return result


def pilot(output, workers, approval, benchmark_record, previous_smoke):
    validate_approval(approval,workers)  # Before creating a directory or environment.
    source = lock()
    if (benchmark_record.get('status') != 'completed_engineering_only'
            or benchmark_record['registration']['runtime'] != source
            or not benchmark_record.get('exact_fixed_configuration_reproduction')
            or benchmark_record['recommended_workers'] != workers):
        raise ValueError('Matching complete benchmark/reproduction record required')
    rows = [r for r in benchmark_record['cases'] if r['batched_inference'] and r['workers']==workers]
    rows.append(benchmark_record['repeat_best'])
    # Scale full startup-inclusive engineering measurements; retain measured historical
    # serial validation cost, since no validation partitions are opened for engineering.
    training = 2*512*max(r['seconds']/r['decisions'] for r in rows)
    validation = 40*900*max(r['seconds']/r['simulated_seconds'] for r in previous_smoke['runs'])
    projected = 2*(training+validation)
    if projected > 10800:
        raise TimeoutError('Projected pilot exceeds three-hour ceiling; revised approval required')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    started = perf_counter()
    deadline = started+10800
    write(output/'registration.json',dict(runtime=source,execution=proposal(workers),
          approval=approval,projected_seconds=projected,training_projection=training,validation_projection=validation))
    from crewmate_v2.learning.evaluate import evaluate_game,validation_contexts
    from crewmate_v2.learning.comparison import compare
    runs = []
    try:
        for arm in ('original','shaped'):
            folder = output/f'{arm}-17'
            result = run_one(folder,arm,workers,deadline,approval)
            policy,_ = V2Policy.load(folder/'checkpoint-512.pt')
            games = []
            for context in validation_contexts():
                game = evaluate_game(policy,deadline=deadline,**context)
                append(folder/'validation.jsonl',game)
                games.append(game)
            runs.append(dict(arm=arm,initialization_seed=17,result=result,games=games))
        write(output/'comparison.json',compare(runs))
        write(output/'pilot.json',dict(status='pilot_complete_acceptance_pending',runs=runs,
                                      seconds=perf_counter()-started,final_test_opened=False))
    except BaseException as error:
        write(output/'failure.json',dict(status='stopped_preserve_outputs',reason=repr(error),completed_runs=len(runs)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,choices=(1,2,4,8),required=True)
    parser.add_argument('--approval',type=Path,required=True)
    parser.add_argument('--benchmark-record',type=Path,required=True)
    parser.add_argument('--previous-smoke',type=Path,required=True)
    args = parser.parse_args()
    pilot(args.output,args.workers,json.loads(args.approval.read_text()),
          json.loads(args.benchmark_record.read_text()),json.loads(args.previous_smoke.read_text()))


if __name__ == '__main__':
    main()
