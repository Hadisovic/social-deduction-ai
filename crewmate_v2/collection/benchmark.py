"""Bounded engineering benchmarks; never the reward-comparison training pilot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import numpy as np
import torch
from crewmate_v2.learning.policy import V2Policy
from crewmate_v2.learning.train import write, state_hash
from .collector import Collector
from .config import lock
from .ppo import update, save
from .resources import Resources, hardware


def benchmark_one(output, workers, decisions=64, baseline=False, ceiling=180, batched=True):
    if decisions not in (32, 64) or not 0 < ceiling <= 240:
        raise ValueError('Engineering benchmark only: 32/64 decisions and at most four minutes')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source = lock()
    started = perf_counter()
    with Resources(include_parent_startup=True) as resources:
        with Collector(workers=workers, deadline=started + ceiling, on_spawn=resources.watch) as collector:
            initialized = perf_counter()
            torch.manual_seed(17)
            policy = V2Policy()
            optimizer = torch.optim.Adam(policy.parameters(), lr=.0003)
            rng = np.random.default_rng(17)
            initial = state_hash(policy)
            collection_started = perf_counter()
            rollout = collector.collect(policy, decisions, batched=batched and not baseline, original_inference=baseline)
            collected = perf_counter()
            diagnostics = update(policy, optimizer, rollout, rng)
            optimized = perf_counter()
            final = state_hash(policy)
            if initial == final:
                raise ValueError('Engineering update did not change disposable weights')
            trace = [t.hexdigest() for t in collector.traces]
            snapshot = collector.snapshot()
            save(output/'checkpoint.pt', policy, optimizer, rng, collector, decisions)
            loaded, _ = V2Policy.load(output/'checkpoint.pt')
            if state_hash(loaded) != final or lock() != source:
                raise ValueError('Checkpoint roundtrip/source check failed')
            end = perf_counter()
            if end > started + ceiling:
                raise TimeoutError('Engineering wall ceiling reached')
    seconds = perf_counter() - started
    result = dict(status='engineering_only_not_training_evidence', workers=workers, logical_environments=8,
                  baseline_unbatched_double_forward=baseline, decisions=decisions, optimizer_updates=1,
                  batched_inference=batched and not baseline,
                  epochs=4, minibatches_per_epoch=1, seconds=seconds,
                  startup_seconds=initialized-started, collection_seconds=collected-collection_started,
                  optimization_seconds=optimized-collected, checkpoint_seconds=end-optimized,
                  simulated_seconds=float(sum(rollout['durations'])), decisions_per_second=decisions/seconds,
                  simulated_seconds_per_real_second=float(sum(rollout['durations']))/seconds,
                  steady_collection_decisions_per_second=decisions/(collected-collection_started),
                  initial_sha256=initial, final_sha256=final, actor_action_trace_sha256=trace,
                  stream_packet_sha256={s:r['packet_sha256'] for s,r in snapshot['environments'].items()},
                  completed_episodes=len(rollout['episodes']), partial_streams=8,
                  forced_windows=sum(t['info']['coalesced_forced_windows'] for t in rollout['transition_audits']),
                  reward_components={k:sum(t['info']['reward_components'].get(k,0.) for t in rollout['transition_audits'])
                                     for k in sorted({k for t in rollout['transition_audits'] for k in t['info']['reward_components']})},
                  diagnostics=diagnostics, resources=resources.result,
                  runtime=source, checkpoint_sha256=hashlib.sha256((output/'checkpoint.pt').read_bytes()).hexdigest(),
                  final_test_opened=False, pilot_started=False)
    write(output/'result.json', result)
    print(json.dumps({k:result[k] for k in ('workers','baseline_unbatched_double_forward','seconds','decisions_per_second','simulated_seconds_per_real_second')}), flush=True)
    return result


def fresh_case(output, workers, baseline, batched, ceiling):
    """Every configuration pays its interpreter/import and process startup cost."""
    command = [sys.executable,'-m','crewmate_v2.collection.benchmark','--case',
               '--workers',str(workers),'--output',str(output),'--wall-seconds',str(ceiling)]
    if baseline:
        command.append('--baseline')
    if not batched:
        command.append('--unbatched')
    began = perf_counter()
    process = subprocess.Popen(command)
    try:
        code = process.wait(timeout=ceiling+15)
    except BaseException:
        if os.name == 'nt':
            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
        else:
            process.kill()
        process.wait()
        raise
    seconds = perf_counter()-began
    if code:
        raise RuntimeError(f'Engineering case exited with code {code}; preserve its directory')
    path = Path(output)/'result.json'
    result = json.loads(path.read_text())
    result['process_startup_and_import_seconds'] = seconds-result['seconds']
    result['seconds'] = seconds
    result['decisions_per_second'] = result['decisions']/seconds
    result['simulated_seconds_per_real_second'] = result['simulated_seconds']/seconds
    cpu = result['resources']['cpu_seconds']
    if cpu is not None:
        result['resources']['average_cpu_cores'] = cpu/seconds
        result['resources']['average_machine_cpu_percent'] = 100*cpu/seconds/os.cpu_count()
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def suite(output, ceiling=900):
    if not 0 < ceiling <= 900:
        raise ValueError('Suite engineering ceiling is fifteen minutes')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    registration = dict(status='engineering_only', hardware=hardware(), runtime=lock(),
                        decisions_per_case=64, logical_environments=8, suite_ceiling_seconds=ceiling,
                        seed_partition=[2740000,2749999], initialization=17,
                        no_pilot_or_validation=True)
    write(output/'registration.json', registration)
    rows = []
    try:
        for name, workers, baseline, batched in [('baseline',1,True,False),('unbatched-cached',1,False,False),
                ('workers-1',1,False,True),('workers-2',2,False,True),('workers-4',4,False,True),('workers-8',8,False,True)]:
            remaining = ceiling - (perf_counter()-started)
            if remaining < 15:
                raise TimeoutError('Engineering suite has no safe remaining case budget')
            rows.append(fresh_case(output/name,workers,baseline,batched,min(240,remaining-5)))
        best = min([r for r in rows if r['batched_inference']], key=lambda r:r['seconds'])
        remaining = ceiling-(perf_counter()-started)
        repeated = fresh_case(output/'repeat-best',best['workers'],False,True,min(240,remaining-5))
        keys = ('initial_sha256','final_sha256','actor_action_trace_sha256','stream_packet_sha256','simulated_seconds','forced_windows')
        if any(best[k] != repeated[k] for k in keys):
            raise ValueError('Fixed-configuration deterministic reproduction failed')
        result = dict(status='completed_engineering_only', seconds=perf_counter()-started, registration=registration,
                      cases=rows, repeat_best=repeated, recommended_workers=best['workers'],
                      exact_fixed_configuration_reproduction=True,
                      speedup_vs_matched_serial=rows[0]['seconds']/best['seconds'])
        write(output/'summary.json', result)
        return result
    except BaseException as error:
        write(output/'failure.json', dict(status='stopped_preserve_outputs', reason=repr(error), completed_cases=len(rows)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wall-seconds',type=float,default=900)
    parser.add_argument('--case',action='store_true')
    parser.add_argument('--workers',type=int,choices=(1,2,4,8),default=1)
    parser.add_argument('--baseline',action='store_true')
    parser.add_argument('--unbatched',action='store_true')
    args = parser.parse_args()
    if args.case:
        benchmark_one(args.output,args.workers,baseline=args.baseline,ceiling=args.wall_seconds,batched=not args.unbatched)
    else:
        suite(args.output,args.wall_seconds)


if __name__ == '__main__':
    main()
