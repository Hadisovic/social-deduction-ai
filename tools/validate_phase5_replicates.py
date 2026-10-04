"""Measure replicated checkpoints and reserve final tests for a later locked selection.

Safe to restart only between completed jobs: completed metrics are reused; a
partial journal is surfaced for inspection rather than appended or erased.
"""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase5_evaluation import evaluate
from phase5_options import PROTOCOL

OUTPUT = ROOT / 'docs/benchmarks/phase5/expanded'
RUNS = ('options-17-v2', 'options-29-v2', 'options-43-v2', 'options-17-v2-ablation')


def measure(name, checkpoint, matches):
    destination = OUTPUT / name
    if (destination / 'metrics.json').exists():
        return json.loads((destination / 'metrics.json').read_text())
    return evaluate(destination, matches, [Path(checkpoint)] if checkpoint else [],
                    validation=True, include_baselines=checkpoint is None,
                    include_idle=checkpoint is None, random_protocol=PROTOCOL)


def diagnostic_name(run, step):
    return f'options-17-{step}' if run == 'options-17-v2' else f'{run}-{step}'


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for run in RUNS:
        for step in (2048, 4096, 6144, 8192):
            # The seed-17 diagnostic worker owns these directories already.
            if run == 'options-17-v2':
                continue
            checkpoint = f'artifacts/phase5/runs/{run}/checkpoint-{step}.pt'
            jobs.append((diagnostic_name(run, step), checkpoint, 20))
    in_flight = {}
    with ProcessPoolExecutor(max_workers=3) as pool:
        while jobs or in_flight:
            for name, checkpoint, matches in list(jobs):
                if len(in_flight) >= 3:
                    break
                path = ROOT / checkpoint
                if not path.exists() or time.time() - path.stat().st_mtime < 2:
                    continue
                future = pool.submit(measure, name, checkpoint, matches)
                in_flight[future] = name
                jobs.remove((name, checkpoint, matches))
                print(f'Started validation: {name}', flush=True)
            for future in list(in_flight):
                if future.done():
                    future.result()
                    print(f'Finished validation: {in_flight.pop(future)}', flush=True)
            if jobs or in_flight:
                time.sleep(5)
        while not all((OUTPUT/diagnostic_name('options-17-v2',s)/'metrics.json').exists()
                      for s in (2048,4096,6144,8192)):
            time.sleep(5)
        finalists = []
        for run in RUNS:
            candidates = []
            for step in (2048,4096,6144,8192):
                record = json.loads((OUTPUT/diagnostic_name(run,step)/'metrics.json').read_text())
                method, families = next(iter(record['metrics'].items()))
                r = families['ID']
                if r['illegal_actions'] or r['navigation_failures']:
                    continue
                candidates.append((r['crew_win_rate'], r['mean_own_tasks'],
                                   r['voting_accuracy'] or 0., -step, method))
            if not candidates:
                raise RuntimeError(f'No mechanically valid candidate for {run}')
            finalists.append(dict(run=run, checkpoint=max(candidates)[-1]))
        selection = dict(status='checkpoint_screening_only', finalists=finalists,
                         basis='20 shared validation seeds; 100-match validation follows',
                         final_test_opened=False)
        (OUTPUT/'screening-selection.json').write_text(json.dumps(selection,indent=2)+'\n')
        jobs = [('selection-baselines', None, 100)]
        jobs += [(f"selection-{f['run']}", f['checkpoint'], 100) for f in finalists]
        jobs += [('selection-v1-17', 'artifacts/phase5/runs/full-17-approved/policy.pt', 100)]
        futures = {pool.submit(measure, *job): job[0] for job in jobs}
        for future, name in futures.items():
            future.result()
            print(f'Finished selection validation: {name}', flush=True)
    from phase5_research_report import build
    build()
    print(json.dumps({'status':'validation_ready_for_review','final_test_opened':False,
                      'next':'Inspect paired comparisons and learning curves before more training or final tests.'}),flush=True)


if __name__ == '__main__':
    main()
