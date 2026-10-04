"""Validation-only worker. Never opens final ID/patient test partitions."""
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase5_evaluation import evaluate
from phase5_options import PROTOCOL

output = ROOT / 'docs/benchmarks/phase5/expanded'
output.mkdir(parents=True, exist_ok=True)
jobs = [('diagnostic-baselines', [], True)]
jobs += [(f'options-17-{step}', [Path(f'artifacts/phase5/runs/options-17-v2/checkpoint-{step}.pt')], False)
         for step in (2048, 4096, 6144, 8192)]
jobs += [('v1-17-final', [Path('artifacts/phase5/runs/full-17-approved/policy.pt')], False)]
while jobs:
    progressed = False
    for name, checkpoints, baselines in list(jobs):
        if not all((ROOT / path).exists() for path in checkpoints):
            continue
        destination = output / name
        if (destination / 'metrics.json').exists():
            jobs.remove((name, checkpoints, baselines))
            continue
        print(f'Validation job: {name}', flush=True)
        evaluate(destination, 20, checkpoints, validation=True,
                 include_baselines=baselines, random_protocol=PROTOCOL, include_idle=baselines)
        jobs.remove((name, checkpoints, baselines))
        progressed = True
    if jobs and not progressed:
        time.sleep(10)
print(json.dumps({'status': 'diagnostic_validation_complete', 'final_test_opened': False}), flush=True)
