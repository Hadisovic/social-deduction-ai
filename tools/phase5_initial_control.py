"""Reconstruct the exact random initialization for a completed/in-progress V2 run.

This control distinguishes learning from a lucky deterministic initial preference.
It follows the trainer's RNG and construction order and verifies unchanged sources.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase5_options import CrewmateOptionsEnv
from phase5_policy import StrategicPolicy
from phase5_evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    run = ROOT / f'artifacts/phase5/runs/options-{args.seed}-v2'
    config = json.loads((run/'run-config.json').read_text())
    for name, digest in config['sources_sha256'].items():
        current = hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        if current != digest:
            raise ValueError(f'Cannot reconstruct initialization after source change: {name}')
    directory = ROOT / f'artifacts/phase5/runs/options-{args.seed}-initial-control'
    directory.mkdir(parents=True,exist_ok=True)
    checkpoint = directory/'policy.pt'
    if not checkpoint.exists():
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.use_deterministic_algorithms(True)
        env = CrewmateOptionsEnv(ablation=False)
        policy = StrategicPolicy()
        metadata = {**config, 'steps':0, 'initial_control':True,
                    'initialization_reconstructed':True,
                    'notice':'Same source hashes, seed, environment-before-policy construction order; no PPO updates.'}
        policy.save(checkpoint,metadata)
        (directory/'run-config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    output = ROOT / f'docs/benchmarks/phase5/expanded/initial-{args.seed}'
    evaluate(output,20,[checkpoint.relative_to(ROOT)],validation=True,include_baselines=False)


if __name__ == '__main__':
    main()
