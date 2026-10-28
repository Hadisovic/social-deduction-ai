"""Bounded infrastructure smoke trainer; large batches require a later budget gate."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from phase5_options_training import ppo_update
from phase5_policy import StrategicPolicy, tensors
from phase5_training import time_aware_gae
from .config import BASELINE_SHA, FROZEN, ROOT, digest, load_experiment, seeds, source_hashes, verify_frozen
from .env import Phase6Env


def load_policy(path):
    verify_frozen()
    policy, metadata = StrategicPolicy.load(path)
    if metadata.get('phase6_sources') != source_hashes():
        raise ValueError('Phase 6 checkpoint source/runtime mismatch')
    from .config import Experiment
    return policy, Experiment(**metadata['experiment']), metadata


def train(output, experiment, *, steps=256, seed=17):
    if type(steps) is not int or steps not in (256, 512):
        raise ValueError('Approved development scope: 256 or 512 decisions only')
    verify_frozen()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    env = Phase6Env(experiment)
    policy = StrategicPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=experiment.learning_rate)
    frozen = {k: v.clone() for k, v in env.belief_model.network.state_dict().items()}
    initial = {k: v.clone() for k, v in policy.state_dict().items()}
    partition = seeds('smoke_train', 100)
    episode = 0
    packet, _ = env.reset(seed=partition[episode])
    began = perf_counter()
    completed = 0
    simulated = 0.
    episodes, updates = [], []
    metadata = dict(experiment=experiment.to_dict(), phase6_sources=source_hashes(),
                    baseline_sha=BASELINE_SHA, initialization_seed=seed, steps=steps,
                    seed_partition='smoke_train', frozen_sha256=FROZEN, smoke_only=True)
    (output / 'config.json').write_text(json.dumps(metadata, indent=2) + '\n')
    while completed < steps:
        rollout = {k: [] for k in ('packets', 'actions', 'logp', 'values', 'next_values',
                                   'rewards', 'discounts', 'dones', 'durations')}
        for _ in range(min(experiment.rollout, steps - completed)):
            action, logp, value = policy.act(packet)
            new, reward, done, _, info = env.step(action)
            with torch.no_grad():
                next_value = 0. if done else float(policy(tensors(new))[1].item())
            for key, item in zip(rollout, (packet, action, logp, value, next_value, reward,
                                          info['discount'], done, info['elapsed_seconds'])):
                rollout[key].append(item)
            completed += 1
            simulated += info['elapsed_seconds']
            if done:
                episodes.append(dict(seed=partition[episode], **info['episode']))
                episode += 1
                if episode >= len(partition):
                    raise ValueError('Smoke seed partition exhausted')
                packet, _ = env.reset(seed=partition[episode])
            else:
                packet = new
        diagnostics = ppo_update(policy, optimizer, rollout, .02, rng)
        advantages, returns = time_aware_gae(*[np.asarray(rollout[k]) for k in
            ('rewards', 'values', 'next_values', 'discounts', 'dones', 'durations')])
        diagnostics.update(advantage_mean=float(advantages.mean()),
                           advantage_std=float(advantages.std()), return_std=float(returns.std()))
        updates.append(dict(decisions=completed, simulated_seconds=simulated,
                            seconds=perf_counter() - began, **diagnostics))
        print(json.dumps(updates[-1]), flush=True)
        with (output / 'updates.jsonl').open('a') as f:
            f.write(json.dumps(updates[-1]) + '\n')
    assert all(torch.equal(v, frozen[k]) for k, v in env.belief_model.network.state_dict().items())
    assert any(not torch.equal(v, initial[k]) for k, v in policy.state_dict().items())
    verify_frozen()
    policy.save(output / 'policy.pt', metadata)
    result = dict(status='infrastructure_smoke_only', decisions=completed, episodes=episodes,
                  simulated_seconds=simulated, seconds=perf_counter() - began,
                  checkpoint_sha256=digest(output / 'policy.pt'), belief_frozen=True,
                  policy_weights_changed=True, updates=updates)
    (output / 'training.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    train(args.output, load_experiment(args.config), steps=args.steps, seed=args.seed)


if __name__ == '__main__':
    main()
