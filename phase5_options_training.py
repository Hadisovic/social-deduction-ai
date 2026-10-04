"""PPO option experiments with optimization diagnostics and immutable run records."""
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from phase5_options import CrewmateOptionsEnv, PROTOCOL, MAX_OPTION_SECONDS
from phase5_policy import StrategicPolicy, tensors
from phase5_training import time_aware_gae

ROOT = Path(__file__).resolve().parent


def ppo_update(policy, optimizer, rollout, entropy, rng):
    packets = {k: torch.as_tensor(np.stack([p[k] for p in rollout['packets']]))
               for k in rollout['packets'][0]}
    actions = torch.tensor(rollout['actions'], dtype=torch.long)
    old = torch.tensor(rollout['logp'], dtype=torch.float32)
    advantage, returns = time_aware_gae(*[np.asarray(rollout[k]) for k in
        ('rewards', 'values', 'next_values', 'discounts', 'dones', 'durations')])
    advantage = torch.from_numpy(advantage)
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    returns = torch.from_numpy(returns.astype(np.float32))
    losses, policy_losses, value_losses, norms = [], [], [], []
    for _ in range(4):
        for ids in np.array_split(rng.permutation(len(actions)), max(1, int(np.ceil(len(actions) / 64)))):
            distribution, value = policy({k: v[ids] for k, v in packets.items()})
            ratio = (distribution.log_prob(actions[ids]) - old[ids]).exp()
            actor_loss = -torch.minimum(ratio * advantage[ids], ratio.clamp(.8, 1.2) * advantage[ids]).mean()
            critic_loss = (value - returns[ids]).square().mean()
            loss = actor_loss + .5 * critic_loss - entropy * distribution.entropy().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite PPO loss')
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
            optimizer.step()
            losses.append(float(loss.detach()))
            policy_losses.append(float(actor_loss.detach()))
            value_losses.append(float(critic_loss.detach()))
            norms.append(float(norm))
    with torch.no_grad():
        distribution, value = policy(packets)
        log_ratio = distribution.log_prob(actions) - old
        ratio = log_ratio.exp()
        variance = returns.var(unbiased=False)
        explained = 1 - (returns - value).var(unbiased=False) / variance if variance > 1e-8 else None
    return dict(loss=float(np.mean(losses)), policy_loss=float(np.mean(policy_losses)),
                value_loss=float(np.mean(value_losses)), gradient_norm=float(np.mean(norms)),
                entropy=float(distribution.entropy().mean()),
                approximate_kl=float(((ratio - 1) - log_ratio).mean()),
                clip_fraction=float(((ratio - 1).abs() > .2).float().mean()),
                explained_variance=float(explained) if explained is not None else None,
                entropy_coefficient=entropy)


def train(directory, steps=8192, seed=17, ablation=False):
    if steps < 256 or steps % 256:
        raise ValueError('Use a positive multiple of 256 transitions')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError('Use an empty output directory; never overwrite a run')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    config = json.loads((ROOT / 'artifacts/phase5/config.json').read_text())
    config.update(seed=seed, ablation=ablation, steps=steps, quick=False,
                  execution_protocol=PROTOCOL, max_option_seconds=MAX_OPTION_SECONDS,
                  selection_split='validation only; never final-test seeds')
    names = ('phase5_features.py', 'phase5_env.py', 'phase5_options.py', 'phase5_policy.py',
             'phase5_runner.py', 'phase5_training.py', 'phase5_options_training.py', 'among_us_map.py')
    sources = {name: hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
               for name in names}
    config.update(sources_sha256=sources, features_sha256=sources['phase5_features.py'],
                  geometry_sha256=sources['among_us_map.py'], options_sha256=sources['phase5_options.py'])
    (directory / 'run-config.json').write_text(json.dumps(config, indent=2) + '\n')
    env = CrewmateOptionsEnv(ablation=ablation)
    policy = StrategicPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    rng = np.random.default_rng(seed)
    frozen = {k: v.clone() for k, v in env.belief_model.network.state_dict().items()}
    initial = {k: v.clone() for k, v in policy.state_dict().items()}
    episode_index = 0
    packet, _ = env.reset(seed=600000)
    completed = 0
    simulated_seconds = 0.
    episodes, history = [], []
    began = perf_counter()
    while completed < steps:
        rollout = {k: [] for k in ('packets', 'actions', 'logp', 'values', 'next_values',
                                   'rewards', 'discounts', 'dones', 'durations')}
        for _ in range(min(256, steps - completed)):
            action, logp, value = policy.act(packet)
            new, reward, done, _, info = env.step(action)
            with torch.no_grad():
                next_value = float(policy(tensors(new))[1].item()) if not done else 0.
            for key, item in zip(rollout, (packet, action, logp, value, next_value, reward,
                                           info['discount'], done, info['elapsed_seconds'])):
                rollout[key].append(item)
            completed += 1
            simulated_seconds += info['elapsed_seconds']
            if done:
                episodes.append(dict(match_seed=600000 + episode_index, training_step=completed,
                                     simulated_seconds=simulated_seconds, **info['episode']))
                with (directory / 'episodes.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(episodes[-1]) + '\n')
                episode_index += 1
                if episode_index >= 20000:
                    raise ValueError('Training seed partition exhausted')
                packet, _ = env.reset(seed=600000 + episode_index)
            else:
                packet = new
        # Keep exploration during the shorter diagnostic experiment.
        entropy = .02 + (.005 - .02) * completed / steps
        diagnostics = ppo_update(policy, optimizer, rollout, entropy, rng)
        row = dict(steps=completed, episodes=len(episodes), simulated_seconds=simulated_seconds,
                   mean_option_seconds=float(np.mean(rollout['durations'])),
                   seconds=perf_counter() - began, **diagnostics)
        history.append(row)
        with (directory / 'updates.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        if completed % 2048 == 0:
            policy.save(directory / f'checkpoint-{completed}.pt', {**config, 'steps': completed})
    assert all(torch.equal(v, frozen[k]) for k, v in env.belief_model.network.state_dict().items())
    assert any(not torch.equal(v, initial[k]) for k, v in policy.state_dict().items())
    config.update(belief_frozen_verified=True, weights_changed=True)
    policy.save(directory / 'policy.pt', config)
    result = dict(config=config, history=history, episodes=episodes, seconds=perf_counter()-began,
                  checkpoint_sha256=hashlib.sha256((directory/'policy.pt').read_bytes()).hexdigest(),
                  status='trained_not_evaluated')
    (directory / 'training.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--ablation', action='store_true')
    args = parser.parse_args()
    train(args.output, args.steps, args.seed, args.ablation)
