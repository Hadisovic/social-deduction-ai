"""Same PPO coefficients/update budgets, with GAE isolated at each stream boundary."""
import numpy as np
import torch
from phase5_training import time_aware_gae
from crewmate_v2.learning.train import state_hash
from crewmate_v2.learning.policy import V2Policy
from crewmate_v2.learning.provenance import runtime_lock
from crewmate_v2.config import LobbySettings
from .config import lock


def advantages(rollout):
    output = [np.empty(len(rollout['actions']), np.float32) for _ in range(2)]
    cursor = 0
    for lo, hi in rollout['segments']:
        if lo != cursor or hi <= lo:
            raise ValueError('Rollout segments must partition contiguous stream trajectories')
        values = time_aware_gae(*[np.asarray(rollout[k][lo:hi]) for k in
                                ('rewards', 'values', 'next_values', 'discounts', 'dones', 'durations')])
        for target, value in zip(output, values):
            target[lo:hi] = value
        cursor = hi
    if cursor != len(rollout['actions']):
        raise ValueError('Incomplete stream partition')
    return output


def update(policy, optimizer, rollout, rng):
    packets = {k: torch.as_tensor(np.stack([p[k] for p in rollout['packets']])) for k in rollout['packets'][0]}
    actions = torch.tensor(rollout['actions'], dtype=torch.long)
    old = torch.tensor(rollout['logp'], dtype=torch.float32)
    advantage, returns = [torch.from_numpy(a) for a in advantages(rollout)]
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    losses, norms = [], []
    for _ in range(4):
        for ids in np.array_split(rng.permutation(len(actions)), max(1, int(np.ceil(len(actions) / 64)))):
            distribution, value = policy({k: v[ids] for k, v in packets.items()})
            ratio = (distribution.log_prob(actions[ids]) - old[ids]).exp()
            actor_loss = -torch.minimum(ratio * advantage[ids], ratio.clamp(.8, 1.2) * advantage[ids]).mean()
            critic_loss = (value - returns[ids]).square().mean()
            loss = actor_loss + .5 * critic_loss - .02 * distribution.entropy().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite PPO loss')
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
            optimizer.step()
            losses.append(float(loss.detach()))
            norms.append(float(norm))
    with torch.no_grad():
        distribution, value = policy(packets)
        log_ratio = distribution.log_prob(actions) - old
        ratio = log_ratio.exp()
        variance = returns.var(unbiased=False)
        explained = 1 - (returns - value).var(unbiased=False) / variance if variance > 1e-8 else None
    diagnostics = dict(loss=float(np.mean(losses)), gradient_norm=float(np.mean(norms)),
                       entropy=float(distribution.entropy().mean()),
                       approximate_kl=float(((ratio - 1) - log_ratio).mean()),
                       clip_fraction=float(((ratio - 1).abs() > .2).float().mean()),
                       explained_variance=float(explained) if explained is not None else None)
    if any(v is not None and not np.isfinite(v) for v in diagnostics.values()):
        raise FloatingPointError('Nonfinite PPO diagnostic')
    if diagnostics['approximate_kl'] > .10 or diagnostics['clip_fraction'] > .80 or diagnostics['gradient_norm'] > 100:
        raise RuntimeError('Registered optimizer stability stop')
    if any(not torch.isfinite(p).all() for p in policy.parameters()):
        raise FloatingPointError('Nonfinite optimizer weights')
    return diagnostics


def save(path, policy, optimizer, rng, collector, completed, evidence=None):
    policy.save(path, dict(runtime=runtime_lock(LobbySettings()), collection_runtime=lock(),
                           policy_sha256=state_hash(policy), training_state=dict(
                               optimizer=optimizer.state_dict(), update_rng=rng.bit_generator.state,
                               torch_rng=torch.get_rng_state(), collector=collector.snapshot(), completed=completed,
                               evidence=evidence or {})))


def resume(path, optimizer_factory, collector):
    policy, metadata = V2Policy.load(path)
    if metadata.get('collection_runtime') != lock():
        raise ValueError('Collection source/runtime drift')
    if state_hash(policy) != metadata['policy_sha256']:
        raise ValueError('Collection checkpoint policy mismatch')
    state = metadata['training_state']
    optimizer = optimizer_factory(policy.parameters())
    optimizer.load_state_dict(state['optimizer'])
    rng = np.random.default_rng()
    rng.bit_generator.state = state['update_rng']
    collector.restore(state['collector'])
    torch.set_rng_state(state['torch_rng'])
    return policy, optimizer, rng, state['completed'], state['evidence']
