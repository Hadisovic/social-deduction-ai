"""Approved seed-17 training; the separate smoke command keeps its original cap."""
from collections import Counter
import hashlib
from pathlib import Path
from time import perf_counter, time
import numpy as np
import torch
from phase5_options_training import ppo_update
from phase5_policy import StrategicPolicy, tensors
from phase5_training import time_aware_gae
from .config import BASELINE_SHA, FROZEN, Experiment, digest, seeds, source_hashes
from .env import Phase6Env
from .study_support import (append_json, episode_metrics, finite, guard, packet_guard,
                            runtime_versions, state_hash, trace_step, transition_guard, write_json)

CONDITIONS = ('full', 'history_no_belief', 'current_only')
CHECKPOINTS = (2048, 4096, 6144, 8192)


def train_run(output, condition, expected_sources, revision, deadline, *, verification=False):
    if condition not in CONDITIONS or type(verification) is not bool:
        raise ValueError('Unregistered staged training condition')
    steps = 32 if verification else 8192
    partition_name = 'smoke_train' if verification else 'train'
    partition = seeds(partition_name, 100 if verification else 20000)
    experiment = Experiment(memory=condition)
    guard(expected_sources, deadline)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    began = perf_counter()
    completed = simulated = episode = 0
    episodes, updates, checkpoints = [], [], []
    torch.set_num_threads(1); torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(17)
    metadata = dict(experiment=experiment.to_dict(), phase6_sources=expected_sources,
                    baseline_sha=BASELINE_SHA, research_revision=revision, initialization_seed=17,
                    steps=steps, seed_partition=partition_name, frozen_sha256=FROZEN,
                    smoke_only=verification, entropy_coefficient=.02, gamma=.99,
                    runtime_versions=runtime_versions(),
                    selection='validation only: win rate, own tasks, earliest checkpoint')
    write_json(output/'config.json', metadata, exclusive=True)
    try:
        env = Phase6Env(experiment); policy = StrategicPolicy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=experiment.learning_rate)
        frozen = {k: v.clone() for k, v in env.belief_model.network.state_dict().items()}
        initial_hash = state_hash(policy)
        packet, _ = env.reset(seed=partition[episode])
        trace = hashlib.sha256()
        actions, ends, entropies = Counter(), Counter(), []
        episode_decisions = episode_simulated = 0
        while completed < steps:
            guard(expected_sources, deadline)
            rollout = {k: [] for k in ('packets', 'actions', 'logp', 'values', 'next_values',
                                      'rewards', 'discounts', 'dones', 'durations')}
            for _ in range(min(experiment.rollout, steps-completed)):
                if time() > deadline: raise TimeoutError('Training deadline exceeded')
                packet_guard(packet)
                action, logp, value = policy.act(packet)
                if not packet['mask'][action]: raise ValueError('Masked policy action')
                with torch.no_grad():
                    entropies.append(float(policy(tensors(packet))[0].entropy().item()))
                actions[env.options[action].kind] += 1
                new, reward, done, _, info = env.step(action)
                transition_guard(env, info, reward)
                trace_step(trace, packet, action, reward, info, done)
                with torch.no_grad():
                    next_value = 0. if done else float(policy(tensors(new))[1].item())
                finite((logp, value, next_value))
                for key, item in zip(rollout, (packet, action, logp, value, next_value, reward,
                                              info['discount'], done, info['elapsed_seconds'])):
                    rollout[key].append(item)
                completed += 1; simulated += info['elapsed_seconds']
                episode_decisions += 1; episode_simulated += info['elapsed_seconds']
                ends[info['option_end']] += 1
                if done:
                    row = dict(seed=partition[episode], training_step=completed,
                               **episode_metrics(env, info['episode'], actions, ends,
                                                 episode_decisions, episode_simulated, entropies))
                    episodes.append(row); append_json(output/'episodes.jsonl', row)
                    episode += 1
                    if episode >= len(partition): raise ValueError('Training partition exhausted')
                    packet, _ = env.reset(seed=partition[episode])
                    actions, ends, entropies = Counter(), Counter(), []
                    episode_decisions = episode_simulated = 0
                else: packet = new
            diagnostics = ppo_update(policy, optimizer, rollout, .02, rng)
            advantages, returns = time_aware_gae(*[np.asarray(rollout[k]) for k in
                ('rewards', 'values', 'next_values', 'discounts', 'dones', 'durations')])
            diagnostics.update(advantage_mean=float(advantages.mean()), advantage_std=float(advantages.std()),
                               return_std=float(returns.std()), mean_option_seconds=float(np.mean(rollout['durations'])))
            finite(diagnostics)
            if any(not torch.isfinite(p).all() for p in policy.parameters()):
                raise FloatingPointError('Nonfinite policy weights')
            if not all(torch.equal(v, frozen[k]) for k, v in env.belief_model.network.state_dict().items()):
                raise ValueError('Frozen belief weights changed')
            guard(expected_sources, deadline)
            row = dict(decisions=completed, episodes=len(episodes), simulated_seconds=simulated,
                       seconds=perf_counter()-began, **diagnostics)
            updates.append(row); append_json(output/'updates.jsonl', row)
            print(f'{condition}: '+__import__('json').dumps(row), flush=True)
            if completed in CHECKPOINTS:
                checkpoint = output/f'checkpoint-{completed}.pt'
                if checkpoint.exists(): raise FileExistsError(checkpoint)
                policy.save(checkpoint, {**metadata, 'steps': completed})
                checkpoints.append(dict(decisions=completed, name=checkpoint.name, sha256=digest(checkpoint)))
            write_json(output/'status.json', dict(status='training', **row))
        if state_hash(policy) == initial_hash: raise ValueError('Policy weights did not change')
        policy.save(output/'policy.pt', metadata)
        result = dict(status='trained_not_selected', condition=condition, decisions=completed,
                      simulated_seconds=simulated, seconds=perf_counter()-began,
                      episodes_completed=len(episodes), partial_episode_decisions=episode_decisions,
                      partial_episode_simulated_seconds=episode_simulated,
                      partial_episode_own_tasks=env.completed_tasks if episode_decisions else 0,
                      initial_state_sha256=initial_hash, final_state_sha256=state_hash(policy),
                      transition_trace_sha256=trace.hexdigest(), checkpoints=checkpoints,
                      checkpoint_sha256=digest(output/'policy.pt'), belief_frozen=True,
                      policy_weights_changed=True, seed_partition=partition_name,
                      used_match_seeds=list(partition[:episode+1]), updates=updates)
        write_json(output/'training.json', result, exclusive=True)
        write_json(output/'status.json', result)
        return result
    except BaseException as error:
        write_json(output/'failure.json', dict(status='failed', error=repr(error), decisions=completed,
                   simulated_seconds=simulated, seconds=perf_counter()-began, preserved_checkpoints=checkpoints))
        raise
