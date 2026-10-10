"""Paired validation workers; no route to final-test partitions exists here."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter, time
import numpy as np
import torch
from phase5_policy import StrategicPolicy, tensors
from .config import Experiment, ROOT, digest, seeds
from .env import Phase6Env
from .train import load_policy
from .study_support import (append_json, episode_metrics, guard, packet_guard,
                            runtime_versions, trace_step, transition_guard, write_json)

CONTROLS = ('idle', 'random', 'scripted', 'phase5_release_transfer')


def validation_manifest(count, offset=0):
    if type(count) is not int or count <= 0 or type(offset) is not int or offset < 0:
        raise ValueError('Invalid validation offset')
    manifest = seeds('validation', count+offset)
    return manifest[offset:]


def play_match(env, method, policy, seed, deadline, stop_file=None):
    packet, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed ^ 999)
    actions, ends, entropies = Counter(), Counter(), []
    decisions = simulated = 0
    trace = hashlib.sha256()
    while True:
        if stop_file and Path(stop_file).exists(): raise RuntimeError('Another validation job failed; stop')
        if time() > deadline: raise TimeoutError('Evaluation deadline exceeded')
        packet_guard(packet)
        before = packet
        if method == 'scripted':
            intent = env.match.controllers[env.focal].decide(env.observation)
            actions[intent.kind.value] += 1
            action = intent.kind.value
            packet, reward, done, _, info = env.step_scripted(intent)
        else:
            action = 0 if method == 'idle' else int(rng.choice(np.flatnonzero(packet['mask']))) if policy is None else policy.act(packet, True)[0]
            if not packet['mask'][action]: raise ValueError('Masked evaluation action')
            actions[env.options[action].kind] += 1
            if policy is not None:
                with torch.no_grad(): entropies.append(float(policy(tensors(packet))[0].entropy().item()))
            packet, reward, done, _, info = env.step(action)
        transition_guard(env, info, reward)
        trace_step(trace, before, action, reward, info, done)
        decisions += 1; simulated += info['elapsed_seconds']; ends[info['option_end']] += 1
        if done: break
    return dict(method=method, family='ID', seed=seed, actor_transition_sha256=trace.hexdigest(),
                **episode_metrics(env, info['episode'], actions, ends, decisions, simulated, entropies))


def evaluate_job(job):
    output = Path(job['output']); output.mkdir(parents=True, exist_ok=False)
    rows = []; began = perf_counter()
    try:
        if job.get('stop_file') and Path(job['stop_file']).exists():
            raise RuntimeError('Study stop flag exists')
        if job.get('partition') != 'validation':
            raise ValueError('Only validation is authorized; final partitions remain closed')
        manifest = validation_manifest(job['count'], job.get('offset', 0))
        guard(job['sources'], job['deadline'])
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        method = job['method']
        if method in CONTROLS:
            experiment = Experiment(); policy = None
            if method == 'phase5_release_transfer':
                policy, _ = StrategicPolicy.load(ROOT/'artifacts/phase5/release/policy.pt')
        else:
            policy, experiment, metadata = load_policy(job['checkpoint'])
            if metadata.get('smoke_only') or metadata.get('initialization_seed') != 17:
                raise ValueError('Study only accepts authorized seed-17 research checkpoints')
            if metadata.get('runtime_versions') != runtime_versions():
                raise ValueError('Checkpoint runtime dependency versions changed')
            if experiment != Experiment(memory=job['condition']):
                raise ValueError('Checkpoint conditions differ from the registered comparison')
            if digest(job['checkpoint']) != job['checkpoint_sha256']:
                raise ValueError('Checkpoint digest changed')
        env = Phase6Env(experiment)
        write_json(output/'job.json', {**job, 'match_seeds':manifest, 'experiment':experiment.to_dict()}, exclusive=True)
        for seed in manifest:
            guard(job['sources'], job['deadline'])
            row = play_match(env, method, policy, seed, job['deadline'], job.get('stop_file'))
            rows.append(row); append_json(output/'matches.jsonl', row)
            write_json(output/'status.json', dict(status='validation', matches=len(rows), seconds=perf_counter()-began))
        guard(job['sources'], job['deadline'])
        write_json(output/'complete.json', dict(status='validation_complete', matches=len(rows),
                   seconds=perf_counter()-began, matches_sha256=digest(output/'matches.jsonl')), exclusive=True)
        print(f'{method}: validation {len(rows)} matches in {perf_counter()-began:.1f}s', flush=True)
        return rows
    except BaseException as error:
        if job.get('stop_file'):
            try: write_json(job['stop_file'], dict(error=repr(error), method=job['method']), exclusive=True)
            except FileExistsError: pass
        write_json(output/'failure.json', dict(status='failed', error=repr(error), matches=len(rows), seconds=perf_counter()-began))
        raise


def select_checkpoint(condition, rows, checkpoints):
    """Frozen ranking: wins, own tasks, earlier step. No final-test values."""
    scored = []
    for checkpoint in checkpoints:
        method = f"{condition}@{checkpoint['decisions']}"
        subset = [row for row in rows if row['method'] == method]
        expected = set(validation_manifest(20))
        if len(subset) != 20 or {r['seed'] for r in subset} != expected:
            raise ValueError('Selection requires exactly the registered 20 validation seeds')
        if any(r['family'] != 'ID' or r['illegal_actions'] or r['navigation_failures'] for r in subset):
            raise ValueError('Unsafe or nonvalidation selection data')
        scored.append((sum(r['crew_win'] for r in subset), sum(r['own_tasks'] for r in subset),
                       -checkpoint['decisions'], checkpoint))
    return max(scored, key=lambda item: item[:3])[-1]
