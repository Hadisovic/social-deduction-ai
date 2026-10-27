"""Deterministic barrier collection with one central policy and independent samplers."""
import hashlib
import multiprocessing as mp
from time import perf_counter

import numpy as np
import torch
from torch.distributions import Categorical
from crewmate_v2.learning.policy import tensors, check_packet
from crewmate_v2.learning.train import actor_trace
from crewmate_v2.config import LobbySettings
from .bank import EnvironmentBank, worker_main

FIELDS = ('packets', 'actions', 'logp', 'values', 'next_values', 'rewards', 'discounts', 'dones', 'durations')


@torch.no_grad()
def forward(policy, packets, batched=True):
    for packet in packets:
        check_packet(packet)
    if batched:
        inputs = {k: torch.as_tensor(np.stack([p[k] for p in packets])) for k in packets[0]}
        distribution, values = policy(inputs)
        return distribution.probs, values
    rows = [policy(tensors(p)) for p in packets]
    return torch.cat([d.probs for d, _ in rows]), torch.cat([v for _, v in rows])


class Collector:
    def __init__(self, workers=1, streams=8, arm='original', partition='engineering',
                 initialization=17, deadline=None, lobby=None, on_spawn=None):
        if workers not in (1, 2, 4, 8) or streams < workers or streams % workers:
            raise ValueError('Use equally partitioned supported workers')
        self.workers, self.streams = workers, streams
        self.arm, self.partition = arm, partition
        self.lobby = lobby or LobbySettings()
        self.deadline = deadline or perf_counter() + 180
        self.connections, self.processes = [], []
        self.bank = None
        self.generators = []
        for slot in range(streams):
            seed = int.from_bytes(hashlib.sha256(f'v2-actions:{initialization}:{slot}'.encode()).digest()[:8], 'big')
            self.generators.append(torch.Generator().manual_seed(seed))
        self.traces = [hashlib.sha256() for _ in range(streams)]
        options = dict(streams=streams, arm=arm, partition=partition, deadline=self.deadline, lobby=self.lobby)
        try:
            if workers == 1:
                self.bank = EnvironmentBank(slots=list(range(streams)), **options)
                self.packets = self.bank.initial()
            else:
                ctx = mp.get_context('spawn')
                for rank in range(workers):
                    parent, child = ctx.Pipe()
                    process = ctx.Process(target=worker_main, args=(child, dict(slots=list(range(rank, streams, workers)), **options)))
                    process.start()
                    if on_spawn is not None:
                        on_spawn(process.pid)
                    child.close()
                    self.connections.append(parent)
                    self.processes.append(process)
                self.packets = {}
                for connection in self.connections:
                    self.packets.update(self._receive(connection))
        except BaseException:
            self.close()
            raise

    def _receive(self, connection):
        if not connection.poll(max(0., self.deadline - perf_counter())):
            raise TimeoutError('Collector wall ceiling reached')
        status, result = connection.recv()
        if status != 'ok':
            raise RuntimeError(f'Worker failed: {result}')
        return result

    def _command(self, command, payload=None):
        if self.bank is not None:
            return self.bank.snapshot() if command == 'snapshot' else getattr(self.bank, command)(payload)
        for rank, connection in enumerate(self.connections):
            part = None if payload is None else {k: v for k, v in payload.items() if k % self.workers == rank}
            connection.send((command, part))
        result = {}
        for connection in self.connections:
            result.update(self._receive(connection))
        return result

    def collect(self, policy, decisions=128, batched=True, original_inference=False):
        if decisions <= 0 or decisions % self.streams:
            raise ValueError('Equal per-stream actionable budgets are required')
        rollouts = [{k: [] for k in FIELDS} for _ in range(self.streams)]
        episodes, transitions = [], []
        for _ in range(decisions // self.streams):
            if perf_counter() >= self.deadline:
                raise TimeoutError('Collector wall ceiling reached')
            packets = [self.packets[s] for s in range(self.streams)]
            probabilities, values = forward(policy, packets, batched)
            actions = {s: int(torch.multinomial(probabilities[s], 1, generator=self.generators[s])) for s in range(self.streams)}
            logs = Categorical(probs=probabilities).log_prob(torch.tensor(list(actions.values())))
            rows = self._command('step', actions)
            self.packets = {s: rows[s]['packet'] for s in range(self.streams)}
            immediate = None
            if original_inference:
                immediate = torch.zeros(self.streams)
                live = [s for s in range(self.streams) if not rows[s]['done']]
                if live:
                    immediate[live] = forward(policy, [self.packets[s] for s in live], False)[1]
            for slot in range(self.streams):
                row, rollout = rows[slot], rollouts[slot]
                actor_trace(self.traces[slot], packets[slot], actions[slot])
                if rollout['values'] and not original_inference:
                    rollout['next_values'][-1] = 0. if rollout['dones'][-1] else float(values[slot])
                items = (packets[slot], actions[slot], float(logs[slot]), float(values[slot]),
                         0. if row['done'] else (float(immediate[slot]) if immediate is not None else None),
                         row['reward'], row['info']['discount'], row['done'], row['info']['elapsed_seconds'])
                for key, value in zip(FIELDS, items):
                    rollout[key].append(value)
                if row['done']:
                    episodes.append(dict(slot=slot, seed=row['seed'], rotation=row['rotation'],
                                         episode_index=row['episode_index'], **row['info']['episode']))
                transitions.append(dict(slot=slot, **{k: row[k] for k in ('seed', 'rotation', 'reward', 'done', 'info', 'audit')}))
        if not original_inference:
            bootstrap = forward(policy, [self.packets[s] for s in range(self.streams)], batched)[1]
            for s, rollout in enumerate(rollouts):
                rollout['next_values'][-1] = 0. if rollout['dones'][-1] else float(bootstrap[s])
        flat = {k: sum([r[k] for r in rollouts], []) for k in FIELDS}
        length = decisions // self.streams
        flat['segments'] = [(s * length, (s + 1) * length) for s in range(self.streams)]
        flat['episodes'] = episodes
        flat['transition_audits'] = transitions
        if not np.isfinite([flat[k] for k in ('logp', 'values', 'next_values', 'rewards', 'discounts', 'durations')]).all():
            raise FloatingPointError('Nonfinite collector transition')
        return flat

    def snapshot(self):
        return dict(workers=self.workers, streams=self.streams, arm=self.arm, partition=self.partition,
                    lobby=self.lobby.to_dict(),
                    environments=self._command('snapshot'), generators=[g.get_state() for g in self.generators])

    def restore(self, saved):
        if any(saved[k] != getattr(self, k) for k in ('workers', 'streams', 'arm', 'partition')):
            raise ValueError('Resume configuration mismatch')
        if saved['lobby'] != self.lobby.to_dict():
            raise ValueError('Resume lobby mismatch')
        self.packets = self._command('restore', saved['environments'])
        for generator, state in zip(self.generators, saved['generators'], strict=True):
            generator.set_state(state)
        # Trace hashes cover this continuation; checkpoint prefix hash is separate.
        self.traces = [hashlib.sha256() for _ in range(self.streams)]

    def close(self):
        for connection in self.connections:
            try:
                connection.send(('close', None))
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for connection in self.connections:
            connection.close()
        self.connections.clear()
        self.processes.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
