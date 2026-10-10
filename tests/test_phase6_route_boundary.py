import numpy as np
import torch
from dataclasses import replace
from phase6.config import Experiment
from phase6.env import Phase6Env, route_descriptors


def test_route_descriptors_only_computed_at_policy_boundary(monkeypatch):
    import phase6.env as module
    torch.set_num_threads(1)
    env = Phase6Env(Experiment(distance='route'))
    env.config = replace(env.config, initial_kill_cooldown=1000, kill_cooldown=1000)
    env.reset(seed=1250000)
    calls = []
    original = module.route_descriptors
    def record(packet, obs, options, planner):
        calls.append(obs.tick)
        return original(packet, obs, options, planner)
    monkeypatch.setattr(module, 'route_descriptors', record)
    action = next(i for i, c in enumerate(env.options) if c.kind == 'task')
    packet, _, _, _, info = env.step(action)
    assert info['elapsed_seconds'] > .6
    assert calls == [env.observation.tick]
    # Omitting unused intermediate computations must not alter the consumed
    # packet: compare with a fresh explicit public-route projection.
    expected = {k: v.copy() for k, v in packet.items()}
    route_descriptors(expected, env.observation, env.options, env.planner)
    assert all(np.array_equal(packet[k], expected[k]) for k in packet)
