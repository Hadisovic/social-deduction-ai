import numpy as np
import pytest
import torch
from phase3_engine import GameConfig
from phase5_env import CrewmateStrategicEnv
from phase5_options import CrewmateOptionsEnv, environment_for, PROTOCOL


def test_option_composes_same_physics_rewards_and_discount_as_manual_continuation():
    torch.set_num_threads(1)
    config = GameConfig(initial_kill_cooldown=1000, kill_cooldown=1000, timeout=120)
    option = CrewmateOptionsEnv(config=config, render_trace=True)
    base = CrewmateStrategicEnv(planner=option.planner, belief_model=option.belief_model, config=config)
    option.reset(seed=600000)
    base.reset(seed=600000)
    index = next(i for i, c in enumerate(option.options) if c.kind == 'task')
    task = option.options[index].task_id
    packet, reward, done, _, info = option.step(index)
    elapsed = total = raw = 0.
    discount = 1.
    while elapsed < info['elapsed_seconds'] - 1e-8:
        other, r, base_done, _, part = base.step(index if elapsed == 0 else 0)
        total += discount * r
        discount *= part['discount']
        elapsed += part['elapsed_seconds']
        raw += part['undiscounted_reward']
    assert info['elapsed_seconds'] > .6
    assert task in option._credited
    assert info['option_end'] == 'task_completed'
    assert reward == pytest.approx(total)
    assert info['discount'] == pytest.approx(discount)
    assert info['undiscounted_reward'] == pytest.approx(raw)
    assert done == base_done
    assert all(np.array_equal(packet[k], other[k]) for k in packet)
    assert option.match.game.metrics['illegal_actions'] == 0
    assert option.visual_frames[0][0] == 0
    assert option.visual_frames[-1][0] == pytest.approx(elapsed)


def test_protocol_factory_preserves_v1_and_rejects_unknown_versions():
    assert type(environment_for({})) is CrewmateStrategicEnv
    assert type(environment_for({'execution_protocol': PROTOCOL})) is CrewmateOptionsEnv
    with pytest.raises(ValueError):
        environment_for({'execution_protocol': 'unknown'})


def test_option_runtime_rejects_changed_source_and_ppo_records_real_diagnostics():
    from phase5_runtime import checked_environment
    from phase5_options_training import ppo_update
    from phase5_policy import StrategicPolicy
    with pytest.raises(ValueError, match='runtime mismatch'):
        checked_environment({'options_sha256': '0'*64})
    torch.set_num_threads(1)
    torch.manual_seed(19)
    env = CrewmateOptionsEnv()
    packet, _ = env.reset(seed=600000)
    policy = StrategicPolicy()
    before = {k:v.clone() for k,v in policy.state_dict().items()}
    rollout = {k:[] for k in ('packets','actions','logp','values','next_values',
                               'rewards','discounts','dones','durations')}
    for i in range(16):
        action, logp, value = policy.act(packet)
        for k,v in zip(rollout,(packet,action,logp,value,0.,float(i%3),.99,i==15,.6)):
            rollout[k].append(v)
    diagnostics = ppo_update(policy, torch.optim.Adam(policy.parameters(), lr=.001),
                             rollout, .01, np.random.default_rng(19))
    assert all(v is None or np.isfinite(v) for v in diagnostics.values())
    assert diagnostics['approximate_kl'] >= -1e-7
    assert 0 <= diagnostics['clip_fraction'] <= 1
    assert diagnostics['entropy'] > 0
    assert any(not torch.equal(before[k],v) for k,v in policy.state_dict().items())
