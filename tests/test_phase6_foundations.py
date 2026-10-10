from dataclasses import replace
import json
import numpy as np
import pytest
import torch

from test_phase3_boundary import planner, scene, change_others
from belief.memory import ActorMemory
from belief.model import BeliefModel
from phase3_engine import GameConfig
from phase5_env import CrewmateStrategicEnv
from phase5_features import choices, encode, KINDS
from phase5_policy import StrategicPolicy, tensors
from phase6.config import Experiment, PARTITIONS, seeds, verify_frozen
from phase6.env import Phase6Env, route_descriptors
from social_deduction.actor import Point, Role
from social_deduction.phase3_observation import project_game
from social_deduction.truth import BodyTruth, TaskTruth


def test_fresh_partitions_disjoint_and_final_seeds_not_exposed_by_development_cli():
    ranges = list(PARTITIONS.values())
    for a, (lo, hi) in enumerate(ranges):
        assert hi >= lo and lo > 971999
        assert all(hi < x or lo > y for x, y in ranges[a + 1:])
    assert seeds('development', 2) == (1250000, 1250001)
    for args in (('development', 21), ('final_id', 501), ('unknown', 1)):
        with pytest.raises(ValueError): seeds(*args)
    from phase6.train import train
    with pytest.raises(ValueError, match='scope'):
        train('must-not-create', Experiment(), steps=8192)


@pytest.mark.parametrize('role', [Role.CREWMATE, Role.IMPOSTOR])
@pytest.mark.parametrize('hidden', ['role', 'position', 'tasks', 'cooldown', 'death'])
def test_phase6_counterfactual_hidden_truth_stays_out_of_packets(scene, role, hidden):
    game, world, actor = scene
    # Own role is legitimate; the fixture is a boundary test, not a game setup.
    world = replace(world, players=tuple(replace(p, role=role) if p.identity.player_id == actor else p for p in world.players))
    if hidden == 'role':
        changed = change_others(world, actor, lambda p: replace(p, role=Role.IMPOSTOR if p.role is Role.CREWMATE else Role.CREWMATE))
    elif hidden == 'position':
        changed = change_others(world, actor, lambda p: replace(p, position=Point(99., 99.)))
    elif hidden == 'tasks':
        changed = change_others(world, actor, lambda p: replace(p, tasks=(TaskTruth('private', 'hidden', .9),)))
    elif hidden == 'cooldown':
        changed = change_others(world, actor, lambda p: replace(p, private_cooldown=77.))
    else:
        changed = change_others(world, actor, lambda p: replace(p, active=False))
        p = next(p for p in changed.players if p.identity.player_id != actor)
        changed = replace(changed, bodies=(BodyTruth(p.identity.player_id, p.position, p.room, 1, 'secret'),))
    from social_deduction.phase3_api import ObservationSettings
    settings = ObservationSettings(sight_range=Experiment().game_config().effective_sight_range(role))
    a = project_game(world, actor, game.map, game.planner, settings)
    b = project_game(changed, actor, game.map, game.planner, settings)
    assert a == b
    if role is Role.IMPOSTOR:
        return  # Impostor policy/memory is a later milestone, not Phase 4 memory.
    torch.set_num_threads(1)
    policy = StrategicPolicy()
    for memory_condition in ('full', 'history_no_belief', 'current_only'):
        packets = []
        for obs in (a, b):
            memory = ActorMemory(); memory.update(obs)
            options = choices(obs, memory)
            packet = encode(obs, memory, BeliefModel.load().predict(memory), options, ablation=memory_condition != 'full')
            route_descriptors(packet, obs, options, game.planner)
            packets.append(packet)
        assert all(np.array_equal(packets[0][k], packets[1][k]) for k in packets[0])
        da, va = policy(tensors(packets[0])); db, vb = policy(tensors(packets[1]))
        assert torch.equal(da.probs, db.probs) and torch.equal(va, vb)


def test_controlled_ablations_drop_belief_separately_from_history(planner):
    torch.set_num_threads(1)
    envs = [Phase6Env(Experiment(memory=m), planner=planner) for m in ('full', 'history_no_belief', 'current_only')]
    for env in envs:
        env.reset(seed=1250000)
        other = next(p for p in env.match.game.players.values() if p.player_id != env.focal)
        env.match.game.players[env.focal].motion.spawn((-.7, -2.8))
        other.motion.spawn((-1.02, -4.8)); env.match.game.tick = 1; env.match.game._invalidate()
        env._observe(); env._packet()
        other.motion.spawn((-1.38, -10.8)); env.match.game.tick = 2; env.match.game._invalidate()
        env._observe(); env._packet()
        assert (other.player_id in env.memory.last_seen) == (env.experiment.memory != 'current_only')
    full, history, current = envs
    assert np.array_equal(full.packet['players'][:, :22], history.packet['players'][:, :22])
    assert np.all(history.packet['players'][:, 22] == .25)
    assert np.all(history.packet['players'][:, 23] == 0.)
    assert np.all(current.packet['players'][:, 22] == .25)
    assert not np.array_equal(history.packet['players'][:, :22], current.packet['players'][:, :22])


@pytest.mark.parametrize('seconds', [12, 24])
def test_new_option_composes_existing_physics_discount_and_reward(planner, seconds):
    torch.set_num_threads(1)
    experiment = Experiment(max_option_seconds=seconds)
    option = Phase6Env(experiment, planner=planner)
    # Isolate the mechanical test from eliminations; no hidden information goes
    # into policy inputs and no historical configuration is overwritten.
    option.config = replace(option.config, initial_kill_cooldown=1000, kill_cooldown=1000)
    base = CrewmateStrategicEnv(config=option.config, planner=planner, belief_model=option.belief_model)
    option.reset(seed=1250000); base.reset(seed=1250000)
    index = next(i for i, c in enumerate(option.options) if c.kind == 'task')
    _, reward, done, _, info = option.step(index)
    elapsed = total = raw = 0.; discount = 1.
    while elapsed < info['elapsed_seconds'] - 1e-8:
        _, r, base_done, _, part = base.step(index if elapsed == 0 else 0)
        total += discount * r; discount *= part['discount']
        elapsed += part['elapsed_seconds']; raw += part['undiscounted_reward']
    assert info['elapsed_seconds'] <= seconds + .6
    assert reward == pytest.approx(total) and info['discount'] == pytest.approx(discount)
    assert info['undiscounted_reward'] == pytest.approx(raw) and done == base_done
    assert option.match.game.metrics['illegal_actions'] == 0


def test_asymmetric_replay_deterministic_and_config_version_recorded(planner):
    from phase3_runner import ScriptedMatch
    runs = [ScriptedMatch(1250000, config=Experiment().game_config(), planner=planner) for _ in range(2)]
    for run in runs: run.run()
    assert runs[0].trajectory_hash == runs[1].trajectory_hash
    assert runs[0].replay_document()['schema'] == 'phase6-replay-v1'
    assert runs[0].replay_document()['config']['impostor_sight_range'] == 6.75
    assert all(run.game.metrics['illegal_actions'] == run.game.metrics['navigation_failures'] == 0 for run in runs)


def test_identity_color_spawn_coverage_and_frozen_artifacts():
    from phase6.audit import audit
    result = audit()
    assert result['matches_audited'] == 1000
    assert set(result['impostor_counts']['public_id']) == {'p0', 'p1', 'p2', 'p3', 'p4'}
    verify_frozen()


@pytest.mark.parametrize('kwargs', [dict(memory='omniscient'), dict(distance='truth'),
    dict(max_option_seconds=48), dict(rollout=1024), dict(learning_rate=.01),
    dict(crewmate_sight_range=7.), dict(impostor_sight_range=True)])
def test_invalid_experiments_rejected(kwargs):
    with pytest.raises(ValueError): Experiment(**kwargs)
