"""Registered complete-episode PPO; only the impostor optimizer may change weights."""
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
import torch

from crewmate_v2.learning.train import state_hash
from frozen_handoff.match import FrozenCrewMatch
from frozen_handoff.models import load_model, MODELS
from social_deduction.actor import Role
from spectator.recorder import MatchRecorder
from .checkpoints import optimizer_for
from .controller import Controller
from .coordinator import ContextGate, ImpostorLedger
from .time_math import gae
from .weights import tensor_hash


@dataclass(frozen=True)
class PPOConfig:
    epochs: int = 4
    minibatch: int = 64
    learning_rate: float = 3e-4
    clip: float = .2
    value_coefficient: float = .5
    family_entropy_start: float = .02
    family_entropy_end: float = .005
    conditional_entropy: float = .001
    actor_grad_norm: float = .5
    critic_grad_norm: float = 1.
    lambda_per_second: float = .995
    gamma_per_second: float = 1.


def source_signature():
    return {p.name: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py'))}


def check_sources(expected):
    if source_signature() != expected:
        raise RuntimeError('Registered learner source changed during execution')


def transition_batch(rows):
    if not rows or not all(r['settled'] for r in rows):
        raise ValueError('Only settled, complete episodes may enter PPO')
    if not rows[-1]['done'] or rows[-1]['next_value'] != 0 or any(r['done'] for r in rows[:-1]):
        raise ValueError('Exactly one terminal transition and zero bootstrap required')
    packets = {k: torch.as_tensor(np.stack([r['packet'][k] for r in rows])) for k in rows[0]['packet']}
    actions = torch.tensor([r['action'] for r in rows], dtype=torch.long)
    if not packets['mask'].bool().gather(1, actions[:, None]).all():
        raise ValueError('Masked training action')
    advantage, returns = gae([r['reward'] for r in rows], [r['value'] for r in rows],
        [r['next_value'] for r in rows], np.ones(len(rows)), [r['done'] for r in rows],
        [r['duration'] for r in rows])
    return packets, actions, torch.tensor([r['logp'] for r in rows]), torch.from_numpy(advantage), torch.from_numpy(returns)


def ppo_update(policy, optimizer, rows, generator, *, completed_episode, config=PPOConfig()):
    if not 1 <= completed_episode <= 8:
        raise ValueError('Unregistered episode budget')
    data, actions, old, advantage, returns = transition_batch(rows)
    actor = policy.actor_parameters(); critic = policy.critic_parameters()
    actual = [p for g in optimizer.param_groups for p in g['params']]
    if (set(map(id, actor)) & set(map(id, critic)) or len(actual) != len(list(policy.parameters()))
            or set(map(id, actual)) != set(map(id, policy.parameters()))):
        raise ValueError('Wrong optimizer ownership or overlapping actor/critic')
    if any(not p.requires_grad for p in actual):
        raise ValueError('Impostor training parameters must be trainable')
    with torch.no_grad():
        initial, _ = policy(data)
        error = float((initial.log_prob(actions) - old).abs().max())
        if error > 2e-4:
            raise ValueError('Off-policy collection/log-probability mismatch')
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    coefficient = config.family_entropy_start + (config.family_entropy_end - config.family_entropy_start) * (completed_episode - 1) / 7
    stats = []; started = perf_counter(); cpu_started = process_time(); before = tensor_hash(policy)
    for _ in range(config.epochs):
        for ids in torch.randperm(len(rows), generator=generator).split(config.minibatch):
            distribution, value, families, _, conditional = policy.evaluate({k: v[ids] for k, v in data.items()})
            ratio = (distribution.log_prob(actions[ids]) - old[ids]).exp()
            loss = -torch.minimum(ratio * advantage[ids], ratio.clamp(1-config.clip, 1+config.clip) * advantage[ids]).mean()
            value_loss = (value - returns[ids]).square().mean()
            objective = loss - coefficient * families.entropy().mean() - config.conditional_entropy * conditional.mean()
            if not torch.isfinite(objective + value_loss):
                raise FloatingPointError('Nonfinite PPO objective')
            optimizer.zero_grad(set_to_none=True)
            (objective + config.value_coefficient * value_loss).backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in actual):
                raise FloatingPointError('Nonfinite gradient')
            an = torch.nn.utils.clip_grad_norm_(actor, config.actor_grad_norm)
            cn = torch.nn.utils.clip_grad_norm_(critic, config.critic_grad_norm)
            optimizer.step()
            stats.append([float(loss.detach()), float(value_loss.detach()), float(an), float(cn)])
    if any(not torch.isfinite(p).all() for p in policy.parameters()):
        raise FloatingPointError('Nonfinite impostor weights')
    with torch.no_grad():
        distribution, value, families, _, conditional = policy.evaluate(data)
        logratio = distribution.log_prob(actions) - old; ratio = logratio.exp()
        variance = returns.var(unbiased=False)
        explained = float(1 - (returns-value).var(unbiased=False)/variance) if variance > 1e-8 else None
    means = np.asarray(stats).mean(0)
    result = dict(minibatch_updates=len(stats), decisions=len(rows), completed_episode=completed_episode,
        policy_loss=float(means[0]), value_loss=float(means[1]), actor_gradient_norm=float(means[2]),
        critic_gradient_norm=float(means[3]), approximate_kl=float(((ratio-1)-logratio).mean()),
        clip_fraction=float(((ratio-1).abs() > config.clip).float().mean()), explained_variance=explained,
        family_entropy=float(families.entropy().mean()), joint_entropy=float(distribution.entropy().mean()),
        conditional_entropy=float(conditional.mean()), family_entropy_coefficient=coefficient,
        collection_logp_error=error, optimization_seconds=perf_counter()-started,
        optimization_cpu_seconds=process_time()-cpu_started,
        policy_hash_before=before, policy_hash_after=tensor_hash(policy))
    if before == result['policy_hash_after']:
        raise AssertionError('PPO did not update any impostor tensor')
    return result


class TrainingLedger(ImpostorLedger):
    def __init__(self, *args, training_decisions=0, policy_update=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_decisions = training_decisions; self.policy_update = policy_update

    def decision(self, decision):
        recorder = self.recorder
        try:
            self.recorder = None
            super().decision(decision)
        finally:
            self.recorder = recorder
        if recorder:
            recorder.record_neural(self.controller.player_id, decision['kind'], decision['target'],
                probabilities=decision['probabilities'], value_estimate=decision['value'],
                entropy=decision['entropy'], legal_actions=decision['labels'],
                training_step=self.training_decisions+self.controller.decisions,
                policy_update=self.policy_update, selected_action_index=decision['index'])


def event_metrics(game, impostor):
    events = game.event_log
    kills = [e for e in events if e['kind'] == 'elimination' and e.get('killer_id') == impostor]
    own_victims = {e['victim_id']: e for e in kills}
    reports = [e for e in events if e['kind'] == 'report' and e.get('reporter_id') == impostor]
    self_reports = [e for e in reports if e.get('victim_id') in own_victims]
    latencies = [(e['tick']-own_victims[e['victim_id']]['tick'])*game.config.dt for e in self_reports]
    return dict(kills=len(kills), accepted_reports=len(reports), accepted_self_reports=len(self_reports),
        self_report_latencies=latencies,
        witnessed_kill_events=sum(e['kind'] == 'elimination_witnessed' and e.get('killer_id') == impostor for e in events),
        fake_task_starts=sum(e['kind'] == 'fake_task_start' and e.get('player_id') == impostor for e in events),
        fake_task_completions=sum(e['kind'] == 'fake_task_end' and e.get('player_id') == impostor for e in events),
        claims=dict(Counter(str(e.get('claim_kind')) for e in events if e['kind'] == 'claim' and e.get('speaker_id') == impostor)))


def collect_episode(policy, crew_policy, *, seed, crew='B', generator=None, deadline=None,
                    replay=None, policy_update=0, training_decisions=0):
    started = perf_counter(); cpu_started = process_time()
    imp_before = tensor_hash(policy); crew_before = state_hash(crew_policy)
    actors = []
    def factory(*, seed, player_id):
        actor = Controller(policy, seed=seed, player_id=player_id, stochastic=generator is not None)
        if generator is not None:
            actor.generator = generator
        actors.append(actor)
        return actor
    match = FrozenCrewMatch(seed, model=crew, controller_factory=factory, accounting='eligible-v1', deadline=deadline)
    actor = actors[0]; belief = next(iter(match.views.values())).belief_model.network
    belief_before = state_hash(belief)
    recorder = None
    if replay:
        recorder = MatchRecorder(match.match, replay, checkpoint_path='registered shared pilot',
            training_session='Shared pilot: real learned impostor, frozen crew '+crew,
            model_architecture='FamilyPolicy 75075; FourCrewPolicy 51778',
            source_version='shared-pilot-v1',
            controller_types={pid: 'frozen-neural-'+crew if pid in match.crew else 'learned-impostor' for pid in match.game.players},
            metadata=dict(crew_checkpoint_sha256=MODELS[crew][1], impostor_tensor_hash=imp_before,
                optimizer_updates_before_match=policy_update, training_decisions_before_match=training_decisions,
                social_guards=False, study_partition='development-only'))
    ledger = TrainingLedger(match.game, actor, recorder, training_decisions=training_decisions, policy_update=policy_update)
    match.match.controllers[match.impostor] = ContextGate(match.game, actor, ledger)
    match.game.step = ledger.step
    try:
        _, summary = match.collect(crew_policy, initialization=17, policy_version=policy_update, recorder=recorder)
        credit = ledger.check()
    finally:
        if recorder and not recorder.recorder._closed:
            recorder.recorder._stream.close(); recorder.recorder._closed = True
    if (state_hash(crew_policy) != crew_before or state_hash(belief) != belief_before
            or tensor_hash(policy) != imp_before):
        raise AssertionError('Match inference changed neural weights')
    if any(p.requires_grad or p.grad is not None for m in (crew_policy, belief) for p in m.parameters()):
        raise AssertionError('Frozen opponent gradient leak')
    if summary['illegal_actions']:
        raise AssertionError('Illegal learned action')
    summary.update(crew=crew, impostor_win=match.game.result.winner is Role.IMPOSTOR,
        credit=credit, actions=dict(actor.actions), held_callbacks=actor.held_callbacks,
        fake_repeat_continuations=actor.fake_repeats, behavior=event_metrics(match.game, match.impostor),
        neural_decisions=actor.decisions, collection_seconds=perf_counter()-started,
        collection_cpu_seconds=process_time()-cpu_started,
        impostor_hash=imp_before, frozen_crew_hash=crew_before, frozen_belief_hash=belief_before,
        event_sha256=hashlib.sha256(json.dumps(match.game.event_log, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        private_actor_memories=len({id(v.memory) for v in match.views.values()}) == 4)
    return ledger.rows, summary
