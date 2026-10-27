"""Versioned, lossy V2-to-legacy opponent adapter. Never resizes neural inputs."""
import hashlib
import torch
from belief.memory import ActorMemory
from belief.model import BeliefModel
from crewmate_v2.api import validate
from phase5_features import choices, encode, mechanical_intent
from phase6.config import ROOT, digest, verify_frozen
from phase6.train import load_policy
from social_deduction.actor import Phase, Role
from social_deduction.phase3_api import Intent, IntentKind, WitnessedElimination, intent_allowed

CHECKPOINT = 'artifacts/phase6/remaining-six/full-29/checkpoint-8192.pt'
SHA256 = '238904b7fe303b77e885c4cd6fdf4d9d447a62eb5b463436a391c0b4d6e1cb43'
ADAPTER = 'v2-legacy-full29-projection-v1'


def weight_hash(network):
    h = hashlib.sha256()
    for key, value in sorted(network.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class FrozenModels:
    def __init__(self):
        if digest(ROOT / CHECKPOINT) != SHA256:
            raise ValueError('Pinned full-29 checkpoint digest mismatch')
        # Original runtime/map/backbone/source checks are mandatory and unmodified.
        with torch.random.fork_rng(devices=[]):
            self.policy, self.experiment, self.metadata = load_policy(ROOT / CHECKPOINT)
            self.belief = BeliefModel.load()
        if self.experiment.memory != 'full' or self.experiment.distance != 'euclidean':
            raise ValueError('Adapter supports the pinned full/euclidean experiment only')
        self.policy.eval().requires_grad_(False)
        self.belief.network.eval().requires_grad_(False)
        self.initial_hashes = self.hashes()

    def hashes(self):
        return dict(policy=weight_hash(self.policy), belief=weight_hash(self.belief.network))

    def assert_unchanged(self):
        verify_frozen()
        if self.hashes() != self.initial_hashes or digest(ROOT / CHECKPOINT) != SHA256:
            raise RuntimeError('Frozen weights changed')
        for network in (self.policy, self.belief.network):
            if any(p.requires_grad or p.grad is not None for p in network.parameters()):
                raise RuntimeError('Frozen opponent participates in gradients')


class FrozenCrew:
    """One actor's state; shared models contain no mutable actor memory.

    Keep original /240 clock, task counts/progress and feature meanings. V2 bar,
    role confirmations, stage details and visual proofs are deliberately omitted.
    The original .6-second observation cadence and bounded options are retained.
    Readiness is checked against the real actor legal mask before submission.
    """
    def __init__(self, models):
        self.models = models
        self.memory = ActorMemory()
        self.selected = None
        self.started = 0.
        self.initial_phase = None
        self.bodies = set()
        self.witnesses = set()
        self.last_packet = None
        self.last_choice = None
        self.inference_calls = 0

    def packet(self, envelope):
        obs = validate(envelope).actor
        if obs.own.role is not Role.CREWMATE or not obs.own.active or obs.context.phase is Phase.FINISHED:
            raise ValueError('Frozen crew adapter requires a living crewmate')
        self.memory.update(obs)
        options = choices(obs, self.memory)
        packet = encode(obs, self.memory, self.models.belief.predict(self.memory), options)
        return packet, options

    def interrupt(self):
        self.selected = None

    def decide(self, envelope):
        obs = validate(envelope).actor
        if not obs.own.active or obs.context.phase is Phase.FINISHED:
            return Intent(IntentKind.WAIT)
        self.memory.update(obs)
        bodies = {b.victim_id for b in obs.visible_bodies}
        witnesses = {e for e in self.memory.events if type(e.payload) is WitnessedElimination}
        c = self.selected
        extended = c is not None and c.kind in ('task', 'room', 'follow', 'flee', 'approach_body', 'emergency_travel')
        limit = 3. if c and c.kind in ('follow', 'flee') else self.models.experiment.max_option_seconds
        finished_task = c and c.kind == 'task' and not any(t.task_id == c.task_id and t.progress < 1 for t in obs.own.tasks)
        if (not extended or obs.context.phase is not self.initial_phase or bodies - self.bodies
                or witnesses - self.witnesses or finished_task or obs.time_seconds - self.started >= limit - 1e-8
                or (c.kind != 'task' and obs.navigation_status != 'moving')):
            packet, options = self.packet(envelope)
            index, _, _ = self.models.policy.act(packet, deterministic=True)
            c = self.selected = options[index]
            self.last_packet = packet; self.last_choice = c; self.inference_calls += 1
            self.started = obs.time_seconds; self.initial_phase = obs.context.phase
            self.bodies = bodies; self.witnesses = witnesses
        action = mechanical_intent(c, obs)
        # Incubating V2 tasks may be at their console but not interactable yet.
        if action is None or not intent_allowed(obs, action):
            return Intent(IntentKind.WAIT)
        return action
