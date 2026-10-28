"""Reuse Phase 5 physics, policy inputs and PPO; version only new experiments."""
from belief.features import current_memory
from navigation_service import NavTarget, NavStatus
from phase5_env import CrewmateStrategicEnv
from phase5_features import Choice, KINDS, choices, encode
from social_deduction.actor import Phase
from social_deduction.phase3_api import IntentKind, WitnessedElimination
from .config import Experiment, PROTOCOL


def route_descriptors(packet, observation, options, planner):
    """Static navigation service queried only with own/observed/public locations.

    No roles, live unseen positions or world references are accepted here. A
    failed public route is encoded as maximum distance, never a hidden-state mask.
    """
    start = (observation.own.position.x, observation.own.position.y)
    for i, choice in enumerate(options):
        if choice.position is None:
            continue
        if choice.intent and choice.intent.kind is IntentKind.GO_TO_ROOM:
            target = NavTarget.room(choice.intent.target)
        elif choice.intent and choice.intent.kind is IntentKind.GO_TO_TASK:
            target = NavTarget.task(choice.intent.target)
        else:
            target = NavTarget.location((choice.position.x, choice.position.y))
        plan = planner.plan(start, target)
        packet['actions'][i, len(KINDS) + 2] = min(plan.length / 60., 1.) if plan.status in (NavStatus.SUCCESS, NavStatus.MOVING) else 1.
    return packet


class Phase6Env(CrewmateStrategicEnv):
    def __init__(self, experiment=None, **kwargs):
        self.experiment = experiment or Experiment()
        self._inside_option = False
        if 'config' in kwargs or 'ablation' in kwargs:
            raise ValueError('Phase 6 rules/ablation must come from the experiment record')
        super().__init__(config=self.experiment.game_config(), **kwargs)

    def _observe(self):
        previous = self._previous_tick
        super()._observe()
        if self.experiment.memory == 'current_only' and self.observation.own.active and self.observation.context.phase is not Phase.FINISHED:
            self.memory = current_memory(self.observation, previous)
        # Eliminated/terminal actors receive no new model memory, as in Phase 4.
        # Their last safe packet is never used for another live policy decision.

    def _packet(self):
        self.belief = self.belief_model.predict(self.memory)
        self.options = choices(self.observation, self.memory)
        self.packet = encode(self.observation, self.memory, self.belief, self.options,
                             ablation=self.experiment.memory != 'full')
        if self.experiment.distance == 'route' and not self._inside_option:
            route_descriptors(self.packet, self.observation, self.options, self.planner)
        return self.packet

    def _advance(self, selected):
        # Base transitions refresh observations several times during one held
        # goal. Their intermediate packets are never consumed by the policy.
        # Query expensive public routes only at the real decision boundary.
        self._inside_option = True
        try:
            result = self._compose(selected)
        finally:
            self._inside_option = False
        if self.experiment.distance == 'route':
            route_descriptors(result[0], self.observation, self.options, self.planner)
        return result

    def _compose(self, selected):
        # Mechanical option composition matches V2; separate source preserves
        # the version-checked Phase 5 release runtime byte-for-byte.
        initial = self.observation
        known_bodies = {b.victim_id for b in initial.visible_bodies}
        known_kills = {e.tick for e in self.memory.events if type(e.payload) is WitnessedElimination}
        extended = selected.kind in {'task', 'room', 'follow', 'flee', 'approach_body', 'emergency_travel'}
        limit = 3. if selected.kind in {'follow', 'flee'} else self.experiment.max_option_seconds
        duration = reward = raw_reward = 0.
        discount = 1.
        frames = []
        while True:
            packet, r, done, truncated, info = super()._advance(
                selected if duration == 0 else Choice('continue'))
            reward += discount * r
            discount *= info['discount']
            duration += info['elapsed_seconds']
            raw_reward += info['undiscounted_reward']
            if self.render_trace:
                frames.extend(self.visual_frames if not frames else self.visual_frames[1:])
            obs = self.observation
            if done or not extended:
                reason = 'terminal' if done else 'one_decision'
            elif obs.context.phase is not initial.context.phase or obs.context.phase is not Phase.ROAMING:
                reason = 'phase_changed'
            elif {b.victim_id for b in obs.visible_bodies} - known_bodies:
                reason = 'new_visible_body'
            elif any(type(e.payload) is WitnessedElimination and e.tick not in known_kills for e in self.memory.events):
                reason = 'witnessed_elimination'
            elif selected.kind == 'task' and not any(t.task_id == selected.task_id and t.progress < 1 for t in obs.own.tasks):
                reason = 'task_completed'
            elif selected.kind != 'task' and obs.navigation_status != 'moving':
                reason = 'movement_finished'
            elif duration >= limit - 1e-8:
                reason = 'time_limit'
            else:
                continue
            break
        self.last_label = selected.label
        if self.render_trace:
            self.visual_frames = frames
        info.update(elapsed_seconds=duration, discount=discount, undiscounted_reward=raw_reward,
                    option_end=reason, option_kind=selected.kind, execution_protocol=PROTOCOL)
        return packet, float(reward), done, truncated, info
