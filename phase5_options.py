"""Versioned mechanical options; the network still selects every strategic goal.

V1 is deliberately preserved for reproducible baseline runs. Durations and
interruptions use only the focal actor's observations, never hidden simulator
state. Rewards and discount factors are composed over real elapsed game time.
"""
from phase5_env import CrewmateStrategicEnv
from phase5_features import Choice
from social_deduction.actor import Phase
from social_deduction.phase3_api import WitnessedElimination

PROTOCOL = 'strategic-options-v2'
MAX_OPTION_SECONDS = 24.


class CrewmateOptionsEnv(CrewmateStrategicEnv):
    def _advance(self, selected):
        # Scripts retain their original decision frequency for a fair baseline.
        extended = selected.kind in {
            'task', 'room', 'follow', 'flee', 'approach_body', 'emergency_travel'}
        initial = self.observation
        known_bodies = {b.victim_id for b in initial.visible_bodies}
        known_kills = {e.tick for e in self.memory.events
                       if type(e.payload) is WitnessedElimination}
        limit = 3. if selected.kind in {'follow', 'flee'} else MAX_OPTION_SECONDS
        duration = reward = raw_reward = 0.
        discount = 1.
        frames = []
        reason = 'one_decision'
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
                break
            if obs.context.phase is not initial.context.phase or obs.context.phase is not Phase.ROAMING:
                reason = 'phase_changed'
                break
            if {b.victim_id for b in obs.visible_bodies} - known_bodies:
                reason = 'new_visible_body'
                break
            if any(type(e.payload) is WitnessedElimination and e.tick not in known_kills
                   for e in self.memory.events):
                reason = 'witnessed_elimination'
                break
            if selected.kind == 'task':
                task = next((t for t in obs.own.tasks if t.task_id == selected.task_id), None)
                if task is None or task.progress >= 1:
                    reason = 'task_completed'
                    break
            elif obs.navigation_status != 'moving':
                reason = 'movement_finished'
                break
            if duration >= limit - 1e-8:
                reason = 'time_limit'
                break
        self.last_label = selected.label
        if self.render_trace:
            self.visual_frames = frames
        info.update(elapsed_seconds=duration, discount=discount,
                    undiscounted_reward=raw_reward, option_end=reason,
                    option_kind=selected.kind, execution_protocol=PROTOCOL)
        return packet, float(reward), done, truncated, info


def environment_for(metadata, **kwargs):
    protocol = metadata.get('execution_protocol', 'strategic-decisions-v1')
    if protocol not in {'strategic-decisions-v1', PROTOCOL}:
        raise ValueError(f'Unknown execution protocol: {protocol}')
    cls = CrewmateOptionsEnv if protocol == PROTOCOL else CrewmateStrategicEnv
    return cls(**kwargs)
