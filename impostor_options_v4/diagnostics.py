"""Observation-independent audit counters; no policy/physics changes."""
from social_deduction.actor import Phase
from .config import ready_seconds
from .env import OptionsEnv
from impostor_research.strategic_v2 import StrategicEnv


class Audit:
    def reset(self,**kwargs):
        self.fake_selection_streak=0;self.max_fake_selection_streak=0
        self.fake_selections=0;self.fake_selections_no_visible=0
        self.fake_activity_seconds=0.;self.ready_faking_seconds=0.;self.measured_ready_seconds=0.
        result=super().reset(**kwargs)
        original=self.match.tick

        def audited_tick(intent=None):
            g=self.match.game;own=g.players[self.match.focal]
            t=g.time;phase=g.phase;active=own.active;cooldown=own.cooldown
            before=own.interaction_task;remaining=own.interaction_remaining
            cursor=len(g.event_log);kills=g.metrics['eliminations']
            original(intent)
            dt=g.time-t
            completion=any(e['kind']=='fake_task_end' and e.get('player_id')==self.match.focal for e in g.event_log[cursor:])
            after_fake=own.interaction_task is not None and own.interaction_task.startswith('fake:')
            before_fake=before is not None and before.startswith('fake:')
            entered_meeting=g.phase not in (Phase.ROAMING,Phase.FINISHED)
            self.measured_ready_seconds+=ready_seconds(dt,cooldown,roaming=phase is Phase.ROAMING,
                active=active and own.active,entered_meeting=entered_meeting,
                killed=g.metrics['eliminations']>kills,kill_cooldown=g.config.kill_cooldown)
            terminal_fake=before_fake and g.is_terminal and dt>0 and intent is None
            if phase is Phase.ROAMING and active and own.active and not entered_meeting and (after_fake or completion or terminal_fake):
                span=min(dt,remaining) if before_fake and (completion or own.interaction_task==before) else dt
                self.fake_activity_seconds+=span
                self.ready_faking_seconds+=ready_seconds(span,cooldown,roaming=True,active=True,
                    entered_meeting=entered_meeting,
                    killed=g.metrics['eliminations']>kills,kill_cooldown=g.config.kill_cooldown)
        self.match.tick=audited_tick
        return result

    def step(self,action):
        if not self.finished and self.action_space.contains(action) and self.packet['mask'][int(action)]:
            fake=self.options[int(action)].kind=='fake_task'
            self.fake_selection_streak=self.fake_selection_streak+1 if fake else 0
            self.max_fake_selection_streak=max(self.max_fake_selection_streak,self.fake_selection_streak)
            if fake:
                self.fake_selections+=1
                self.fake_selections_no_visible+=not bool(self.envelope.actor.visible_players)
        return super().step(action)

    def summary(self):
        result=super().summary()
        own_events=[e for e in self.match.game.event_log if e.get('player_id')==self.match.focal]
        result.update(fake_selections=self.fake_selections,fake_selections_no_visible=self.fake_selections_no_visible,
            max_fake_selection_streak=self.max_fake_selection_streak,fake_activity_seconds=self.fake_activity_seconds,
            ready_faking_seconds=self.ready_faking_seconds,measured_cooldown_ready_seconds=self.measured_ready_seconds,
            fake_starts=sum(e['kind']=='fake_task_start' for e in own_events),
            fake_completions=sum(e['kind']=='fake_task_end' for e in own_events))
        return result


class DiagnosticEnv(Audit,OptionsEnv):
    pass


class LegacyEnv(Audit,StrategicEnv):
    pass
