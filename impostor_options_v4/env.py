"""New reward/action interface, unchanged lobby, historical policy and game engine."""
from collections import Counter
from dataclasses import asdict
import numpy as np
from social_deduction.actor import Phase,Role
from social_deduction.phase3_api import IntentKind
from impostor_research.strategic_v2 import StrategicEnv,family
from impostor_research.env import RewardConfig
from . import SCHEMA
from .config import Rewards,Options,ready_seconds
from .shaping import PursuitPotential


COMPONENTS=('terminal','kill','ejection','cooldown_ready','pursuit')


class OptionsEnv(StrategicEnv):
    """Same neural candidate choices. Repeating the active fake becomes continue.

    Policy control is retained every 0.6s during fake activity: continue, repeat,
    cancel, travel, kill and report are still voluntary legal choices. No target is
    selected automatically and no action is removed to discourage faking.
    """
    def __init__(self,*,rewards=Rewards(),options=Options(),**kwargs):
        self.rewards=rewards;self.option_config=options
        super().__init__(**kwargs)
        self.reward_config=RewardConfig(win=rewards.win,loss=rewards.loss,timeout=rewards.timeout,
                                       per_second=0.,gamma_per_second=1.)

    def reset(self,**kwargs):
        self.pursuit=PursuitPotential(self.rewards.pursuit_scale,self.rewards.pursuit_min_progress)
        self.pending_pursuit=0.;self.reward_totals=Counter({k:0. for k in COMPONENTS})
        self.ready_elapsed=0.;self.fake_repeats=0;self.decision_journal=[]
        return super().reset(**kwargs)

    def _observe(self):
        super()._observe()
        obs=self.envelope.actor
        self.pending_pursuit+=self.pursuit.advance(obs)
        if obs.context.phase is not Phase.FINISHED:
            # Own timer only: exposing one's own activity is legitimate. The engine
            # observation already exposes its console; never inspect other timers.
            own=self.match.game.players[self.match.focal]
            fake=obs.interaction_task is not None and obs.interaction_task.startswith('fake:')
            self.packet['global_'][3]=min(1.,own.interaction_remaining/self.match.game.config.fake_task_seconds) if fake else 0.
            self.packet['extra'][2]=self.pursuit.best
            self.packet['extra'][3]=self.pursuit.credit/self.rewards.pursuit_scale if self.rewards.pursuit_scale else 0.
        # Include new observation semantics in the replay digest.
        for key in ('global_','extra'):self.actor_trace.update(self.packet[key].tobytes())

    def _persistent_intent(self,intent):
        obs=self.envelope.actor
        if (intent is not None and intent.kind is IntentKind.FAKE_TASK and
            obs.own.active and obs.context.phase is Phase.ROAMING and
            obs.interaction_task=='fake:'+intent.target):
            self.fake_repeats+=1
            return None
        return intent

    def _advance(self,intent):
        intent=self._persistent_intent(intent)
        components=Counter({k:0. for k in COMPONENTS});elapsed=ready=0.;forced=0
        while True:
            for index in range(self.match.decision_ticks):
                g=self.match.game;own=g.players[self.match.focal]
                start_time=g.time;start_phase=g.phase;active=own.active;cooldown=own.cooldown
                eliminations=g.metrics['eliminations'];was_ejected=own.status=='ejected'
                self.match.tick(intent if elapsed==0 and index==0 else None)
                dt=g.time-start_time
                if self.render_trace:self.visual_frames.append(self._frame())
                killed=g.metrics['eliminations']-eliminations
                available=ready_seconds(dt,cooldown,roaming=start_phase is Phase.ROAMING,
                    active=active and own.active,entered_meeting=g.phase not in (Phase.ROAMING,Phase.FINISHED),
                    killed=bool(killed),kill_cooldown=g.config.kill_cooldown)
                ready+=available;components['cooldown_ready']+=self.rewards.cooldown_ready_per_second*available
                components['kill']+=self.rewards.successful_kill*killed
                if not was_ejected and own.status=='ejected':components['ejection']+=self.rewards.own_ejection
                elapsed+=dt
                if g.is_terminal:
                    components['terminal']+=(self.rewards.win if g.result.winner is Role.IMPOSTOR else
                        self.rewards.loss if g.result.winner is Role.CREWMATE else self.rewards.timeout)
                    break
            self.windows+=1;self._observe()
            components['pursuit']+=self.pending_pursuit;self.pending_pursuit=0.
            if g.is_terminal or not self.coalesce or len(self.options)>1:break
            forced+=1
        self.finished=g.is_terminal
        reward=sum(components.values());self.total_reward+=reward
        self.reward_totals.update(components);self.ready_elapsed+=ready
        return self.packet,float(reward),self.finished,False,dict(elapsed_seconds=elapsed,discount=1.,
            undiscounted_reward=reward,forced_windows_coalesced=forced,reward_components=dict(components),
            cooldown_ready_seconds=ready)

    def step(self,action):
        if self.finished:raise RuntimeError('Reset after terminal')
        if not self.action_space.contains(action) or not self.packet['mask'][int(action)]:
            raise ValueError('Invalid masked selection')
        choice=self.options[int(action)];self.last_kind=choice.kind;self.policy_decisions+=1
        self.decision_journal.append(int(action))
        if family(choice.kind)=='travel':self.goal=choice.position
        elif choice.kind=='cancel_navigation':self.goal=None
        initial=self.envelope.actor
        known_players={p.player_id for p in initial.visible_players}
        known_bodies={b.victim_id for b in initial.visible_bodies}
        self.visual_frames=[self._frame()] if self.render_trace else []
        reward=elapsed=ready=0.;components=Counter({k:0. for k in COMPONENTS})
        first=True
        while True:
            _,r,done,_,info=self._advance(choice.intent if first else None);first=False
            elapsed+=info['elapsed_seconds'];reward+=r;ready+=info['cooldown_ready_seconds']
            components.update(info['reward_components']);obs=self.envelope.actor
            if any(a.kind is IntentKind.KILL for a in obs.actions):self.legal_kill_windows+=1
            if (done or family(choice.kind)!='travel' or elapsed>=self.option_config.travel_seconds-1e-8
                or obs.context.phase is not Phase.ROAMING or obs.context.phase is not initial.context.phase
                or obs.navigation_status!='moving' or {p.player_id for p in obs.visible_players}-known_players
                or {b.victim_id for b in obs.visible_bodies}-known_bodies
                or any(a.kind is IntentKind.KILL for a in obs.actions)):
                break
        self.strategic_return+=reward
        info.update(elapsed_seconds=elapsed,discount=1.,undiscounted_reward=reward,
            reward_components=dict(components),cooldown_ready_seconds=ready,
            kill_bonus=components['kill'],ejection_penalty=components['ejection'])
        if done:
            self.models.assert_unchanged();info['episode']=self.summary()
        return self.packet,float(reward),done,False,info

    def summary(self):
        result=super().summary();g=self.match.game
        focal_events=[e for e in g.event_log if e.get('player_id')==self.match.focal]
        result.update(schema=SCHEMA,rewards=asdict(self.rewards),options=asdict(self.option_config),
            reward_components=dict(self.reward_totals),cooldown_ready_seconds=self.ready_elapsed,
            fake_repeated_continuations=self.fake_repeats,
            fake_starts=sum(e['kind']=='fake_task_start' for e in focal_events),
            fake_completions=sum(e['kind']=='fake_task_end' for e in focal_events),
            fake_interruptions=sum(e['kind']=='task_interrupted' and str(e.get('task_id','')).startswith('fake:') for e in focal_events),
            pursuit_positive_total=self.pursuit.positive_total)
        return result
