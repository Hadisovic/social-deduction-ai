"""Trusted projection/reward orchestration; no simulator object crosses to the NN."""
from collections import Counter
from dataclasses import asdict
import numpy as np
from frozen_handoff.controller import LegalImpostor
from social_deduction.actor import Phase,Role
from spectator.replay_format import ReplayReward
from .controller import ActorContext
from .rewards import Rewards,ready_seconds
from .time_math import gae

COMPONENTS=('terminal','kill','ejection','cooldown_ready','pursuit')

class ImpostorLedger:
    def __init__(self,game,controller,recorder=None):
        self.game=game;self.controller=controller;self.recorder=recorder;self.rewards=Rewards()
        self.totals=Counter({k:0. for k in COMPONENTS});self.rows=[];self.elapsed=0.;self.ready=0.;self.return_=0.
        self.step_components=[];self._original_step=game.step

    def decision(self,decision):
        self.settle(decision['value'],False)
        self.rows.append(dict(packet=decision['packet'],action=decision['index'],logp=decision['logp'],
            value=decision['value'],start_time=self.game.time,reward=0.,duration=0.,next_value=None,done=False,settled=False))
        if self.recorder:
            self.recorder.record_neural(self.controller.player_id,decision['kind'],decision['target'],
                probabilities=decision['probabilities'],value_estimate=decision['value'],entropy=decision['entropy'],
                legal_actions=decision['labels'],training_step=8192,policy_update=0,selected_action_index=decision['index'])

    def settle(self,next_value,done):
        if self.rows and not self.rows[-1]['settled']:
            row=self.rows[-1];row.update(next_value=float(next_value),done=bool(done),settled=True)
            if row['duration']<0 or row['duration']==0 and not done:raise ValueError('Invalid elapsed transition')

    def step(self,actions=None):
        if self.game.is_terminal:return self.game.result
        g=self.game;pid=self.controller.player_id;own=g.players[pid]
        start=g.time;phase=g.phase;active=own.active;cooldown=own.cooldown
        kills=g.metrics['eliminations'];ejected=own.status=='ejected'
        result=self._original_step(actions)
        dt=g.time-start;killed=g.metrics['eliminations']-kills
        c=Counter({k:0. for k in COMPONENTS})
        seconds=ready_seconds(dt,cooldown,roaming=phase is Phase.ROAMING,active=active and own.active,
            entered_meeting=g.phase not in (Phase.ROAMING,Phase.FINISHED),killed=bool(killed),kill_cooldown=g.config.kill_cooldown)
        c['cooldown_ready']=self.rewards.cooldown_ready_per_second*seconds
        c['kill']=self.rewards.successful_kill*killed
        if not ejected and own.status=='ejected':c['ejection']=self.rewards.own_ejection
        c['pursuit']=self.controller.pending_pursuit;self.controller.pending_pursuit=0.
        if g.is_terminal:
            c['terminal']=self.rewards.win if g.result.winner is Role.IMPOSTOR else self.rewards.loss if g.result.winner is Role.CREWMATE else self.rewards.timeout
            c['pursuit']+=self.controller.pursuit.advance(g.observe(pid))
        reward=sum(c.values());self.totals.update(c);self.ready+=seconds;self.elapsed+=dt;self.return_+=reward
        self.step_components.append(dict(seconds=dt,phase=phase.value,ready_seconds=seconds,components=dict(c)))
        if self.rows:
            self.rows[-1]['reward']+=reward;self.rows[-1]['duration']+=dt
        if self.recorder and any(c.values()):
            self.recorder.recorder.write_reward(ReplayReward(tick=g.tick,sim_time=g.time,player_id=pid,
                reward=reward,cumulative_reward=self.return_,components=dict(c)))
        if g.is_terminal:self.settle(0.,True)
        return result

    def check(self):
        if not self.game.is_terminal or not self.rows or not all(r['settled'] for r in self.rows):raise ValueError('Incomplete rollout')
        if not self.rows[-1]['done'] or self.rows[-1]['next_value']!=0:raise ValueError('Terminal bootstrap leak')
        if abs(sum(r['reward'] for r in self.rows)-self.return_)>1e-7:raise ValueError('Reward attribution mismatch')
        if abs(self.totals['pursuit'])>1e-7 or self.controller.pursuit.positive_total>.05000001:raise ValueError('Pursuit bound/cancellation failed')
        advantage,returns=gae([r['reward'] for r in self.rows],[r['value'] for r in self.rows],
            [r['next_value'] for r in self.rows],np.ones(len(self.rows)),[r['done'] for r in self.rows],
            [r['duration'] for r in self.rows])
        if not np.isfinite(advantage).all() or not np.isfinite(returns).all():raise ValueError('Nonfinite rollout credit')
        return dict(transitions=len(self.rows),terminal_bootstrap=0.,finite_advantages=True,
            zero_duration_terminal_transitions=sum(r['done'] and r['duration']==0 for r in self.rows),
            reward_sum=self.return_,components=dict(self.totals),elapsed_seconds=self.elapsed,cooldown_ready_seconds=self.ready,
            objective=asdict(self.rewards),math='impostor-terminal-time-gae-v2',optimizer_updates=0)

class ContextGate:
    """Trusted bridge owns world/recorder. The implementation gets immutable values."""
    def __init__(self,game,implementation,ledger):
        self.game=game;self.implementation=implementation;self.ledger=ledger
        self.checked=LegalImpostor(implementation,implementation.player_id);self.calls=0

    def decide(self,observation):
        if not observation.own.active or observation.context.phase is Phase.FINISHED:raise ValueError('Inactive bridge call')
        envelope=self.game.observe_v2(observation.own.player_id)
        own=self.game.players[observation.own.player_id]
        fake=observation.interaction_task is not None and observation.interaction_task.startswith('fake:')
        self.implementation.supply(ActorContext(envelope,float(own.interaction_remaining) if fake else 0.))
        before=self.implementation.decisions
        action=self.checked.decide(observation);self.calls+=1
        if self.implementation.decisions!=before:self.ledger.decision(self.implementation.last_decision)
        return action
