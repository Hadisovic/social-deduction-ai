"""Trusted coordinator. Only actor-local tensors enter shared-policy inference."""
from collections import Counter
from types import SimpleNamespace
from time import perf_counter
import hashlib
import numpy as np
import torch
from belief.memory import ActorMemory
from belief.model import BeliefModel
from crewmate_v2.runtime import PublicMemory,V2StrategicMatch
from crewmate_v2.learning.env import LearningEnv
from .rewards import ActorLedger
from crewmate_v2.learning.opponents import ReporterController
from phase3_runner import behavior_seed
from phase3_bots import BotStyle
from phase5_features import mechanical_intent
from social_deduction.actor import Phase,Role
from social_deduction.phase3_api import IntentKind,intent_allowed,WitnessedElimination
from .config import Rewards,check_seed
from crewmate_v2.learning.train import state_hash

FIELDS=('packets','actions','logp','values','next_values','rewards','discounts','dones','durations')
EXTENDED={'task','room','follow','flee','approach_body','emergency_travel'}


class ActorView(LearningEnv):
    """Reuses the five-player encoder, binding a fresh memory to exactly one owner."""
    def __init__(self,game,pid,belief):
        super().__init__(lobby=game.lobby,planner=game.planner,belief_model=belief)
        self.match=SimpleNamespace(game=game);self.focal=pid
        self.memory=ActorMemory();self.public_memory=PublicMemory();self.pending=None
        self._previous_tick=-1;self.finished=False;self._inside_option=False
        self.refresh()

    def refresh(self):
        if self.match.game.players[self.focal].active and not self.match.game.is_terminal:self._observe()


class FourCrewMatch:
    def __init__(self,seed,*,lobby=None,mixture=None,rewards=None,rotation=0,deadline=None):
        check_seed(seed)
        self.match=V2StrategicMatch(seed,lobby=lobby,mixture=mixture)
        self.game=self.match.game;self.seed=seed;self.deadline=deadline or perf_counter()+120
        self.crew=tuple(sorted(pid for pid,p in self.game.players.items() if p.role is Role.CREWMATE))
        imp=next(pid for pid,p in self.game.players.items() if p.role is Role.IMPOSTOR)
        self.impostor=imp
        self.match.controllers={imp:ReporterController(behavior_seed(seed,imp),BotStyle('nearest','hunter','evidence'),rotation)}
        # Each script sees only its own observation; no other crewmate is scripted.
        belief=BeliefModel.load();belief.network.requires_grad_(False)
        self.views={pid:ActorView(self.game,pid,belief) for pid in self.crew}
        roles={pid:p.role for pid,p in self.game.players.items()}
        self.ledgers={pid:ActorLedger(pid,roles,rewards or Rewards(),self.game.config.dt) for pid in self.crew}
        self.options={};self.selected={};self.starts={};self.histories={pid:[] for pid in self.crew}
        self.rows={pid:[] for pid in self.crew};self.actions={pid:Counter() for pid in self.crew}
        self.reporting={pid:Counter() for pid in self.crew};self.opportunities={pid:{k:set() for k in ('visible','range','legal','choice_legal','held_legal')} for pid in self.crew}
        self.max_candidates=0;self.policy_version=None;self.decisions=0

    def _lease(self,pid,choice):
        view=self.views[pid];obs=view.observation
        self.selected[pid]=choice
        self.starts[pid]=dict(tick=self.game.tick,phase=obs.context.phase,
            bodies={b.victim_id for b in obs.visible_bodies},reports={a.target for a in obs.actions if a.kind is IntentKind.REPORT},
            kills={e.tick for e in view.memory.events if type(e.payload) is WitnessedElimination})
        if choice.kind!='continue':view.pending=choice if choice.kind=='task' else None

    def _ready(self,pid):
        if pid not in self.starts:return True
        view=self.views[pid];obs=view.observation;s=self.starts[pid];choice=self.selected[pid]
        if obs.context.phase is not s['phase']:
            view.pending=None;return True
        elapsed=(self.game.tick-s['tick'])*self.game.config.dt
        if elapsed<.6-1e-8:return False
        if choice.kind not in EXTENDED:return True
        if {b.victim_id for b in obs.visible_bodies}-s['bodies']:return True
        if {a.target for a in obs.actions if a.kind is IntentKind.REPORT}-s['reports']:return True
        if any(type(e.payload) is WitnessedElimination and e.tick not in s['kills'] for e in view.memory.events):return True
        if choice.kind=='task':
            if not any(t.task_id==choice.task_id and t.progress<1 for t in obs.own.tasks):return True
        elif obs.navigation_status!='moving':return True
        return elapsed>=(3. if choice.kind in ('follow','flee') else view.experiment.max_option_seconds)-1e-8

    def _settle(self,pid,next_value=0.,done=False):
        if not self.rows[pid]:return
        row=self.rows[pid][-1]
        if row.get('settled'):return
        score=self.ledgers[pid].rollup(row.pop('frame_start'))
        row.update(rewards=score['reward'],discounts=score['discount'],durations=score['elapsed_seconds'],
                   next_values=next_value,dones=done,settled=True)
        if self.recorder is not None:
            from spectator.replay_format import ReplayReward
            self.recorder.recorder.write_reward(ReplayReward(tick=self.game.tick,sim_time=self.game.time,
                player_id=pid,reward=score['undiscounted_reward'],
                cumulative_reward=sum(self.ledgers[pid].components.values()),components=score['components']))

    def collect(self,policy,*,initialization=17,policy_version=0,recorder=None,deterministic=False):
        if self.policy_version is not None:raise RuntimeError('A match cannot be collected twice')
        self.policy_version=policy_version
        self.recorder=recorder
        frozen_parameters=state_hash(policy)
        sampling_seeds={pid:int.from_bytes(hashlib.sha256(
            f'fourcrew:{initialization}:{self.seed}:{pid}'.encode()).digest()[:8],'big') for pid in self.crew}
        generators={pid:torch.Generator().manual_seed(seed) for pid,seed in sampling_seeds.items()}
        if recorder:recorder.after_step()
        while not self.game.is_terminal:
            if perf_counter()>self.deadline:raise TimeoutError('Engineering match wall ceiling')
            for pid,view in self.views.items():view.refresh()
            ready=[];packets=[]
            for pid in self.crew:
                view=self.views[pid]
                if not self.game.players[pid].active or not self._ready(pid):continue
                packet=view._packet();self.max_candidates=max(self.max_candidates,len(view.options))
                if len(view.options)==1:
                    self._lease(pid,view.options[0]);continue
                ready.append(pid);packets.append(packet)
            if ready:
                tensors={k:torch.as_tensor(np.stack([p[k] for p in packets])) for k in packets[0]}
                with torch.no_grad():distribution,values=policy(tensors)
                for i,pid in enumerate(ready):
                    probs=distribution.probs[i];index=int(probs.argmax()) if deterministic else int(torch.multinomial(probs,1,generator=generators[pid]))
                    view=self.views[pid];choice=view.options[index]
                    self._settle(pid,float(values[i]))
                    self.rows[pid].append(dict(packets=packets[i],actions=index,logp=float(probs[index].log()),values=float(values[i]),frame_start=len(self.ledgers[pid].frames)))
                    self.actions[pid][choice.kind]+=1;self.decisions+=1
                    legal={a.target for a in view.observation.actions if a.kind is IntentKind.REPORT}
                    self.opportunities[pid]['choice_legal'].update(legal)
                    if choice.kind=='report':self.reporting[pid]['selected']+=1
                    if recorder:
                        labels=[f'{j:02d}: {c.label}' for j,c in enumerate(view.options)]
                        recorder.record_neural(pid,choice.kind,choice.target_player,probabilities={labels[j]:float(probs[j]) for j in range(len(labels))},
                            value_estimate=float(values[i]),entropy=float(distribution.entropy()[i]),legal_actions=labels,
                            training_step=self.decisions,policy_update=policy_version,selected_action_index=index,
                            suspicion_probabilities=view.belief.by_player)
                    self._lease(pid,choice)
            submitted={};legal={}
            for pid,view in self.views.items():
                if not self.game.players[pid].active:continue
                obs=view.observation;opps=self.opportunities[pid]
                opps['visible'].update(b.victim_id for b in obs.visible_bodies)
                reports={a.target for a in obs.actions if a.kind is IntentKind.REPORT}
                opps['range'].update(reports);opps['legal'].update(reports)
                if pid not in ready:opps['held_legal'].update(reports)
                selected=self.selected.get(pid)
                choice=selected if self.game.tick==self.starts.get(pid,{}).get('tick') else view.pending
                intent=mechanical_intent(choice,obs)
                if intent is not None and intent.kind is not IntentKind.WAIT:
                    if not intent_allowed(obs,intent):raise ValueError('Actor produced stale/illegal intent')
                    submitted[pid]=intent;legal[pid]=True
                    if intent.kind is IntentKind.REPORT:self.reporting[pid]['submitted']+=1
            if self.game.tick%self.match.decision_ticks==0:
                obs=self.game.observe(self.impostor)
                if obs.own.active and obs.actions:
                    intent=self.match.controllers[self.impostor].decide(obs)
                    if intent.kind is not IntentKind.WAIT:submitted[self.impostor]=intent
            mark=len(self.game.event_log);self.game.step(submitted);events=self.game.event_log[mark:]
            for pid,ledger in self.ledgers.items():
                ledger.consume(events,active=self.game.players[pid].active,terminal=self.game.is_terminal,
                    winner=self.game.result.winner if self.game.is_terminal else None,intent=submitted.get(pid),legal=legal.get(pid,False))
                reports=sum(e['kind']=='report' and e.get('reporter_id')==pid for e in events)
                self.reporting[pid]['accepted']+=reports;self.reporting[pid]['meeting_triggered']+=reports
            if recorder:recorder.after_step()
            if recorder and recorder.recorder.error:raise OSError(recorder.recorder.error)
        for pid in self.crew:self._settle(pid,done=True)
        if state_hash(policy)!=frozen_parameters:raise RuntimeError('Policy version changed inside a match')
        if recorder:recorder.finish()
        if recorder and recorder.recorder.error:raise OSError(recorder.recorder.error)
        flat={k:[] for k in FIELDS};segments=[];keys=[]
        for pid in self.crew:
            lo=len(flat['actions'])
            for row in self.rows[pid]:
                for k in FIELDS:flat[k].append(row[k])
            if len(flat['actions'])>lo:segments.append((lo,len(flat['actions'])));keys.append((self.seed,pid))
        flat.update(segments=segments,trajectory_keys=keys)
        summary=dict(seed=self.seed,decisions=self.decisions,reason=self.game.result.reason,
            crew_win=self.game.result.winner is Role.CREWMATE,simulated_seconds=self.game.time,
            actors={pid:dict(actions=dict(self.actions[pid]),reward_components=dict(self.ledgers[pid].components),
                reward_counts=dict(self.ledgers[pid].counts),reporting=dict(self.reporting[pid]),
                opportunities={k:len(v) for k,v in self.opportunities[pid].items()},memory_events=len(self.views[pid].memory.events)) for pid in self.crew},
            max_candidates=self.max_candidates,policy_version=policy_version,sampling_seeds=sampling_seeds,
            illegal_actions=self.game.metrics['illegal_actions'])
        return flat,summary
