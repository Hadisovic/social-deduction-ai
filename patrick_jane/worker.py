"""Simulation only. This module never imports Torch or owns an optimizer."""
from collections import defaultdict,Counter
import hashlib
import json
import math
import traceback
from pathlib import Path
from phase3_engine import stream_seed,json_value
from social_deduction.actor import Role,Phase
from social_deduction.phase3_api import IntentKind
from .environment import PatrickJaneGame
from .memory import Memory
from .navigation import NavigationMemory
from .features import encode,model_packet
from .rewards import RewardLedger
from .opponents import Opponent


class Match:
    def __init__(self,seed,experiment,*,held_out=False,replay=None,metadata=None,live_dir=None):
        self.game=PatrickJaneGame(seed,experiment=experiment)
        self.crew=tuple(pid for pid,p in self.game.players.items() if p.role is Role.CREWMATE)
        self.memory={p:Memory(p,experiment.event_capacity) for p in self.crew}
        self.nav={p:NavigationMemory(self.game.map,stream_seed(seed,'pj:nav:'+p)) for p in self.crew}
        self.opponents={p:Opponent(stream_seed(seed,'pj:script:'+p),held_out,
            getattr(experiment,'allowed_sabotages',('lights','oxygen','reactor'))) for p in self.game.players if p not in self.crew}
        if str(getattr(experiment,'profile','')).startswith('patrick-jane-lights-learning-'):
            from .lights_setup import LightsLedger
            self.ledger=LightsLedger(self.game)
        else:self.ledger=RewardLedger(self.game)
        self.packets={};self.rewards=[];self.recorder=None;self.trace=hashlib.sha256()
        self.decisions=0;self.all_decisions=defaultdict(int);self.last_phase=self.game.phase
        self.last_direct={p:0 for p in self.crew};self.recording_seconds=0.
        self.live=None;self.cumulative=defaultdict(float);self.bootstrap_packets={}
        self.exposure={p:Counter() for p in self.crew};self.selected={p:Counter() for p in self.crew}
        if live_dir:
            from .live import LiveBridge
            self.live=LiveBridge(self,live_dir,metadata)
        if replay:
            from .replay import Recorder
            self.recorder=Recorder(self,replay,metadata or {})

    def _bonus(self,pid,obs):
        memory=self.memory[pid];nav=self.nav[pid];memory.update(obs);fresh=nav.update(obs)
        bonus=self.ledger.observation_bonus(pid,obs,fresh,nav,memory)
        if bonus:self.rewards.append((self.game.time,pid,bonus))

    def prepare(self):
        self.packets={};game=self.game
        if self.live:self.live.check()
        if game.is_terminal:return self.response()
        # Project every learner and opponent before anybody executes an action.
        self.observations={pid:game.observe_pj(pid) for pid,p in game.players.items() if p.active}
        for pid in self.crew:
            if pid not in self.observations:continue
            obs=self.observations[pid];a=obs.actor
            self._bonus(pid,obs)
            eligible=(a.context.phase is Phase.ROAMING or
                a.context.phase is Phase.DISCUSSION and a.context.speaker_id==pid or
                a.context.phase is Phase.VOTING and pid in a.context.participants and pid not in a.context.voted_ids)
            if eligible:
                self.packets[pid]=encode(obs,self.memory[pid],self.nav[pid])
                counts=self.exposure[pid];counts['decisions']+=1
                counts['roaming' if a.context.phase is Phase.ROAMING else 'discussion' if a.context.phase is Phase.DISCUSSION else 'voting']+=1
                counts['body_visible']+=bool(a.visible_bodies)
                counts['legal_report']+=any(x.kind.value=='report' for x in a.actions)
                counts['legal_repair']+=bool(self.packets[pid].interaction_mask[3])
                if obs.sabotage:counts['during_'+obs.sabotage]+=1
        return self.response()

    def response(self):
        result=dict(time=self.game.time,terminal=self.game.is_terminal,
            packets={p:model_packet(x) for p,x in self.packets.items()},rewards=self.rewards,
            result=self.summary() if self.game.is_terminal else None,
            monitoring=bool(self.live and self.live.active),phase=self.game.phase.value,
            live_error=self.live.error if self.live else None,
            bootstrap_packets=self.bootstrap_packets)
        if self.recorder:
            for time,pid,components in self.rewards:self.recorder.reward(time,pid,components)
        for time,pid,components in self.rewards:
            self.cumulative[pid]+=sum(components.values())
            if self.live:self.live.reward(time,pid,components,self.cumulative[pid])
        self.rewards=[]
        return result

    def advance(self,decisions):
        from .decoder import execute
        actions={}
        for pid,record in decisions.items():
            choice=record['choice'];packet=self.packets[pid]
            actions[pid]=execute(choice,packet,self.nav[pid]);self.decisions+=1;self.all_decisions[pid]+=1
            counts=self.selected[pid]
            if packet.phase==2:
                counts['skip' if choice.voting==14 else 'vote']+=1
                if choice.voting!=14:
                    target=packet.other_ids[choice.voting]
                    counts['vote_target/'+target]+=1
            elif packet.phase==1:
                kind=('silent','share_evidence','accuse','defend','question','strategic_claim')[choice.communication]
                counts[kind]+=1
                if kind=='accuse':
                    target=packet.candidates[choice.target].key
                    counts['accusation_target/'+target]+=1
            else:counts[('travel','task','report','repair','emergency')[choice.interaction]]+=1
            self.trace.update(json.dumps(json_value(dict(player=pid,tick=self.game.tick,action=actions[pid])),sort_keys=True).encode())
            if self.recorder:self.recorder.neural(pid,record,packet,self.nav[pid])
            if self.live:self.live.neural(pid,record,packet,self.nav[pid])
        for pid,script in self.opponents.items():
            if pid in self.observations:actions[pid]=script.decide(self.observations[pid])
        interval=max(1,round(self.game.config.decision_interval/self.game.config.dt))
        first=True;start_phase=self.game.phase
        while not self.game.is_terminal:
            pre={pid:p.active for pid,p in self.game.players.items()}
            previous_time=self.game.time
            cursor=len(self.game.event_log)
            self.game.step(actions if first else {})
            if getattr(self.game.experiment,'record_diagnostics',False):
                for ev in self.game.event_log[cursor:]:
                    if ev['kind']=='task_interrupted' and ev.get('reason')=='new_intention' and ev.get('player_id') in self.crew:
                        action=actions.get(ev['player_id']) if first else None
                        if action is not None and getattr(action,'kind',None) in (
                                IntentKind.GO_TO_TASK,IntentKind.GO_TO_ROOM,
                                IntentKind.GO_TO_LOCATION,IntentKind.GO_TO_BODY):
                            self.game.learning_diagnostics['task_interruptions_navigation']+=1
            for pid,components in self.ledger.collect(self.game).items():
                if components:self.rewards.append((self.game.time,pid,components))
            if self.game.truncation_pending and not self.game.is_terminal:
                # Only the same legitimate private roaming packets as ordinary
                # inference. No action, RNG sample, ghost observation or budget
                # decision is fabricated for the critic bootstrap.
                for pid in self.crew:
                    if not self.game.players[pid].active:continue
                    obs=self.game.observe_pj(pid);self._bonus(pid,obs)
                    self.bootstrap_packets[pid]=model_packet(encode(obs,self.memory[pid],self.nav[pid]))
                self.game.finalize_truncation()
            for pid in decisions:
                p=self.game.players[pid];packet=self.packets[pid]
                waiting_tasks=[t for t,d in zip(packet.observation.actor.own.tasks,packet.observation.own_tasks)
                               if t.progress<1 and d.ready_in_seconds>0]
                incomplete=[t for t in packet.observation.actor.own.tasks if t.progress<1]
                incubation_wait=bool(incomplete) and len(waiting_tasks)==len(incomplete)
                # Measure actual elapsed physics time; exclude incubation, interaction,
                # movement and meetings rather than charging a nominal .6 seconds.
                avoidable=(pre[pid] and p.active and start_phase is Phase.ROAMING and self.game.phase is Phase.ROAMING and
                           p.navigator.status.value!='moving' and p.interaction_task is None and pid not in self.game.repair_holds and not incubation_wait and
                           decisions[pid]['choice'].interaction==0 and decisions[pid]['choice'].navigation==0)
                penalty=self.ledger.waiting(pid,self.game.time-previous_time,avoidable)
                if penalty:self.rewards.append((self.game.time,pid,penalty))
            if self.recorder:self.recorder.step()
            if self.live:self.live.step()
            first=False
            interrupt=self.game.phase is not start_phase or any(len(self.game.direct_events[p])>self.last_direct[p] for p in self.crew)
            if self.game.tick%interval==0 or interrupt:break
        self.last_direct={p:len(self.game.direct_events[p]) for p in self.crew}
        if self.game.is_terminal:
            self.packets={}
            response=self.response()
            if self.live:self.live.publish(force=True)  # Include final delayed rewards in live display.
            if self.recorder:self.recorder.finish()
            return response
        return self.prepare()

    def summary(self):
        g=self.game;result=g.result
        ejections=[e for e in g.event_log if e['kind']=='ejection' and e.get('player_id')]
        sabotages=Counter(e.get('sabotage') for e in g.event_log if e['kind']=='sabotage_started')
        repaired=Counter(e.get('sabotage') for e in g.event_log if e['kind']=='sabotage_repaired')
        choices=Counter()
        for actor_counts in self.selected.values():
            for key,value in actor_counts.items():
                if '/' in key:
                    family,target=key.split('/',1)
                    key=('correct_' if g.players[target].role is Role.IMPOSTOR else 'incorrect_')+('vote' if family=='vote_target' else 'accusation')
                choices[key]+=value
        return dict(seed=g.seed,players=len(g.players),crew=len(self.crew),impostors=len(g.players)-len(self.crew),
            winner=result.winner.value if result and result.winner else None,reason=result.reason if result else None,
            simulation_seconds=g.time,individual_decisions=self.decisions,per_actor=dict(self.all_decisions),
            metrics=g.metrics,truncated=g.truncation_details is not None,truncation=g.truncation_details,
            correct_ejections=sum(g.players[e['player_id']].role is Role.IMPOSTOR for e in ejections),
            innocent_ejections=sum(g.players[e['player_id']].role is Role.CREWMATE for e in ejections),
            skips=sum(e['kind']=='vote_submitted' and e.get('target_id') is None for e in g.event_log),
            mean_coverage=sum(float((n.seen_at>=0).mean()) for n in self.nav.values())/len(self.nav),
            memories={p:m.summary() for p,m in self.memory.items()},trace=self.trace.hexdigest(),
            exposure=dict(sum(self.exposure.values(),Counter())),neural_actions=dict(choices),
            sabotages_by_kind=dict(sabotages),repairs_by_kind=dict(repaired),
            critical_cancellations=sum(e['kind']=='sabotage_cancelled' for e in g.event_log),
            crew_reports=sum(e['kind']=='report' and e.get('reporter_id') in self.crew for e in g.event_log),
            impostor_reports=sum(e['kind']=='report' and e.get('reporter_id') not in self.crew for e in g.event_log),
            ghost_tasks=sum(e['kind']=='task_completed' and e.get('ghost',False) for e in g.event_log),
            recording_seconds=self.recorder.seconds if self.recorder else 0.,
            monitor_seconds=self.live.seconds if self.live else 0.,monitor_error=self.live.error if self.live else None,
            monitor_transient_write_failures=self.live.transient_write_failures if self.live else 0,
            learning_diagnostics={**dict(g.learning_diagnostics),
                'task_starts':sum(e['kind']=='task_start' and e.get('player_id') in self.crew for e in g.event_log),
                'task_interruptions':sum(e['kind']=='task_interrupted' and e.get('player_id') in self.crew for e in g.event_log),
                'task_interruptions_new_intention':sum(e['kind']=='task_interrupted' and e.get('player_id') in self.crew and e.get('reason')=='new_intention' for e in g.event_log),
                'navigation_requests':sum(e['kind']=='navigate' and e.get('player_id') in self.crew for e in g.event_log),
                'lights_actor_seconds':sum(g.lights_actor_seconds.values())})


def serve(connection):
    from time import perf_counter
    match=None
    while True:
        try:
            command,payload=connection.recv()
            began=perf_counter()
            if command=='close':break
            if command=='reset':match=Match(**payload);value=match.prepare()
            elif command=='advance':value=match.advance(payload)
            else:raise ValueError('Unknown simulation command')
            value['worker_simulation_seconds']=perf_counter()-began
            connection.send(('ok',value))
        except EOFError:break
        except BaseException:
            connection.send(('error',traceback.format_exc()));break
    connection.close()
