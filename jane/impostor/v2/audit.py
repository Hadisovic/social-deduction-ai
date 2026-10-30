"""Append-only compressed decisions and separate offline effects, never actor input."""
from dataclasses import asdict,is_dataclass
from enum import Enum
import gzip
import json
from pathlib import Path
from jane.impostor.v1.session import Actor as HistoricalActor,JointRecorder as HistoricalRecorder


def serial(value):
    if isinstance(value,Enum):return value.value
    if is_dataclass(value):return asdict(value)
    raise TypeError(type(value).__name__)


class AuditTrail:
    def __init__(self,replay):
        self.handle=gzip.open(Path(replay).with_suffix('.decisions.jsonl.gz'),'xt',encoding='utf-8',compresslevel=1) if replay else None
        self.count=0
    def write(self,record,**value):
        if self.handle:
            self.handle.write(json.dumps(dict(record=record,**value),default=serial,separators=(',',':'))+'\n')
            self.count+=1
            if self.count%20==0:self.handle.flush()
    def close(self):
        if self.handle:self.write('closed');self.handle.close();self.handle=None


class Actor(HistoricalActor):
    def choose(self,observation,feedback):
        command,transition,diagnostics=super().choose(observation,feedback)
        a=observation.actor;own=a.own
        diagnostics['local']=dict(position=[own.position.x,own.position.y],room=own.room,
            roster=[p.player_id for p in a.roster],visible_players=[p.player_id for p in a.visible_players],
            visible_bodies=[b.victim_id for b in a.visible_bodies],teammates=list(observation.teammates),
            navigation=a.navigation_status,own_route=observation.own_route,cooldown=feedback.cooldown,
            fake_remaining=feedback.fake_remaining,fake_console=feedback.fake_console,
            vent=observation.vent_state,door_blocked=observation.door_blocked,
            speaker=a.context.speaker_id,participants=a.context.participants,voted_ids=a.context.voted_ids)
        return command,transition,diagnostics


class JointRecorder(HistoricalRecorder):
    def __init__(self,match,path,metadata,live=None,audit=None):
        from collections import Counter
        from spectator.recorder import MatchRecorder
        self.match=match;self.live=live
        self.recorder=MatchRecorder(match,path,training_session=metadata['session'],
            game_schema='lizbon-classic-v2',source_version=metadata['source'],
            model_architecture=metadata['architecture'],controller_types=metadata['controllers'],metadata=metadata)
        self.cumulative=Counter();self.seconds=0.;self.policy_update=metadata.get('policy_update',0)
        self.audit=audit
    def impostor(self,pid,transition,diagnostics):
        super().impostor(pid,transition,diagnostics)
        p=transition.packet
        self.audit.write('decision',input_scope='actor-local',owner=pid,tick=p.tick,time=p.time,phase=p.phase,
            local=diagnostics['local'],selected=transition.action,selected_kind=p.candidates[transition.action].kind,
            log_probability=transition.logp,value=transition.value,entropy=diagnostics['entropy'],
            selection='argmax' if self.match.opponents[pid].actor.deterministic else 'sampled',
            self_features=p.self_features.tolist(),players=p.players.tolist(),player_mask=p.player_mask.tolist(),
            events=p.events.tolist(),event_mask=p.event_mask.tolist(),
            candidates=[dict(kind=c.kind,key=c.key,position=c.position,command=c.payload,
                family=int(p.families[i]),features=p.actions[i].tolist(),probability=float(prob))
                for i,(c,prob) in enumerate(zip(p.candidates,diagnostics['probabilities']))])
    def reward(self,when,pid,components):
        super().reward(when,pid,components)
        if pid not in self.match.opponents:return
        self.audit.write('reward',input_scope='learner-only',owner=pid,tick=self.match.game.tick,
            time=when,components=components,total=sum(components.values()))
