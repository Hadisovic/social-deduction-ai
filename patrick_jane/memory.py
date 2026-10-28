"""Owner-bound evidence memory; claims retain their source and assertion time."""
from collections import deque
from dataclasses import dataclass
from social_deduction.actor import PlayerSeen, BodySeen, Report, Meeting, Vote, Ejection, Phase
from social_deduction.phase3_api import StructuredClaim, WitnessedElimination
from .api import Event, Observation


@dataclass(frozen=True)
class LastSeen:
    position: tuple[float,float]
    room: str
    time: float
    velocity: tuple[float,float]
    interacting: bool


class Memory:
    def __init__(self, owner, capacity=64):
        self.owner=owner;self.capacity=capacity;self.events=deque(maxlen=capacity)
        self.last_seen={};self.known_dead=set();self.confirmed={};self.own_statements=[]
        self._delivered=set();self._visible={};self._bodies=set();self.time=-1.;self.own_room=None

    def append(self,event):
        if event.delivered>self.time+1e-8 or event.observed>event.delivered+1e-8:
            raise ValueError('Future evidence rejected')
        if event not in self._delivered:
            self._delivered.add(event);self.events.append(event)
            if event.provenance=='claim' and event.source==self.owner:self.own_statements.append(event)

    def update(self,obs:Observation):
        if type(obs) is not Observation or obs.actor.own.player_id!=self.owner:
            raise ValueError('Private memory cannot change owners or accept truth')
        a=obs.actor;now=a.time_seconds
        if now<self.time:raise ValueError('New match requires new memory')
        self.time=now
        if not a.own.active:return  # no post-death observation ingestion
        if a.own.room!=self.own_room:
            self.append(Event('own_location',self.owner,room=a.own.room,position=(a.own.position.x,a.own.position.y),
                              observed=now,delivered=now,source=self.owner,provenance='direct'))
            self.own_room=a.own.room
        current={p.player_id:p for p in a.visible_players}
        for pid,p in current.items():
            previous=self.last_seen.get(pid);position=(p.position.x,p.position.y)
            if pid not in self._visible or previous is None or previous.room!=p.room:
                self.append(Event('sighting',pid,room=p.room,position=position,observed=now,delivered=now,
                                  source=self.owner,provenance='direct'))
            elif (p.velocity.x,p.velocity.y)!=(0.,0.) and previous.velocity==(0.,0.):
                self.append(Event('movement',pid,room=p.room,position=position,observed=now,delivered=now,
                                  source=self.owner,provenance='direct'))
            self.last_seen[pid]=LastSeen(position,p.room,now,(p.velocity.x,p.velocity.y),p.interacting)
        # Meetings remove local sight by design; avoid inventing a physical departure.
        if a.context.phase is Phase.ROAMING:
            for pid in sorted(self._visible.keys()-current.keys()):
                old=self.last_seen[pid]
                self.append(Event('lost_sight',pid,room=old.room,position=old.position,
                                  observed=now,delivered=now,source=self.owner,provenance='direct'))
        self._visible=current
        for body in a.visible_bodies:
            self.known_dead.add(body.victim_id)
            if body.victim_id not in self._bodies:
                self.append(Event('body',body.victim_id,room=body.room,position=(body.position.x,body.position.y),
                                  observed=now,delivered=now,source=self.owner,provenance='direct'))
                self._bodies.add(body.victim_id)
        for record in a.evidence:
            p=record.payload;t=record.tick*a.settings.dt;event=None
            if type(p) is Report:
                self.known_dead.add(p.victim_id)
                event=Event('report',p.reporter_id,p.victim_id,p.room,observed=t,delivered=t,source=p.reporter_id)
            elif type(p) is Meeting:
                absent={i.player_id for i in a.roster}-set(p.participants)
                self.known_dead.update(absent)
                event=Event('meeting',observed=t,delivered=t)
            elif type(p) is Vote:
                event=Event('vote',p.voter_id,p.target_id or '',observed=t,delivered=t,source=p.voter_id,detail=p.meeting_id)
            elif type(p) is Ejection:
                if p.player_id:self.known_dead.add(p.player_id)
                event=Event('ejection',p.player_id or '',observed=t,delivered=t,detail=p.meeting_id)
            elif type(p) is StructuredClaim:
                event=Event(p.kind.value,p.subject_id or '',room=p.region,observed=p.asserted_tick*a.settings.dt,
                            delivered=t,source=p.speaker_id,provenance='claim')
            elif type(p) is WitnessedElimination and record.provenance.value=='direct':
                self.known_dead.add(p.victim_id)
                event=Event('witness',p.killer_id,p.victim_id,p.region,(p.position.x,p.position.y),t,t,self.owner,'direct')
            if event:self.append(event)
        for event in obs.events:
            self.append(event)
            if event.kind=='confirmed_role':self.confirmed[event.subject]=event.detail

    def summary(self):
        return dict(owner=self.owner,event_count=len(self.events),last_seen_count=len(self.last_seen),
                    own_statement_count=len(self.own_statements))
