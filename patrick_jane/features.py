"""Explicit PJ tensors; no role roster, reward oracle or frozen belief scores."""
from dataclasses import dataclass
import math
import numpy as np
from social_deduction.actor import Phase
from .api import Observation
from .navigation import Goal

SELF_WIDTH=192;PLAYER_WIDTH=16;EVENT_WIDTH=16;TARGET_WIDTH=16;MAX_OTHERS=14
KINDS=('sighting','lost_sight','movement','body','report','meeting','vote','ejection','witness',
       'sabotage','sabotage_repaired','confirmed_role','visual_task','share_evidence','accuse',
       'defend','question','strategic_claim','saw_player','was_in_region','found_body',
       'saw_player_near_body','suspect_player','defend_player','was_at_task','saw_elimination','own_location')
TARGET_TYPES=('player','location','task','explore','body','repair','evidence')


@dataclass(frozen=True)
class Target:
    kind: str
    key: str
    position: tuple[float,float] | None = None
    detail: object = None


@dataclass
class Packet:
    self_features: np.ndarray
    players: np.ndarray
    player_mask: np.ndarray
    events: np.ndarray
    event_mask: np.ndarray
    targets: np.ndarray
    target_types: np.ndarray
    interaction_mask: np.ndarray
    navigation_mask: np.ndarray
    communication_mask: np.ndarray
    vote_mask: np.ndarray
    phase: int
    candidates: tuple[Target,...]
    other_ids: tuple[str,...]
    observation: Observation


def encode(obs,memory,nav):
    if type(obs) is not Observation or memory.owner!=obs.actor.own.player_id:raise ValueError('Owner-bound PJ packet required')
    a=obs.actor;own=a.own;now=a.time_seconds;position=(own.position.x,own.position.y)
    ids=tuple(i.player_id for i in a.roster if i.player_id!=own.player_id)
    if len(ids)>MAX_OTHERS:raise ValueError('Population exceeds registered maximum')
    idn=lambda p:(int(p[1:])+1)/15 if p and p.startswith('p') and p[1:].isdigit() else 0.
    rooms=tuple(a.map.rooms);roomn=lambda r:(rooms.index(r)+1)/(len(rooms)+1) if r in rooms else 0.
    phase=(Phase.ROAMING,Phase.DISCUSSION,Phase.VOTING,Phase.FINISHED).index(a.context.phase)
    values=[position[0]/30,position[1]/30,roomn(own.room),float(own.active),now/900,
            phase/3,len(ids)/14,own.meetings_remaining,obs.task_bar[0]/max(1,obs.task_bar[1]),
            float(obs.sabotage=='lights'),float(obs.sabotage=='oxygen'),float(obs.sabotage=='reactor'),
            (obs.sabotage_remaining or 0)/30,a.settings.sight_range/6.75,
            float(a.interaction_task is not None),float(a.navigation_status=='moving'),
            float(nav.goal is not None),float(nav.original is not None),
            float(np.mean(nav.seen_at>=0)),len(memory.events)/memory.capacity]
    if nav.goal:values.extend([nav.goal.position[0]/30,nav.goal.position[1]/30,math.dist(position,nav.goal.position)/60])
    else:values.extend([0.,0.,0.])
    for task,detail in zip(own.tasks,obs.own_tasks):
        dest=next(d for d in a.destinations if d.id==task.console_id)
        values.extend([task.progress,detail.stage/max(1,detail.stages),detail.ready_in_seconds/60,
                       detail.stage_seconds/60,dest.standing.x/30,dest.standing.y/30,float(detail.visual)])
    # Known utility and vent geometry, not hidden actors. Reserve explicit finite slots.
    for d in a.destinations:
        if d.category=='utility':values.extend([d.standing.x/30,d.standing.y/30])
    values.append(len(obs.own_route)/100)
    for i in range(4):values.extend([v/30 for v in obs.own_route[i]] if i<len(obs.own_route) else [0.,0.])
    for vent in nav.geometry.data.get('vents',{}).values():
        point=vent.get('position')
        if point is not None:values.extend([point['x']/30,point['y']/30])
    values.extend([nav.original.position[0]/30,nav.original.position[1]/30,
                   max(0.,nav.detour_until-now)/8] if nav.original else [0.,0.,0.])
    values.append(idn(own.player_id))
    self_features=np.zeros(SELF_WIDTH,np.float32);self_features[:min(len(values),SELF_WIDTH)]=values[:SELF_WIDTH]
    players=np.zeros((MAX_OTHERS,PLAYER_WIDTH),np.float32);pm=np.zeros(MAX_OTHERS,bool)
    visible={p.player_id:p for p in a.visible_players}
    for i,pid in enumerate(ids):
        pm[i]=True;seen=memory.last_seen.get(pid);p=visible.get(pid)
        players[i,:]=[idn(pid),float(pid not in memory.known_dead),float(p is not None),float(seen is not None),
            seen.position[0]/30 if seen else 0,seen.position[1]/30 if seen else 0,roomn(seen.room) if seen else 0,
            min(1.,(now-seen.time)/120) if seen else 1.,seen.velocity[0]/3 if seen else 0,
            seen.velocity[1]/3 if seen else 0,float(seen.interacting) if seen else 0,
            float(pid in a.context.participants),float(pid in a.context.voted_ids),
            float(memory.confirmed.get(pid)=='crewmate'),float(memory.confirmed.get(pid)=='impostor'),
            sum(e.subject==pid for e in memory.events)/memory.capacity]
    ev=np.zeros((memory.capacity,EVENT_WIDTH),np.float32);em=np.zeros(memory.capacity,bool)
    records=tuple(memory.events)
    for i,e in enumerate(records):
        em[i]=True
        ev[i,:]=[(KINDS.index(e.kind)+1)/len(KINDS) if e.kind in KINDS else 0,idn(e.subject),idn(e.other),
            roomn(e.room),e.position[0]/30 if e.position else 0,e.position[1]/30 if e.position else 0,
            e.observed/900,e.delivered/900,min(1.,(now-e.observed)/900),min(1.,(now-e.delivered)/900),
            idn(e.source),float(e.provenance=='direct'),float(e.provenance=='public'),float(e.provenance=='claim'),
            idn(e.origin_source or e.source),float(e.detail=='impostor')-float(e.detail=='crewmate')
            if e.kind=='confirmed_role' else (KINDS.index(e.detail)+1)/len(KINDS) if e.detail in KINDS else 0.]
    candidates=[]
    for pid in (own.player_id,*ids):candidates.append(Target('player',pid,position if pid==own.player_id else memory.last_seen[pid].position if pid in memory.last_seen else None))
    for d in a.destinations:candidates.append(Target('location',d.id,(d.standing.x,d.standing.y),d))
    for task in own.tasks:
        if task.progress<1:
            d=next(d for d in a.destinations if d.id==task.console_id)
            candidates.append(Target('task',task.task_id,(d.standing.x,d.standing.y),task))
    for index,point,last in nav.candidates(position):candidates.append(Target('explore',str(index),point,last))
    for b in a.visible_bodies:candidates.append(Target('body',b.victim_id,(b.position.x,b.position.y),b))
    for key in obs.repair_locations:
        if key not in obs.repaired_locations:
            d=next(d for d in a.destinations if d.id==key)
            candidates.append(Target('repair',key,(d.standing.x,d.standing.y),d))
    for i,e in enumerate(records):candidates.append(Target('evidence',str(i),e.position,e))
    tf=np.zeros((len(candidates),TARGET_WIDTH),np.float32);types=np.zeros((len(candidates),len(TARGET_TYPES)),bool)
    for i,t in enumerate(candidates):
        types[i,TARGET_TYPES.index(t.kind)]=True;tf[i,:7]=types[i]
        tf[i,7:]=[t.position[0]/30 if t.position else 0,t.position[1]/30 if t.position else 0,
            math.dist(position,t.position)/60 if t.position else 0,idn(t.key) if t.kind=='player' else 0,
            float(t.position is not None),float(t.kind=='task' and t.key in a.interactable_tasks),
            float(t.key in a.interactable_destinations),roomn(getattr(t.detail,'room','')),
            min(1.,(now-t.detail.observed)/900) if t.kind=='evidence' else 0]
    im=np.array([True,bool(a.interactable_tasks),any(x.kind.value=='report' for x in a.actions),
                 any(k in a.interactable_destinations and k not in obs.repaired_locations for k in obs.repair_locations),
                 any(x.kind.value=='call_meeting' for x in a.actions)],bool)
    may_change=now-nav.last_change>=2.4 or nav.goal is None or a.navigation_status!='moving'
    nm=np.array([nav.goal is not None,nav.goal is not None and nav.original is None and may_change,may_change],bool)
    social_targets=bool(set(ids)&set(a.context.participants))
    cm=np.array([True,bool(records),social_targets,social_targets,social_targets,True],bool)
    vm=np.zeros(15,bool)
    for i,pid in enumerate(ids):vm[i]=pid in a.context.participants and pid not in memory.known_dead
    vm[14]=True
    return Packet(self_features,players,pm,ev,em,tf,types,im,nm,cm,vm,phase,tuple(candidates),ids,obs)


def target_mask(packet,kind,choice):
    """Conditional masks are reproducible from the stored pre-action packet."""
    if hasattr(packet,'conditional_masks'):return packet.conditional_masks[(kind,choice)]
    a=packet.observation.actor
    mask=np.zeros(len(packet.candidates),bool)
    for i,t in enumerate(packet.candidates):
        if kind=='navigation':
            mask[i]=t.position is not None and t.kind in ('location','task','explore','body','repair')
            if choice==1 and mask[i]:mask[i]=.5<=math.dist((a.own.position.x,a.own.position.y),t.position)<=6.
        elif kind=='interaction':
            mask[i]=(choice==1 and t.kind=='task' and t.key in a.interactable_tasks or
                     choice==2 and t.kind=='body' and any(x.kind.value=='report' and x.target==t.key for x in a.actions) or
                     choice==3 and t.kind=='repair' and t.key in a.interactable_destinations or
                     choice==4 and t.kind=='location' and any(x.kind.value=='call_meeting' and x.target==t.key for x in a.actions))
        elif kind=='communication':
            mask[i]=(choice==1 and t.kind=='evidence' or choice in (2,3,4) and t.kind=='player' and t.key in a.context.participants or
                     choice==5 and t.kind in ('location','evidence'))
            if choice in (2,4) and t.kind=='player' and t.key==a.own.player_id:mask[i]=False
    return mask


def goal_for(target):
    if target.kind in ('location','repair'):destination=target.key
    elif target.kind=='task':destination=target.detail.console_id
    else:destination=None
    return Goal(target.key,target.position,destination)


@dataclass
class ModelPacket:
    """Compact IPC/training view; explicit actor memories stay inside their worker."""
    self_features:np.ndarray
    players:np.ndarray
    player_mask:np.ndarray
    events:np.ndarray
    event_mask:np.ndarray
    targets:np.ndarray
    interaction_mask:np.ndarray
    navigation_mask:np.ndarray
    communication_mask:np.ndarray
    vote_mask:np.ndarray
    phase:int
    conditional_masks:dict


def model_packet(packet):
    masks={(k,c):target_mask(packet,k,c) for k,cs in
           (('navigation',(1,2)),('interaction',(1,2,3,4)),('communication',(1,2,3,4,5))) for c in cs}
    return ModelPacket(*(getattr(packet,k) for k in ('self_features','players','player_mask','events','event_mask',
        'targets','interaction_mask','navigation_mask','communication_mask','vote_mask','phase')),masks)
