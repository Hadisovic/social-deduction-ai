"""Actor-only policy inputs and choices. No simulator, roles or reward queries."""
from dataclasses import dataclass
import math
import numpy as np
from belief.features import encode as belief_features
from social_deduction.actor import Phase, Point
from social_deduction.phase3_api import ClaimDraft, ClaimKind, Intent, IntentKind, WitnessedElimination

SCHEMA='strategic-candidates-v1'
MAX_ACTIONS=96
KINDS=('continue','wait','task','room','follow','flee','approach_body','report','emergency_travel',
       'emergency','vote','skip','claim_suspect','claim_defend','claim_sighting','claim_location','claim_kill')
REGIONS=('Cafeteria','Weapons','O2','Navigation','Shields','Communications','Storage','Admin',
         'Electrical','Lower Engine','Security','Reactor','Upper Engine','MedBay','NorthHallway',
         'CrossHallway','AdminHallway','BigYHallway','SouthHallway','Unknown')
GLOBAL_NAMES=('x','y','active','moving','interacting','own_remaining','own_completed',
              'nearest_task_distance','has_task','interaction_progress','meeting_allowance','time',
              'speaker','voted','body_visible','body_distance')+tuple('room_'+r for r in REGIONS)+tuple('phase_'+p.value for p in Phase)
PLAYER_FEATURES=26
ACTION_FEATURES=len(KINDS)+6


@dataclass(frozen=True)
class Choice:
    kind: str
    intent: Intent | None = None
    target_player: str | None = None
    position: Point | None = None
    task_id: str | None = None
    progress: float = 0.

    @property
    def label(self):
        target=self.intent.target if self.intent else None
        return self.kind+(' / '+str(self.target_player or target) if target or self.target_player else '')


def choices(obs,memory):
    """Parameterized mechanical actions. Every social selection is left to PPO."""
    result=[Choice('continue')]
    if not obs.own.active or obs.context.phase is Phase.FINISHED: return tuple(result)
    available={a.kind for a in obs.actions}
    if obs.context.phase is Phase.ROAMING:
        result.append(Choice('wait',Intent(IntentKind.CANCEL_NAVIGATION)))
        destinations={d.id:d for d in obs.destinations}
        for t in sorted(obs.own.tasks,key=lambda t:t.task_id):
            if t.progress<1:
                d=destinations[t.console_id]
                result.append(Choice('task',Intent(IntentKind.GO_TO_TASK,d.id),position=d.standing,
                                     task_id=t.task_id,progress=t.progress))
        for a in obs.actions:
            if a.kind is IntentKind.GO_TO_ROOM:
                # Public console centroid identifies the room to the network.
                # This is a location descriptor, not the planner's standing goal.
                # Without it every room action would have an identical input row.
                points=[d.standing for d in obs.destinations if d.room==a.target]
                # Fourteen rooms have public consoles. Hallways remain traversable
                # by navigation, but are not standalone strategic room choices.
                if points:
                    point=Point(sum(p.x for p in points)/len(points),sum(p.y for p in points)/len(points))
                    result.append(Choice('room',a,position=point))
            elif a.kind is IntentKind.REPORT:
                result.append(Choice('report',a,target_player=a.target))
            elif a.kind is IntentKind.GO_TO_BODY:
                b=next(b for b in obs.visible_bodies if b.victim_id==a.target)
                result.append(Choice('approach_body',a,a.target,b.position))
            elif a.kind is IntentKind.CALL_MEETING:
                result.append(Choice('emergency',a))
        for p in sorted(obs.visible_players,key=lambda p:p.player_id):
            result.append(Choice('follow',Intent(IntentKind.GO_TO_LOCATION,position=p.position),p.player_id,p.position))
            # A public console gives a reachable refuge. Choice of whom to flee
            # is learned; this geometry rule never sees a belief or hidden role.
            d=max(obs.destinations,key=lambda d:math.hypot(d.standing.x-p.position.x,d.standing.y-p.position.y))
            result.append(Choice('flee',Intent(IntentKind.GO_TO_TASK,d.id),p.player_id,d.standing))
        if obs.own.meetings_remaining:
            d=next(d for d in obs.destinations if d.name=='EmergencyConsole')
            result.append(Choice('emergency_travel',Intent(IntentKind.GO_TO_TASK,d.id),position=d.standing))
    elif obs.context.phase is Phase.VOTING:
        for a in obs.actions:
            if a.kind is IntentKind.VOTE: result.append(Choice('vote',a,a.target))
            elif a.kind is IntentKind.SKIP: result.append(Choice('skip',a))
    elif IntentKind.CLAIM in available:
        def claim(kind,draft,target=None):
            result.append(Choice(kind,Intent(IntentKind.CLAIM,claim=draft),target))
        claim('claim_location',ClaimDraft(ClaimKind.WAS_IN_REGION,obs.own.player_id,obs.own.room,obs.tick))
        for pid in memory.candidates:
            claim('claim_suspect',ClaimDraft(ClaimKind.SUSPECT_PLAYER,pid,obs.own.room,obs.tick),pid)
            claim('claim_defend',ClaimDraft(ClaimKind.DEFEND_PLAYER,pid,obs.own.room,obs.tick),pid)
            last=memory.last_seen.get(pid)
            if last:
                claim('claim_sighting',ClaimDraft(ClaimKind.SAW_PLAYER,pid,last.payload.room,last.tick),pid)
        seen=set()
        for e in reversed(memory.events):
            if type(e.payload) is WitnessedElimination and e.payload.killer_id not in seen:
                p=e.payload; seen.add(p.killer_id)
                claim('claim_kill',ClaimDraft(ClaimKind.SAW_ELIMINATION,p.killer_id,p.region,e.tick),p.killer_id)
    if len(result)>MAX_ACTIONS: raise ValueError('Action schema capacity exceeded; never silently truncate')
    return tuple(result)


def mechanical_intent(choice,obs):
    if choice is None or not obs.own.active: return None
    if choice.kind=='task':
        if obs.context.phase is not Phase.ROAMING: return None
        task=next((t for t in obs.own.tasks if t.task_id==choice.task_id and t.progress<1),None)
        if task is None: return None
        if task.task_id in obs.interactable_tasks:
            return None if obs.interaction_task==task.task_id else Intent(IntentKind.INTERACT,task.task_id)
        if obs.navigation_status=='moving' and obs.navigation_target==task.console_id: return None
        return Intent(IntentKind.GO_TO_TASK,task.console_id)
    a=choice.intent
    if a and a.kind in (IntentKind.GO_TO_TASK,IntentKind.GO_TO_ROOM):
        target=('room:' if a.kind is IntentKind.GO_TO_ROOM else '')+a.target
        if obs.navigation_status=='moving' and obs.navigation_target==target:return None
    return a


def encode(obs,memory,belief,options,*,ablation=False):
    own=obs.own; ids=memory.candidates
    x,y=own.position.x,own.position.y
    ds={d.id:d for d in obs.destinations}
    tasks=[t for t in own.tasks if t.progress<1]
    distance=lambda p:min(1.,math.hypot(p.x-x,p.y-y)/60)
    active=next((t.progress for t in own.tasks if t.task_id==obs.interaction_task),0.)
    values=[x/30,y/30,float(own.active),float(obs.navigation_status=='moving'),float(obs.interaction_task is not None),
        len(tasks)/8,sum(t.progress>=1 for t in own.tasks)/8,
        min((distance(ds[t.console_id].standing) for t in tasks),default=0.),float(bool(tasks)),active,
        float(own.meetings_remaining),min(1.,obs.time_seconds/240),float(obs.context.speaker_id==own.player_id),
        float(own.player_id in obs.context.voted_ids),float(bool(obs.visible_bodies)),
        min((distance(b.position) for b in obs.visible_bodies),default=0.)]
    region=own.room if own.room in REGIONS else 'Unknown'
    values += [float(r==region) for r in REGIONS]+[float(p is obs.context.phase) for p in Phase]
    base=belief_features(memory,ids)/8
    visible={p.player_id:p for p in obs.visible_players}
    logits=dict(zip(belief.player_ids,belief.logits)); probs=belief.by_player
    center=sum(logits.values())/4
    players=[]
    for i,pid in enumerate(ids):
        p=visible.get(pid)
        players.append([*base[i],.25 if ablation else probs[pid],0. if ablation else math.tanh((logits[pid]-center)/10),
            (p.position.x-x)/60 if p else 0.,(p.position.y-y)/60 if p else 0.])
    actions=np.zeros((MAX_ACTIONS,ACTION_FEATURES),np.float32)
    targets=np.full(MAX_ACTIONS,-1,np.int64); mask=np.zeros(MAX_ACTIONS,np.int8)
    for i,c in enumerate(options):
        mask[i]=1; actions[i,KINDS.index(c.kind)]=1
        p=c.position
        actions[i,len(KINDS):]=[(p.x-x)/60 if p else 0.,(p.y-y)/60 if p else 0.,distance(p) if p else 0.,
                                c.progress,float(p is not None),float(c.target_player==own.player_id)]
        if c.target_player in ids:targets[i]=ids.index(c.target_player)
    return {'global':np.asarray(values,np.float32),'players':np.asarray(players,np.float32),
            'actions':actions,'targets':targets,'mask':mask}
