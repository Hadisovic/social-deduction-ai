"""Immutable actor-only V2 extension. No world/ghost/trainer objects are accepted."""
from dataclasses import dataclass, fields, is_dataclass
import math
from social_deduction.actor import Provenance, Role
from social_deduction.phase3_api import Phase3Observation, validate_observation
from . import SCHEMA


@dataclass(frozen=True)
class ConfirmedEjection:
    meeting_id: str
    player_id: str
    role: Role


@dataclass(frozen=True)
class VisualTaskProof:
    player_id: str
    task_type: str
    console_id: str


@dataclass(frozen=True)
class V2Event:
    tick: int
    provenance: Provenance
    payload: ConfirmedEjection | VisualTaskProof


@dataclass(frozen=True)
class TaskBar:
    completed: int
    total: int


@dataclass(frozen=True)
class OwnTaskDetail:
    task_id: str
    task_type: str
    category: str
    stage: int
    stages: int
    ready_in_seconds: float
    stage_seconds: float
    visual: bool


@dataclass(frozen=True)
class V2Observation:
    schema: str
    actor: Phase3Observation
    task_bar: TaskBar
    events: tuple[V2Event,...]
    own_tasks: tuple[OwnTaskDetail,...]


def validate(observation):
    if type(observation) is not V2Observation or type(observation.schema) is not str or observation.schema!=SCHEMA:
        raise TypeError('Expected exact V2 actor observation')
    validate_observation(observation.actor)
    def closed(value):
        if is_dataclass(value):
            attributes=getattr(value,'__dict__',None)
            if attributes is not None and set(attributes)!={f.name for f in fields(value)}:
                raise TypeError('Actor values cannot carry extra attributes or back-references')
            for f in fields(value):closed(getattr(value,f.name))
        elif type(value) is tuple:
            for item in value:closed(item)
    closed(observation)
    bar=observation.task_bar
    if type(bar) is not TaskBar or type(bar.completed) is not int or type(bar.total) is not int or not 0<=bar.completed<=bar.total or bar.total<=0:
        raise ValueError('Invalid public task bar')
    if type(observation.events) is not tuple or type(observation.own_tasks) is not tuple:
        raise TypeError('V2 actor collections must be immutable tuples')
    ids={p.player_id for p in observation.actor.roster}
    for event in observation.events:
        if type(event) is not V2Event or type(event.tick) is not int or not 0<=event.tick<=observation.actor.tick:
            raise ValueError('Invalid or future V2 event')
        p=event.payload
        if type(p) is ConfirmedEjection:
            if event.provenance is not Provenance.PUBLIC or type(p.role) is not Role or type(p.meeting_id) is not str:
                raise ValueError('Confirmed role requires a public authenticated ejection')
        elif type(p) is VisualTaskProof:
            if event.provenance is not Provenance.DIRECT or type(p.task_type) is not str or type(p.console_id) is not str:
                raise ValueError('Visual proof must be direct')
        else:raise TypeError('Unrecognized/privileged V2 event payload')
        if type(p.player_id) is not str or p.player_id not in ids:raise ValueError('Unknown event identity')
    if any(type(t) is not OwnTaskDetail for t in observation.own_tasks):
        raise TypeError('Own tasks must contain only exact immutable details')
    own_ids={t.task_id for t in observation.actor.own.tasks}
    if len(observation.own_tasks)!=len(own_ids) or {t.task_id for t in observation.own_tasks}!=own_ids:
        raise ValueError('V2 task details must describe only own assignments')
    for task in observation.own_tasks:
        if type(task) is not OwnTaskDetail or any(type(getattr(task,k)) is not str for k in ('task_id','task_type','category')) or task.category not in ('Common','Long','Short'):
            raise ValueError('Invalid own task detail')
        if (type(task.stage) is not int or type(task.stages) is not int or not 0<=task.stage<=task.stages
                or type(task.ready_in_seconds) not in (float,int) or not math.isfinite(task.ready_in_seconds) or task.ready_in_seconds<0
                or type(task.stage_seconds) not in (float,int) or not math.isfinite(task.stage_seconds) or task.stage_seconds<=0 or type(task.visual) is not bool):
            raise ValueError('Invalid own stage')
    return observation
