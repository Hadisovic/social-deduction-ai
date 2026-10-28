"""Metadata-backed task types; durations and interaction stages are authored abstractions."""
from dataclasses import dataclass
import hashlib
from random import Random
from phase3_engine import stream_seed


@dataclass(frozen=True)
class Stage:
    console_id: str
    seconds: float
    wait_before_seconds: float = 0.


@dataclass(frozen=True)
class TaskPlan:
    owner_id: str
    task_type: str
    category: str
    stages: tuple[Stage,...]
    visual: bool = False


@dataclass
class OwnedTask:
    task_id: str
    plan: TaskPlan
    stage: int = 0
    work: float = 0.
    ready_at: float = 0.

    @property
    def owner_id(self): return self.plan.owner_id
    @property
    def console_id(self): return self.plan.stages[min(self.stage,len(self.plan.stages)-1)].console_id
    @property
    def duration(self): return self.plan.stages[min(self.stage,len(self.plan.stages)-1)].seconds
    @property
    def progress(self):
        if self.stage>=len(self.plan.stages):return 1.
        return (sum(s.seconds for s in self.plan.stages[:self.stage])+self.work)/sum(s.seconds for s in self.plan.stages)

    def advance(self,seconds,now):
        if self.progress>=1 or now+1e-9<self.ready_at:return False
        self.work=min(self.duration,self.work+seconds)
        if self.work>=self.duration-1e-9:
            self.stage+=1;self.work=0.
            if self.stage<len(self.plan.stages):
                self.ready_at=now+self.plan.stages[self.stage].wait_before_seconds
        return self.progress>=1


def task_catalog(geometry):
    """Group existing console geometry by task type; no new console coordinates."""
    groups={}
    for destination in geometry.destinations:
        if destination.category=='task':
            groups.setdefault(destination.name,[]).append(destination.id)
    return {name:tuple(sorted(ids)) for name,ids in groups.items()}


def make_plan(owner,kind,category,catalog,rng):
    def one(name):return rng.choice(catalog[name])
    visual=kind in ('ClearAsteroids','SubmitScan','PrimeShields','EmptyGarbage','EmptyChute')
    if kind=='FixWiring':stages=tuple(Stage(i,4.) for i in rng.sample(catalog[kind],3))
    elif kind=='AlignEngineOutput':stages=tuple(Stage(i,8.) for i in catalog[kind])
    elif kind=='FuelEngines':
        ids=catalog[kind]
        # Native supplemental console names include Storage then both engine rooms.
        stages=tuple(Stage(i,6.) for i in (ids[0],ids[1],ids[0],ids[2]))
    elif kind=='InspectSample':stages=(Stage(one(kind),3.),)
    elif kind=='UploadData':stages=(Stage(one('DownloadData'),8.),Stage(one(kind),8.))
    elif kind=='DivertPower':stages=(Stage(one(kind),5.),Stage(one('AcceptDivertedPower'),3.))
    elif kind in ('EmptyGarbage','EmptyChute'):
        stages=(Stage(one(kind),5.),Stage(one('EmptyGarbage/EmptyChute'),5.))
    else:
        seconds={'SwipeCard':4.,'ClearAsteroids':20.,'SubmitScan':10.,'StartReactor':15.,
            'CalibrateDistributor':6.,'ChartCourse':5.,'CleanO2Filter':6.,'UnlockManifolds':6.,
            'StabilizeSteering':5.,'PrimeShields':5.}[kind]
        stages=(Stage(one(kind),seconds),)
    if kind=='InspectSample':stages+=(Stage(stages[0].console_id,3.,60.),)
    return TaskPlan(owner,kind,category,stages,visual)


def assignments(geometry,seed,roster):
    """Hypothetical plans for EVERY identity, before selecting crewmate owners."""
    catalog=task_catalog(geometry)
    common_rng=Random(stream_seed(seed,'crewmate-v2:common'))
    common=common_rng.choice(('SwipeCard','FixWiring'))
    common_plan=make_plan('',common,'Common',catalog,common_rng)
    categories={}
    for task in geometry.data['tasks'].values():
        categories.setdefault(task['length'],set()).add(task['taskType'])
    supported_long=sorted(categories['Long']-{'VentCleaning'})
    supported_short=sorted(categories['Short']-{'VentCleaning'})
    result={}
    for pid in sorted(roster):
        rng=Random(stream_seed(seed,'crewmate-v2:tasks:'+pid))
        specs=[TaskPlan(pid,common,'Common',common_plan.stages,common_plan.visual)]
        specs.append(make_plan(pid,rng.choice(supported_long),'Long',catalog,rng))
        specs.extend(make_plan(pid,kind,'Short',catalog,rng) for kind in rng.sample(supported_short,2))
        result[pid]=tuple(OwnedTask('v2-'+hashlib.sha256(f'{stream_seed(seed,"crewmate-v2:task-ids")}:{pid}:{i}'.encode()).hexdigest()[:16],spec)
                          for i,spec in enumerate(specs))
    return result
