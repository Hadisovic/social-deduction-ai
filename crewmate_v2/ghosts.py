"""Task-only capability: no world, roles, RNG, evidence, communication or memory inputs."""
from dataclasses import dataclass
import math
from .tasks import OwnedTask


@dataclass(frozen=True)
class GhostState:
    owner_id: str
    position: tuple[float,float]
    bounds: tuple[float,float,float,float]
    speed: float

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in (*self.position,*self.bounds,self.speed))
                or self.speed<=0 or not self.bounds[0]<=self.position[0]<=self.bounds[2]
                or not self.bounds[1]<=self.position[1]<=self.bounds[3]):
            raise ValueError('Ghost must start inside finite world bounds')


def advance_ghost(state:GhostState,tasks:tuple[OwnedTask,...],goals:dict,dt:float,now:float):
    """Return new state and completed own task. No externally selected actions exist."""
    if not math.isfinite(dt) or dt<=0:raise ValueError('Invalid ghost timestep')
    if any(t.owner_id!=state.owner_id for t in tasks):raise ValueError('Foreign ghost task')
    remaining=sorted((t for t in tasks if t.progress<1 and t.ready_at<=now+1e-9),key=lambda t:t.task_id)
    if not remaining:return state,None
    task=remaining[0];goal=tuple(goals[task.console_id])
    # `now` is the end of this physics interval; do not backdate work into incubation.
    available=min(dt,max(0.,now-task.ready_at))
    if not state.bounds[0]<=goal[0]<=state.bounds[2] or not state.bounds[1]<=goal[1]<=state.bounds[3]:
        raise ValueError('Ghost destination outside world')
    distance=math.dist(state.position,goal)
    travel=min(available,distance/state.speed)
    fraction=min(1.,state.speed*available/distance) if distance else 1.
    position=tuple(a+(b-a)*fraction for a,b in zip(state.position,goal))
    new=GhostState(state.owner_id,position,state.bounds,state.speed)
    completed=None
    if travel<available or distance<1e-9:
        if task.advance(max(0.,available-travel),now):completed=task
    return new,completed
