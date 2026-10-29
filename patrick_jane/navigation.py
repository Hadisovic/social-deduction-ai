"""Known-map exploration and private retained goals; no hidden world inputs."""
from dataclasses import dataclass
import math
from random import Random
import numpy as np
import shapely
from time import perf_counter
from navigation_service import NavigationPlanner,NavStatus,Plan,route_length
from social_deduction.actor import Phase, Point
from social_deduction.phase3_api import Intent, IntentKind
from social_deduction.phase3_observation import visible_between


class PatrickPlanner(NavigationPlanner):
    """Reuse map A-star for one-off exploration goals; cache fields for consoles."""
    def plan(self,start,target):
        if target.kind!='location':return super().plan(start,target)
        began=perf_counter()
        if not self.map.contains(start):return Plan(NavStatus.INVALID_START)
        try:region=self.arrival_region(target)
        except ValueError:return Plan(NavStatus.INVALID_DESTINATION)
        if region.contains(start):return Plan(NavStatus.SUCCESS,(tuple(start),),0.,perf_counter()-began)
        route=self.map.astar(start,region.anchor)
        if not route:return Plan(NavStatus.NO_ROUTE,planning_seconds=perf_counter()-began)
        route=tuple(self._trim_to_arrival(self._simplify(route),region))
        if not region.contains(route[-1]) or not all(self.map.segment_clear(a,b) for a,b in zip(route,route[1:])):
            return Plan(NavStatus.NO_ROUTE,planning_seconds=perf_counter()-began)
        return Plan(NavStatus.MOVING,route,route_length(route),perf_counter()-began)


@dataclass(frozen=True)
class Goal:
    key: str
    position: tuple[float,float]
    destination: str | None = None


class NavigationMemory:
    def __init__(self,geometry,seed):
        self.geometry=geometry;self.rng=Random(seed);self.goal=None;self.original=None
        self.detour_until=0.;self.last_change=-math.inf;self.cursor=0;self.commitment=None
        # Every reachable grid node is eligible, including edges/corners/corridors.
        self.nodes=np.asarray(geometry.nodes,dtype=np.float64)
        shapely.prepare(geometry.ray_barriers)
        # Coverage is an explicit .75-map-unit spatial abstraction. Every original
        # grid node remains selectable; this avoids raycasting 67k goals per actor.
        _,first,self.node_cells=np.unique(np.floor(self.nodes/.75).astype(int),axis=0,return_index=True,return_inverse=True)
        self.coverage_nodes=self.nodes[first]
        self.seen_at=np.full(len(first),-1.,dtype=np.float64)
        self.order=list(range(len(self.nodes)));self.rng.shuffle(self.order)

    def update(self,obs):
        a=obs.actor
        if not a.own.active or a.context.phase is not Phase.ROAMING:return 0
        position=(a.own.position.x,a.own.position.y);sight=a.settings.sight_range
        close=np.flatnonzero(np.sum((self.coverage_nodes-position)**2,axis=1)<=sight*sight)
        if not len(close):return 0
        rays=shapely.linestrings(np.stack((np.broadcast_to(position,(len(close),2)),self.coverage_nodes[close]),axis=1))
        seen=close[~shapely.intersects(self.geometry.ray_barriers,rays)]
        fresh=int(np.sum(self.seen_at[seen]<0));self.seen_at[seen]=a.time_seconds
        return fresh

    def candidates(self,position,count=32):
        # Cyclic coverage is independent of roles. Old areas remain revisitable.
        selected=[]
        for _ in range(min(count,len(self.order))):
            i=self.order[self.cursor%len(self.order)];self.cursor+=1;selected.append(i)
        # Bounded randomized nearby alternatives suitable for temporary detours.
        close=np.flatnonzero(np.sum((self.nodes-position)**2,axis=1)<=36.)
        if len(close):selected+=self.rng.sample(close.tolist(),min(8,len(close)))
        return tuple((int(i),tuple(map(float,self.nodes[i])),float(self.seen_at[self.node_cells[i]])) for i in dict.fromkeys(selected))

    def command(self,mode,goal,now,obs):
        if mode==0:
            if obs.actor.interaction_task is not None:return Intent(IntentKind.WAIT)
            if self.commitment and self.commitment.task and self.commitment.incubating and obs.actor.navigation_status!='moving':return Intent(IntentKind.WAIT)
            if self.original is not None and (obs.actor.navigation_status!='moving' or now>=self.detour_until):
                self.goal=self.original;self.original=None
                return self._intent(self.goal)
            if self.goal is not None and obs.actor.navigation_status in ('cancelled','stuck','timeout','idle'):
                return self._intent(self.goal)
            return Intent(IntentKind.WAIT)
        if goal is None:raise ValueError('Navigation target required')
        if mode==1:
            if self.goal is None or self.original is not None:raise ValueError('Detour requires retained goal')
            self.original=self.goal;self.detour_until=now+8.
        elif mode==2:self.original=None
        else:raise ValueError('Unknown navigation mode')
        self.goal=goal;self.last_change=now
        return self._intent(goal)

    @staticmethod
    def _intent(goal):
        if goal.destination:return Intent(IntentKind.GO_TO_TASK,goal.destination)
        return Intent(IntentKind.GO_TO_LOCATION,position=Point(*goal.position))
