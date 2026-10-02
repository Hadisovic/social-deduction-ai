"""Data-backed Skeld geometry, physical queries and diagnostic A*.

Native game coordinates: x points right, y points up. Rendering is separate.
Source chains remain inspectable; vents and open doors are not solid objects.
"""
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import heapq
import json
import math
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points, unary_union

ASSET_DIR = Path(__file__).resolve().parent / 'assets' / 'skeld'
ROOMS = ('Cafeteria','Weapons','O2','Navigation','Shields','Communications','Storage',
         'Admin','Electrical','Lower Engine','Security','Reactor','Upper Engine','MedBay')
DIRECTIONS = ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1))
REGION_ALIASES = {'UpperEngine':'Upper Engine','LowerEngine':'Lower Engine','LifeSupp':'O2',
                  'Nav':'Navigation','Comms':'Communications'}


def outline(record):
    points = record['points']
    if record['closed']:
        return Polygon(points).boundary
    return LineString(points)


def polygons(geometry):
    if geometry.geom_type == 'Polygon':
        return [geometry]
    return [p for g in getattr(geometry, 'geoms', ()) for p in polygons(g)]


@dataclass(frozen=True)
class Destination:
    id: str
    name: str
    room: str
    position: tuple[float, float]
    standing: tuple[float, float]
    radius: float
    category: str
    provenance: str

    @property
    def world_pos(self):
        return self.standing


class AmongUsMap:
    """Immutable static map. Door changes create a fresh map, never hidden state."""
    def __init__(self, player_radius=None, closed_doors=()):
        self.data = json.loads((ASSET_DIR / 'among_us_map.json').read_text(encoding='utf-8'))
        defaults = self.data['simulation_defaults']
        self.radius = float(defaults['player_radius'] if player_radius is None else player_radius)
        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError('player_radius must be finite and positive')
        self.cell_size = defaults['grid_cell']
        self.closed_doors = frozenset(closed_doors)
        known = {d['id'] for d in self.data['doors']}
        if self.closed_doors - known:
            raise ValueError('Unknown door id')
        self.regions = [(r['room'], shapely.make_valid(Polygon(r['points']))) for r in self.data['regions']]
        envelope = unary_union([p for _, p in self.regions]).buffer(defaults['area_envelope_padding'])
        walls = [outline(r) for r in self.data['walls']]
        walls += [LineString(p) for p in self.data['seam_repairs']]
        self.wall_lines = unary_union(walls)
        objects = [shapely.make_valid(Polygon(r['points'])) if r['closed'] else outline(r)
                   for r in self.data['obstacles']]
        self.objects = unary_union(objects)
        self.barriers = unary_union([self.wall_lines, self.objects])
        # AreaCollider regions are classification envelopes, not authoritative floor.
        # Select the connected interior AFTER applying native collider clearance.
        candidate = envelope.difference(self.barriers.buffer(self.radius, quad_segs=12))
        self.center_components = sorted(polygons(candidate), key=lambda p: p.area, reverse=True)
        if not self.center_components:
            raise ValueError('No floor survives player clearance')
        self.center_domain = self.center_components[0]
        self.excluded_components = len(self.center_components) - 1
        closed = [Polygon(d['points']) for d in self.data['doors'] if d['id'] in self.closed_doors]
        if closed:
            doors = unary_union(closed)
            self.center_domain = self.center_domain.difference(doors.buffer(self.radius, quad_segs=12))
            self.objects = unary_union([self.objects, doors])
            self.barriers = unary_union([self.barriers, doors])
        # Physical floor used by raycasts/rendering; native walls still take precedence.
        self.floor = self.center_domain.buffer(self.radius, quad_segs=12).difference(self.objects)
        self.ray_barriers = unary_union([self.barriers, self.floor.boundary])
        self.visibility_lines = unary_union([outline(r) for r in self.data['visibility']])
        self._calibrate_artwork()
        self._build_grid()
        self.destinations = self._destinations()

    def _calibrate_artwork(self):
        art = self.data['artwork']
        anchors = [(self.data['vents'][k]['position'], p) for k,p in art['vent_landmarks'].items()]
        native = np.array([[p['x'],p['y']] for p,_ in anchors])
        pixels = np.array([p for _,p in anchors], dtype=float)
        self.art_x = np.linalg.lstsq(np.column_stack([native[:,0],np.ones(len(native))]),pixels[:,0],rcond=None)[0]
        self.art_y = np.linalg.lstsq(np.column_stack([native[:,1],np.ones(len(native))]),pixels[:,1],rcond=None)[0]
        predicted = np.column_stack([native[:,0]*self.art_x[0]+self.art_x[1],native[:,1]*self.art_y[0]+self.art_y[1]])
        self.art_residuals = np.linalg.norm(predicted-pixels,axis=1)
        w,h = art['landmarks_preview_size']
        left,top = self.art_to_world((0,0))
        right,bottom = self.art_to_world((w,h))
        self.bounds = (left,bottom,right,top)
        self.width,self.height = right-left,top-bottom

    def art_to_world(self, pixel):
        return ((pixel[0]-self.art_x[1])/self.art_x[0],(pixel[1]-self.art_y[1])/self.art_y[0])

    def contains(self, position):
        return bool(self.center_domain.covers(Point(position)))

    def segment_clear(self, a, b):
        return bool(self.center_domain.covers(LineString([a,b])))

    def move(self, position, displacement):
        """Swept circle via radius-inflated domain; no tunneling, axis sliding."""
        p = np.asarray(position,dtype=float)
        delta = np.asarray(displacement,dtype=float)
        if not np.all(np.isfinite(delta)) or not self.contains(p):
            raise ValueError('Movement requires finite displacement and a valid start')
        target = p + delta
        if self.segment_clear(p,target):
            return tuple(target), False
        # Small substeps preserve sliding around corners; each is swept, not sampled.
        steps = max(1,math.ceil(float(np.linalg.norm(delta))/(self.radius*.45)))
        for _ in range(steps):
            for axis in (0,1):
                d = np.zeros(2); d[axis]=delta[axis]/steps
                if self.segment_clear(p,p+d):
                    p += d
                else:
                    lo,hi=0.,1.
                    for _ in range(10):
                        mid=(lo+hi)/2
                        if self.segment_clear(p,p+d*mid): lo=mid
                        else: hi=mid
                    p += d*lo
        return tuple(p), True

    def region(self, position):
        p=Point(position)
        for name,shape in self.regions:
            if name in ROOMS and shape.covers(p): return name
        for name,shape in self.regions:
            if shape.covers(p): return name
        return 'Hallway' if self.contains(position) else 'Exterior'

    def raycast(self, position, direction, distance=5.5):
        end=np.asarray(position)+np.asarray(direction)*distance
        hit=LineString([position,end]).intersection(self.ray_barriers)
        return distance if hit.is_empty else min(distance,Point(position).distance(hit))

    def _build_grid(self):
        x0,y0,x1,y1=self.center_domain.bounds
        cs=self.cell_size
        self.xs=np.arange(math.floor(x0/cs)*cs,x1,cs)
        self.ys=np.arange(math.floor(y0/cs)*cs,y1,cs)
        xx,yy=np.meshgrid(self.xs,self.ys)
        self.grid=shapely.contains_xy(self.center_domain,xx,yy)
        self.nodes=np.column_stack([xx[self.grid],yy[self.grid]])
        self.edges=np.zeros((*self.grid.shape,8),dtype=bool)
        # Check actual swept geometry for every edge, including diagonals.
        gy,gx=np.nonzero(self.grid)
        for k,(dx,dy) in enumerate(DIRECTIONS):
            nx,ny=gx+dx,gy+dy
            mask=(nx>=0)&(ny>=0)&(nx<len(self.xs))&(ny<len(self.ys))
            ii=np.flatnonzero(mask)
            ii=ii[self.grid[ny[ii],nx[ii]]]
            start=np.column_stack([self.xs[gx[ii]],self.ys[gy[ii]]])
            end=np.column_stack([self.xs[nx[ii]],self.ys[ny[ii]]])
            valid=shapely.covers(self.center_domain,shapely.linestrings(np.stack([start,end],axis=1)))
            self.edges[gy[ii],gx[ii],k]=valid

    def nearest_node(self, point, max_distance=None):
        point=np.asarray(point)
        dist=np.sum((self.nodes-point)**2,axis=1)
        for idx in np.argsort(dist):
            node=self.nodes[idx]
            if max_distance is not None and dist[idx]>max_distance**2: break
            if not self.contains(point) or self.segment_clear(point,node):
                return tuple(node)
        raise ValueError(f'No reachable standing point near {tuple(point)}')

    def _destinations(self):
        records=[]; seen=set()
        for task_id, task in self.data['tasks'].items():
            for console in task.get('consoles',[]):
                p=console['position']; pos=(p['x'],p['y'])
                key=(task['taskType'],pos)
                if key in seen: continue
                seen.add(key)
                radius=console['usableDistance']
                stand=self.nearest_node(pos,radius)
                records.append(Destination(f'task:{task_id}:{console["id"]}',task['taskType'],
                    REGION_ALIASES.get(console['room'], console['room']),pos,stand,radius,'task','Impostor 2026.8.18'))
        for stage in self.data['supplemental_task_stages']:
            pos = tuple(stage['position']) if 'position' in stage else self.art_to_world(stage['preview_position'])
            approach = self.art_to_world(stage['standing_preview']) if 'standing_preview' in stage else pos
            stand = self.nearest_node(approach, stage['radius'])
            if math.dist(pos, stand) > stage['radius']:
                raise ValueError(f'Standing point outside interaction radius: {stage["id"]}')
            records.append(Destination('stage:'+stage['id'], stage['name'], stage['room'],
                pos, stand, stage['radius'], 'task', stage['provenance']))
        # Utility markers absent from normal-task metadata, including both reactor hands.
        utilities=('EmergencyConsole','MapRoomConsole','SurvConsole','FixCommsConsole',
                   'NoOxyConsole','SwitchConsole','UpperHandConsole','LowerHandConsole')
        for c in self.data['consoles']:
            if not any(c['name'].endswith(n) for n in utilities): continue
            if c['position'] is None:
                pos=self.art_to_world(self.data['utility_artwork_positions'][c['name'].split('/')[-1]])
                provenance='image-estimated utility; source SVG circle lacks a world transform'
            else:
                pos=tuple(c['position'])
                provenance='2021 trigger bounds; utility use radius estimated'
            stand=self.nearest_node(pos,1.5)
            records.append(Destination('utility:'+c['id'],c['name'].split('/')[-1],c['room'],pos,
                stand,1.5,'utility',provenance))
        return tuple(records)

    def astar(self, start, goal):
        """Validated edges with exact endpoint connectors; returns native coordinates."""
        if not self.contains(start) or not self.contains(goal): return []
        if self.segment_clear(start,goal): return [tuple(start),tuple(goal)]
        a=self.nearest_node(start); b=self.nearest_node(goal)
        def cell(p): return (round((p[0]-self.xs[0])/self.cell_size),round((p[1]-self.ys[0])/self.cell_size))
        src,dst=cell(a),cell(b)
        queue=[(0.,0.,src)]; costs={src:0.}; parents={}
        while queue:
            _,g,current=heapq.heappop(queue)
            if g>costs[current]: continue
            if current==dst:
                cells=[dst]
                while cells[-1]!=src: cells.append(parents[cells[-1]])
                route=[tuple(start)]+[(self.xs[x],self.ys[y]) for x,y in reversed(cells)]+[tuple(goal)]
                return route
            x,y=current
            for k,(dx,dy) in enumerate(DIRECTIONS):
                if not self.edges[y,x,k]: continue
                nxt=(x+dx,y+dy); ng=g+math.hypot(dx,dy)
                if ng<costs.get(nxt,float('inf')):
                    costs[nxt]=ng; parents[nxt]=current
                    heapq.heappush(queue,(ng+math.dist(nxt,dst),ng,nxt))
        return []

    def connected_grid_count(self):
        y,x=np.argwhere(self.grid)[0]
        queue=deque([(x,y)]); seen={(x,y)}
        while queue:
            x,y=queue.popleft()
            for k,(dx,dy) in enumerate(DIRECTIONS):
                nxt=(x+dx,y+dy)
                if self.edges[y,x,k] and nxt not in seen:
                    seen.add(nxt); queue.append(nxt)
        return len(seen)


@lru_cache(maxsize=4)
def get_map(player_radius=None):
    return AmongUsMap(player_radius)
