"""Source-backed doors/vents; immutable revision maps, actor-local route knowledge."""
from copy import copy
from functools import lru_cache
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union
from among_us_map import get_map,DIRECTIONS
from .navigation import PatrickPlanner


@lru_cache(maxsize=16)
def geometry(closed=()):
    base=get_map();closed=frozenset(closed)
    if not closed:return base
    m=copy(base);m.closed_doors=closed
    polygons=[Polygon(d['points']) for d in base.data['doors'] if d['id'] in closed]
    if len(polygons)!=len(closed):raise ValueError('Unknown Skeld door')
    solid=unary_union(polygons);inflated=solid.buffer(base.buffer_radius,quad_segs=12)
    m.center_domain=base.center_domain.difference(inflated)
    m.objects=unary_union([base.objects,solid]);m.barriers=unary_union([base.barriers,solid])
    m.floor=m.center_domain.buffer(base.radius,quad_segs=12).difference(m.objects)
    m.ray_barriers=unary_union([m.barriers,m.floor.boundary])
    m.grid=base.grid.copy();ys,xs=np.nonzero(base.grid)
    good=~shapely.intersects_xy(inflated,base.xs[xs],base.ys[ys]);m.grid[ys,xs]=good
    gy,gx=np.nonzero(m.grid);m.nodes=np.column_stack([m.xs[gx],m.ys[gy]])
    m.edges=base.edges.copy();m.edges[~m.grid]=False
    for k,(dx,dy) in enumerate(DIRECTIONS):
        yy,xx=np.nonzero(m.edges[:,:,k]);endx=xx+dx;endy=yy+dy
        lines=shapely.linestrings(np.stack([np.column_stack([m.xs[xx],m.ys[yy]]),np.column_stack([m.xs[endx],m.ys[endy]])],axis=1))
        m.edges[yy,xx,k]&=m.grid[endy,endx]&~shapely.intersects(inflated,lines)
    return m


@lru_cache(maxsize=16)
def planner(closed=()):return PatrickPlanner(geometry(tuple(sorted(closed))),cache_size=48)


def door_groups():
    groups={}
    for d in get_map().data['doors']:groups.setdefault(d['room'],[]).append(d['id'])
    return {k:tuple(v) for k,v in sorted(groups.items())}


def vents():
    data=get_map().data['vents'];result={}
    for key,v in data.items():
        p=v['position'];point=(p['x'],p['y']) if isinstance(p,dict) else tuple(p)
        result[str(key)]=dict(position=tuple(map(float,get_map().nearest_node(point))),source_position=point,
            links=tuple(str(v[x]) for x in ('left','center','right') if v.get(x) is not None))
    for key,v in result.items():
        if any(key not in result[other]['links'] for other in v['links']):raise ValueError('Asymmetric vent graph')
    return result
