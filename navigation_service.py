"""Known-map navigation; accepts public geometry and own-motion feedback only.

Coordinates and emitted displacements are native (x right, y up). This module
never receives a simulator/role state and never writes a player's position.
"""
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
import heapq
import math
from time import perf_counter

import numpy as np
import shapely
from shapely.geometry import LineString, Point

from among_us_map import DIRECTIONS, AmongUsMap, polygons


class NavStatus(str, Enum):
    IDLE = 'idle'
    MOVING = 'moving'
    SUCCESS = 'success'
    CANCELLED = 'cancelled'
    INVALID_START = 'invalid_start'
    INVALID_DESTINATION = 'invalid_destination'
    NO_ROUTE = 'no_route'
    STUCK = 'stuck'
    TIMEOUT = 'timeout'


@dataclass(frozen=True)
class NavTarget:
    kind: str
    key: str = ''
    position: tuple[float, float] | None = None
    radius: float = 0.1

    @classmethod
    def task(cls, destination_id):
        """Task or utility destination ID; does not imply task assignment."""
        return cls('destination', str(destination_id))

    @classmethod
    def room(cls, room_id):
        return cls('room', str(room_id))

    @classmethod
    def location(cls, position, radius=0.1):
        return cls('location', position=tuple(position), radius=radius)


@dataclass(frozen=True)
class ArrivalRegion:
    target: NavTarget
    shape: object
    anchor: tuple[float, float]

    def contains(self, position):
        return bool(self.shape.covers(Point(position)))

    def distance(self, position):
        return float(self.shape.distance(Point(position)))


@dataclass(frozen=True)
class Plan:
    status: NavStatus
    route: tuple[tuple[float, float], ...] = ()
    length: float = 0.0
    planning_seconds: float = 0.0
    cache_hit: bool = False


def route_length(route):
    return sum(math.dist(a, b) for a, b in zip(route, route[1:]))


def _finite_point(position):
    try:
        return len(position) == 2 and all(math.isfinite(float(v)) for v in position)
    except (TypeError, ValueError):
        return False


class NavigationPlanner:
    """Static validated grid plus bounded per-target reverse distance fields.

    Share one planner across agents. Replace it with a fresh planner when known
    geometry changes; do not mutate its map or cached arrays in place.
    """
    def __init__(self, public_map: AmongUsMap, cache_size=8):
        if cache_size < 1:
            raise ValueError('cache_size must be positive')
        self.map = public_map
        self.cache_size = cache_size
        self._regions = OrderedDict()
        self._fields = OrderedDict()
        self.field_builds = 0
        self.field_build_seconds = 0.0
        self.destinations = {d.id: d for d in public_map.destinations}
        # Preparation accelerates predicates without changing physical geometry.
        shapely.prepare(public_map.center_domain)
        grid = public_map.grid
        ids = np.full(grid.shape, -1, dtype=np.int32)
        ids[grid] = np.arange(len(public_map.nodes))
        self._ids = ids
        ys, xs = np.nonzero(grid)
        neighbors = [[] for _ in public_map.nodes]
        for k, (dx, dy) in enumerate(DIRECTIONS):
            valid = public_map.edges[ys, xs, k]
            src = ids[ys[valid], xs[valid]]
            dst = ids[ys[valid] + dy, xs[valid] + dx]
            cost = math.hypot(dx, dy) * public_map.cell_size
            for a, b in zip(src.tolist(), dst.tolist()):
                neighbors[a].append((b, cost))
        self._neighbors = neighbors
        self._points = shapely.points(public_map.nodes)

    def _remember(self, cache, key, value):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.cache_size:
            cache.popitem(last=False)

    def arrival_region(self, target: NavTarget) -> ArrivalRegion:
        if target in self._regions:
            self._regions.move_to_end(target)
            return self._regions[target]
        m = self.map
        if target.kind == 'destination':
            if target.key not in self.destinations:
                raise ValueError('Unknown destination ID')
            dest = self.destinations[target.key]
            anchor = dest.standing
            shape = Point(dest.position).buffer(dest.radius, quad_segs=64).intersection(m.center_domain)
            # Select the same approach side as the established legal standing point.
            parts = [p for p in polygons(shape) if p.covers(Point(anchor))]
            if not parts:
                raise ValueError('Destination has no valid interaction region')
            shape = parts[0]
        elif target.kind == 'room':
            regions = [shape for name, shape in m.regions if name == target.key]
            if not regions:
                raise ValueError('Unknown room/region ID')
            shape = shapely.union_all(regions).intersection(m.center_domain)
            if shape.is_empty or shape.area <= 0:
                raise ValueError('Room has no walkable arrival area')
            anchor = tuple(shape.representative_point().coords[0])
        elif target.kind == 'location':
            if (not _finite_point(target.position) or not math.isfinite(target.radius)
                    or target.radius <= 0 or not m.contains(target.position)):
                raise ValueError('Location needs a legal center and positive finite radius')
            anchor = target.position
            shape = Point(anchor).buffer(target.radius, quad_segs=64).intersection(m.center_domain)
            shape = next(p for p in polygons(shape) if p.covers(Point(anchor)))
        else:
            raise ValueError('Unknown target kind')
        shapely.prepare(shape)
        region = ArrivalRegion(target, shape, tuple(anchor))
        self._remember(self._regions, target, region)
        return region

    def _node_id(self, position):
        x, y = self.map.nearest_node(position)
        ix = round((x - self.map.xs[0]) / self.map.cell_size)
        iy = round((y - self.map.ys[0]) / self.map.cell_size)
        return int(self._ids[iy, ix])

    def _field(self, region):
        key = region.target
        if key in self._fields:
            self._fields.move_to_end(key)
            return self._fields[key], True
        started = perf_counter()
        n = len(self.map.nodes)
        distances = [math.inf] * n
        following = np.full(n, -1, dtype=np.int32)
        seeds = np.flatnonzero(shapely.covers(region.shape, self._points))
        endpoints = {}
        if len(seeds):
            queue = [(0.0, int(i)) for i in seeds]
            for i in seeds:
                distances[int(i)] = 0.0
        else:
            # Sub-grid location regions still have an exact, swept endpoint connector.
            i = self._node_id(region.anchor)
            distance = math.dist(self.map.nodes[i], region.anchor)
            queue = [(distance, i)]
            distances[i] = distance
            endpoints[i] = region.anchor
        heapq.heapify(queue)
        while queue:
            distance, current = heapq.heappop(queue)
            if distance > distances[current]:
                continue
            for neighbor, cost in self._neighbors[current]:
                candidate = distance + cost
                if candidate < distances[neighbor]:
                    distances[neighbor] = candidate
                    following[neighbor] = current
                    heapq.heappush(queue, (candidate, neighbor))
        field = (np.asarray(distances), following, endpoints)
        self._remember(self._fields, key, field)
        self.field_builds += 1
        self.field_build_seconds += perf_counter() - started
        return field, False

    def _simplify(self, route):
        # Greedy visible waypoint skipping; no shortcut can bypass swept clearance.
        simple = [route[0]]
        coordinates = np.asarray(route)
        i = 0
        while i < len(route) - 1:
            # Batch predicates avoid thousands of Python/Shapely object calls
            # while retaining the exact furthest-visible waypoint choice.
            ends = coordinates[i+1:]
            starts = np.broadcast_to(coordinates[i], ends.shape)
            segments = shapely.linestrings(np.stack([starts, ends], axis=1))
            visible = np.flatnonzero(shapely.covers(self.map.center_domain, segments))
            j = i + 1 + int(visible[-1])
            simple.append(route[j])
            i = j
        return simple

    def _trim_to_arrival(self, route, region):
        trimmed = [route[0]]
        for a, b in zip(route, route[1:]):
            line = LineString([a, b])
            hit = line.intersection(region.shape)
            if not hit.is_empty:
                # Earliest contact along this segment, plus a tiny interior margin
                # to avoid floating-point boundary disagreement during execution.
                points = shapely.get_coordinates(hit)
                distance = min(line.project(Point(p)) for p in points)
                distance = min(line.length, distance + 1e-7)
                end = tuple(line.interpolate(distance).coords[0])
                if region.contains(end):
                    trimmed.append(end)
                    return trimmed
            trimmed.append(b)
        return trimmed

    def plan(self, start, target: NavTarget) -> Plan:
        began = perf_counter()
        if not _finite_point(start) or not self.map.contains(start):
            return Plan(NavStatus.INVALID_START, planning_seconds=perf_counter() - began)
        try:
            region = self.arrival_region(target)
        except (ValueError, TypeError, StopIteration):
            return Plan(NavStatus.INVALID_DESTINATION, planning_seconds=perf_counter() - began)
        start = tuple(float(v) for v in start)
        if region.contains(start):
            return Plan(NavStatus.SUCCESS, (start,), planning_seconds=perf_counter() - began)
        try:
            (distance, following, endpoints), cached = self._field(region)
            current = self._node_id(start)
        except ValueError:
            return Plan(NavStatus.NO_ROUTE, planning_seconds=perf_counter() - began)
        if not math.isfinite(distance[current]):
            return Plan(NavStatus.NO_ROUTE, planning_seconds=perf_counter() - began, cache_hit=cached)
        route = [start, tuple(self.map.nodes[current])]
        while following[current] != -1:
            current = int(following[current])
            route.append(tuple(self.map.nodes[current]))
        if current in endpoints:
            route.append(endpoints[current])
        route = [p for i, p in enumerate(route) if i == 0 or math.dist(route[i - 1], p) > 1e-12]
        route = self._trim_to_arrival(self._simplify(route), region)
        if (not region.contains(route[-1]) or
                not all(self.map.segment_clear(a, b) for a, b in zip(route, route[1:]))):
            return Plan(NavStatus.NO_ROUTE, planning_seconds=perf_counter() - began, cache_hit=cached)
        return Plan(NavStatus.MOVING, tuple(route), route_length(route), perf_counter() - began, cached)


@dataclass(frozen=True)
class ControllerConfig:
    speed: float = 2.5
    timeout_seconds: float = 60.0
    stuck_seconds: float = 1.0
    max_replans: int = 2
    waypoint_tolerance: float = 1e-8

    def __post_init__(self):
        if any(not math.isfinite(x) or x <= 0 for x in
               (self.speed, self.timeout_seconds, self.stuck_seconds, self.waypoint_tolerance)):
            raise ValueError('Controller limits must be positive and finite')
        if self.max_replans < 0:
            raise ValueError('max_replans must be nonnegative')


class NavigationService:
    """One intention per agent, shared planner. command -> physics -> feedback.

    Terminal outcomes persist until navigate() or cancel(); no silent retries.
    Diagnostics are for evaluation, not automatic actor observation fields.
    """
    def __init__(self, planner: NavigationPlanner, config=ControllerConfig()):
        self.planner = planner
        self.config = config
        self.status = NavStatus.IDLE
        self.target = None
        self.route = ()
        self._pending = None

    def navigate(self, position, target):
        self.target = target
        self.elapsed = self.travelled = self.planning_seconds = 0.0
        self.collisions = self.blocked_steps = self.steps = self.replans = 0
        self.last_replan_reason = None
        self._stalled = 0.0
        self._pending = None
        self.initial_plan = self._plan(position)
        self.planned_length = self.initial_plan.length
        return self.status

    @property
    def awaiting_feedback(self):
        """True only when command() emitted motion that physics must consume."""
        return self._pending is not None

    def _plan(self, position):
        plan = self.planner.plan(position, self.target)
        self.planning_seconds += plan.planning_seconds
        self.route = plan.route
        self._waypoint = 1
        self.status = plan.status
        self._best_remaining = plan.length
        self._stalled = 0.0
        return plan

    def cancel(self):
        self.status = NavStatus.CANCELLED
        self.route = ()
        self._pending = None
        return self.status

    def replan(self, position, *, planner=None, reason='requested'):
        """Replace known geometry explicitly; never query hidden door state."""
        if self.status is not NavStatus.MOVING:
            return self.status
        self._pending = None
        if self.replans >= self.config.max_replans:
            self.status = NavStatus.STUCK
            return self.status
        if planner is not None:
            if not math.isclose(planner.map.radius, self.planner.map.radius):
                raise ValueError('Cannot change player radius during navigation')
            self.planner = planner
        self.replans += 1
        self.last_replan_reason = reason
        self._plan(position)
        return self.status

    def command(self, position, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive and finite')
        if self._pending is not None:
            raise RuntimeError('Supply motion feedback before requesting another command')
        if self.status is not NavStatus.MOVING:
            return (0.0, 0.0)
        if not _finite_point(position) or not self.planner.map.contains(position):
            self.status = NavStatus.INVALID_START
            return (0.0, 0.0)
        if self.planner.arrival_region(self.target).contains(position):
            self.status = NavStatus.SUCCESS
            return (0.0, 0.0)
        while (self._waypoint < len(self.route) and
               math.dist(position, self.route[self._waypoint]) <= self.config.waypoint_tolerance):
            self._waypoint += 1
        if self._waypoint >= len(self.route):
            self.replan(position, reason='arrival_missed')
            return (0.0, 0.0)
        waypoint = self.route[self._waypoint]
        if not self.planner.map.segment_clear(position, waypoint):
            self.replan(position, reason='route_invalidated')
            return (0.0, 0.0)
        diff = np.asarray(waypoint) - position
        length = float(np.linalg.norm(diff))
        displacement = tuple(diff * min(1.0, self.config.speed * dt / length))
        self._pending = (tuple(position), dt, math.dist((0, 0), displacement))
        return displacement

    def feedback(self, position, collided=False):
        if self._pending is None:
            raise RuntimeError('No outstanding motion command')
        previous, dt, intended_distance = self._pending
        self._pending = None
        if not _finite_point(position):
            raise ValueError('Feedback must be a finite 2D position')
        moved = math.dist(previous, position)
        if moved > self.config.speed * dt + 1e-7:
            raise ValueError('Feedback violates the finite speed-bounded motion contract')
        self.steps += 1
        self.elapsed += dt
        self.travelled += moved
        self.collisions += int(collided)
        self.blocked_steps += int(moved < intended_distance * 0.2)
        if not self.planner.map.contains(position):
            self.status = NavStatus.INVALID_START
        elif self.planner.arrival_region(self.target).contains(position):
            self.status = NavStatus.SUCCESS
        elif self.elapsed >= self.config.timeout_seconds - 1e-9:
            self.status = NavStatus.TIMEOUT
        else:
            remaining = math.dist(position, self.route[self._waypoint]) + route_length(self.route[self._waypoint:])
            # Progress toward the route, not displacement from a moving anchor:
            # oscillations cannot keep resetting this best-progress timer.
            if remaining < self._best_remaining - self.config.waypoint_tolerance:
                self._best_remaining = remaining
                self._stalled = 0.0
            else:
                self._stalled += dt
            if self._stalled >= self.config.stuck_seconds - 1e-9:
                self.replan(position, reason='no_progress')
        return self.status
