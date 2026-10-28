"""Trusted projection onto the additive Phase 3 actor API and native Skeld sight."""

from dataclasses import replace
from functools import lru_cache
import math

from shapely.geometry import LineString

from among_us_map import AmongUsMap
from navigation_service import NavTarget, NavigationPlanner
from .actor import BodySeen, Phase, PlayerSeen, Point, Provenance, Role, SelfView, TaskView
from .truth import WorldTruth
from .phase3_api import (
    DestinationView, EventView, ObservationSettings, Phase3Observation,
    StructuredClaim, legal_actions, validate_event, validate_observation,
)


def _xy(point):
    return (point.x, point.y) if type(point) is Point else tuple(point)


def _point(point):
    """Native map/grid coordinates can be numpy scalars; packets use builtins."""
    x, y = _xy(point)
    return Point(float(x), float(y))


def visible_between(geometry: AmongUsMap, origin, target, sight_range: float) -> bool:
    """Point-to-point 360-degree sight; wall/prop/floor-boundary contact blocks.

    Uses the same native barriers as physics, without inflating sight by the
    player's collision radius. Interactions separately use arrival regions.
    """
    a, b = _xy(origin), _xy(target)
    if math.dist(a, b) > sight_range:
        return False
    return not geometry.ray_barriers.intersects(LineString((a, b)))


@lru_cache(maxsize=4)
def destination_views(geometry: AmongUsMap) -> tuple[DestinationView, ...]:
    return tuple(DestinationView(d.id, d.name, d.room, _point(d.position),
                                 _point(d.standing), float(d.radius), d.category)
                 for d in sorted(geometry.destinations, key=lambda d: d.id))


def project_game(world: WorldTruth, player_id: str, geometry: AmongUsMap,
                 planner: NavigationPlanner, settings: ObservationSettings, *,
                 navigation_status: str = 'idle', navigation_target: str | None = None,
                 interaction_task: str | None = None,
                 publications: tuple[EventView, ...] = (),
                 direct_events: tuple[EventView, ...] = ()) -> Phase3Observation:
    """Build a detached value packet from explicit allowlisted fields.

    Public envelopes must have authenticated rule triggers. Direct historical
    events must already be filtered for this actor at event time by the engine.
    No hidden events, counts, sequence identifiers or policy helpers are copied.
    """
    ids = [player.identity.player_id for player in world.players]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate public player IDs')
    own = next((player for player in world.players if player.identity.player_id == player_id), None)
    if own is None:
        raise ValueError('Unknown actor')
    if world.tick < 0:
        raise ValueError('Negative simulation tick')
    if planner.map is not geometry:
        raise ValueError('Visibility and interaction geometry must share the same public map')
    roster = tuple(sorted((player.identity for player in world.players), key=lambda p: p.player_id))
    local_sight = own.active and world.context.phase is Phase.ROAMING
    players = tuple(PlayerSeen(player.identity.player_id, _point(player.position), player.room,
                               _point(player.velocity), player.interacting)
                    for player in sorted(world.players, key=lambda p: p.identity.player_id)
                    if local_sight and player.active and player.identity.player_id != player_id
                    and visible_between(geometry, own.position, player.position, settings.sight_range))
    bodies = tuple(BodySeen(body.victim_id, _point(body.position), body.room)
                   for body in sorted(world.bodies, key=lambda b: b.victim_id)
                   if local_sight and visible_between(geometry, own.position, body.position,
                                                     settings.sight_range))
    events = []
    for records, public in ((publications, True), (direct_events, False)):
        previous_tick = -1
        for record in records:
            # Future payload/type/order cannot become a present-time side channel.
            if record.tick > world.tick:
                continue
            validate_event(record, world.tick)
            if record.tick < previous_tick:
                raise ValueError('Delivered events must be ordered by delivery tick')
            previous_tick = record.tick
            if public == (record.provenance is Provenance.DIRECT):
                raise ValueError('Public and private-direct event streams must remain separate')
            events.append(record)
    events.extend(EventView(world.tick, Provenance.DIRECT, payload) for payload in (*players, *bodies))
    destinations = destination_views(geometry)
    position = _xy(own.position)
    interactable_destinations = tuple(
        destination.id for destination in destinations
        if local_sight and math.dist(position, _xy(destination.position)) <= destination.radius
        and planner.arrival_region(NavTarget.task(destination.id)).contains(position))
    nearby = set(interactable_destinations)
    tasks = tuple(TaskView(task.task_id, task.console_id, task.progress)
                  for task in sorted(own.tasks, key=lambda t: t.task_id))
    own_view = SelfView(player_id, own.role, _point(own.position), own.room, own.active,
                        tasks, own.meetings_remaining)
    packet = Phase3Observation(
        world.match_id, world.tick, world.tick * settings.dt, own_view, roster,
        replace(world.map, sight_range=settings.sight_range), world.context,
        players, bodies, tuple(events), destinations, settings,
        tuple(task.task_id for task in tasks if task.progress < 1 and task.console_id in nearby),
        interactable_destinations,
        max(0., own.private_cooldown) if own.role is Role.IMPOSTOR else None,
        navigation_status, navigation_target, interaction_task,
    )
    packet = replace(packet, actions=legal_actions(packet))
    validate_observation(packet)
    return packet
