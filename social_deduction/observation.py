"""Trusted crew-only observation boundary. No dependency on navigation or ML."""

from dataclasses import replace
import math

from .actor import (
    Action, ActionKind, ActorObservation, BodySeen, Claim, Ejection, Evidence,
    MatchEnded, Meeting, Phase, PlayerSeen, Point, Provenance, Report,
    Role, SelfView, TaskView, Vote, Wall, validate_observation,
)
from .truth import WorldTruth


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _blocked(a: Point, b: Point, wall: Wall) -> bool:
    """Closed segment/AABB intersection, including touching a wall."""
    lo, hi = 0.0, 1.0
    for start, end, lower, upper in (
        (a.x, b.x, wall.left, wall.right),
        (a.y, b.y, wall.top, wall.bottom),
    ):
        delta = end - start
        if delta == 0:
            if not lower <= start <= upper:
                return False
        else:
            t0, t1 = sorted(((lower - start) / delta, (upper - start) / delta))
            lo, hi = max(lo, t0), min(hi, t1)
            if lo > hi:
                return False
    return True


def _visible(origin: Point, target: Point, public_map) -> bool:
    return (_distance(origin, target) <= public_map.sight_range and
            not any(_blocked(origin, target, wall) for wall in public_map.walls))


def legal_actions(observation: ActorObservation) -> tuple[Action, ...]:
    """All candidates depend solely on actor-safe fields; no truth queries."""
    validate_observation(observation)
    own, context, public_map = observation.own, observation.context, observation.map
    if not own.active or context.phase is Phase.FINISHED:
        return ()
    if context.phase is Phase.DISCUSSION:
        return ((Action(ActionKind.CLAIM),)
                if context.speaker_id == own.player_id and own.player_id in context.participants else ())
    if context.phase is Phase.VOTING:
        if own.player_id not in context.participants or own.player_id in context.voted_ids:
            return ()
        return tuple(Action(ActionKind.VOTE, p) for p in sorted(context.participants)) + (Action(ActionKind.SKIP),)

    actions = [Action(ActionKind.WAIT)]
    actions.extend(Action(ActionKind.NAVIGATE, "room:" + room) for room in sorted(public_map.rooms))
    actions.extend(Action(ActionKind.NAVIGATE, "console:" + c.console_id)
                   for c in sorted(public_map.consoles, key=lambda c: c.console_id))
    actions.extend(Action(ActionKind.FOLLOW, p.player_id) for p in observation.visible_players)
    nearby = {c.console_id for c in public_map.consoles
              if _distance(own.position, c.position) <= public_map.interaction_range
              and _visible(own.position, c.position, public_map)}
    actions.extend(Action(ActionKind.INTERACT, t.task_id) for t in own.tasks
                   if t.progress < 1.0 and t.console_id in nearby)
    actions.extend(Action(ActionKind.REPORT, b.victim_id) for b in observation.visible_bodies
                   if _distance(own.position, b.position) <= public_map.interaction_range)
    if own.meetings_remaining > 0:
        actions.extend(Action(ActionKind.CALL_MEETING, c.console_id)
                       for c in sorted(public_map.consoles, key=lambda c: c.console_id)
                       if c.meeting and c.console_id in nearby)
    return tuple(actions)


def observe(world: WorldTruth, player_id: str) -> ActorObservation:
    """Project a truth fixture into a value-only packet. Future runner calls this."""
    ids = [p.identity.player_id for p in world.players]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate public player IDs")
    own = next((p for p in world.players if p.identity.player_id == player_id), None)
    if own is None:
        raise ValueError("Unknown actor")
    if own.role is not Role.CREWMATE:
        raise ValueError("Phase 1 actor contract supports crewmates only")
    if world.tick < 0:
        raise ValueError("Negative simulation tick")

    # No global alive list; roster is the full match roster independent of deaths.
    roster = tuple(sorted((p.identity for p in world.players), key=lambda p: p.player_id))
    local_sight = own.active and world.context.phase is Phase.ROAMING
    players = tuple(PlayerSeen(p.identity.player_id, p.position, p.room, p.velocity, p.interacting)
                    for p in sorted(world.players, key=lambda p: p.identity.player_id)
                    if local_sight and p.active and p.identity.player_id != player_id
                    and _visible(own.position, p.position, world.map))
    bodies = tuple(BodySeen(b.victim_id, b.position, b.room)
                   for b in sorted(world.bodies, key=lambda b: b.victim_id)
                   if local_sight and _visible(own.position, b.position, world.map))
    events = []
    previous_tick = -1
    for publication in world.publications:
        # Scheduled future publications are not yet part of the public history.
        # Do not inspect their payload or let their ordering affect this packet.
        if publication.tick > world.tick:
            continue
        if publication.tick < previous_tick or publication.tick < 0:
            raise ValueError("Publications must be ordered by public delivery tick")
        previous_tick = publication.tick
        if type(publication.payload) not in (Report, Meeting, Claim, Vote, Ejection, MatchEnded):
            raise TypeError("Non-public payload in publication log")
        provenance = Provenance.CLAIM if type(publication.payload) is Claim else Provenance.PUBLIC
        events.append(Evidence(publication.tick, provenance, publication.payload))
    events.extend(Evidence(world.tick, Provenance.DIRECT, payload) for payload in (*players, *bodies))
    snapshot = ActorObservation(
        world.match_id, world.tick,
        SelfView(player_id, own.role, own.position, own.room, own.active,
                 tuple(TaskView(t.task_id, t.console_id, t.progress)
                       for t in sorted(own.tasks, key=lambda t: t.task_id)), own.meetings_remaining),
        roster, world.map, world.context, players, bodies, tuple(events),
    )
    validate_observation(snapshot)
    return replace(snapshot, actions=legal_actions(snapshot))
