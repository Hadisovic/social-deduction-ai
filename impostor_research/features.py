"""Impostor-only candidate representation from validated actor-local values."""
from dataclasses import dataclass
import math
import numpy as np
from crewmate_v2.api import validate
from phase5_features import REGIONS
from social_deduction.actor import Phase, Point, Role
from social_deduction.phase3_api import ClaimDraft, ClaimKind, Intent, IntentKind, intent_allowed

MAX_ACTIONS = 256
KINDS = ('continue',) + tuple(k.value for k in IntentKind if k is not IntentKind.CLAIM) + tuple('claim_' + k.value for k in ClaimKind)
GLOBAL_COUNT = 11 + len(Phase) + len(REGIONS)
PLAYER_COUNT = 12 + 9
ACTION_COUNT = len(KINDS) + 6 + len(REGIONS)


@dataclass(frozen=True)
class Candidate:
    kind: str
    intent: Intent | None = None
    target: str | None = None
    position: Point | None = None
    region: str = 'Unknown'


def candidates(envelope, memory):
    obs = validate(envelope).actor
    if obs.own.role is not Role.IMPOSTOR:
        raise ValueError('Impostor interface requires own impostor role')
    result = [Candidate('continue')]
    if not obs.own.active or obs.context.phase is Phase.FINISHED:
        return tuple(result)
    destinations = {d.id: d for d in obs.destinations}
    players = {p.player_id: p for p in obs.visible_players}
    bodies = {b.victim_id: b for b in obs.visible_bodies}
    def add(action, target=None, point=None, region='Unknown'):
        if intent_allowed(obs, action):
            kind = 'claim_' + action.claim.kind.value if action.claim else action.kind.value
            result.append(Candidate(kind, action, target, point, region))
    for a in obs.actions:
        if a.kind in (IntentKind.WAIT, IntentKind.GO_TO_LOCATION):
            continue
        if a.kind is IntentKind.CLAIM:
            # Public vocabulary allows lying; no hidden truth is consulted.
            for room in sorted(obs.map.rooms):
                add(Intent(IntentKind.CLAIM, claim=ClaimDraft(ClaimKind.WAS_IN_REGION, obs.own.player_id, room, obs.tick)), region=room)
            for pid in memory.candidates:
                for kind in (ClaimKind.SUSPECT_PLAYER, ClaimKind.DEFEND_PLAYER, ClaimKind.SAW_ELIMINATION):
                    add(Intent(IntentKind.CLAIM, claim=ClaimDraft(kind, pid, obs.own.room, obs.tick)), pid, region=obs.own.room)
                last = memory.last_seen.get(pid)
                if last:
                    add(Intent(IntentKind.CLAIM, claim=ClaimDraft(ClaimKind.SAW_PLAYER, pid, last.payload.room, last.tick)), pid, region=last.payload.room)
        elif a.kind in (IntentKind.GO_TO_TASK, IntentKind.FAKE_TASK, IntentKind.CALL_MEETING):
            d = destinations[a.target]; add(a, point=d.standing, region=d.room)
        elif a.kind is IntentKind.GO_TO_ROOM:
            points = [d.standing for d in obs.destinations if d.room == a.target]
            point = Point(sum(p.x for p in points)/len(points), sum(p.y for p in points)/len(points)) if points else None
            add(a, point=point, region=a.target)
        else:
            seen = players.get(a.target) or bodies.get(a.target)
            add(a, a.target if a.target in memory.candidates else None, seen.position if seen else None)
    if obs.context.phase is Phase.ROAMING:
        for p in sorted(obs.visible_players, key=lambda p: p.player_id):
            add(Intent(IntentKind.GO_TO_LOCATION, position=p.position), p.player_id, p.position, p.room)
    if len(result) > MAX_ACTIONS:
        raise ValueError('Impostor action capacity exceeded; truncation prohibited')
    return tuple(result)


def encode(envelope, memory, public_memory, options):
    obs = validate(envelope).actor
    own = obs.own; ids = memory.candidates
    x, y = own.position.x, own.position.y
    bar = envelope.task_bar
    global_values = [x/30, y/30, float(obs.navigation_status == 'moving'), float(obs.interaction_task is not None),
        min((obs.kill_cooldown or 0)/45, 1), min(obs.time_seconds/900, 1), bar.completed/bar.total, float(own.meetings_remaining),
        *memory.self_social()]
    global_values += [float(obs.context.phase is p) for p in Phase]
    global_values += [float(own.room == r) for r in REGIONS]
    # Generic legitimate evidence summaries, NOT a crew suspicion predictor.
    base = memory.player_features()
    visible = {p.player_id: p for p in obs.visible_players}
    public = public_memory.player_features(ids)
    players = []
    for i, pid in enumerate(ids):
        p = visible.get(pid); last = memory.last_seen.get(pid)
        players.append([*base[i], float(p is not None), (p.position.x-x)/60 if p else 0,
            (p.position.y-y)/60 if p else 0, (last.payload.position.x-x)/60 if last else 0,
            (last.payload.position.y-y)/60 if last else 0,
            min((obs.tick-last.tick)*obs.settings.dt/900, 1) if last else 1, *public[i]])
    actions = np.zeros((MAX_ACTIONS, ACTION_COUNT), np.float32)
    mask = np.zeros(MAX_ACTIONS, np.int8); targets = np.full(MAX_ACTIONS, -1, np.int64)
    for i, c in enumerate(options):
        mask[i] = 1; actions[i, KINDS.index(c.kind)] = 1
        p = c.position
        actions[i, len(KINDS):len(KINDS)+6] = [(p.x-x)/60 if p else 0, (p.y-y)/60 if p else 0,
            min(math.hypot(p.x-x, p.y-y)/60, 1) if p else 0, float(p is not None),
            float(c.intent is not None and c.intent.target == own.player_id),
            min((obs.tick-c.intent.claim.asserted_tick)*obs.settings.dt/900, 1) if c.intent and c.intent.claim else 0]
        actions[i, len(KINDS)+6:] = [float(c.region == r) for r in REGIONS]
        if c.target in ids: targets[i] = ids.index(c.target)
    return dict(global_=np.asarray(global_values, np.float32), players=np.asarray(players, np.float32),
                actions=actions, targets=targets, mask=mask)
