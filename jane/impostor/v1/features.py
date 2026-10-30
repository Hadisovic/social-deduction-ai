"""Actor-local impostor observations and legal candidates; no strategic overrides."""
from dataclasses import dataclass
import math
import numpy as np

SELF_WIDTH, PLAYER_WIDTH, EVENT_WIDTH, ACTION_WIDTH = 192, 24, 16, 48
KINDS = ('wait', 'cancel', 'destination', 'room', 'visible_player', 'remembered_player',
         'body_travel', 'fake', 'kill', 'report', 'emergency', 'claim', 'vote', 'skip',
         'lights', 'repair', 'doors', 'vent_enter', 'vent_move', 'vent_exit')
FAMILIES = ('continue', 'cancel', 'travel', 'fake', 'kill', 'report', 'emergency', 'claim',
            'vote', 'skip', 'lights', 'repair', 'doors', 'vent_enter', 'vent_move', 'vent_exit')
KIND_FAMILY = {kind: FAMILIES.index('travel' if kind in
    ('destination', 'room', 'visible_player', 'remembered_player', 'body_travel') else
    'continue' if kind == 'wait' else kind) for kind in KINDS}


@dataclass(frozen=True)
class OwnFeedback:
    owner: str
    match_id: str
    tick: int
    cooldown: float
    fake_remaining: float
    fake_console: str | None


@dataclass(frozen=True)
class Candidate:
    kind: str
    key: str = ''
    position: tuple | None = None
    payload: object = None


@dataclass
class Packet:
    self_features: np.ndarray
    players: np.ndarray
    player_mask: np.ndarray
    events: np.ndarray
    event_mask: np.ndarray
    actions: np.ndarray
    families: np.ndarray
    candidates: tuple
    owner: str
    tick: int
    time: float
    phase: str

    def tensors(self):
        return {name: getattr(self, name) for name in
                ('self_features', 'players', 'player_mask', 'events', 'event_mask', 'actions', 'families')}


def encode(obs, feedback, memory, planner, settings, last_goal=None, last_kind='wait'):
    from patrick_jane.api import Observation, Action, Event
    from social_deduction.phase3_api import Intent, IntentKind, ClaimKind, ClaimDraft
    from navigation_service import NavTarget
    from patrick_jane.features import event_code
    if type(obs) is not Observation or type(feedback) is not OwnFeedback:
        raise ValueError('Exact detached observation and own-feedback types required')
    a = obs.actor; own = a.own; owner = own.player_id
    if (feedback.owner, feedback.match_id, feedback.tick) != (owner, a.match_id, a.tick) or memory.owner != owner:
        raise ValueError('Cross-owner or stale feedback')
    if own.role.value != 'impostor' or not own.active:
        raise ValueError('Only living impostor observations are eligible')
    if not all(math.isfinite(v) and v >= 0 for v in (feedback.cooldown, feedback.fake_remaining)):
        raise ValueError('Invalid private own timers')
    if feedback.cooldown != float(a.kill_cooldown):
        raise ValueError('Cooldown must match the actor-local native observation')
    memory.update(obs)
    now = a.time_seconds; phase = a.context.phase.value
    if phase=='finished':raise ValueError('Terminal actors have no policy decisions')
    xy = (float(own.position.x), float(own.position.y))
    ids = tuple(p.player_id for p in a.roster if p.player_id != owner)
    if len(ids) > 14: raise ValueError('Fifteen-player capacity exceeded')
    allies = set(obs.teammates)
    visible = {p.player_id: p for p in a.visible_players}
    rooms = tuple(a.map.rooms)
    room_code = lambda r: (rooms.index(r) + 1) / (len(rooms) + 1) if r in rooms else 0.
    identity = lambda pid: (tuple(p.player_id for p in a.roster).index(pid) + 1) / 15 if pid in {p.player_id for p in a.roster} else 0.
    v = [xy[0]/30, xy[1]/30, room_code(own.room), now/900, float(a.navigation_status=='moving'),
         float(own.active), len(ids)/14, identity(owner)]
    v += [float(phase == p) for p in ('roaming', 'discussion', 'voting', 'finished')]
    v += [feedback.cooldown/60, feedback.fake_remaining/4, float(obs.vent_state is not None),
          float(obs.vent_transition_ready), float(obs.can_sabotage), float(obs.door_blocked),
          float(any(x.kind is IntentKind.KILL for x in a.actions)), float(any(x.kind is IntentKind.REPORT for x in a.actions))]
    l, m, t = settings.lobby, settings.meetings, settings.tasks
    v += [l.players/15, l.impostors/3, l.player_speed/3, l.crewmate_vision/5, l.impostor_vision/5,
          l.kill_cooldown_seconds/60, own.meetings_remaining/9, m.emergency_cooldown_seconds/60,
          m.discussion_seconds/120, m.voting_seconds/300, t.common/2, t.long/3, t.short/5,
          float(l.confirm_ejects), float(t.visual_tasks), obs.task_bar[0]/max(1, obs.task_bar[1]),
          float(obs.sabotage=='lights'), obs.emergency_ready_in/60,
          len(memory.known_dead)/15, len(memory.events)/64, len(visible)/14,
          sum(p in allies for p in visible)/3]
    v += [float(last_kind == kind) for kind in KINDS]
    v += [float(own.room == r) for r in rooms[:32]]
    v += [c/30 for point in obs.own_route[:4] for c in point]
    if len(v) > SELF_WIDTH: raise ValueError('Self feature budget exceeded')
    sf = np.zeros(SELF_WIDTH, np.float32); sf[:len(v)] = v
    players = np.zeros((14, PLAYER_WIDTH), np.float32); pm = np.zeros(14, bool)
    for i, pid in enumerate(ids):
        seen = memory.last_seen.get(pid); current = visible.get(pid)
        position = (current.position.x, current.position.y) if current else seen.position if seen else None
        records = tuple(memory.events)
        row = [identity(pid), float(current is not None), float(pid not in memory.known_dead),
               float(seen is not None), (position[0]-xy[0])/60 if position else 0,
               (position[1]-xy[1])/60 if position else 0, min(1., (now-seen.time)/120) if seen else 1.,
               seen.velocity[0]/3 if seen else 0, seen.velocity[1]/3 if seen else 0,
               room_code(seen.room) if seen else 0, float(pid in allies),
               float(memory.confirmed.get(pid)=='crewmate'), float(memory.confirmed.get(pid)=='impostor'),
               float(pid in a.context.participants), float(pid in a.context.voted_ids),
               sum(e.subject == pid for e in records)/64,
               sum(e.subject == pid and e.source == owner for e in records)/64,
               float(seen.interacting) if seen else 0,
               float(any(e.kind=='vent_enter' and e.subject==pid and e.provenance=='direct' for e in records)),
               float(pid in memory.known_dead), float(position is not None), 0., 0., 0.]
        players[i] = row; pm[i] = True
    ev = np.zeros((64, 16), np.float32); em = np.zeros(64, bool)
    for i, e in enumerate(tuple(memory.events)[-64:]):
        ev[i] = [event_code(e.kind, True), identity(e.subject), identity(e.other), room_code(e.room),
                 e.position[0]/30 if e.position else 0, e.position[1]/30 if e.position else 0,
                 e.observed/900, e.delivered/900, min(1., (now-e.observed)/900), min(1., (now-e.delivered)/900),
                 identity(e.source), float(e.provenance=='direct'), float(e.provenance=='public'),
                 float(e.provenance=='claim'), identity(e.origin_source or e.source), event_code(e.detail, True)]
        em[i] = True
    candidates = [Candidate('wait', payload=Intent(IntentKind.WAIT))]
    def add(kind, key='', position=None, payload=None):
        candidates.append(Candidate(kind, str(key), position, payload))
    if phase == 'roaming' and obs.vent_state is None and obs.vent_transition_ready:
        if a.navigation_status == 'moving' or feedback.fake_remaining > 0:
            add('cancel', payload=Intent(IntentKind.CANCEL_NAVIGATION))
        for action in a.actions:
            kind = action.kind
            if kind in (IntentKind.GO_TO_TASK, IntentKind.GO_TO_ROOM):
                try:
                    target = NavTarget.task(action.target) if kind is IntentKind.GO_TO_TASK else NavTarget.room(action.target)
                    arrival = planner.arrival_region(target)
                    key = ('destination' if kind is IntentKind.GO_TO_TASK else 'room', str(action.target))
                    if arrival.contains(xy) or key == last_goal and a.navigation_status == 'moving': continue
                    add(key[0], action.target, tuple(arrival.anchor), action)
                except (ValueError, KeyError):
                    continue
            elif kind is IntentKind.KILL and action.target not in allies:
                p = visible.get(action.target)
                if p: add('kill', action.target, (p.position.x, p.position.y), action)
            elif kind is IntentKind.REPORT:
                b = next((b for b in a.visible_bodies if b.victim_id == action.target), None)
                if b: add('report', action.target, (b.position.x, b.position.y), action)
            elif kind is IntentKind.CALL_MEETING:
                if obs.sabotage is None and obs.emergency_ready_in<=1e-9:
                    add('emergency', action.target, payload=action)
            elif kind is IntentKind.FAKE_TASK:
                if feedback.fake_remaining > 0 and feedback.fake_console == action.target: continue
                add('fake', action.target, payload=action)
        for pid, seen in sorted(memory.last_seen.items()):
            if pid in memory.known_dead or now-seen.time > 120: continue
            p = visible.get(pid); pos = (float(p.position.x), float(p.position.y)) if p else seen.position
            if math.dist(xy, pos) <= .1: continue
            if a.navigation_status == 'moving' and obs.own_route and math.dist(pos,obs.own_route[-1]) <= .1:
                continue
            add('visible_player' if p else 'remembered_player', pid, pos,
                Intent(IntentKind.GO_TO_LOCATION, position=__import__('social_deduction.actor', fromlist=['Point']).Point(*pos)))
        for body in a.visible_bodies:
            pos = (float(body.position.x), float(body.position.y))
            if math.dist(xy, pos) > .3:
                add('body_travel', body.victim_id, pos, Intent(IntentKind.GO_TO_BODY, body.victim_id))
        if obs.can_sabotage: add('lights', 'lights', payload=Action('sabotage', 'lights'))
        for key in obs.door_actions: add('doors', key, payload=Action('doors', key))
        for key in obs.vent_actions: add('vent_enter', key, payload=Action('vent_enter', key))
        for key in obs.repair_locations:
            if key in a.interactable_destinations and key not in obs.repaired_locations:
                add('repair', key, payload=Action('repair', key))
    elif phase == 'roaming' and obs.vent_state is not None and obs.vent_transition_ready:
        for key in obs.vent_links: add('vent_move', key, payload=Action('vent_move', key))
        if obs.vent_can_exit: add('vent_exit', obs.vent_state, payload=Action('vent_exit', obs.vent_state))
    elif phase == 'discussion' and a.context.speaker_id == owner:
        for pid in a.context.participants:
            for kind in (ClaimKind.SUSPECT_PLAYER, ClaimKind.DEFEND_PLAYER, ClaimKind.SAW_ELIMINATION):
                if pid == owner and kind is not ClaimKind.DEFEND_PLAYER: continue
                add('claim', pid, payload=Intent(IntentKind.CLAIM, claim=ClaimDraft(kind, pid, own.room, a.tick)))
        for room in rooms:
            add('claim', room, payload=Intent(IntentKind.CLAIM,
                claim=ClaimDraft(ClaimKind.WAS_IN_REGION, owner, room, a.tick)))
        for e in tuple(memory.events):
            statement = Event('share_evidence', e.subject, e.other, e.room, e.position,
                              e.observed, now, owner, 'claim', e.kind, e.origin_source or e.source)
            add('claim', e.subject, e.position, Action('communication', statement=statement))
    elif phase == 'voting' and owner in a.context.participants and owner not in a.context.voted_ids:
        for action in a.actions:
            if action.kind is IntentKind.VOTE and action.target!=owner: add('vote', action.target, payload=action)
            elif action.kind is IntentKind.SKIP: add('skip', payload=action)
    af = np.zeros((len(candidates), ACTION_WIDTH), np.float32)
    for i, c in enumerate(candidates):
        af[i, KINDS.index(c.kind)] = 1.
        claim = getattr(c.payload, 'claim', None)
        if claim: af[i, 20 + tuple(ClaimKind).index(claim.kind)] = 1.
        pos = c.position
        af[i, 28:] = [(pos[0]-xy[0])/60 if pos else 0, (pos[1]-xy[1])/60 if pos else 0,
                     math.dist(xy, pos)/60 if pos else 0, float(pos is not None),
                     float(c.key==owner), float(c.key in allies), float(c.key in visible),
                     identity(c.key), float(c.key in memory.known_dead), feedback.cooldown/60,
                     feedback.fake_remaining/4, float(c.key==feedback.fake_console),
                     float(last_goal==(c.kind,c.key)), float(c.kind=='claim'),
                     room_code(claim.region) if claim else 0, len(visible)/14,
                     sum(pid not in allies for pid in visible)/14,
                     float(c.kind.startswith('vent_')), float(obs.door_blocked), 0.]
    return Packet(sf, players, pm, ev, em, af,
                  np.array([KIND_FAMILY[c.kind] for c in candidates], np.int64),
                  tuple(candidates), owner, a.tick, now, phase)
