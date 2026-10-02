"""Persistent legitimate memory; deliberately no engine, truth or label imports."""
from dataclasses import asdict, dataclass
import json

from social_deduction.actor import (
    BodySeen, Ejection, Identity, MatchEnded, Meeting, Phase, PlayerSeen, Point,
    Provenance, Report, Role, Vote,
)
from social_deduction.phase3_api import (
    ClaimKind, EventView, Phase3Observation, StructuredClaim, WitnessedElimination,
    validate_event, validate_observation,
)

MEMORY_SCHEMA = 'actor-memory-v1'
PAYLOADS = {c.__name__: c for c in (PlayerSeen, BodySeen, Report, Meeting, Vote,
                                  Ejection, StructuredClaim, WitnessedElimination)}


@dataclass(frozen=True)
class Contradiction:
    speaker_id: str
    subject_id: str
    strength: str
    claim_ref: int
    support_ref: str


def location_claim(claim):
    if claim.kind in (ClaimKind.WAS_IN_REGION, ClaimKind.WAS_AT_TASK):
        return claim.speaker_id
    if claim.kind is ClaimKind.SAW_PLAYER:
        return claim.subject_id
    return None


class ActorMemory:
    """One actor, one match. Updates accept only validated canonical observations.

    `events` is the compressed timeline. `direct_rooms` stores only exact observed
    room/time pairs for conservative claim checks, without inventing intermediate
    positions. Dead/finished observations freeze this observer's evidence.
    """
    def __init__(self):
        self.match_id = None
        self.player_id = None
        self.roster = ()
        self.tick = -1
        self.dt = .2
        self.phase = Phase.ROAMING
        self.events = []
        self.last_seen = {}
        self.direct_rooms = {}
        self.visible = frozenset()
        self._known = set()
        self._keyframes = {}
        self._bodies = set()

    @property
    def candidates(self):
        return tuple(i.player_id for i in self.roster if i.player_id != self.player_id)

    def update(self, observation: Phase3Observation):
        validate_observation(observation)
        if observation.own.role is not Role.CREWMATE:
            raise ValueError('Phase 4 memory requires a crewmate focal actor')
        if self.match_id is not None and (
            observation.match_id != self.match_id or observation.own.player_id != self.player_id
            or observation.roster != self.roster or observation.settings.dt != self.dt
        ):
            raise ValueError('Cannot mix actor, match, roster or timestep in memory')
        if observation.tick < self.tick:
            raise ValueError('Memory cannot move backwards in time')
        if observation.context.phase is Phase.FINISHED or not observation.own.active:
            return ()
        if observation.tick == self.tick:
            return ()
        if self.match_id is None:
            self.match_id, self.player_id = observation.match_id, observation.own.player_id
            self.roster, self.dt = observation.roster, observation.settings.dt
            if len(self.roster) != 5 or len(set(i.player_id for i in self.roster)) != 5:
                raise ValueError('Phase 4 requires five unique public identities')
            if self.player_id not in {i.player_id for i in self.roster}:
                raise ValueError('Focal actor missing from roster')
        self.tick, self.phase = observation.tick, observation.context.phase
        self.visible = frozenset(p.player_id for p in observation.visible_players)
        # Own location is legitimate support for statements about the focal actor.
        self.direct_rooms[(self.player_id, self.tick)] = observation.own.room
        added = []
        for event in observation.evidence:
            p = event.payload
            if type(p) is MatchEnded:
                continue
            if type(p) is StructuredClaim and not 0 <= p.asserted_tick <= event.tick:
                raise ValueError('Claim asserts a time outside its delivered history')
            if type(p) is PlayerSeen:
                self.last_seen[p.player_id] = event
                self.direct_rooms[(p.player_id, event.tick)] = p.room
                prior = self._keyframes.get(p.player_id)
                if prior and (event.tick - prior.tick) * self.dt < 2 and (
                    p.room, p.interacting) == (prior.payload.room, prior.payload.interacting):
                    continue
                self._keyframes[p.player_id] = event
            elif type(p) is BodySeen:
                if p.victim_id in self._bodies:
                    continue
                self._bodies.add(p.victim_id)
            if event in self._known:
                continue
            self._known.add(event)
            self.events.append(event)
            added.append(event)
        return tuple(added)

    def contradictions(self):
        result = []
        earlier = {}
        for index, event in enumerate(self.events):
            c = event.payload
            if type(c) is not StructuredClaim:
                continue
            subject = location_claim(c)
            if subject is None:
                continue
            key = (subject, c.asserted_tick)
            room = self.direct_rooms.get(key)
            if room is not None and room != c.region:
                result.append(Contradiction(c.speaker_id, subject, 'direct', index,
                                           f'sighting:{subject}:{c.asserted_tick}'))
            for prior_index, prior in earlier.get(key, []):
                if prior.region != c.region:
                    # A conflict alone does not establish which speaker is wrong.
                    for speaker, ref, support in ((c.speaker_id, index, prior_index),
                                                  (prior.speaker_id, prior_index, index)):
                        result.append(Contradiction(speaker, subject, 'claim', ref,
                                                   f'claim:{support}'))
            earlier.setdefault(key, []).append((index, c))
        return tuple(result)

    def to_dict(self):
        def pack(event):
            return {'tick': event.tick, 'provenance': event.provenance.value,
                    'type': type(event.payload).__name__, 'payload': asdict(event.payload)}
        return {'schema': MEMORY_SCHEMA, 'match_id': self.match_id, 'player_id': self.player_id,
                'roster': [asdict(i) for i in self.roster], 'tick': self.tick, 'dt': self.dt,
                'phase': self.phase.value, 'events': [pack(e) for e in self.events],
                'last_seen': {pid: pack(e) for pid, e in sorted(self.last_seen.items())},
                'direct_rooms': [[pid, t, r] for (pid, t), r in sorted(self.direct_rooms.items())],
                'visible': sorted(self.visible)}

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text):
        data = json.loads(text)
        if data.get('schema') != MEMORY_SCHEMA:
            raise ValueError('Incompatible actor memory schema')
        memory = cls()
        expected = set(memory.to_dict())
        if set(data) != expected:
            raise ValueError('Unexpected memory fields')
        memory.match_id, memory.player_id = data['match_id'], data['player_id']
        memory.tick, memory.dt = data['tick'], data['dt']
        if type(memory.tick) is not int or memory.tick < -1 or not 0 < memory.dt < 10:
            raise ValueError('Invalid memory time')
        memory.phase = Phase(data['phase'])
        if memory.phase is Phase.FINISHED:
            raise ValueError('Terminal outcome is not a belief input')
        memory.roster = tuple(Identity(**i) for i in data['roster'])
        ids = {i.player_id for i in memory.roster}
        if len(ids) != 5 or len(memory.roster) != 5 or memory.player_id not in ids:
            raise ValueError('Invalid candidate identities')
        def unpack(record):
            if set(record) != {'tick', 'provenance', 'type', 'payload'}:
                raise ValueError('Invalid event envelope')
            p = dict(record['payload'])
            kind = PAYLOADS.get(record['type'])
            if kind is None:
                raise ValueError('Unrecognized or privileged event type')
            for name in ('position', 'velocity'):
                if name in p:
                    p[name] = Point(**p[name])
            if kind is Meeting:
                p['participants'] = tuple(p['participants'])
            if kind is StructuredClaim:
                p['kind'] = ClaimKind(p['kind'])
            event = EventView(record['tick'], Provenance(record['provenance']), kind(**p))
            validate_event(event, memory.tick)
            if kind is StructuredClaim and not 0 <= p['asserted_tick'] <= event.tick:
                raise ValueError('Invalid claim time')
            return event
        memory.events = [unpack(e) for e in data['events']]
        memory._known = set(memory.events)
        if len(memory._known) != len(memory.events):
            raise ValueError('Duplicate memory events')
        memory.last_seen = {pid: unpack(e) for pid, e in data['last_seen'].items()}
        for pid, event in memory.last_seen.items():
            if pid not in ids or type(event.payload) is not PlayerSeen or event.payload.player_id != pid:
                raise ValueError('Invalid last sighting')
        for pid, tick, room in data['direct_rooms']:
            if pid not in ids or type(tick) is not int or not 0 <= tick <= memory.tick or type(room) is not str:
                raise ValueError('Invalid observed location index')
            memory.direct_rooms[(pid, tick)] = room
        memory.visible = frozenset(data['visible'])
        if not memory.visible <= ids - {memory.player_id}:
            raise ValueError('Invalid visible identities')
        for event in memory.events:
            if type(event.payload) is PlayerSeen:
                memory._keyframes[event.payload.player_id] = event
            if type(event.payload) is BodySeen:
                memory._bodies.add(event.payload.victim_id)
        return memory
