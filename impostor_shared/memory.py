"""Independent impostor memory. No fake crew role and no access to crew memories."""
import numpy as np
from social_deduction.actor import BodySeen, Ejection, MatchEnded, Meeting, Phase, PlayerSeen, Report, Role, Vote
from social_deduction.phase3_api import ClaimKind, StructuredClaim, validate_observation


class ImpostorMemory:
    def __init__(self):
        self.owner = None; self.match_id = None; self.tick = -1; self.dt = .2
        self.candidates = (); self.events = []; self.last_seen = {}; self.visible = set()
        self._known = set(); self._keyframes = {}

    def update(self, obs):
        validate_observation(obs)
        if obs.own.role is not Role.IMPOSTOR:
            raise ValueError('Impostor memory requires own impostor role')
        if self.owner is not None and (self.owner != obs.own.player_id or self.match_id != obs.match_id):
            raise ValueError('Cannot share actor memory')
        if obs.tick < self.tick:
            raise ValueError('Memory time reversed')
        if not obs.own.active or obs.context.phase is Phase.FINISHED or obs.tick == self.tick:
            return
        self.owner = obs.own.player_id; self.match_id = obs.match_id; self.tick = obs.tick; self.dt = obs.settings.dt
        self.candidates = tuple(i.player_id for i in obs.roster if i.player_id != self.owner)
        if len(self.candidates) != 4:
            raise ValueError('Only five-player games supported')
        self.visible = {p.player_id for p in obs.visible_players}
        for e in obs.evidence:
            p = e.payload
            if type(p) is MatchEnded: continue
            if type(p) is PlayerSeen:
                self.last_seen[p.player_id] = e
                prior = self._keyframes.get(p.player_id)
                if prior and (e.tick-prior.tick)*self.dt < 2 and (p.room,p.interacting) == (prior.payload.room,prior.payload.interacting):
                    continue
                self._keyframes[p.player_id] = e
            if e not in self._known:
                self._known.add(e); self.events.append(e)

    def self_social(self):
        counts = [0., 0., 0.]
        for e in self.events:
            p = e.payload
            if type(p) is StructuredClaim and p.subject_id == self.owner:
                counts[0] += p.kind is ClaimKind.SUSPECT_PLAYER
                counts[1] += p.kind is ClaimKind.SAW_ELIMINATION
            elif type(p) is Vote and p.target_id == self.owner: counts[2] += 1
        return [min(v/8,1) for v in counts]

    def player_features(self):
        rows = {pid: [0.]*12 for pid in self.candidates}
        for pid, row in rows.items():
            last = self.last_seen.get(pid)
            row[0] = float(last is not None)
            row[1] = min((self.tick-last.tick)*self.dt/900,1) if last else 1.
            row[2] = float(pid in self.visible)
        for e in self.events:
            p = e.payload
            def add(pid, index):
                if pid in rows: rows[pid][index] += .125
            if type(p) is PlayerSeen:
                add(p.player_id,3)
                if p.interacting: add(p.player_id,4)
            elif type(p) is BodySeen: add(p.victim_id,5)
            elif type(p) is Meeting:
                for pid in rows:
                    if pid not in p.participants: rows[pid][6] = 1.
            elif type(p) is Ejection: add(p.player_id,6)
            elif type(p) is Report: add(p.reporter_id,7)
            elif type(p) is Vote:
                add(p.voter_id,8); add(p.target_id,9)
            elif type(p) is StructuredClaim:
                if p.kind in (ClaimKind.SUSPECT_PLAYER,ClaimKind.SAW_ELIMINATION): add(p.subject_id,10)
                elif p.kind is ClaimKind.DEFEND_PLAYER: add(p.subject_id,11)
        return np.clip(np.asarray(list(rows.values()),np.float32),0,1)
