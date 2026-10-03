"""Identity-free candidate features from legitimate memory only."""
from dataclasses import replace
import math
import numpy as np
from social_deduction.actor import BodySeen, Ejection, Meeting, PlayerSeen, Report, Vote
from social_deduction.phase3_api import ClaimKind, StructuredClaim, WitnessedElimination
from .memory import ActorMemory

FEATURE_SCHEMA = 'belief-features-v1'
FEATURE_NAMES = (
    'seen', 'seen_age', 'visible', 'sightings', 'interaction_fraction',
    'body_known', 'public_absent', 'ejected', 'direct_elimination',
    'direct_near_body', 'near_report', 'reports', 'votes_cast', 'votes_received', 'skips',
    'claimed_elimination', 'claimed_near_body', 'accused', 'defended',
    'location_claims', 'direct_contradiction', 'claim_conflict',
)
VIEWS = ('full', 'current', 'no_claims', 'collapsed')
INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def current_memory(observation, previous_tick):
    """Drop historical publications but retain the currently public meeting roster.

    Its canonical Meeting envelope materializes information also present in the
    current PublicContext; it does not restore old claims, votes or sightings.
    """
    fresh = replace(observation, evidence=tuple(e for e in observation.evidence
        if e.tick > previous_tick or (type(e.payload) is Meeting and
            e.payload.meeting_id == observation.context.meeting_id)))
    memory = ActorMemory()
    memory.update(fresh)
    return memory


def encode(memory, candidates=None, view='full'):
    if type(memory) is not ActorMemory or memory.tick < 0:
        raise TypeError('Encode requires initialized ActorMemory')
    candidates = tuple(candidates or memory.candidates)
    if len(candidates) != 4 or set(candidates) != set(memory.candidates):
        raise ValueError('Candidates must be a permutation of the four non-self identities')
    if view not in VIEWS:
        raise ValueError('Unknown evidence view')
    rows = {pid: {n: 0. for n in FEATURE_NAMES} for pid in candidates}
    def add(pid, key, value=1.):
        if pid in rows:
            rows[pid][key] += value
    seen = [e for e in memory.events if type(e.payload) is PlayerSeen]
    for pid, row in rows.items():
        last = memory.last_seen.get(pid)
        if last:
            row['seen'] = 1
            row['seen_age'] = min(1., (memory.tick - last.tick) * memory.dt / 60.)
        row['visible'] = float(pid in memory.visible)
        sightings = [e for e in seen if e.payload.player_id == pid]
        row['sightings'] = math.log1p(len(sightings)) / 4
        row['interaction_fraction'] = (sum(e.payload.interacting for e in sightings) / len(sightings)
                                       if sightings else 0.)
    for event in memory.events:
        p = event.payload
        if type(p) is BodySeen:
            add(p.victim_id, 'body_known')
            # Only sightings preceding discovery, within 3 seconds and 2 map units.
            implicated = {e.payload.player_id for e in seen if 0 <= (event.tick-e.tick)*memory.dt <= 3
                          and math.hypot(e.payload.position.x-p.position.x,
                                         e.payload.position.y-p.position.y) <= 2}
            for pid in implicated - {p.victim_id}:
                add(pid, 'direct_near_body')
        elif type(p) is WitnessedElimination:
            add(p.killer_id, 'direct_elimination')
            add(p.victim_id, 'body_known')
        elif type(p) is Report:
            add(p.victim_id, 'body_known')
            add(p.reporter_id, 'reports')
            implicated = {e.payload.player_id for e in seen if 0 <= (event.tick-e.tick)*memory.dt <= 3
                          and e.payload.room == p.room}
            for pid in implicated - {p.victim_id}:
                add(pid, 'near_report')
        elif type(p) is Meeting:
            for pid in rows:
                if pid not in p.participants:
                    rows[pid]['public_absent'] = 1
        elif type(p) is Ejection:
            add(p.player_id, 'ejected')
        elif type(p) is Vote:
            add(p.voter_id, 'votes_cast')
            add(p.target_id, 'votes_received')
            if p.target_id is None:
                add(p.voter_id, 'skips')
        elif type(p) is StructuredClaim and view != 'no_claims':
            key = {ClaimKind.SAW_ELIMINATION: 'claimed_elimination',
                   ClaimKind.SAW_PLAYER_NEAR_BODY: 'claimed_near_body',
                   ClaimKind.SUSPECT_PLAYER: 'accused', ClaimKind.DEFEND_PLAYER: 'defended'}.get(p.kind)
            if key:
                add(p.subject_id, key)
            if p.kind in (ClaimKind.WAS_IN_REGION, ClaimKind.WAS_AT_TASK):
                add(p.speaker_id, 'location_claims')
    if view != 'no_claims':
        for conflict in memory.contradictions():
            add(conflict.speaker_id, 'direct_contradiction' if conflict.strength == 'direct' else 'claim_conflict')
    if view == 'collapsed':
        for row in rows.values():
            row['direct_elimination'] += row['claimed_elimination']
            row['direct_near_body'] += row['claimed_near_body']
            row['claim_conflict'] += row['direct_contradiction']
            for key in ('claimed_elimination', 'claimed_near_body', 'direct_contradiction'):
                row[key] = 0.
    result = np.asarray([[min(8., row[n]) for n in FEATURE_NAMES] for row in rows.values()], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite belief evidence')
    return result


RULE_WEIGHTS = {'body_known': -4., 'direct_elimination': 6., 'direct_near_body': 2.,
                'near_report': .5, 'reports': .25, 'claimed_elimination': .7,
                'claimed_near_body': .4, 'accused': .1, 'defended': -.2,
                'direct_contradiction': 2.5, 'claim_conflict': .25}


def rule_logits(features):
    weights = np.array([RULE_WEIGHTS.get(n, 0.) for n in FEATURE_NAMES])
    return np.asarray(features) @ weights


def probabilities(logits, temperature=1.):
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= scaled.max(axis=-1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=-1, keepdims=True)
