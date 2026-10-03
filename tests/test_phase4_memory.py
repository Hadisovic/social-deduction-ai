from dataclasses import replace
import json
import numpy as np
import pytest
from belief.memory import ActorMemory
from belief.features import INDEX, current_memory, encode, rule_logits, probabilities
from phase3_engine import Phase3Game
from social_deduction.actor import (BodySeen, Ejection, MatchEnded, Meeting, Phase,
    PlayerSeen, Point, Provenance, PublicContext, Role, Vote)
from social_deduction.phase3_api import ClaimKind, EventView, StructuredClaim, WitnessedElimination


@pytest.fixture
def obs():
    game = Phase3Game(15)
    return next(game.observe(pid) for pid in game.players if game.observe(pid).own.role is Role.CREWMATE)


def packet(obs, tick, events=(), phase=Phase.ROAMING):
    return replace(obs, tick=tick, time_seconds=tick*.2, evidence=tuple(events),
                   visible_players=tuple(e.payload for e in events if type(e.payload) is PlayerSeen),
                   visible_bodies=tuple(e.payload for e in events if type(e.payload) is BodySeen),
                   context=PublicContext(phase))


def sight(pid, tick, room='Electrical', interacting=False):
    return EventView(tick, Provenance.DIRECT, PlayerSeen(pid, Point(0., 0.), room, Point(0., 0.), interacting))


def claim(pid, tick, asserted=10, kind=ClaimKind.WAS_IN_REGION, subject=None, room='Navigation'):
    return EventView(tick, Provenance.CLAIM, StructuredClaim(kind, pid, subject or pid, room, tick, asserted))


def test_memory_last_seen_provenance_conflicts_persist_and_roundtrip(obs):
    pid = next(i.player_id for i in obs.roster if i.player_id != obs.own.player_id)
    memory = ActorMemory()
    memory.update(packet(obs, 10, [sight(pid, 10)]))
    memory.update(packet(obs, 20, [claim(pid, 20)], Phase.DISCUSSION))
    assert memory.last_seen[pid].tick == 10
    assert memory.contradictions()[0].strength == 'direct'
    assert memory.events[-1].provenance is Provenance.CLAIM
    restored = ActorMemory.from_json(memory.to_json())
    assert restored.to_json() == memory.to_json()
    assert np.array_equal(encode(restored), encode(memory))
    for m in (memory, restored):
        m.update(packet(obs, 30, [claim(pid, 20)], Phase.VOTING))
        m.update(packet(obs, 40))
    assert restored.to_json() == memory.to_json()
    assert len(memory.events) == 2
    assert memory.last_seen[pid].payload.room == 'Electrical'


def test_contradiction_is_exact_time_and_belongs_to_speaker(obs):
    a, b = [i.player_id for i in obs.roster if i.player_id != obs.own.player_id][:2]
    m = ActorMemory(); m.update(packet(obs, 10, [sight(a, 10)]))
    m.update(packet(obs, 20, [claim(b, 20, asserted=11, kind=ClaimKind.SAW_PLAYER, subject=a)]))
    assert not m.contradictions()
    m.update(packet(obs, 21, [claim(b, 21, asserted=10, kind=ClaimKind.SAW_PLAYER, subject=a)]))
    assert m.contradictions()[0].speaker_id == b
    x = encode(m)
    assert x[m.candidates.index(b), INDEX['direct_contradiction']] == 1
    assert x[m.candidates.index(a), INDEX['direct_contradiction']] == 0


def test_hearsay_conflict_is_weak_and_symmetric(obs):
    a, b = [i.player_id for i in obs.roster if i.player_id != obs.own.player_id][:2]
    m = ActorMemory(); m.update(packet(obs, 20, [claim(a, 20, subject=b, kind=ClaimKind.SAW_PLAYER)]))
    m.update(packet(obs, 21, [claim(b, 21, room='Electrical')]))
    assert {c.strength for c in m.contradictions()} == {'claim'}
    assert {c.speaker_id for c in m.contradictions()} == {a, b}


@pytest.mark.parametrize('kind', ['sight', 'body', 'claim', 'vote', 'ejection', 'meeting'])
def test_future_event_rejected(obs, kind):
    a = obs.roster[1].player_id
    events = {'sight': sight(a, 30), 'body': EventView(30, Provenance.DIRECT, BodySeen(a, Point(0,0), 'x')),
              'claim': claim(a, 30), 'vote': EventView(30, Provenance.PUBLIC, Vote('m', a, None)),
              'ejection': EventView(30, Provenance.PUBLIC, Ejection('m', a)),
              'meeting': EventView(30, Provenance.PUBLIC, Meeting('m', (a,)))}
    with pytest.raises(ValueError, match='outside observation time'):
        ActorMemory().update(packet(obs, 20, [events[kind]]))


def test_provenance_spoof_future_assertion_and_truth_rejected(obs):
    m = ActorMemory()
    with pytest.raises(TypeError):
        m.update(Phase3Game(15).snapshot())
    with pytest.raises(ValueError, match='provenance'):
        m.update(packet(obs, 20, [replace(claim('p1', 20), provenance=Provenance.DIRECT)]))
    with pytest.raises(ValueError, match='asserts'):
        ActorMemory().update(packet(obs, 20, [claim('p1', 20, asserted=30)]))


def test_terminal_and_dead_packets_do_not_update(obs):
    m = ActorMemory(); m.update(packet(obs, 10))
    before = m.to_json()
    m.update(packet(obs, 20, [EventView(20, Provenance.PUBLIC, MatchEnded(Role.CREWMATE))], Phase.FINISHED))
    m.update(replace(packet(obs, 20), own=replace(obs.own, active=False)))
    assert m.to_json() == before


def test_current_ablation_drops_historical_public_prefix(obs):
    a = next(i.player_id for i in obs.roster if i.player_id != obs.own.player_id)
    packet_now = packet(obs, 30, [claim(a, 20)])
    assert not current_memory(packet_now, 27).events
    m = ActorMemory(); m.update(packet_now)
    assert m.events
    # The live meeting context still discloses its participants; removing that
    # state would unfairly weaken a current-observation-only baseline.
    participants = tuple(i.player_id for i in obs.roster if i.player_id != a)
    meeting = EventView(10,Provenance.PUBLIC,Meeting('current-meeting',participants))
    in_meeting = replace(packet(obs,30,[meeting,claim(a,20)],Phase.DISCUSSION),
                         context=PublicContext(Phase.DISCUSSION,'current-meeting',participants))
    current = current_memory(in_meeting,27)
    assert current.events == [meeting]
    assert encode(current)[current.candidates.index(a),INDEX['public_absent']] == 1


def test_equivariant_features_and_uniform_ambiguous_rules(obs):
    m = ActorMemory(); m.update(packet(obs, 0))
    x = encode(m)
    assert np.array_equal(probabilities(rule_logits(x)), np.full(4, .25))
    pid = m.candidates[0]
    m.update(packet(obs, 10, [sight(pid, 10)]))
    assert np.array_equal(encode(m, tuple(reversed(m.candidates))), encode(m)[::-1])


def test_no_claims_and_collapsed_features(obs):
    m = ActorMemory(); m.update(packet(obs, 20, [claim('p4', 20, kind=ClaimKind.SAW_ELIMINATION, subject='p1')]))
    if 'p1' not in m.candidates:
        pytest.skip('fixture focal')
    row = m.candidates.index('p1')
    assert encode(m)[row, INDEX['claimed_elimination']] == 1
    assert encode(m, view='no_claims')[row, INDEX['claimed_elimination']] == 0
    assert encode(m, view='collapsed')[row, INDEX['direct_elimination']] == 1
    assert encode(m)[row, INDEX['direct_elimination']] == 0


def test_bad_serialization_schema_and_privileged_event(obs):
    m = ActorMemory(); m.update(packet(obs, 0))
    data = m.to_dict(); data['schema'] = 'future'
    with pytest.raises(ValueError, match='schema'):
        ActorMemory.from_json(json.dumps(data))
    data = m.to_dict(); data['events'] = [{'type': 'WorldTruth', 'tick': 0, 'provenance': 'public', 'payload': {}}]
    with pytest.raises(ValueError, match='privileged'):
        ActorMemory.from_json(json.dumps(data))
