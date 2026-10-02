"""Policy tests use immutable inputs without constructing a simulator."""
import ast
from dataclasses import replace
from pathlib import Path

import pytest

from phase3_bots import BotStyle, ScriptedController
from social_deduction.actor import (
    BodySeen, Console, Identity, Phase, PlayerSeen, Point, Provenance,
    PublicContext, PublicMap, Role, SelfView, TaskView,
)
from social_deduction.phase3_api import (
    ClaimKind, DestinationView, EventView, Intent, IntentKind, ObservationSettings,
    Phase3Observation, StructuredClaim, WitnessedElimination, intent_allowed, legal_actions,
)


def packet(**changes):
    public_map = PublicMap(("Cafeteria", "Electrical"), (), (), (
        Console("d0", "Cafeteria", Point(1.0, 0.0)),
        Console("d1", "Electrical", Point(8.0, 0.0)),
    ), 4.5, 1.3)
    obs = Phase3Observation(
        "opaque", 20, 4.0,
        SelfView("p0", Role.CREWMATE, Point(0.0, 0.0), "Cafeteria", True,
                 (TaskView("t0", "d0", 0.0), TaskView("t1", "d1", 0.0)), 1),
        tuple(Identity(f"p{i}", f"Color{i}") for i in range(5)), public_map,
        PublicContext(), (), (), (),
        (DestinationView("d0", "First", "Cafeteria", Point(1.0, 0.0), Point(1.0, 0.0), .8, "task"),
         DestinationView("d1", "Second", "Electrical", Point(8.0, 0.0), Point(8.0, 0.0), .8, "task")),
        ObservationSettings(),
    )
    obs = replace(obs, **changes)
    return replace(obs, actions=legal_actions(obs))


def impostor_packet(**changes):
    own = replace(packet().own, role=Role.IMPOSTOR, tasks=())
    return packet(own=own, kill_cooldown=0.0, **changes)


def voting_packet(evidence=()):
    return packet(evidence=evidence, context=PublicContext(
        Phase.VOTING, "meeting:1", ("p0", "p1", "p2", "p3"), None, (),
    ))


def test_controller_imports_only_safe_values_and_standard_library():
    source = Path(__file__).with_name("phase3_bots.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imports.update(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
    assert imports <= {"dataclasses", "math", "random", "social_deduction.actor",
                       "social_deduction.phase3_api"}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == '__import__' for node in ast.walk(tree))


@pytest.mark.parametrize("crew", ("nearest", "shuffled", "social"))
@pytest.mark.parametrize("impostor", ("hunter", "patient", "self_report"))
def test_variant_actions_are_deterministic_and_legal(crew, impostor):
    style = BotStyle(crew, impostor, "evidence")
    for obs in (packet(), impostor_packet()):
        first = ScriptedController(42, style).decide(obs)
        second = ScriptedController(42, style).decide(obs)
        assert first == second
        assert intent_allowed(obs, first)


def test_nearest_task_is_chosen_and_moving_route_is_preserved():
    bot = ScriptedController(2, BotStyle())
    assert bot.decide(packet()) == Intent(IntentKind.GO_TO_TASK, "d0")
    assert bot.decide(packet(navigation_status="moving", navigation_target="d0")) == Intent(IntentKind.WAIT)


def test_ready_task_starts_and_timed_interaction_is_allowed_to_continue():
    bot = ScriptedController(1, BotStyle())
    assert bot.decide(packet(interactable_tasks=("t0",))).kind is IntentKind.INTERACT
    assert bot.decide(packet(interactable_tasks=("t0",), interaction_task="t0")).kind is IntentKind.WAIT


def test_body_report_interrupts_task_intention():
    body = BodySeen("p4", Point(.2, 0.0), "Cafeteria")
    obs = packet(visible_bodies=(body,), navigation_status="moving", interaction_task="t0")
    action = ScriptedController(0, BotStyle()).decide(obs)
    assert action == Intent(IntentKind.REPORT, "p4")
    assert intent_allowed(obs, action)


def test_out_of_range_body_is_approached_without_hidden_victim_target():
    body = BodySeen("p4", Point(3.0, 0.0), "Cafeteria")
    action = ScriptedController(0).decide(packet(visible_bodies=(body,)))
    assert action == Intent(IntentKind.GO_TO_BODY, "p4")


def test_hunter_only_kills_visible_legal_candidate():
    seen = PlayerSeen("p1", Point(.5, 0.0), "Cafeteria", Point(0.0, 0.0), False)
    bot = ScriptedController(0, BotStyle(impostor="hunter"))
    obs = impostor_packet(visible_players=(seen,))
    assert bot.decide(obs) == Intent(IntentKind.KILL, "p1")
    hidden = impostor_packet()
    assert ScriptedController(0, BotStyle(impostor="hunter")).decide(hidden).kind is not IntentKind.KILL


def test_cooldown_prevents_kill_and_visible_chase_uses_observed_position():
    seen = PlayerSeen("p1", Point(3.0, 0.0), "Cafeteria", Point(0.0, 0.0), False)
    obs = impostor_packet(visible_players=(seen,))
    bot = ScriptedController(0, BotStyle(impostor="hunter"))
    assert bot.decide(obs) == Intent(IntentKind.GO_TO_LOCATION, position=seen.position)
    cooling = replace(obs, kill_cooldown=10.0)
    cooling = replace(cooling, actions=legal_actions(cooling))
    assert ScriptedController(0).decide(cooling).kind is not IntentKind.KILL


def test_hearsay_does_not_enter_direct_sighting_memory():
    claim = StructuredClaim(ClaimKind.SAW_PLAYER, "p1", "p2", "Electrical", 20, 10)
    obs = packet(evidence=(EventView(20, Provenance.CLAIM, claim),))
    bot = ScriptedController(0)
    bot.decide(obs)
    assert bot.recent_players == {}
    assert bot.witnesses == {}


def test_direct_kill_witness_generates_typed_claim_with_original_time():
    witness = WitnessedElimination("p1", "p4", Point(.3, 0.0), "Cafeteria")
    event = EventView(10, Provenance.DIRECT, witness)
    obs = packet(evidence=(event,), context=PublicContext(
        Phase.DISCUSSION, "meeting:1", ("p0", "p1", "p2", "p3"), "p0", (),
    ))
    bot = ScriptedController(0, BotStyle())
    action = bot.decide(obs)
    assert action.kind is IntentKind.CLAIM
    assert action.claim.kind is ClaimKind.SAW_ELIMINATION
    assert action.claim.subject_id == "p1" and action.claim.asserted_tick == 10
    assert intent_allowed(obs, action)
    assert bot.decide(obs).kind is IntentKind.WAIT  # One statement per meeting.


@pytest.mark.parametrize("meeting", ("evidence", "cautious", "skeptical"))
def test_direct_witness_votes_but_no_evidence_skips(meeting):
    bot = ScriptedController(0, BotStyle(meeting=meeting))
    assert bot.decide(voting_packet()).kind is IntentKind.SKIP
    event = EventView(10, Provenance.DIRECT,
                      WitnessedElimination("p1", "p4", Point(.3, 0.0), "Cafeteria"))
    assert bot.decide(voting_packet((event,))) == Intent(IntentKind.VOTE, "p1")


def test_repeated_claim_does_not_gain_duplicate_weight():
    claim = StructuredClaim(ClaimKind.SUSPECT_PLAYER, "p2", "p1", "Cafeteria", 20, 10)
    event = EventView(20, Provenance.CLAIM, claim)
    bot = ScriptedController(0, BotStyle(meeting="evidence"))
    assert bot.decide(voting_packet((event,) * 10)).kind is IntentKind.SKIP


def test_tied_evidence_skips_and_dead_public_candidate_cannot_receive_vote():
    events = tuple(EventView(10, Provenance.DIRECT,
                             WitnessedElimination(killer, "p4", Point(.3, 0.0), "Cafeteria"))
                   for killer in ("p1", "p2"))
    bot = ScriptedController(0, BotStyle())
    assert bot.decide(voting_packet(events)).kind is IntentKind.SKIP
    obs = voting_packet(events)
    obs = replace(obs, context=replace(obs.context, participants=("p0", "p2", "p3")))
    obs = replace(obs, actions=legal_actions(obs))
    assert ScriptedController(0, BotStyle()).decide(obs) == Intent(IntentKind.VOTE, "p2")


def test_inactive_policy_never_submits_gameplay_action():
    obs = packet(own=replace(packet().own, active=False))
    assert ScriptedController(0).decide(obs).kind is IntentKind.WAIT


def test_invalid_style_rejected():
    with pytest.raises(ValueError):
        BotStyle(crew="oracle")


def test_batch_summary_requires_replay_and_reports_timeouts_separately():
    from run_phase3_batch import summarize
    row = {"completed": 1, "winner": "draw", "reason": "timeout", "timeouts": 1,
           "replay_checked": 1}
    summary = summarize([row], {}, 1.0)
    assert summary["acceptance"] == "FAIL_OR_UNCHECKED"
    assert summary["winners"] == {"draw": 1}
    assert summary["totals"]["timeouts"] == 1
