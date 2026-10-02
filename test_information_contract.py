"""Phase 1 paired-world tests; fixtures intentionally do not simulate gameplay."""

from dataclasses import FrozenInstanceError, asdict, replace
import inspect
import json

import pytest

from social_deduction import actor
from social_deduction.actor import (
    Action, ActionKind, Claim, Console, Ejection, Evidence, Identity, Knowledge,
    MatchEnded, Meeting, Phase, Point, Provenance, PublicContext, PublicMap,
    Report, Role, Vote, Wall, remember, validate_observation,
)
from social_deduction.observation import legal_actions, observe
from social_deduction.training import role_labels
from social_deduction.truth import (
    BodyTruth, InternalEvent, PlayerTruth, Publication, TaskTruth, WorldTruth,
    assign_identities, assign_roles,
)


@pytest.fixture
def world():
    public_map = PublicMap(
        ("A", "B"), (("A", "B"),), (Wall(50, -10, 55, 10),),
        (Console("task", "A", Point(5, 0)), Console("button", "A", Point(0, 5), True)),
    )
    players = (
        PlayerTruth(Identity("p0", "Red"), Role.CREWMATE, Point(0, 0), "A",
                    tasks=(TaskTruth("mine", "task"),)),
        PlayerTruth(Identity("p1", "Blue"), Role.CREWMATE, Point(20, 0), "A"),
        PlayerTruth(Identity("p2", "Green"), Role.IMPOSTOR, Point(900, 0), "B"),
        PlayerTruth(Identity("p3", "Yellow"), Role.CREWMATE, Point(950, 0), "B"),
        PlayerTruth(Identity("p4", "Purple"), Role.CREWMATE, Point(990, 0), "B"),
    )
    return WorldTruth("opaque-match", 10, public_map, players)


def change_player(world, player_id, **changes):
    return replace(world, players=tuple(replace(p, **changes) if p.identity.player_id == player_id
                                        else p for p in world.players))


def packet(world):
    return observe(world, "p0")


def test_role_swap_changes_labels_but_not_complete_actor_packet(world):
    swapped = change_player(change_player(world, "p1", role=Role.IMPOSTOR), "p2", role=Role.CREWMATE)
    assert packet(world) == packet(swapped)
    assert role_labels(world) != role_labels(swapped)
    assert json.dumps(asdict(packet(world))) == json.dumps(asdict(packet(swapped)))


@pytest.mark.parametrize("changes", [
    {"position": Point(400, 100), "room": "elsewhere"},
    {"tasks": (TaskTruth("private-secret", "task", 0.8),)},
    {"private_cooldown": 81.0},
    {"interacting": True, "velocity": Point(3, 8)},
    {"active": False},
])
def test_hidden_player_facts_do_not_change_packet_or_actions(world, changes):
    assert packet(world) == packet(change_player(world, "p2", **changes))


def test_occluded_player_is_hidden_even_inside_range(world):
    occluded = change_player(world, "p2", position=Point(100, 0))
    assert packet(occluded) == packet(world)
    assert packet(change_player(occluded, "p2", position=Point(110, 0))) == packet(occluded)


def test_visible_players_still_keep_private_tasks_and_cooldowns(world):
    altered = change_player(world, "p1", tasks=(TaskTruth("secret", "task", 0.7),), private_cooldown=99)
    assert packet(world) == packet(altered)


def test_occluded_body_produces_no_report_candidate(world):
    occluded = replace(world, bodies=(BodyTruth("p3", Point(60, 0), "A", 2, "p2"),))
    assert packet(occluded) == packet(world)


def test_visible_position_and_animation_are_observed(world):
    changed = packet(change_player(world, "p1", position=Point(21, 0), interacting=True))
    assert changed != packet(world)
    assert changed.visible_players[0].position == Point(21, 0)
    assert changed.visible_players[0].interacting
    assert not hasattr(changed.visible_players[0], "tasks")
    assert not hasattr(changed.visible_players[0], "role")


@pytest.mark.parametrize("tasks", [(TaskTruth("new-own", "task"),), (TaskTruth("mine", "task", 1.0),)])
def test_own_tasks_and_progress_are_observed(world, tasks):
    changed = packet(change_player(world, "p0", tasks=tasks))
    assert changed != packet(world)
    if tasks[0].progress == 1.0:
        assert Action(ActionKind.INTERACT, "mine") not in changed.actions


def test_public_report_changes_packet_without_hidden_report_details(world):
    reported = replace(world, publications=(Publication(8, Report("p1", "p3", "B")),))
    observation = packet(reported)
    assert observation != packet(world)
    event = observation.evidence[0]
    assert event == Evidence(8, Provenance.PUBLIC, Report("p1", "p3", "B"))
    assert "death_tick" not in json.dumps(asdict(observation))


def test_claim_is_not_verified_or_promoted_to_direct_evidence(world):
    claim = Claim("p1", "p2", "A", 2)
    heard = replace(world, publications=(Publication(9, claim),))
    event = packet(heard).evidence[0]
    assert event.provenance is Provenance.CLAIM
    assert event.observed_tick == 9 and event.payload.claimed_tick == 2
    assert packet(heard) == packet(change_player(heard, "p2", position=Point(800, 9)))
    assert all(p.player_id != "p2" for p in packet(heard).visible_players)


def test_future_publications_not_delivered(world):
    future = replace(world, publications=(Publication(11, Report("p1", "p3", "B")),))
    assert packet(future) == packet(world)


def test_future_payloads_and_order_are_not_an_input_side_channel(world):
    future = replace(world, publications=(Publication(15, InternalEvent(15, "secret", "p2")),
                                         Publication(11, Report("p1", "p3", "B"))))
    assert packet(future) == packet(world)


def test_hidden_log_insertion_and_container_order_do_not_change_ordering(world):
    changed = replace(world, players=tuple(reversed(world.players)), internal_events=(
        InternalEvent(1, "private-task", "p4"), InternalEvent(2, "hidden-kill", "p3"),
    ))
    assert packet(changed) == packet(world)


def test_same_tick_public_order_is_preserved_without_internal_sequence_ids(world):
    first, second = Claim("p1", "p2", "A", 1), Claim("p3", "p4", "B", 2)
    heard = replace(world, publications=(Publication(9, first), Publication(9, second)))
    assert [e.payload for e in packet(heard).evidence[:2]] == [first, second]


def test_unseen_death_and_body_do_not_leak(world):
    dead = change_player(world, "p3", active=False)
    dead = replace(dead, bodies=(BodyTruth("p3", Point(950, 0), "B", 4, "p2"),))
    assert packet(dead) == packet(world)
    assert len(packet(dead).roster) == 5


def test_visible_body_has_no_death_time_or_killer_and_local_report_target(world):
    body = BodyTruth("p3", Point(10, 0), "A", 3, "p2")
    seen = replace(change_player(world, "p3", active=False), bodies=(body,))
    different_truth = replace(seen, bodies=(replace(body, death_tick=8, killer_id="p1"),))
    assert packet(seen) == packet(different_truth)
    assert Action(ActionKind.REPORT, "p3") in packet(seen).actions
    assert packet(seen).visible_bodies[0].victim_id == "p3"


def test_follow_targets_only_currently_visible_and_navigation_is_public(world):
    observation = packet(world)
    assert Action(ActionKind.FOLLOW, "p1") in observation.actions
    assert Action(ActionKind.FOLLOW, "p2") not in observation.actions
    assert Action(ActionKind.NAVIGATE, "room:B") in observation.actions
    assert legal_actions(observation) == observation.actions


def test_interaction_cannot_reach_through_wall(world):
    wall_map = replace(world.map, walls=(Wall(2, -1, 3, 1),))
    assert Action(ActionKind.INTERACT, "mine") not in packet(replace(world, map=wall_map)).actions


def test_own_meeting_allowance_is_local_action_information(world):
    assert Action(ActionKind.CALL_MEETING, "button") in packet(world).actions
    assert Action(ActionKind.CALL_MEETING, "button") not in packet(
        change_player(world, "p0", meetings_remaining=0)).actions
    assert packet(world) == packet(change_player(world, "p1", meetings_remaining=0))


def test_vote_targets_use_public_roster_not_hidden_life_or_role(world):
    participants = ("p0", "p1", "p2", "p3", "p4")
    voting = replace(world, context=PublicContext(Phase.VOTING, "m1", participants),
                     publications=(Publication(5, Meeting("m1", participants)),))
    assert packet(voting) == packet(change_player(voting, "p3", active=False))
    assert Action(ActionKind.VOTE, "p3") in packet(voting).actions
    assert Action(ActionKind.SKIP) in packet(voting).actions
    cast = replace(voting, context=replace(voting.context, voted_ids=("p0",)),
                   publications=voting.publications + (Publication(10, Vote("m1", "p0", None)),))
    assert packet(cast).actions == ()


def test_discussion_turn_and_ejection_disclose_no_role(world):
    discussion = replace(world, context=PublicContext(Phase.DISCUSSION, "m1", ("p0", "p1"), "p0"),
                         publications=(Publication(8, Ejection("m0", "p4")),))
    assert packet(discussion).actions == (Action(ActionKind.CLAIM),)
    assert not hasattr(packet(discussion).evidence[0].payload, "role")
    assert packet(replace(discussion, context=replace(discussion.context, speaker_id="p1"))).actions == ()


def test_no_ghost_vision_and_no_actions_after_finish(world):
    dead = packet(change_player(world, "p0", active=False))
    assert not dead.actions and not dead.visible_players and not dead.visible_bodies
    finished = replace(world, context=PublicContext(Phase.FINISHED),
                       publications=(Publication(10, MatchEnded(Role.CREWMATE)),))
    assert not packet(finished).actions and not packet(finished).visible_players
    assert not hasattr(packet(finished), "labels")


@pytest.mark.parametrize("position,visible", [(Point(0, 220), True), (Point(0, 220.01), False),
                                              (Point(50, 0), False)])
def test_range_and_wall_contact_edges(world, position, visible):
    changed = packet(change_player(world, "p1", position=position))
    assert bool(changed.visible_players) is visible


def test_memory_keeps_sighting_without_refreshing_unseen_player(world):
    initial = packet(world)
    history = remember(Knowledge(world.match_id, "p0"), initial)
    later = replace(change_player(world, "p1", position=Point(700, 0), room="B"), tick=20)
    history = remember(history, packet(later))
    assert history.history[0].visible_players[0].position == Point(20, 0)
    assert history.history[1].visible_players == ()
    alternate = replace(change_player(later, "p1", position=Point(850, 0)), tick=20)
    base = remember(Knowledge(world.match_id, "p0"), initial)
    assert remember(base, packet(alternate)) == history


def test_claim_provenance_survives_memory_and_public_prefix_repetition(world):
    heard = replace(world, publications=(Publication(5, Claim("p1", "p2", "A", 1)),))
    memory = remember(Knowledge(world.match_id, "p0"), packet(heard))
    memory = remember(memory, packet(replace(heard, tick=11)))
    for snapshot in memory.history:
        assert snapshot.evidence[0].provenance is Provenance.CLAIM
        assert snapshot.evidence[0].observed_tick == 5


def test_memory_rejects_labels_wrong_actor_wrong_match_and_time(world):
    observation = packet(world)
    history = remember(Knowledge(world.match_id, "p0"), observation)
    with pytest.raises(TypeError):
        remember(history, role_labels(world))
    for bad in (replace(observation, match_id="another"),
                replace(observation, own=replace(observation.own, player_id="p1")), observation):
        with pytest.raises(ValueError):
            remember(history, bad)


def test_prepopulated_memory_cannot_mix_other_actors(world):
    foreign = observe(world, "p1")
    tainted = Knowledge(world.match_id, "p0", (foreign,))
    with pytest.raises(ValueError, match="different actor"):
        remember(tainted, packet(replace(world, tick=20)))


def test_snapshot_is_immutable_and_not_aliased_to_privileged_state(world):
    observation = packet(world)
    with pytest.raises(FrozenInstanceError):
        observation.own.room = "changed"
    with pytest.raises(FrozenInstanceError):
        observation.visible_players[0].position.x = 1000
    assert type(observation.own) is not type(world.players[0])
    assert not hasattr(observation, "world") and not hasattr(observation, "internal_events")
    assert "private_cooldown" not in json.dumps(asdict(observation))


@pytest.mark.parametrize("payload", [InternalEvent(1, "secret", "p2"), {"role": "impostor"}])
def test_public_log_rejects_privileged_or_untyped_payload(world, payload):
    with pytest.raises(TypeError):
        packet(replace(world, publications=(Publication(9, payload),)))


def test_malformed_nested_payload_and_mutable_map_are_rejected(world):
    with pytest.raises(TypeError):
        packet(replace(world, publications=(Publication(9, Report("p1", "p2", {"secret": 1})),)))
    with pytest.raises(TypeError):
        packet(replace(world, map=replace(world.map, rooms=["A", "B"])))
    with pytest.raises(TypeError):
        validate_observation(replace(packet(world), actions=(role_labels(world),)))


def test_bad_provenance_and_time_order_fail_closed(world):
    bad = Evidence(9, Provenance.DIRECT, Claim("p1", "p2", "A", 2))
    with pytest.raises(ValueError):
        validate_observation(replace(packet(world), evidence=(bad,)))
    with pytest.raises(ValueError):
        packet(replace(world, publications=(Publication(9, Report("p1", "p2", "A")),
                                           Publication(8, Report("p1", "p3", "A")))))


def test_unknown_duplicate_and_unsupported_actors_fail_closed(world):
    with pytest.raises(ValueError):
        observe(world, "missing")
    with pytest.raises(ValueError, match="crewmates only"):
        observe(world, "p2")
    with pytest.raises(ValueError):
        packet(replace(world, players=world.players + (world.players[0],)))


def test_identity_assignment_is_independent_reproducible_and_role_balanced_over_seeds():
    identities = assign_identities(5, seed=901)
    assert identities == assign_identities(5, seed=901)
    assert len({p.player_id for p in identities}) == 5
    assert len({p.display_name for p in identities}) == 5
    assert assign_roles(5, seed=902) == assign_roles(5, seed=902)
    possible_impostors = set()
    for seed in range(100):
        roles = assign_roles(5, seed=seed)
        assert roles.count(Role.IMPOSTOR) == 1
        possible_impostors.add(identities[roles.index(Role.IMPOSTOR)].player_id)
    assert possible_impostors == {p.player_id for p in identities}
    assert assign_identities(5, seed=903) != identities
    assert "role" not in inspect.signature(assign_identities).parameters


def test_identity_mapping_varies_without_changing_role_assignment():
    roles = assign_roles(5, seed=500)
    slot = roles.index(Role.IMPOSTOR)
    public_ids = {assign_identities(5, seed=s)[slot].player_id for s in range(100)}
    assert public_ids == {f"p{i}" for i in range(5)}
    assert roles == assign_roles(5, seed=500)


def test_actor_module_has_no_truth_or_training_dependency():
    # Architecture guard: policy-facing types must not import privileged modules.
    import ast
    tree = ast.parse(inspect.getsource(actor))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert "truth" not in imports and "training" not in imports
