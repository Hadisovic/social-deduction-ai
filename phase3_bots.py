"""Small scripted policies whose only game input is an immutable observation.

There is deliberately no engine, truth, map-geometry, navigation or renderer
import here. State is limited to recent direct sightings and a current intention;
this is a reproducible simulator fixture, not a learned memory/belief model.
"""
from dataclasses import dataclass
import math
from random import Random

from social_deduction.actor import BodySeen, Phase, PlayerSeen, Provenance, Role
from social_deduction.phase3_api import (
    ClaimDraft, ClaimKind, Intent, IntentKind, Phase3Observation,
    StructuredClaim, WitnessedElimination,
)


CREW_VARIANTS = ("nearest", "shuffled", "social")
IMPOSTOR_VARIANTS = ("hunter", "patient", "self_report")
MEETING_VARIANTS = ("evidence", "cautious", "skeptical")


def _distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


@dataclass(frozen=True)
class BotStyle:
    crew: str = "nearest"
    impostor: str = "hunter"
    meeting: str = "evidence"

    def __post_init__(self):
        for name, choices in (("crew", CREW_VARIANTS), ("impostor", IMPOSTOR_VARIANTS),
                              ("meeting", MEETING_VARIANTS)):
            if getattr(self, name) not in choices:
                raise ValueError(f"Unknown {name} script")


class ScriptedController:
    """A replacement point for a future policy: ``decide(observation) -> Intent``.

    The private RNG seed is independently derived by the trusted runner. It is
    not the match/role seed. All variants are selected before observing a role.
    """
    def __init__(self, seed: int, style: BotStyle | None = None):
        self.rng = Random(seed)
        self.style = style or BotStyle(
            self.rng.choice(CREW_VARIANTS), self.rng.choice(IMPOSTOR_VARIANTS),
            self.rng.choice(MEETING_VARIANTS),
        )
        self.recent_players = {}
        self.recent_bodies = {}
        self.witnesses = {}
        self.task_order = {}
        self.last_room = None
        self.last_exploration_tick = 0
        self.last_patrol = None
        self.last_chase_time = -math.inf
        self.last_claim_meeting = None

    @staticmethod
    def _available(observation, kind):
        return tuple(action for action in observation.actions if action.kind is kind)

    def _observe(self, obs):
        # Only DIRECT evidence can enter direct-observation state. Hearsay remains
        # a StructuredClaim and is considered separately by the voting rule.
        for event in obs.evidence:
            if event.provenance is not Provenance.DIRECT:
                continue
            payload = event.payload
            if isinstance(payload, PlayerSeen):
                prior = self.recent_players.get(payload.player_id)
                if prior is None or event.tick >= prior[1]:
                    self.recent_players[payload.player_id] = (payload, event.tick)
            elif isinstance(payload, BodySeen):
                prior = self.recent_bodies.get(payload.victim_id)
                if prior is None or event.tick < prior[1]:
                    self.recent_bodies[payload.victim_id] = (payload, event.tick)
            elif isinstance(payload, WitnessedElimination):
                self.witnesses[(payload.killer_id, payload.victim_id)] = (payload, event.tick)
        if obs.context.phase is Phase.ROAMING:
            self.last_room = obs.own.room
            self.last_exploration_tick = obs.tick

    def decide(self, obs: Phase3Observation) -> Intent:
        self._observe(obs)
        if not obs.own.active or not obs.actions:
            return Intent(IntentKind.WAIT)
        if obs.context.phase is Phase.DISCUSSION:
            if (self._available(obs, IntentKind.CLAIM) and
                    self.last_claim_meeting != obs.context.meeting_id):
                self.last_claim_meeting = obs.context.meeting_id
                return self._claim(obs)
            return Intent(IntentKind.WAIT)
        if obs.context.phase is Phase.VOTING:
            return self._vote(obs)
        if obs.context.phase is not Phase.ROAMING:
            return Intent(IntentKind.WAIT)
        if obs.own.role is Role.IMPOSTOR:
            return self._impostor(obs)
        return self._crew(obs)

    def _report(self, obs):
        actions = self._available(obs, IntentKind.REPORT)
        if actions:
            return actions[0]
        approaches = self._available(obs, IntentKind.GO_TO_BODY)
        if approaches:
            positions = {body.victim_id: body.position for body in obs.visible_bodies}
            selected = min(approaches, key=lambda action: (
                _distance(obs.own.position, positions[action.target]), action.target,
            ))
            if obs.navigation_status != 'moving' or obs.navigation_target != 'body:' + selected.target:
                return selected
        return None

    def _crew(self, obs):
        report = self._report(obs)
        if report is not None:
            return report
        interactions = self._available(obs, IntentKind.INTERACT)
        if interactions and obs.interaction_task is None:
            return interactions[0]
        if obs.interaction_task is not None or obs.navigation_status == "moving":
            return Intent(IntentKind.WAIT)
        remaining = [task for task in obs.own.tasks if task.progress < 1.0]
        destinations = {destination.id: destination for destination in obs.destinations}
        valid = {action.target: action for action in self._available(obs, IntentKind.GO_TO_TASK)}
        remaining = [task for task in remaining if task.console_id in valid]
        if remaining:
            for task in sorted(remaining, key=lambda task: task.task_id):
                if task.task_id not in self.task_order:
                    self.task_order[task.task_id] = self.rng.random()

            def cost(task):
                destination = destinations[task.console_id]
                distance = _distance(obs.own.position, destination.standing)
                if self.style.crew == "shuffled":
                    return self.task_order[task.task_id], task.task_id
                if self.style.crew == "social" and obs.visible_players:
                    distance += .35 * min(_distance(player.position, destination.standing)
                                          for player in obs.visible_players)
                return distance, task.task_id

            return valid[min(remaining, key=cost).console_id]
        # Finished crew patrol public rooms so undiscovered bodies can still be
        # found while teammates work. No hidden body/living-player list is used.
        return self._patrol(obs, rooms=True)

    def _impostor(self, obs):
        kills = self._available(obs, IntentKind.KILL)
        if kills and (self.style.impostor != "patient" or len(obs.visible_players) <= 2):
            seen = {player.player_id: player for player in obs.visible_players}
            return min(kills, key=lambda action: (
                _distance(obs.own.position, seen[action.target].position), action.target,
            ))
        # Reporting is a strategy choice. Hunter/patient sometimes report while
        # cooling down; self_report always pursues an observed body.
        if self.style.impostor == "self_report" or (obs.kill_cooldown or 0) > 6:
            report = self._report(obs)
            if report is not None:
                return report
        if obs.interaction_task is not None:
            return Intent(IntentKind.WAIT)
        if obs.visible_players and (obs.kill_cooldown or 0) <= 1.2:
            target = min(obs.visible_players, key=lambda player: (
                _distance(obs.own.position, player.position), player.player_id,
            ))
            # Every chase point was just legitimately observed. Replacements are
            # bounded in frequency; the trusted engine owns movement and routing.
            if (self._available(obs, IntentKind.GO_TO_LOCATION) and
                    (obs.navigation_status != "moving" or
                     obs.time_seconds - self.last_chase_time >= 2.4)):
                self.last_chase_time = obs.time_seconds
                return Intent(IntentKind.GO_TO_LOCATION, position=target.position)
        if obs.navigation_status == "moving":
            return Intent(IntentKind.WAIT)
        fakes = self._available(obs, IntentKind.FAKE_TASK)
        if fakes and (obs.kill_cooldown or 0) > 1.2:
            return fakes[0]
        return self._patrol(obs)

    def _patrol(self, obs, rooms=False):
        kind = IntentKind.GO_TO_ROOM if rooms else IntentKind.GO_TO_TASK
        choices = [action for action in self._available(obs, kind)
                   if action.target != self.last_patrol]
        if rooms:
            choices = [action for action in choices if action.target != obs.own.room]
        if not choices:
            return Intent(IntentKind.WAIT)
        # Stable candidate order before RNG use makes ordering bugs visible in
        # boundary tests and avoids accidental dictionary-iteration dependence.
        choices.sort(key=lambda action: action.target)
        selected = self.rng.choice(choices)
        self.last_patrol = selected.target
        return selected

    def _claim(self, obs):
        participants = set(obs.context.participants)
        others = sorted(participants - {obs.own.player_id})
        region = self.last_room or obs.own.room
        tick = min(self.last_exploration_tick, obs.tick)
        if obs.own.role is Role.IMPOSTOR:
            if self.style.impostor == "patient" or not others:
                # This is an explicit fabricated alibi chosen from public rooms,
                # not a leak of another player's actual hidden position.
                rooms = sorted(obs.map.rooms)
                draft = ClaimDraft(ClaimKind.WAS_IN_REGION, obs.own.player_id,
                                   self.rng.choice(rooms) if rooms else region, tick)
            else:
                draft = ClaimDraft(ClaimKind.SUSPECT_PLAYER, self.rng.choice(others), region, tick)
            return Intent(IntentKind.CLAIM, claim=draft)
        witnesses = [item for item in self.witnesses.values()
                     if item[0].killer_id in participants and item[0].killer_id != obs.own.player_id]
        if witnesses:
            event, tick = max(witnesses, key=lambda item: (item[1], item[0].killer_id))
            return Intent(IntentKind.CLAIM, claim=ClaimDraft(
                ClaimKind.SAW_ELIMINATION, event.killer_id, event.region, tick,
            ))
        nearby = self._near_body(obs)
        if nearby:
            player, tick, room = nearby[0]
            return Intent(IntentKind.CLAIM, claim=ClaimDraft(
                ClaimKind.SAW_PLAYER_NEAR_BODY, player, room, tick,
            ))
        if self.recent_bodies:
            body, tick = max(self.recent_bodies.values(), key=lambda item: (item[1], item[0].victim_id))
            return Intent(IntentKind.CLAIM, claim=ClaimDraft(
                ClaimKind.FOUND_BODY, body.victim_id, body.room, tick,
            ))
        sightings = [item for item in self.recent_players.values() if item[0].player_id in participants]
        if sightings:
            player, tick = max(sightings, key=lambda item: (item[1], item[0].player_id))
            return Intent(IntentKind.CLAIM, claim=ClaimDraft(
                ClaimKind.SAW_PLAYER, player.player_id, player.room, tick,
            ))
        return Intent(IntentKind.CLAIM, claim=ClaimDraft(
            ClaimKind.WAS_IN_REGION, obs.own.player_id, region, tick,
        ))

    def _near_body(self, obs):
        result = []
        for player, tick in self.recent_players.values():
            if player.player_id not in obs.context.participants or player.player_id == obs.own.player_id:
                continue
            for body, body_tick in self.recent_bodies.values():
                # Tick distance is a small bounded script heuristic, not access to
                # privileged death time. It compares two personal observations.
                if (abs(tick - body_tick) <= 15 and player.room == body.room and
                        _distance(player.position, body.position) <= 2.0):
                    result.append((player.player_id, tick, body.room))
                    break
        return sorted(result, key=lambda item: (-item[1], item[0]))

    def _vote(self, obs):
        votes = {action.target: action for action in self._available(obs, IntentKind.VOTE)}
        skip = self._available(obs, IntentKind.SKIP)
        fallback = skip[0] if skip else Intent(IntentKind.WAIT)
        votes.pop(obs.own.player_id, None)
        if not votes:
            return fallback
        scores = {player: 0 for player in votes}
        for event, _ in self.witnesses.values():
            if event.killer_id in scores:
                scores[event.killer_id] = 8
        for player, _, _ in self._near_body(obs):
            if player in scores:
                scores[player] = max(scores[player], 3)
        # One speaker contributes at most one weight to a candidate. A claim is
        # never converted into direct evidence or privileged truth.
        hearsay = {}
        for event in obs.evidence:
            claim = event.payload
            if (event.provenance is not Provenance.CLAIM or
                    not isinstance(claim, StructuredClaim) or claim.subject_id not in scores or
                    claim.speaker_id == obs.own.player_id):
                continue
            weight = {ClaimKind.SAW_ELIMINATION: 3, ClaimKind.SAW_PLAYER_NEAR_BODY: 2,
                      ClaimKind.SUSPECT_PLAYER: 1}.get(claim.kind, 0)
            pair = (claim.speaker_id, claim.subject_id)
            hearsay[pair] = max(hearsay.get(pair, 0), weight)
        for (_, target), weight in hearsay.items():
            scores[target] += weight
        threshold = {"evidence": 3, "cautious": 4, "skeptical": 6}[self.style.meeting]
        if obs.own.role is Role.IMPOSTOR:
            # Deception can use only public vote candidates and public claims.
            threshold = 1
            if self.style.impostor == "patient":
                return fallback
        ranked = sorted(scores, key=lambda player: (-scores[player], player))
        best = ranked[0]
        if scores[best] < threshold or (len(ranked) > 1 and scores[best] == scores[ranked[1]]):
            return fallback
        return votes[best]


# Explicit names for the eventual learner replacement, with one shared small
# implementation so own-role handling cannot silently bypass the input contract.
ScriptedCrewmateController = ScriptedController
ScriptedImpostorController = ScriptedController
