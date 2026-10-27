"""Opponent-only safeguards. This module has no game/truth/controller registry."""
from dataclasses import dataclass
import math
from crewmate_v2.api import validate
from social_deduction.actor import Phase, Role
from social_deduction.phase3_api import Intent, IntentKind, WitnessedElimination, intent_allowed


@dataclass(frozen=True)
class Safeguards:
    bodies: bool = True
    eyewitness: bool = True

    def __post_init__(self):
        if type(self.bodies) is not bool or type(self.eyewitness) is not bool:
            raise ValueError('Safeguards must be boolean')


class OpponentGuard:
    def __init__(self, config=Safeguards()):
        self.config = config
        self.owner = None
        self.match_id = None
        self.tick = -1
        self.witnesses = {}
        self.pursuing = None
        self.records = []  # Private engineering audit; never an actor input.

    def apply(self, envelope, proposed):
        obs = validate(envelope).actor
        if self.owner is not None and (self.owner != obs.own.player_id or self.match_id != obs.match_id):
            raise ValueError('Opponent memories cannot change owners or matches')
        if obs.tick < self.tick:
            raise ValueError('Opponent memory cannot travel backwards')
        self.owner = obs.own.player_id; self.match_id = obs.match_id; self.tick = obs.tick
        if obs.own.role is not Role.CREWMATE:
            raise ValueError('Safeguards apply only to crewmate opponents')
        if not obs.own.active or obs.context.phase is Phase.FINISHED:
            self.pursuing = None
            return Intent(IntentKind.WAIT)
        # Validated event provenance distinguishes direct evidence from claims.
        for e in obs.evidence:
            if type(e.payload) is WitnessedElimination:
                self.witnesses[e.payload.victim_id] = (e.tick, e.payload.killer_id)
        action = proposed
        reason = None
        if self.config.eyewitness and obs.context.phase is Phase.VOTING:
            for _, killer in sorted(self.witnesses.values(), reverse=True):
                vote = Intent(IntentKind.VOTE, killer)
                if intent_allowed(obs, vote):
                    action, reason = vote, 'direct_witness_vote'
                    break
        if obs.context.phase is Phase.ROAMING:
            bodies = sorted((b for b in obs.visible_bodies if self.config.bodies or
                (self.config.eyewitness and b.victim_id in self.witnesses)),
                key=lambda b: (math.hypot(b.position.x-obs.own.position.x, b.position.y-obs.own.position.y), b.victim_id))
            if bodies:
                body = bodies[0]
                report = Intent(IntentKind.REPORT, body.victim_id)
                approach = Intent(IntentKind.GO_TO_BODY, body.victim_id)
                if intent_allowed(obs, report):
                    action, reason = report, 'visible_body_report'
                    self.pursuing = None
                elif intent_allowed(obs, approach):
                    self.pursuing = body.victim_id
                    # Keep the existing route; suppress competing task proposals.
                    action = (Intent(IntentKind.WAIT) if obs.navigation_status == 'moving'
                        and obs.navigation_target == 'body:' + body.victim_id else approach)
                    reason = 'visible_body_approach'
            elif self.pursuing is not None:
                # An out-of-view/resolved body is no longer a valid pursuit target.
                self.pursuing = None
                action, reason = Intent(IntentKind.CANCEL_NAVIGATION), 'body_pursuit_ended'
        else:
            self.pursuing = None
        if action is None or (action.kind is not IntentKind.WAIT and not intent_allowed(obs, action)):
            action = Intent(IntentKind.WAIT)
        if reason:
            self.records.append(dict(tick=obs.tick, proposed=proposed, executed=action,
                                     reason=reason, changed=proposed != action))
        return action
