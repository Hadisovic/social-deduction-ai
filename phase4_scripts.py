"""Small actor-safe communication variants; movement/rules stay Phase 3 scripted."""
from phase3_bots import ScriptedController
from social_deduction.actor import Role
from social_deduction.phase3_api import ClaimDraft, ClaimKind, Intent, IntentKind


class StudyController(ScriptedController):
    """Vary claim formats so an accusation token is not an impostor label.

    Crew preserve witnessed elimination/body evidence. Otherwise they alternate
    truthful self-alibis, sightings, and explicitly uncertain accusations. Impostors
    fabricate from public identities/rooms and their own observation timestamps.
    No world, other controller or hidden player state is accessible here.
    """
    def _claim(self, obs):
        original = super()._claim(obs)
        if obs.own.role is Role.CREWMATE and original.claim.kind in (
            ClaimKind.SAW_ELIMINATION, ClaimKind.SAW_PLAYER_NEAR_BODY, ClaimKind.FOUND_BODY
        ):
            return original
        others = sorted(set(obs.context.participants) - {obs.own.player_id})
        tick = min(self.last_exploration_tick, obs.tick)
        room = self.last_room or obs.own.room
        draw = self.rng.random()
        if draw < .4 or not others:
            if obs.own.role is Role.IMPOSTOR:
                room = self.rng.choice(sorted(obs.map.rooms))
            claim = ClaimDraft(ClaimKind.WAS_IN_REGION, obs.own.player_id, room, tick)
        elif draw < .7:
            if obs.own.role is Role.CREWMATE and self.recent_players:
                seen, seen_tick = max(self.recent_players.values(), key=lambda item: item[1])
                claim = ClaimDraft(ClaimKind.SAW_PLAYER, seen.player_id, seen.room, seen_tick)
            else:
                claim = ClaimDraft(ClaimKind.SAW_PLAYER, self.rng.choice(others),
                                   self.rng.choice(sorted(obs.map.rooms)), tick)
        else:
            kind = ClaimKind.SUSPECT_PLAYER
            if obs.own.role is Role.IMPOSTOR and draw > .9:
                kind = ClaimKind.SAW_ELIMINATION
            claim = ClaimDraft(kind, self.rng.choice(others), room, tick)
        return Intent(IntentKind.CLAIM, claim=claim)
