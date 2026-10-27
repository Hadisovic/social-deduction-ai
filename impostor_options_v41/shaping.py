"""Arm proximity history when killing becomes available, not at crowded spawn."""
from social_deduction.actor import Phase
from impostor_options_v4.shaping import PursuitPotential as OriginalPotential


class PursuitPotential(OriginalPotential):
    """Same0.05 lifetime credit/terminal correction; cooldown never refills credit.

    Initial nearby crew cannot saturate the frontier during the45s cooldown.
    Cooldown suspends tracking and clears only geometry history. Once ready, the
    first visible observation establishes a non-paying baseline; actual subsequent
    own approach may pay. A successful kill starts a new approach period without
    replenishing the lifetime budget. Meetings do not reset its frontier/budget.
    """
    def advance(self,obs):
        if obs.context.phase is Phase.FINISHED or not obs.own.active:
            return super().advance(obs)
        if obs.context.phase is not Phase.ROAMING:
            return super().advance(obs)
        if obs.kill_cooldown is None or obs.kill_cooldown>1e-9:
            self.best=0.;self.previous={};self.own=None;self.last_time=obs.time_seconds
            return 0.
        return super().advance(obs)
