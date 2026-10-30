"""Simulated-time reward accounting. No action preferences or world observations."""
import math


def ready_seconds(elapsed,cooldown,*,active_before=True,active_after=True,
                  roaming_before=True,roaming_after=True,vented_before=False,
                  vented_after=False,transition_before=False,transition_after=False,
                  killed=False,kill_cooldown=45.):
    if any(not math.isfinite(x) or x<0 for x in (elapsed,cooldown,kill_cooldown)):
        raise ValueError('Invalid simulated time')
    eligible=(active_before and active_after and roaming_before and roaming_after
              and not vented_before and not vented_after and not transition_before and not transition_after)
    return max(0.,elapsed-(kill_cooldown if killed else cooldown)) if eligible else 0.
