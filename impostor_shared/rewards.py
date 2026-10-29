"""Explicit new semantics. Historical V2 remains in impostor_research unchanged."""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class Rewards:
    version: str = 'impostor-rewards-v4'
    win: float = 10.
    loss: float = -10.
    timeout: float = -12.
    successful_kill: float = 3.
    own_ejection: float = -6.
    cooldown_ready_per_second: float = -.02
    pursuit_scale: float = .05
    pursuit_min_progress: float = .05

    def __post_init__(self):
        if self.version != 'impostor-rewards-v4':
            raise ValueError('Unknown reward version; use the historical V2 environment for V2')
        if not all(math.isfinite(v) for k,v in asdict(self).items() if k != 'version'):
            raise ValueError('Rewards must be finite')
        if self.cooldown_ready_per_second > 0 or self.pursuit_scale < 0 or self.pursuit_min_progress <= 0:
            raise ValueError('Invalid time penalty or pursuit configuration')


@dataclass(frozen=True)
class Options:
    version: str = 'impostor-actions-v4'
    observation_version: str = 'impostor-observations-v4'
    travel_seconds: float = 3.

    def __post_init__(self):
        if (self.version, self.observation_version) != ('impostor-actions-v4', 'impostor-observations-v4'):
            raise ValueError('Unknown action/observation version')
        if not math.isfinite(self.travel_seconds) or self.travel_seconds <= 0:
            raise ValueError('Invalid option duration')


def ready_seconds(elapsed, cooldown, *, roaming, active, entered_meeting=False, killed=False, kill_cooldown=45.):
    """Overlap of an eligible physical interval with zero cooldown, including crossings.

    A successful kill resets cooldown at the start of its physics interval. A report
    enters a meeting before physics, so its interval is excluded. Termination that
    consumes no engine ticks consumes no time here. No decision-count multiplier.
    """
    if elapsed < 0 or cooldown < 0:
        raise ValueError('Time cannot be negative')
    if not roaming or not active or entered_meeting:
        return 0.
    anchor = kill_cooldown if killed else cooldown
    return max(0., elapsed - anchor)
