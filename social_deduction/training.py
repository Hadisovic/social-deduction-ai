"""Privileged training/evaluation outputs. Not exported through the actor API."""

from dataclasses import dataclass

from .actor import Role
from .truth import WorldTruth


@dataclass(frozen=True, slots=True)
class RoleLabels:
    match_id: str
    tick: int
    roles: tuple[tuple[str, Role], ...]


def role_labels(world: WorldTruth) -> RoleLabels:
    return RoleLabels(world.match_id, world.tick,
                      tuple(sorted((p.identity.player_id, p.role) for p in world.players)))
