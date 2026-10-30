# Jane Impostor: independent model, reviewed reference

Patrick Jane is an architectural reference only. Its actual pushed source has
been reviewed; no network/weights are copied or assumed compatible with the old
75K/five-player impostor.

The existing crew encoder omits impostor cooldown/team/vent-action information
needed for strategic impostor control. Its five interaction outputs are crew
actions, not kill/sabotage/vent decisions. A new role-specific interface and
policy design are required. Current A2 opponents remain scripts.

Follow [PLAN.md](PLAN.md) and the shared rules/interface documents. The model,
output layout and folder structure remain provisional. Native configuration
translation works separately from the still-unimplemented learned pair.
