# Jane: one shared game contract, two learned roles

**Status: provisional shared workspace.** Draft settings, strict validation,
inventory and plans are implemented. Both models remain in development. Patrick
Jane is a reference, not a copied base or a finalized architecture. When the owner
announces the teammate's progress push, pull/read it before adapting either role.
The currently fetched repository excludes that code. No new network was trained.
A configuration supporting 15 slots does not make the old five-player engine
support 15 players.

## Folder ownership

```text
jane/
  shared/        common settings, compatibility, semantic contract, catalog
  configs/       editable match profiles; both roles use the SAME file/hash
  crewmate/      Patrick Jane source registration and future adapter
  impostor/      Jane Impostor registration and development plan
docs/jane/
  GAME_RULES.md  shared classic rules, settings and task inventory
  INTERFACE.md   observation/action/privacy/checkpoint agreements
  FUTURE_WORK.md disabled mechanics and optional-role roadmap
  CHECKLIST.md   completed work and remaining integration gates
  VERIFICATION.md commands and observed verification results
```

Historical packages, checkpoints, raw replays, studies and Navigation V2 work
stay in their original locations. Nothing is moved or reinterpreted.

## Choose a profile

| File | Population | Intended use | Runtime status |
|---|---|---|---|
| `historical_5p.json` | 4 crew + 1 impostor | Preserve V2.2 settings | Constructor adapter verified separately; no new runner |
| `jane_12p_base.json` | 10 crew + 2 impostors | Shared initial Jane target | Needs current Jane source and both adapters |
| `jane_15p_base.json` | 13 crew + 2 impostors | Capacity/custom-lobby target | Needs variable-population engine verification |
| `future_15p_full_classic.json` | 12 crew + 3 impostors | Future full classic mechanics example | Planning only; extensions enabled explicitly; not runnable here |

Base Jane profiles disable vents, sabotage, doors, devices, Clean Vent and
optional roles. Those are staged learning variants, not a claim of full classic
fidelity. The future full-classic profile documents the destination. It does not
implement the mechanics.

Edit a copy of a profile, give it a new `profile_id`, validate it, then register
its fingerprint. Never edit the profile used by a frozen study in place.

## Commands: inspect configuration, not gameplay

From the repository root with Python 3.12+:

```powershell
python -m jane jane/configs/jane_12p_base.json
python -m jane jane/configs/jane_15p_base.json
python -m jane jane/configs/historical_5p.json --engine historical-v2.2
python -m jane jane/configs/jane_12p_base.json --engine patrick-jane --check-pair
```

The last command intentionally exits **2** until actual source, weights and
adapters pass registration. Unknown keys, invalid values and unsupported runtime
settings fail explicitly. The CLI never starts a match, imports a trainer or
loads a model. The historical adapter needs the existing pinned dependencies.

The current game viewer remains `run_impostor_shared.py` on the preserved
five-player implementation. This branch does not advertise a twelve-player
viewer that does not exist.

## Work together

```powershell
git fetch origin
git switch --track origin/codex/jane-shared-foundation
# Then create your own codex/jane-crewmate-* or codex/jane-impostor-* branch.
```

Read [the shared rules](../docs/jane/GAME_RULES.md),
[interface agreement](../docs/jane/INTERFACE.md),
[impostor plan](impostor/PLAN.md) and [checklist](../docs/jane/CHECKLIST.md).
Shared settings are not shared trained weights. Every actor keeps private state.
These folders and interfaces are editable proposals; review the teammate's work
before freezing the joint design. Keep crew and impostor work equally separate.
