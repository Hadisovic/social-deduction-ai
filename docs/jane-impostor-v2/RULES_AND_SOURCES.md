# Classic scope, sources and remaining differences

This is a versioned training approximation of Classic Skeld, not a verified
retail-identical game. Do not label every setting or task minigame complete.

## Verified reference facts

- Innersloth describes **4–15 total players**, crew task/ejection victories,
  reports/meetings/votes, impostor kills, faking, vents and door sabotage:
  https://www.innersloth.com/games/among-us/
- The official historical 2018.9.6 note explicitly gives the **one short task
  fallback when all task counts are zero**, ghost chat isolation and dead
  impostor sabotage (the latter is deferred here):
  https://www.innersloth.com/bug-fixes-in-among-us-2018-9-6/
- That same historical note used 6/8-player minima for 2/3 impostors. Our immutable
  shared validator uses conservative **7/9** minima. Current retail build/version
  limits require a direct capture or current authoritative specification before
  relaxing this restriction. We do not claim the conservative limits cover every
  retail-supported configuration.
- Historical `ShipStatus.Begin` reference selects common tasks once, then allocates
  long/short per player; the task-list routine tracks used task types:
  https://github.com/polusgg/assets-unity/blob/f26e7d9948e3a438853d3f70aff4a9d893b0cb9d/Assets/OfficialAssets/Scripts/ShipStatus.cs
  This is a community-hosted historical reference, not a current official retail
  source release. The allocator here is our own implementation.

## Task pool decision

Use the pinned Skeld task metadata already used by both developers. Its category
labels differ from some older draft tables. Do not silently change old categories
or checkpoints to match an unverified website/build. Common pool: Fix Wiring,
Swipe Card. Long pool includes Align Engine Output, Clear Asteroids, garbage/chute,
Fuel Engines, Inspect Sample, Start Reactor and Submit Scan. Short pool includes
Calibrate Distributor, Chart Course, Clean O2 Filter, Divert Power, Prime Shields,
Stabilize Steering, Unlock Manifolds and Upload Data. Vent Cleaning is excluded
until vent eviction/occupancy semantics are implemented.

Empty Chute and Empty Garbage are canonical alternatives, not two independent
assignments for one player. Legitimate task stages can revisit consoles. Each
player receives distinct mutable task objects; common plans do not share progress.

## Implemented core versus future work

| Area | Current core | Still deferred / approximate |
|---|---|---|
| Map/movement | Existing Skeld geometry, walls/objects, A-star goal execution | Exact retail collision/animation calibration for every prop |
| Roles/populations | Base crew/impostor, conservative 4–15-player lobby profiles | Every special role and role probability/effect |
| Tasks | Configurable counts, unique families, owned staged progress, crew ghosts | Actual minigame controls/difficulty and current-build category verification |
| Kills | Role/range/visibility legality, cooldown, witnesses, corpses | Short/long distance calibration; retail-perfect animation timing |
| Meetings | Reports, emergency limits/cooldown, discussion, ballots, skip/ties/ejection, confirm setting | Inherited A2 mask excludes self-voting; infinite voting; retail-perfect UI/network timing |
| Information | Own observations/history, separate ghost progression, anonymous ballots, task-bar modes | Security/Admin observations and device strategy |
| Lights | Enabled; role-local reduced crew sight and repairs | Full retail switch-panel minigame |
| Doors | Native room closures/collision/expiry/cooldown, animated viewer | Exact retail animation cadence and every door/sabotage interaction |
| Vents | Enter/move/exit, private underground observation, linked network | Clean Vent eviction and specialist role abilities |
| Other sabotage | Disabled to keep both learned roles compatible | O2, reactor, communications and dead-impostor assistance |
| End conditions | Crew task/ejection, impostor parity; technical cuts separately labelled | Unimplemented sabotage victory types |
| Host settings | Existing immutable validated ranges; native medium kill distance | 0.25 vision currently fails native report-range validation; zero/infinite voting; unsupported mechanics |
| Language | Structured claim actions, event provenance | Free-form dialogue/LLM deception |

A new capability needs rules, actor observations, legal candidates, privacy tests,
both-role response tests, viewer support, a versioned receipt and new evaluation.
Keep old settings/checkpoints reproducible instead of retroactively changing them.
