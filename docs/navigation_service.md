# Phase 2 — Reliable navigation service

## Starting audit and design

Started from `381936787236ada3ddfda3705521fe3d2b680864`, fast-forwarded from
`0b12c09` on clean `main`. Teammate commit `3819367` adds the separate source-backed
map, native-coordinate simulator/renderer, assets, builder and 15 map tests.
The initial full baseline was **152 passed, 2 failed**. The failures were Windows
CRLF conversion of source-hashed JSON and the byte-identical legacy snapshot;
`.gitattributes` preserves those repository bytes without weakening either test.

The current map has 63 unique task/utility destinations, 14 rooms, 7 hallway
regions, 42 wall chains, 19 prop records and 13 optional door polygons. Its 0.06-unit
grid has 67,924 connected nodes at radius 0.22. All standing points are legal and
inside their destination radii. This is the Phase 1.5 geometry accepted for
navigation; matching a commercial client is not claimed. Artwork/source geometry
and all destination metadata are preserved.

The teammate already fixed swept diagonal/endpoint legality and includes floor
boundaries in raycasts. The existing inspector follower is not a service: it lacks
cancellation outcomes, consistent interaction regions, bounded replanning and an
exhaustive fixed-timestep benchmark. The Gym wrapper terminates near a standing
anchor rather than the console's interaction region. No existing all-pairs
execution benchmark was found.

## Architecture choice

Considered retaining point-goal A* for every request, a new visibility/navmesh
graph, and cached reverse shortest-path fields on the existing validated grid.
Choose the third: measurements of existing long-route A* were about 89–271 ms;
many agents will revisit the same consoles. A bounded LRU cache amortizes one
multi-source Dijkstra search per destination over subsequent starts. New geometry
would duplicate the already tested collision model and introduce new failure modes.

Use the existing swept `AmongUsMap.move` physics. The service emits bounded native
displacements and consumes measured movement feedback; it never owns or overwrites
the player's position. A public environment motion method is shared by Gym steps,
manual control, the navigator and benchmarks. No new gameplay/training is involved.

Arrival uses the destination radius clipped to the walkable component containing
its established standing point. This accepts a real interaction region while
preventing success through an adjacent wall. Room targets use their walkable region;
location targets use an explicitly specified radius around a valid location.
All target fields and waypoints come from public geometry, not simulator truth.

## Planner and physical geometry

`NavigationPlanner` reuses every cardinal and diagonal edge from `AmongUsMap`.
Both endpoints being legal is insufficient: each edge must be covered by the
radius-inflated legal-center domain. Endpoint connectors and every simplified
segment undergo the same swept check. The geometry audit enumerates all such
edges, including rejected diagonals with two legal endpoints. The static graph is
built once; reverse multi-source Dijkstra fields seed all nodes inside the arrival
region. A tiny location region with no grid node gets a swept connector to its
exact center. Fields and region shapes each use a bounded LRU (default eight).

Paths follow decreasing field distance and then skip to the farthest visible
waypoint. Batched Shapely predicates preserve the same clearance rule. The last
segment stops at first entry into the arrival region, with a 1e-7-unit inward
numerical margin. This is a shortest grid field followed by geometric smoothing,
not a claim of globally shortest continuous paths. Disconnected known geometry
returns `NO_ROUTE`; invalid target identifiers/coordinates return
`INVALID_DESTINATION`.

The map's radius expansion approximates circles with 12 segments per quadrant;
its maximum radial chord error at radius 0.22 is about 0.000471 units. Navigation
uses that existing physical model exactly. Interaction disks use 64 segments per
quadrant (conservatively inscribed), clipped to legal-center space. For consoles,
the component containing the established standing anchor is selected. This is an
explicit controlled-simulator interaction rule, not verified commercial-client
line-of-sight/use logic. No console-center line-of-sight test is imposed because
some source consoles lie inside wall/prop geometry.

The teammate's rays already intersect walls, props and reconstructed floor
boundaries. Ray distances run to physical surfaces; center clearance also accounts
for player radius. The navigator needs neither rays nor sensor redesign. These
rays are not the future social-visibility projector.

## Public API and ownership

```python
from among_us_map_simulation import AmongUsMapEnv
from navigation_service import (
    ControllerConfig, NavigationPlanner, NavigationService, NavStatus, NavTarget,
)

env = AmongUsMapEnv()
env.spawn((-.7, -2.8))  # Episode initialization only.
planner = NavigationPlanner(env.map)  # Share across players using this known map.
nav = NavigationService(planner, ControllerConfig(speed=env.speed))
nav.navigate(env.position, NavTarget.room("Electrical"))

while nav.status is NavStatus.MOVING:
    displacement = nav.command(env.position, env.dt)
    if nav.awaiting_feedback:
        actual_position, collided = env.advance_motion(displacement, env.dt)
        nav.feedback(actual_position, collided)
print(nav.status)
env.close()
```

Target constructors:

- `NavTarget.task(destination_id)`: any of the 63 unique task/utility IDs. This
  names a physical destination, not an assigned task or task-completion action.
- `NavTarget.room(region_name)`: walkable part of a room or hallway region.
- `NavTarget.location((x, y), radius=0.1)`: valid native coordinate and arrival radius.

All service coordinates/displacements use native x-right, y-up units. Gym actions
retain their original y-down direction and full-speed normalization. Do not pass
service displacements directly to Gym `step`: that legacy normalization would
overshoot short final steps. `advance_motion` is the shared speed-bounded movement
primitive used by Gym, the inspector and this service. It calls `map.move` and
updates actual player position, elapsed time and traveled distance. It does not
generate ray observations, Gym rewards or standing-anchor termination; the service
owns interaction-region completion. `spawn` is the shared episode initializer
under Gym `reset`, never a recovery tool. Do not mix Gym termination with service
termination inside one episode.

The controller never receives an environment or writes a player's coordinates.
Each emitted displacement is at most `speed * dt`, capped at the next waypoint.
Every emitted motion must receive feedback before another command. A zero return
after terminal status or a replan has no pending motion; check `awaiting_feedback`.
Bad timestep, missing/duplicate feedback or speed-violating feedback raises a
programming-contract error rather than silently corrupting state.

## Cancellation, replanning and outcomes

`cancel()` discards the route and pending command without moving the player.
`navigate(actual_position, new_target)` replaces any previous intention and resets
its counters. Do intention changes between physics ticks; if cancelling an issued
but unconsumed command, the runner must discard that displacement too.

`replan(actual_position, planner=new_known_map_planner, reason="observed_change")`
can replace geometry explicitly. A new planner invalidates old cached fields;
player radius must stay the same during a request. Geometry must reflect publicly
known or legitimately observed changes, not a hidden simulator door query. Maps
are immutable by convention; do not modify their arrays in place. No door timers,
sabotage system or multi-player obstacle avoidance was added.

The controller checks its next segment before issuing motion. An invalid segment
triggers a bounded replan. Progress is a new best remaining route distance, not
mere displacement: back-and-forth oscillation cannot reset the stall timer forever.
One second without meaningful progress triggers recovery, with at most two replans
per request and a 60-second simulated timeout. These named defaults are configurable;
there is no teleport recovery. Timers advance from consumed physics feedback.

| Status | Meaning |
|---|---|
| `IDLE` | No intention issued |
| `MOVING` | Executing or deterministically replanning |
| `SUCCESS` | Actual position is in the valid arrival region |
| `CANCELLED` | Caller discarded the intention |
| `INVALID_START` | Actual position is non-finite or outside known legal geometry |
| `INVALID_DESTINATION` | Unknown ID, illegal location or empty arrival region |
| `NO_ROUTE` | No validated grid/endpoint route in known geometry |
| `STUCK` | Recovery budget exhausted |
| `TIMEOUT` | Simulated execution time exhausted |

Terminal states persist until a new request or cancellation. `last_replan_reason`
records `requested`, `route_invalidated`, `arrival_missed`, `no_progress` or the
caller's explicit reason. Diagnostics include elapsed time, travel, planned length,
planning wall time, controller steps, collisions, blocked steps and replan count.

## Information boundary

`navigation_service.py` imports only public map geometry and numerical libraries;
it does not import `social_deduction.truth`, observation helpers or training labels.
Its inputs are a public destination, own position, elapsed motion interval and own
collision feedback. It knows console locations, not private assignments or other
players' states. The future trusted runner should own environment and planner,
accept macro-intentions, and expose approved outcomes to the actor. Do not hand a
policy simulator references, arbitrary diagnostic dictionaries or omniscient map
updates. Phase 1 interfaces/tests remain unchanged; Phase 3 wrapper-level leakage
tests are still required.

## Benchmark methodology and reproduction

The fixed configuration is seed `20261002`, radius `0.22`, speed `2.5`, dt `1/30`,
grid `0.06`, timeout `60`, stall threshold `1`, maximum replans `2`, cache size `8`.
There are **63 destinations**, hence **3,906 directed pairs**. Sources spawn at
their legal standing anchors. Different IDs may share an interaction region;
already-inside starts correctly succeed with zero motion and remain in the count.

The random suite has **48 continuous spawns per each of 21 room/hallway regions**,
or **1,008 cases**. Half sample regional grid cells uniformly, half sample cells
within 0.12 units of the legal-center boundary. Valid sub-cell jitter avoids a
grid-node-only evaluation. Target choices cycle nearest, farthest and uniform
console by Euclidean distance. This covers near/far travel and narrow approaches;
the exhaustive pairs cover every named task approach and connecting chokepoint.
Region, destination category and spawn-stratum counts are reported separately.

Each case initializes an episode, plans once, and executes fixed-rate commands
through normal `advance_motion` physics until a real arrival or explicit failure.
The benchmark independently checks every physical segment, speed bound, final
region membership and console radius. No player-position assignment occurs during
navigation. Sensor/reward computation is omitted because it is unused, not because
collision or control is bypassed. Fault-injection tests separately cover blocked
motion, oscillation, timeout, changing intentions and known-door disconnection.

Cases are grouped by destination to reuse distance fields; this is intentionally
a warm-cache throughput workload. Cold/cached planning distributions and field
build totals are reported separately. Arbitrary interleaving across more than eight
distinct targets can rebuild fields. Controller timings include its own checks;
execution wall time also includes independent benchmark assertions. Simulated
travel time and measured wall time are separate columns. Path efficiency means
planned executable route length / actual travel, not optimality against a separate
continuous shortest-path oracle.

```powershell
# Full acceptance run; writes cases.csv, summary.json and README.md
.venv\Scripts\python.exe benchmark_navigation.py

# Full regression suite and focused failure-contract tests
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m pytest test_navigation_service.py -q

# Rerun an exact case ID from cases.csv (same seed/config)
.venv\Scripts\python.exe benchmark_navigation.py --case "random:Electrical:001" --output .pytest_cache/nav_case

# Fast smoke subset; explicitly reports SUBSET_ONLY, never a full gate pass
.venv\Scripts\python.exe benchmark_navigation.py --max-cases 20 --output .pytest_cache/nav_smoke

# Interactive service: Tab selects a console, click selects a location,
# Space starts, WASD/arrows cancel and resume manual motion.
.venv\Scripts\python.exe among_us_map_simulation.py
```

CSV contains every case's identity, coordinates, source region, target, outcome,
failure reason, plan/execute times, planned/actual length, efficiency, collisions,
blocked steps, replans, final distances, steps and seed. JSON contains environment
versions, parameters, map/code byte hashes, deterministic case-manifest hash,
timing distributions, regional summaries and ten longest routes. `base_git_commit`
is the checked-out parent while uncommitted implementation files were benchmarked;
the recorded source hashes identify the tested implementation precisely. Wall times
are machine-dependent; source byte hashes can differ with checkout line endings.

## Measured acceptance result (2026-10-02)

Final implementation commit: `9dcc0a217da468c6d906816937fbd5bd11af128f`.

**PHASE 2 COMPLETE.** Full regression: **197 passed, one existing Gym registration
warning**; 43 navigation tests passed; `pip check` and `git diff --check` passed.
The original 139 Phase 1/legacy checks and 15 map checks remain green.

| Metric | Measured result |
|---|---|
| Directed pairs | 3,906 / 3,906 successful |
| Random spawns | 1,008 / 1,008 successful |
| Total / physical steps | 4,914 / 4,914; 1,535,435 steps |
| Collisions / blocked steps / replans | 0 / 0 / 0 |
| Failure categories / repeated failing regions | None |
| Mean / p95 plan cost | 63.84 / 206.63 ms |
| Mean cold / cached plan cost | 234.98 / 63.58 ms |
| Controller / physics time per step | 179.71 / 90.60 microseconds |
| Mean execution wall / simulated time | 0.1033 s / 10.4154 s per case |
| Total benchmark wall time | 826.13 s |
| Fields built / total field build time | 63 / 8.81 s |
| Planner adjacency setup (excludes map construction) | 0.143 s |

The longest class was cross-ship O2 to Reactor. Worst length case
`random:O2:007` traveled 53.55 units in 21.77 simulated seconds and passed. No
regional success rate hid a chokepoint failure; all region/category/stratum groups
are 100%. Some different destination IDs share arrival space: 150 cases start
already inside their target region (see CSV for exact counts if regenerated).
Every nontrivial case physically executes; early success only uses actual region
membership.

Reproduce the longest case with:

```powershell
python benchmark_navigation.py --case "random:O2:007" --output .pytest_cache/nav_longest
```

Measured timings include periods of concurrent regression/documentation work.
They are not dedicated microbenchmarks. Cached route extraction and geometric
simplification still cost time; planning happens at intention changes/recovery,
not each frame. A future high-throughput match runner may profile that path before
adding asynchronous planning or more caching. Source hashes describe the measured
files; the JSON explicitly annotates a subsequent docstring-only clarification of
the episode spawn API. No executable statements changed after measurement.

## Acceptance and remaining limits

The gate is stronger than the suggested aggregate 99%: **all 4,914 default cases
must succeed with all physical assertions passing**, plus explicit failure-path
regressions and the existing test suite. Partial CLI runs cannot claim gate success.
Committed [benchmark evidence](benchmarks/phase2/README.md) records the measured
result. This empirical gate covers this static map and configuration; it is not a
proof for every continuous start, player radius or future dynamic map.

No full social match, player-to-player collisions, sabotage, minigame learning or
strategic training is included. The map remains a historical-source approximation
with labeled estimated stations. Room arrival can mean reaching its doorway-side
walkable edge; callers needing a particular interior point should use a location
target. Phase 3 should integrate this service through the trusted runner and test
actual visibility, event publication, tick ordering and legal task interaction.
