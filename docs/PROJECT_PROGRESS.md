# Project Progress & Technical Evolution

> **Canonical Document**: Detailed technical history, experimental results, architectural pivot, and roadmap status for the **Social Deduction AI** project.

## 2026-10-02: Phase 4 event memory and belief experiment COMPLETE

Started from clean, synchronized `main` at `e8c5d664a3f17684c3fd844a4ef27ad089b65b12`;
the fresh baseline was 291 passing tests. Memory/data checkpoint: `68780df`;
training/evaluation/observer checkpoint: `e16d8ea`. The current handoff records
the final result revision. No Phase 1/2/3 engine, geometry, navigation, controller
or original renderer file changed.

The actor-safe memory retains canonical DIRECT/PUBLIC/CLAIM events, compressed
sightings, exact observed room/time support, last sightings, claims, public votes
and conservative contradictions across meetings. Four historical non-self
identities remain candidates, including ejected players. Training labels are
joined separately after encoding; hidden state never enters model features.

| Acceptance | Result |
|---|---|
| Full tests | **327 passed**, one existing Gym spec warning |
| Dataset | **1,400 matches / 5,600 focal crews / 64,097 samples** |
| Train / validation / ID test / patient-family test | **800 / 200 / 200 / 200 matches** |
| Failures / timeouts / illegal actions / navigation failures | **0 / 0 / 0 / 0** |
| Exact independent feature + match replays | **5/5** across all partitions |
| Selected model | **2,849-parameter DeepSets scorer, seed 29**, epoch 23 |
| ID calibrated learned accuracy / NLL / Brier / ECE | **70.54% / .5955 / .3189 / .0129** |
| Patient-family calibrated learned | **67.41% / .6375 / .3442 / .0134** |
| Patient-family calibrated rules | **63.15% / .8005 / .4019 / .0501** |
| Exact selected-model retraining | **Identical state dict and temperature** |
| CPU inference, one representative memory | **.246 ms mean**, including feature extraction |
| Real demo matches | **7, 15, 25**, local source artwork and independent belief panel |

Patient-family NLL difference versus calibrated rules is -.1630 (95% match-bootstrap
interval [-.2272,-.1090]). Validation-temperature calibration slightly worsens both
final sets; the released T=1.1016 remains the validation-only choice. Early states
are near chance and maximum entropy; later scripted meetings are often easy. The
corrected current-only ablation retains the live meeting roster but removes
historical claims, votes and sightings; its NLL is 1.2455 (ID) / 1.2546 (patient),
near the uniform prior. No-claims and collapsed-provenance ablations are smaller
degradations. This is evidence within a narrow scripted
distribution, not general social intelligence or improved team win rate.

```powershell
python run_phase4.py
python train_phase4.py
python evaluate_phase4.py
```

The release checkpoint is 15,722 bytes; regenerable shards are ignored. A mixed-DPI
window clipping issue was fixed at the composite display without changing the
frozen renderer. Dataset audit corrected an overstrict timestamp check: inputs
sampled before a terminal killing action can legitimately share its timestamp.
The TIMEOUT guard casing was corrected; all 1,400 existing outcomes were verified
non-timeout before an explicit acceptance-only source amendment, with no array
changes. The amendment and independent replays preserve provenance. A separate
current-context correction updated 26,192 `x_current` public-absence fields to
retain the live meeting roster, followed by five exact feature replays and
retraining/evaluation of only that ablation. Full-model inputs and checkpoint
remain unchanged; see `benchmarks/phase4/current_context_correction.json`.

See [implementation and contract](belief_model.md), [results and plots](benchmarks/phase4/README.md),
and [handoff](../HANDOFF.md). The `predict(actor_memory)` interface is ready for the
separately scoped **Phase 5 strategic policy**. No learned action selection was added.

## 2026-10-02: Phase 3 five-player simulator COMPLETE

Implementation revision: `c2c6ea269fffbefc96993acead6f6127806c354b`.

Started from clean `main` at `deae77c3dda88290659abc3659d5eb472c07aa07` after
fetch/fast-forward verification. The pre-change baseline was **197 passed**.
Phase 1's original interface and Phase 2 navigation/geometry remain unchanged.
The new fixed-tick engine separates physical entities, trusted projection,
immutable actor packets, scripts and spectator rendering. Four crew and one
impostor now play full matches through timed private tasks, local occluded sight,
kills, bodies, reports, structured claims, public voting and centralized wins.

```powershell
python run_phase3.py
python run_phase3_batch.py --matches 1000 --workers 8
```

| Verification | Result |
|---|---|
| Full regression | **291 passed**, one existing Gym spec warning |
| Added checks | **93 Phase 3 tests + 1 legacy-viewer regression** |
| Primary matches / independent replays | **1,000 / 1,000**, seeds 0–999 |
| Crashes / invalid states / timeouts | **0 / 0 / 0** |
| Replay differences / navigation failures / illegal actions | **0 / 0 / 0** |
| Crew task / impostor ejection / impostor parity wins | **870 / 41 / 89** |
| Simulated duration / steps, per match | **58.932 s / 294.749** |
| Worker runtime / full batch wall time | **5.724 s mean / 1,290.18 s**, eight concurrent workers, including replays |
| Real desktop matches inspected | **7, 15, 25**, covering all three winning reasons |

The match deadline is an explicit draw, and the fixed eight-task quota survives
elimination. Unfinished work is reassigned only after absence becomes public at
a meeting, avoiding a hidden-death leak. Meetings freeze positions and timers;
no gameplay teleport is used. Claims retain authenticated speaker/delivery and
separate asserted time; deception is legal, and hearsay is not direct evidence.
Full input/action paired-world tests cover hidden-role/task/position/death/order
changes, future events, native occlusion, provenance and spectator isolation.

The owner's requested clone character/task images are used in a pinned, ignored
local pack: 119 selected PNGs, ten colors, body art and nine task illustrations.
The public repository includes the exact manifest, loader and original fallback,
without republishing ripped artwork. The requested Impostor Skeld files already
matched all five checked-in sources byte-for-byte. The Agentic-Among-Us source
was inspected but provided procedural rendering/reference screenshots rather
than a usable licensed sprite pack. See the [full asset audit](phase3_asset_provenance.md).

Visual QA caught indexed PNG scaling, now normalized without requiring a display.
Concurrent regression exposed a legacy training-viewer close race; one exit guard
and a deterministic regression fix it without weakening old tests. All final
simulation source hashes remained unchanged throughout the 1,000-seed acceptance
run. No learned suspicion, event-memory model, strategic RL, LLM meetings, vents
or sabotage were added. The completed environment is ready for separately scoped
**Phase 4: Event Memory + Suspicion / Belief Model**.

See [architecture/rules/controls](social_simulator.md),
[acceptance metrics and representative timelines](benchmarks/phase3/README.md),
and [current handoff](../HANDOFF.md). Earlier sections below retain historical
milestone results; their “next phase” statements describe those earlier dates.

## 2026-10-02: Phase 2 reliable navigation COMPLETE

Clean `main` was fast-forwarded from `0b12c09` to the teammate's map commit
`381936787236ada3ddfda3705521fe3d2b680864` before implementation. Phase 1.5 is
accepted/frozen for controlled navigation: 63 destinations, 14 rooms, 7 corridor
regions, 67,924 connected grid nodes. No geometry/artwork/destination redesign was
needed. Windows CRLF conversion caused two initial byte-provenance test failures;
`.gitattributes` preserves original bytes. The fresh baseline was 152 passed and
2 failed; after the checkout fix all 15 map tests passed.

The service uses reverse multi-source Dijkstra fields cached over validated grid
edges, swept-clear waypoint simplification, valid interaction regions and a
speed-bounded controller. It supports task/room/location targets, cancellation,
replacement, known-geometry replanning and explicit terminal failure statuses.
`AmongUsMapEnv.advance_motion` is shared with Gym and manual control, so execution
uses normal collision physics. Phase 1 interfaces and legacy navigation remain
unchanged. No social game or training was started.

| Verification | Result |
|---|---|
| Full regression suite | **197 passed**, one existing Gymnasium warning |
| New navigation tests | **43 passed** |
| Directed destination pairs | **3,906 / 3,906 successful** |
| Stratified random spawns | **1,008 / 1,008 successful**, seed 20261002 |
| Combined physical execution | **100% (4,914 / 4,914)**, 1,535,435 steps |
| Collisions / blocked steps / replans | **0 / 0 / 0** in the default static benchmark |
| Failure categories / repeated problem regions | **None**; every regional group passed |
| Longest route | O2 to Reactor, 53.55 units / 21.77 simulated seconds, successful |
| Planning mean / p95 | **63.84 / 206.63 ms** |
| Cold / cached planning mean | **234.98 / 63.58 ms** |
| Controller / physics cost | **179.71 / 90.60 microseconds per executed step** |
| Full measured benchmark wall time | **826.13 s** |

The gate requires all default cases and physical assertions to pass, stronger than
aggregate 99%. Targeted tests additionally establish cancellation, blocked/stuck/
timeout handling, oscillation detection and a known closed-door `NO_ROUTE` outcome.
Timings are machine/workload dependent; some regression work ran concurrently.
Route efficiency is execution fidelity to the planned route, not a continuous
shortest-path optimality result. Dynamic multi-player avoidance and live-client
parity remain outside this evidence.

See [service API, alternatives and reproduction](navigation_service.md),
[benchmark summary](benchmarks/phase2/README.md),
[per-case CSV](benchmarks/phase2/cases.csv), and
[configuration/provenance JSON](benchmarks/phase2/summary.json).
Final Phase 2 implementation revision: `9dcc0a217da468c6d906816937fbd5bd11af128f`.
The subsequent documentation commit records this hash; resolve its own handoff
revision with `git log -1 --format=%H` on the completed checkout.
**Next: Phase 3 scripted five-player simulator**, with trusted integration and
wrapper-level leakage tests. Phase 3 was not started in this change.

## 2026-10-02: separate source-backed map implemented

The owner approved combining sourced geometry/data and supplied artwork to replace
the approximate map without deleting old work. `among_us_map_simulation.py` now
provides a separate native-coordinate Skeld sandbox and Gymnasium interface.
`skeld_config_legacy.py` snapshots the unchanged original map; legacy training and
inspectors remain available. No training or social gameplay was added.

The new blueprint includes 14 rooms, 14 vents, 13 doors, 63 task/utility
destinations, furniture collision, source provenance and diagnostic A*. Fifteen
new tests cover geometry and compatibility. Matching a particular current game
build remains unverified; some metadata and physics defaults are estimates.
See [implementation and handoff details](among_us_map_simulation.md). Earlier
map accuracy claims and benchmark numbers below refer to the legacy environment.

---

## 1. Project Origin

The project originally began as an inquiry into deep reinforcement learning for 2D continuous navigation. Operating in a custom continuous sandbox built with Pygame-CE and wrapped in Gymnasium, the initial objective was to evaluate whether an actor-critic agent (PPO) could discover goal-directed locomotion, obstacle detours, deadlock recovery, and dynamic stealth avoidance end-to-end from raw sensory observations (radial distance rays, spatial coordinates, observer visual cones) without relying on classical pathfinding heuristics (such as A*, navigation meshes, or artificial waypoint graphs).

As the navigation infrastructure matured from simple open arenas (Stage 1) to non-convex obstacles (Stage 2) and ultimately to a full-scale topological recreation of the 14-room *The Skeld* map (Stage 2.5), an important architectural realization emerged: **increasingly complex navigation mechanics do not inherently produce social reasoning.** Navigating between rooms is mechanical execution. The core artificial intelligence challenge of social deduction lies in **reasoning under partial observability, tracking timestamped evidence over time, maintaining explicit role uncertainty, and executing strategic deduction.**

Consequently, the project pivoted: **navigation transitioned into an underlying infrastructure service**, while the primary research program re-centered on building an autonomous crewmate agent capable of playing social-deduction games with strict information boundaries.

---

## 2. Historical Navigation Curriculum (Stages 1 – 2.5)

### Stage 1: Basic Continuous Locomotion (COMPLETE / PRESERVED)

- **Objective**: Learn continuous 2D goal-directed navigation across an unconstrained open domain from raw displacement vectors.
- **Environment**: $1100 \times 700\text{ px}$ arena with $40\text{ px}$ boundary margins ($1020 \times 620\text{ px}$ playable surface). Circular rigid-body agent ($r = 15.0\text{ px}$, constant speed $v = 185.0\text{ px/s}$). Randomized start and goal positions enforcing a minimum Euclidean separation of $350.0\text{ px}$.
- **Observation Space**: 43 continuous `float32` values:
  - `[0:2]`: Normalized player position `[x / 1100, y / 700]`
  - `[2:4]`: Relative goal displacement vector `[dx / 1100, dy / 700]`
  - `[4:20]`: 16 radial obstacle raycasts at $22.5^\circ$ angular intervals (max range $300\text{ px}$)
  - `[20:41]`: Observer patrol sensory slots (zeroed in Stages 1 and 2)
  - `[41]`: Stuck flag ($\ge 15$ consecutive blocked steps)
  - `[42]`: Stagnation progress ($\min(\text{stationary\_steps} / 60, 1.0)$)
- **Action Space**: Continuous 2D displacement $\mathbf{a} \in \text{Box}(-1.0, 1.0, \text{shape}=(2,))$. Actions with $\|\mathbf{a}\| < 0.05$ produce zero movement (deadzone); vectors are normalized to unit magnitude when exceeding 1.0.
- **Reward Formulation**:
  - Step penalty: $-0.01$ per simulation step ($\Delta t = 1/30\text{ s}$).
  - Progress shaping: $+0.05$ per pixel of new-best Euclidean progress toward goal (evaluated every $0.5\text{ s}$, minimum $5.0\text{ px}$ advancement, with anti-farming protection).
  - Goal arrival: $+100.0$ (terminates episode).
  - Episode timeout: $30.0\text{ s}$ (truncates episode, no extra penalty).
- **Verified Benchmark Results** (across 100 stochastic evaluation episodes):
  - **Success Rate**: **100.00%** (100 / 100 episodes)
  - **Average Completion Time**: **2.93 s** (Fastest: 1.83 s, Slowest: 4.83 s)
  - **Average Path Efficiency**: **93.86%** (Initial mean distance: 507.86 px)
  - **Episode Reward**: Mean **120.93** (Std Dev: 6.43, Median: 119.59)
  - **Preserved Brain Checkpoint**: `models/stage1/best_model/best_model.zip`
- **What Stage 1 Demonstrates**: PPO readily masters goal-directed continuous locomotion and smooth wall-sliding along flat arena boundaries using compact vector observations.
- **What Stage 1 Does NOT Demonstrate**: Handling non-convex obstacles, interior dead ends, dynamic stealth avoidance, multi-agent reasoning, or imperfect information.

---

### Stage 2: Obstacle Detours & Stuck Recovery (IMPLEMENTED / OFF-CRITICAL-PATH)

- **Objective**: Discover geometric detour routing around non-convex obstacles without artificial waypoint hints, resist cyclic pacing, and recover from deadlocks.
- **Environment**: Fixed central cross-structure obstacles ($520 \times 95$ top vertical, $520 \times 445$ bottom vertical, $175 \times 320$ left horizontal, $675 \times 320$ right horizontal).
- **Forced Detour Layout Guarantee**: Every episode reset enforces that the straight line connecting spawn to goal intersects at least one obstacle (inflated by player radius $15\text{ px}$). Unobstructed line-of-sight paths are rejected during bounded sampling.
- **Anti-Stagnation & Recovery Reward System**:
  - Short blocked steps ($1 \le n \le 14$): $-0.02$ extra penalty.
  - Prolonged blocked steps ($n \ge 15$): $-0.05$ extra penalty (activates stuck flag = 1.0).
  - Stagnation penalty: $-25.0$ one-time penalty when confined within a $10\text{ px}$ radius for $2.0\text{ s}$ (60 steps).
  - Spatial cell revisit penalty: $-2.0$ penalty applied upon the 3rd confirmed entry into any $30\text{ px}$ spatial grid cell (with 3-step jitter confirmation).
  - Recovery refund: breaking free ($10$ unblocked steps and $\ge 40\text{ px}$ displacement from stuck anchor) refunds 50% of accumulated blocked penalties (capped at $+2.0$).
  - Mathematical anti-farming proof: net reward for entering stuck state and escaping is strictly negative ($\le -0.315$).
  - Timeout penalty: $-50.0$ on reaching $20.0\text{ s}$ (600 steps).
- **Status in the Research Program**:
  - Stage 2 implementation is fully validated by regression suites (`test_stage2_forced_detour_and_revisit.py`, `test_stage2_stuck_stagnation_recovery.py`, `test_stage2_blocked_and_timeout.py`).
  - **Stage 2 is no longer on the critical path for the social deduction project.** It remains a valuable standalone learned-navigation baseline, but full 250k-step training is optional and deferred.

---

### Stage 2.5: The Skeld Navigation Environment (VALIDATED INFRASTRUCTURE)

- **Purpose**: Provide a topologically accurate, full-scale navigation map modeled on *The Skeld* map from *Among Us*, serving as the spatial foundation for multi-agent social deduction simulation.
- **Mathematical Coordinate Separation**:
  - **Reference Space**: $8565.0 \times 4794.0\text{ px}$ (Aspect Ratio $\approx 1.78660826$).
  - **Logical World Space**: $W_{\text{world}} = 1600.0\text{ px}$, $H_{\text{world}} = 895.44658\text{ px}$. Uniform scale $S_{\text{world}} = 1600 / 8565 \approx 0.18680677$. Preserves reference aspect ratio to IEEE floating-point precision ($< 10^{-15}$ relative error).
  - **Display Space**: $1100 \times 700\text{ px}$ window with letterboxing ($s_{\text{display}} = 0.6875$, vertical offset $Y = 42.19\text{ px}$). Exact invertible functions `world_to_screen` and `screen_to_world`.
- **Physical Collision & Walkability Architecture**:
  - 50 solid wall rectangles (`SKELD_ALL_SOLID_RECTS`) defining outer hull boundaries and interior partitions.
  - 32 explicit walkable floor polygons (`SKELD_ALL_WALKABLE_AREAS`) defining room floors and corridors.
  - Dual-layer movement solver: axis-separated movement checks both solid rectangle collision and valid floor boundary containment (`is_point_in_ship_floor`).
  - Exterior non-walkability: the vacuum outside the hull is strictly non-walkable; accidental leaks into space are physically impossible.
- **Calibrated Physics Ratios**:
  - Narrowest doorway: $40.0\text{ px}$.
  - Player radius: $10.0\text{ px}$ (Diameter: $20.0\text{ px}$).
  - Clearance ratio: $\frac{\text{doorway}}{\text{diameter}} = 2.00$ (comfortable bidirectional traversal).
  - Player speed: $160.0\text{ px/s}$ (~10.0s to cross ship width).
  - Goal radius: $12.0\text{ px}$ ($\ge 20\text{ px}$ clearance inside alcoves).
- **Authentic Task Destinations**:
  - 40 authentic task destinations extracted from community reference coordinate markers across all 14 rooms.
  - 37 VERIFIED tasks; 3 ESTIMATED tasks (stand coordinates adjusted by 8–10 px into room interiors to guarantee $\ge 20\text{ px}$ wall clearance).
  - **0 unreachable tasks**: 100% reachability verified by automated tests.
  - All-pairs connectivity: verified path exists across all $40 \times 39 = 1,560$ directed task pairs.
- **Internal Validation Pathfinder**:
  - 4px occupancy grid ($400 \times 224 = 89,600\text{ cells}$) with player radius clearance inflation.
  - Exactly **1 connected walkable component** ($26,754$ walkable cells).
  - 8-connected A* search provides ground-truth shortest path lengths for developer inspection and future diagnostic efficiency metrics (`compute_map_aware_efficiency`).
  - **Representative Route Validation Table (Actual Geometry A*)**:
    | Origin Room | Destination Room | Path Exists | Optimal Route Length | Clearance Status |
    |---|---|:---:|:---:|---|
    | **Cafeteria** | **Electrical** | **YES** | **1001.8 px** | Clear (traverses Storage corridor) |
    | **Cafeteria** | **Navigation** | **YES** | **977.6 px** | Clear (traverses Weapons / East corridor) |
    | **Navigation** | **Reactor** | **YES** | **1491.1 px** | Clear (trans-ship traversal via Cafeteria/Engines) |
    | **MedBay** | **Shields** | **YES** | **1132.8 px** | Clear (traverses Cafeteria & East Hall) |
    | **Security** | **Admin** | **YES** | **985.1 px** | Clear (traverses Lower Engine / Storage) |
    | **Electrical** | **Weapons** | **YES** | **1151.5 px** | Clear (traverses Storage & Cafeteria Hall) |
    | **Lower Engine** | **O2** | **YES** | **1263.9 px** | Clear (traverses Storage & Shields Hall) |
    | **Storage** | **Reactor** | **YES** | **1045.1 px** | Clear (traverses Lower Engine corridor) |
    | **Weapons** | **Electrical** | **YES** | **1151.5 px** | Clear (symmetric route verification) |
- **Interactive Map Inspector**:
  - Command: `python inspect_skeld.py`
  - Keyboard/Mouse Toggles: R (rays), C (collision/walkability overlays), T (task points), L (labels), V (vents), G (A* path overlay), H (HUD), Tab/N (cycle tasks), Click (set target goal), Space (reset), ESC/Q (quit).
- **Current Technical Limitations**:
  - **Path connectivity $\neq$ dynamic controller execution**: Automated tests prove that an obstacle-inflated grid path exists. They do **not** yet prove that a continuous steering controller executing circle-AABB wall-sliding can reliably reach all destinations without getting snagged on diagonal corners. Hardening this controller is the explicit mandate of **Phase 2**.
  - Doors are static open corridors; door sabotage and locking mechanisms are not yet implemented.
  - Vents are recorded as spatial metadata for future research but are not traversable by the player.

---

## 3. The Architecture Pivot

### The Core Insight
In earlier stages, the implicit roadmap assumed that agents would progress through ever-increasing navigation complexity (open field $\to$ static obstacles $\to$ full map $\to$ patrol guards $\to$ multi-agent stealth).

However, rigorous analysis of social deduction games revealed a fundamental gap:
1. **Navigation is a solved mechanical sub-problem**: Finding a route to Electrical, Admin, or MedBay can be handled reliably by standard pathfinding or simple steering controllers. An agent that navigates flawlessly still has zero understanding of who might be an impostor.
2. **The actual learning problem is epistemic and strategic**:
   - Tracking timestamped sightings of other players ("I saw Blue in MedBay at tick 120").
   - Detecting physical impossibilities ("Blue could not have reached Navigation by tick 150 without venting").
   - Evaluating corroborating vs. contradictory verbal claims during emergency meetings.
   - Managing explicit uncertainty over hidden roles without omniscient information leakage.
   - Deciding when to perform tasks, when to group up for safety, when to report bodies, and how to vote.

### The New Architectural Structure
Instead of coupling social deduction to end-to-end continuous pixels or raw locomotion, the project adopts a modular layered hierarchy:

```
+-------------------------------------------------------------------+
|                     Simulated Match Engine                        |
|   (Ground Truth: Roles, Real Positions, Private Cooldowns)        |
+-------------------------------------------------------------------+
                                  |
                                  v  [Strict Information Boundary / Projector]
+-------------------------------------------------------------------+
|                    Legitimate Actor Observation                   |
|   (Local Vision, Public Roster, Typed Public Announcements)       |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                        Event Memory Store                         |
|   (Chronological History, Last-Seen Positions, Claim Tracking)    |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                    Belief & Suspicion Model                       |
|   (Probability Distribution Over Hidden Roles, Consistency)       |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                    Strategic Decision Policy                      |
|   (Goal Selection: Tasks, Patrol, Report, Vote, Discussion Claim) |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                   Reliable Navigation Service                     |
|   (Waypoints, Obstacle Avoidance, Task Station Arrival)           |
+-------------------------------------------------------------------+
```

---

## 4. Phase 1: Project Definition & Information Contract (COMPLETE / FROZEN)

- **Milestone Commit**: `8a0b4c9af33757daf6b9abb52a2e6d46f1f83b0a`  
  *`feat(architecture): establish social-deduction information boundaries`*
- **Objective**: Establish mathematically airtight, executable information boundaries that prevent simulator ground truth, hidden player roles, private cooldowns, and out-of-sight events from contaminating actor observations or memory.

### Four Mutually Exclusive Architectural Domains

| Domain | Module | Content & Role | Allowed Consumers |
|---|---|---|---|
| **WorldTruth** | `social_deduction/truth.py` | Complete objective ground truth: true roles, exact player coordinates, private tasks, kill cooldowns, death ticks, internal simulation events. | Game simulator, trusted observation projector, evaluation metrics. |
| **ActorObservation** | `social_deduction/actor.py` | Filtered, immutable snapshot of legitimate sensory data: local 360° sight occluded by walls, public roster, visible bodies, received public announcements, legal action candidates. | Actor decision policy, actor memory. |
| **Knowledge** | `social_deduction/actor.py` | Chronological, append-only container of an actor's own legitimate historical observations. Rejects cross-match contamination and backwards time steps. | Actor decision policy, belief model. |
| **RoleLabels** | `social_deduction/training.py` | Privileged role targets keyed by match and tick. Kept strictly isolated; never accessible to actor policies at inference time. | Offline training supervision, centralized critic, benchmark evaluation. |

### Architectural Invariants & Guarantees
1. **Zero Ground-Truth Leakage**: If simulator state changes only in hidden attributes (e.g., swapping roles between unseen players, moving an unseen player behind a wall, updating a private kill cooldown), the generated `ActorObservation` and legal action candidate list remain **bit-for-bit identical**.
2. **Independent Identity Randomization**: Per-match public IDs (`p0`..`p4`) and display names are assigned via an identity RNG stream completely independent of the role assignment RNG. Public player order cannot encode hidden roles.
3. **Typed Evidence Provenance**:
   - `DIRECT`: Locally observed entities or bodies, timestamped at sight. Bodies do not disclose time of death. Seeing a task animation does not disclose whether it was genuine or faked.
   - `PUBLIC`: Verified announcements published to all players (body reports, meeting rosters, cast votes, ejections).
   - `CLAIM`: Verbal statements made during discussion. The engine authenticates the speaker and delivery tick, but **never** verifies the claimed fact or claimed timestamp.
4. **Legitimate Action Candidates**: Action candidates (roam, interact, report body, call meeting, claim, vote, skip) are pure functions of the safe observation snapshot. No truth-dependent masks (e.g., "safe room" or "real task" filters) exist.
5. **Phase 1 Test Suite**: 42 exhaustive unit and property tests in `test_information_contract.py` verifying paired-world invariance, occlusion geometry, temporal history appending, role separation, and allowlist serialization.

---

## 5. Current Six-Phase Research Roadmap

```
+---------------------------------------------------------------------------------+
|  Phase 1: Project Definition & Information Contract        [ COMPLETE / FROZEN ] |
|  - Immutable value types, projector, provenance, leakage invariants, 42 tests   |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|  Phase 2: Reliable Navigation Service                       [ COMPLETE ]   |
|  - Interaction arrival, cancellation/replanning, 3,906 pairs + 1,008 spawns   |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|  Phase 3: Five-Player Scripted Game Simulator               [ COMPLETE ]          |
|  - 4 crew + 1 impostor, task assignment, kill cooldowns, body reports, meetings  |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|  Phase 4: Event Memory & Suspicion / Belief Model          [ PLANNED ]          |
|  - Spatiotemporal consistency checks, impossibility detection, role probability  |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|  Phase 5: Learned Crewmate Strategic Policy                 [ PLANNED ]          |
|  - RL/search policy, task prioritization, grouping vs solo, voting strategy      |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|  Phase 6: Multi-Agent Expansion                             [ FUTURE ]           |
|  - Learned impostor policies, self-play, saboteurs, vent networks, language     |
+---------------------------------------------------------------------------------+
```

---

## 6. Historical Phase 1 Test Status (current results above)

As of Phase 1 completion, the project maintains an automated test suite with **139 unique tests passing and 0 failures**:

| Test Category | Suite / File | Test Count | Scope Verified |
|---|---|:---:|---|
| **Social Deduction Information Contract** | `test_information_contract.py` | **42** | Paired-world invariance, hidden role swapping, out-of-sight movement invariance, private cooldown isolation, wall sight occlusion, provenance typing, observation allowlist validation, separate training label interfaces. |
| **The Skeld Environment** | `test_skeld_environment.py` | **34** | Mathematical aspect ratio preservation ($< 10^{-15}$ error), single connected component ($26,754$ cells), 40 task destinations reachability, 9 representative routes traversal via A*, doorway clearance ratio (2.0), hull containment, SB3 `check_env`. |
| **RL Environment & Mechanics** | `test_rl_env.py` | **41** | Gymnasium API compliance, observation range bounds, continuous action mapping, deadzones, deterministic trajectories, reward calculations. |
| **Stage 1 & 2 Randomization** | `test_stage1_randomization.py`<br>`test_stage2_randomization.py` | **41** | Seed reproducibility, spawn margin safety, obstacle clearances, forced detour geometry validation. |
| **Recovery, Blocked & Stagnation** | `test_stage2_blocked_and_timeout.py`<br>`test_stage2_stuck_stagnation_recovery.py`<br>`test_stage2_forced_detour_and_revisit.py` | **37** | Stuck state escalation, stagnation penalties, cell revisit penalties, recovery refund caps, anti-farming mathematical proof. |
| **Pygame Core & Infrastructure** | `test_stealth_env.py`<br>`test_ppo_infrastructure.py`<br>`test_observation_43dim.py` | **44** | Base physics, AABB wall sliding, observation layout, PPO model initialization and checkpointing. |

> *Note on Counting*: Running `pytest -q` executes **139 unique test functions** across all test files in 21.6 seconds. Individual standalone integration scripts contain nested sub-tests and acceptance suites; they verify the same core capabilities without double-counting.

---

## 7. Major Settled Design Decisions

1. **Controlled Custom Simulator as Primary Target**: The project focuses on an internal, fully instrumented, deterministic Python simulation. Directly reverse-engineering or controlling the commercial *Among Us* client is explicitly out of scope.
2. **Navigation is Infrastructure, Not the Learning Objective**: High-level strategic agents will issue macro-intentions (e.g., "navigate to Electrical", "report body", "group with Green"); low-level geometric locomotion is delegated to a deterministic navigation controller.
3. **Strict Information Hiding at the Type Level**: Actors never receive raw world references, debug dictionaries, or hidden simulation flags. The information boundary is enforced at the interface level with immutable dataclasses.
4. **Structured Observations Precede Pixels**: Agents will receive structured spatial coordinates, room identifiers, and discrete event logs rather than raw pixel buffers or end-to-end vision models.
5. **Structured Symbolic Communication Precedes Free-Form NLP**: Meeting discussions will use typed, authenticated claim packets (speaker, subject, room, timestamp) rather than open-ended natural language generation.
6. **Single Learned Crewmate Precedes Multi-Agent Co-Learning**: Phase 5 will train a single focal crewmate against fixed, diverse scripted baselines (honest crew, naive crew, deceptive impostor) before attempting multi-agent self-play.
7. **Phase 1 Information Architecture is Frozen**: The boundary interfaces in `social_deduction/` are locked to prevent retroactive leakage or convenience shortcuts during subsequent simulator development.

---

## 8. Open Technical Questions & Future Considerations

- **Navigation integration after Phase 2**:
  - Static arrival and recovery contracts are settled in `navigation_service.md`.
  - Future moving-player/door mechanics need observed-state integration and a new
    dynamic benchmark; the completed static benchmark does not establish them.
- **Phase 3 rules now settled**:
  - Reports/emergency calls precede kills; public ID breaks simultaneous-action ties.
  - Unfinished tasks retain the fixed quota and are reassigned privately only after public meeting absence/ejection.
  - See `social_simulator.md` for exact timers, voting, timeout and termination rules.
- **Phase 4 (Memory & Beliefs)**:
  - What internal memory representation best compresses long match histories: an explicit spatio-temporal constraint graph, a tabular player-location matrix, or a tokenized event history?
  - Should the suspicion model maintain a Bayesian belief posterior over role permutations, or output continuous heuristic suspicion scores?
- **Phase 5 (Strategic Policy)**:
  - Will policy training succeed with standard continuous/discrete PPO, or will high-level macro-actions require hierarchical reinforcement learning (options/semi-MDPs)?

---

## 9. Key Milestone Commits

| Commit Hash | Date | Milestone Description |
|---|---|---|
| `0bf49b5` | Oct 2026 | Enforce obstacle detours, cell revisit penalties, and stuck recovery (Stage 2). |
| `7511134` | Oct 2026 | Construct and validate full-scale Skeld navigation map with 40 tasks and A* validation (Stage 2.5). |
| `8a0b4c9` | Oct 2026 | Establish social-deduction information boundaries, value types, and provenance (Phase 1). |
| `3819367` | Oct 2026 | Source-backed native Skeld map, renderer, assets and 15 tests (Phase 1.5). |
| `9dcc0a2` | Oct 2026 | Reliable navigation; 197 tests and 4,914 physical executions passing (Phase 2). |
| `c2c6ea2` | Oct 2026 | Five-player simulator; 291 tests, 1,000 seeded matches and 1,000 exact replays; original sprites used locally (Phase 3). |
