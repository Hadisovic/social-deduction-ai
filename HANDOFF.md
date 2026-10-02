# Continuation handoff

## Current state (2026-10-02)

- Repository: `Hadisovic/social-deduction-ai`, branch `main`.
- Synced clean `0b12c09` to teammate map commit
  `381936787236ada3ddfda3705521fe3d2b680864` before Phase 2 changes.
- Phase 1 information boundary remains complete/frozen (`8a0b4c9`).
- Phase 1.5 map is frozen as the controlled navigation target. Commercial-client
  parity remains unverified; geometry and destination metadata are unchanged.
- **PHASE 2 COMPLETE**: 4,914/4,914 physical executions passed; 197 tests pass.
- Phase 3 has not started. No training, social gameplay or model/log changes.
- Final Phase 2 implementation commit: `9dcc0a217da468c6d906816937fbd5bd11af128f`.
- This documentation follow-up records that immutable milestone. Resolve the final
  handoff commit with `git log -1 --format=%H` (a document cannot embed its own hash).

## What changed

`navigation_service.py` provides cached reverse multi-source Dijkstra fields over
swept-validated grid edges, clearance-preserving waypoint simplification,
interaction-region arrival and a bounded physical controller. It exposes
`NavTarget.task(id)`, `.room(name)`, `.location(position, radius)`, `navigate`,
`command`, `feedback`, `cancel`, and explicit `replan` with known geometry.

The service consumes own-motion feedback and emits native x-right/y-up bounded
displacements. It does not hold an environment, teleport or query hidden truth.
`AmongUsMapEnv.spawn` initializes episodes; `advance_motion` performs subsequent
movement through the normal `map.move` collision model. Gym and manual movement
use the same primitive. Gym reset/step shape, direction, rewards and termination
semantics are preserved. Service arrival uses actual console interaction regions
and owns its completion semantics; do not drive it through normalized Gym actions.

The map inspector now uses the service: Tab selects a console, click sets a
location, Space starts, WASD/arrows cancel and move manually. The title shows status.
Diagnostic A* and the old map/environment/inspectors remain available.

The fresh baseline was **152 passed, 2 failed**. Both failures were Windows CRLF
conversion of byte-hashed map sources and the exact legacy snapshot. `.gitattributes`
preserves the original repository bytes; tests and map geometry were not weakened.
All 15 teammate map tests passed after that fix.

## Validation

Using Python 3.12.10 and the existing Windows venv:

| Check | Result |
|---|---|
| `python -m pytest -q` | **197 passed**, one existing Gym registration warning |
| `python -m pytest test_navigation_service.py -q` | **43 passed** |
| `python -m pip check` | No broken requirements |
| `git diff --check` | Passed |
| Full physical benchmark | **4,914/4,914 passed**, zero collisions/blocked steps/replans |

The 43 service tests cover every native room approach, all grid-edge legality,
rejected corner-cutting diagonals, physical motion and radius clearance, interaction
regions, tiny location goals, cancellation/replacement, known-door `NO_ROUTE`,
blocked recovery, oscillation, timeout, invalid inputs, cache bounds, deterministic
case generation and shared Gym physics. Existing Phase 1 and legacy tests pass.

The benchmark has 63 destinations, 3,906 directed pairs and 1,008 deterministic
continuous spawns over 21 regions. It records per-case physical assertions,
outcomes, travel, final distances, steps, collisions, replans, planning/control
costs, seed/configuration and implementation/map hashes. The acceptance rule is
zero failed default cases, stronger than an aggregate 99% target.

Measured on this Windows machine: 826.13 seconds for the full benchmark;
63.84 ms mean / 206.63 ms p95 planning, 234.98 ms mean cold planning and 63.58 ms
mean cached planning. Controller cost was 179.71 microseconds per physics step,
plus 90.60 microseconds for movement physics. Tests/documentation work overlapped
parts of the run, so timings are observations, not isolated performance guarantees.
The longest case was `random:O2:007` to Reactor: 53.55 units, 21.77 simulated seconds,
successful. Every reported region/stratum passed; no unexplained failure remains.

## Reproduction and key files

```powershell
python -m pytest -q
python benchmark_navigation.py
python benchmark_navigation.py --case "random:Electrical:001" --output .pytest_cache/nav_case
python among_us_map_simulation.py
```

Use the repository's `.venv/Scripts/python.exe` on Windows.

- `docs/navigation_service.md`: architecture alternatives, API/ownership,
  arrival/controller details, metrics, reproduction and limitations.
- `docs/benchmarks/phase2/`: committed CSV outcomes, JSON metadata and summary.
- `test_navigation_service.py`: focused service/failure-contract tests.
- `among_us_map.py`, `assets/skeld/among_us_map.json`: frozen map/collision data.
- `docs/among_us_map_simulation.md`, `assets/skeld/NOTICE.md`: fidelity/provenance.
- `docs/PROJECT_PROGRESS.md`: canonical project history and roadmap.
- `docs/information_contract.md`, `social_deduction/`: frozen Phase 1 boundary.

## Preserved decisions and limits

- Main target is our controlled game; classical known-map navigation is allowed.
- Four crew and one impostor, with one learned crewmate later.
- Exact own/visible position and public tick timestamps are accepted abstractions.
- Private task assignments; no live global task bar or hidden death disclosure.
- Public votes, no role reveal on ejection, no actor role roster after the end.
- Claims retain source/asserted time separately from delivery time.
- Actor/memory receive no truth, training labels, debug dictionaries or helper refs.
- Public event publication uses the frozen typed allowlist and authenticated triggers.
- Identity and role RNG streams remain independent.
- No new policy training; preserved model checkpoints and legacy code are unchanged.

This benchmark covers the default static map. Dynamic multi-player avoidance,
door timers, sabotage, task minigames and live-client parity are not established.
Door closure and motion blockage are isolated regression tests, not a full dynamic
game. Path efficiency measures execution fidelity to a planned route, not global
continuous path optimality. Cache workloads/timings are documented explicitly.

## Next work

The Phase 2 gate passed. Phase 3 can now build the five-player scripted
simulator on the frozen information boundary and physical navigation service.
Define transition order, task timers, kill/report/meeting rules and authenticated
publication, then connect Skeld visibility through the trusted projector. Add
wrapper-level paired-world leakage tests. Do not give actors environment/planner
references or refresh known navigation geometry from unobserved world state.

Phase 1 history is minimal snapshot storage; Phase 4 will compress evidence and
implement beliefs. No suspicion model or strategic policy is implemented here.
Review Phase 3 separately before starting it.
