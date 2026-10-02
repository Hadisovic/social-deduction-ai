# Phase 3 acceptance evidence

**PASS**, 2026-10-02. Starting `main` revision:
`deae77c3dda88290659abc3659d5eb472c07aa07`. The measured implementation was
uncommitted during the run; [summary.json](summary.json) identifies its exact
source SHA-256 hashes and confirms they stayed unchanged during validation.
The implementation revision is recorded in the final [handoff](../../../HANDOFF.md).

```powershell
python run_phase3_batch.py --matches 1000 --workers 8
.venv\Scripts\python.exe -m pytest -q
python run_phase3.py
```

## Acceptance results

| Check | Result |
|---|---|
| Primary seeds | **0–999, all 1,000 completed** |
| Independent full replays | **1,000 / 1,000 identical** |
| Crashes / invalid states | **0 / 0** |
| Timeouts / deterministic differences | **0 / 0** |
| Navigation failures / illegal script actions | **0 / 0** |
| Crew wins / impostor wins | **911 / 89** |
| Task completion / impostor ejection / parity | **870 / 41 / 89** |
| Mean simulated match duration | **58.932 s** |
| Physics/rule steps | **294,749**, mean **294.749** per match |
| Eliminations / reports / meetings | **1,325 / 1,212 / 1,212** |
| Public votes / completed tasks | **4,491 / 7,740** |
| Privileged/direct/public audit entries | **136,339** |
| Full regression | **291 passed**, one pre-existing Gym spec warning |
| New tests | **94**: 93 Phase 3 checks and one legacy-viewer regression |
| Package consistency | `pip check`: no broken requirements |

Counts include each primary match once; replay copies do not inflate outcomes.
The 89 impostor wins are valid terminal outcomes, not failures. These results
measure environment correctness against these scripts, not balanced play or
learned social reasoning. Every primary seed is recorded in [matches.csv](matches.csv).
The full test output is [tests.txt](tests.txt).

## Performance

Windows 11, Python 3.12.10, eight worker processes, invariants enabled. Each worker
reuses a bounded Phase 2 planner cache. Tests, visual verification and documentation
work overlapped parts of this run. These are elapsed instrumentation measurements,
not isolated CPU benchmarks or portable performance promises.

| Measurement | Mean per primary match |
|---|---:|
| Total worker runtime | 5.724 s |
| Navigation planning/control | 2.559 s |
| Observation projection, sight and packet validation | 2.517 s |
| Actor-input/action hashing | 0.377 s |
| Physics | 0.0854 s |
| Policy decisions | 0.0109 s |
| Event-log recording | 0.00287 s |

All 2,000 executions took **1,290.18 s** of batch wall time (21.50 minutes),
including worker startup and replay checks. Amortized wall time was **1.290 s per
primary-plus-replay pair** across eight workers. Navigation and projection are
the main optimization targets; no optimization was allowed to weaken the actor
boundary. `visibility_seconds` includes immutable-packet construction and full
validation, not only ray intersections. Rendering is not required.

## Representative complete matches

All three trace hashes were checked against their rows in the acceptance CSV.
The JSONL files are explicitly **privileged spectator audit logs**. They contain
role labels and hidden events and must never be supplied as actor observations.

| Seed | Outcome | Simulated duration | Trace |
|---|---|---:|---|
| 7 | Crew, tasks complete | 53.8 s | [seed_7.jsonl](replays/seed_7.jsonl) |
| 15 | Crew, impostor ejected | 55.2 s | [seed_15.jsonl](replays/seed_15.jsonl) |
| 25 | Impostor, parity | 79.8 s | [seed_25.jsonl](replays/seed_25.jsonl) |

### Seed 15: legitimate evidence changes the second vote

| Time | Privileged world event | What living crew legitimately receive |
|---|---|---|
| 0.0 | Five Cafeteria spawns; White/p4 is impostor | Public colors/IDs, own role/tasks, local sightings; no role roster |
| 2.4–9.6 | Crew arrive and begin four-second tasks | Own progress and visible ambiguous interaction animations |
| 12.0 | White kills Orange/p2 in Upper Engine | No surviving crew witnesses this kill; only killer/victim receive direct evidence |
| 12.6 | White sees/reports the body | Public reporter, victim, region and living meeting roster; no killer/death timestamp |
| 13.2–19.8 | Crew share sightings; White accuses Purple/p3 | Authenticated speaker claims, not verified facts |
| 21.6–27.6 | All four skip; exploration resumes | Public skips and no ejection; positions/timers were frozen during meeting |
| 39.6 | White kills Black/p1 in Upper Engine | Brown/p0 sees both participants and receives direct witnessed-elimination evidence; Purple is elsewhere |
| 40.2 | Brown discovers/reports Black's body | Everyone receives the public report and participant roster |
| 40.8 | Brown claims to have seen White eliminate Black | Purple receives Brown's **CLAIM**, not Brown's direct observation |
| 49.2 | Brown and Purple vote White; White votes Brown | Public 2–1 vote |
| 55.2 | White is ejected and crew win | Identity-only ejection followed by crew winner; no role-reveal payload |

This covers spawn → physical task travel → sightings → elimination → body
discovery → report → claims → vote → continued play → a valid winner. Seed 7
instead ends through all eight tasks, including publicly justified reassignment.
Seed 25 includes two skipped meetings and a third kill producing parity at 79.8 s.

## Visual verification

The literal `python run_phase3.py` command was launched from `C:\Files\RL` using
the system Python and automatically selected the project venv. A real Windows
Pygame window showed the improved reference map, five colored characters, task
progress and the discussion/voting overlay. No flags or configuration prompt were
needed. Further real window runs inspected seeds 7, 15 and 25 through completion.
Captured frames covered spawn, movement, timed task previews, bodies, discussion,
claims, public votes and winner screens. Visuals and batch use the same runner.

The native Computer Use check observed the exact-command window and a paused
single-step transition; automated SDL event tests additionally cover pause,
step, role/route toggles, replay and next seed. Screenshots containing the optional
commercial artwork are kept locally in `.pytest_cache/phase3_visual*`, not
redistributed in this repository. Reproduce with `--capture` as described in
[the simulator guide](../../social_simulator.md).

Visual inspection found indexed-color task PNGs that could not be smoothly scaled;
the loader now normalizes surfaces to 32-bit RGBA without requiring a display.
All nine selected task previews load/scale successfully. Missing/modified art
uses a tested fallback. The selected local assets are verified against 119 pinned
Git blob hashes. See [provenance](../../phase3_asset_provenance.md).

## Regression findings and preserved boundaries

The fresh pre-change baseline passed all **197** tests. A concurrent full-suite
attempt stalled and was stopped; a diagnostic rerun exposed the old training
viewer's initial-render close race: it could enter its episode loop with no
loaded model. A one-line exit guard and deterministic regression test fix that
case. No old test was weakened or removed. The final full run passed **291** tests
in **101.78 s**, with one existing Gymnasium warning about an unregistered spec.

Phase 1's original modules, Phase 2 navigation, the native collision map, source
data and model checkpoints are unchanged. New boundary tests compare complete
serialized actor packets and candidate actions under hidden-world mutations.
They cover hidden roles/tasks/movement/death/cooldowns, order, future events,
body metadata, occlusion, own-role controls, provenance/type confusion, independent
RNG streams and spectator toggles. Rule tests cover task quotas, public-absence
reassignment, witnesses, simultaneous report priority, vote edge cases, ejection,
timeouts and immediate terminal transitions. A large seeded run supplements
these targeted tests; it does not replace or formally prove the boundary.

The phase is ready for a separately scoped Phase 4 memory/belief effort. No Phase 4
or learned-policy implementation is included.
