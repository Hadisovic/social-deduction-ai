# Continuation handoff

## Current state (2026-10-02)

- Repository: `Hadisovic/social-deduction-ai`, branch `main`.
- Phase 3 starting revision: `deae77c3dda88290659abc3659d5eb472c07aa07`.
- Phase 1 complete/frozen: `8a0b4c9`; map milestone: `3819367`;
  Phase 2 implementation: `9dcc0a217da468c6d906816937fbd5bd11af128f`.
- **PHASE 3 COMPLETE**: 291 tests; 1,000 complete seeds and 1,000 identical
  independent replays; all required failure categories zero.
- Phase 3 implementation revision is recorded by the documentation follow-up
  after the implementation commit. Resolve the final handoff with `git log -1`.
- Phase 4 has not started. No learned policy, belief model, training or checkpoint
  changes were made.

## Run the complete game

```powershell
python run_phase3.py
python run_phase3.py --seed 15 --speed 4
python run_phase3.py --seed 25 --speed 4
python run_phase3_batch.py --matches 1000 --workers 8
.venv\Scripts\python.exe -m pytest -q
```

The no-flags visual command was launched in a real Windows window from the
repository root. It automatically uses the existing `.venv` if present. Space
pauses, Right steps, +/- changes speed, R repeats, N changes seed; 1 toggles
spectator roles, G routes, V range, T tasks, L labels, C collision, E log, B bodies,
Tab sidebar, Escape exits. Real complete matches 7, 15 and 25 were inspected.

The owner-requested sprites and task panels are installed locally in ignored
`assets/phase3_local/`. The public repository contains their pinned manifest and
loader, not the ripped game PNGs. An offline clone uses original procedural
characters; `python phase3_assets.py --install` explicitly installs the selected
local pack. Read `docs/phase3_asset_provenance.md` before distributing artwork.
No new package dependencies were needed.

## Architecture and preserved contracts

`Phase3Game` owns physical entities, private tasks, bodies, rules and authenticated
publication. `project_game` creates detached, exact-type immutable
`Phase3Observation` packets. `ScriptedController.decide` receives only those values;
it imports no engine, truth, geometry helper or renderer. `ScriptedMatch` gathers
all actors' inputs before collecting actions, schedules fixed ticks/decisions,
and hashes full actor inputs/actions and privileged event logs. The renderer is
a read-only spectator. Visual and batch modes share that same runner.

The original Phase 1 modules are unchanged. Native visibility and own-impostor
controls live in additive `social_deduction/phase3_api.py` and
`phase3_observation.py`. Phase 2's `NavigationService` and
`AmongUsMapEnv.advance_motion` are reused unchanged, with the same native map.
There is no gameplay teleport or second movement system.

Key files:

- `phase3_engine.py`: fixed-tick rules, entities, transitions, task quota and wins.
- `phase3_bots.py`: independent role-aware scripts with bounded recent observations.
- `phase3_runner.py`: orchestration, trajectory hashes and audit replay output.
- `run_phase3.py`, `phase3_renderer.py`: desktop demo and spectator overlays.
- `run_phase3_batch.py`: seeded acceptance, independent replays, metrics/provenance.
- `phase3_assets.py`, `assets/phase3_assets.json`: selected local artwork and fallback.
- `docs/social_simulator.md`: complete API, rules, architecture and limitations.
- `docs/benchmarks/phase3/`: acceptance CSV/JSON, test output, three representative
  privileged JSONL traces and a truth-versus-actor timeline.
- `test_phase3_*.py`: 93 new checks, including 26 full-packet boundary checks.

## Settled rules

- Four crew/one impostor; independent seeded identities, colors, roles, tasks and
  controller behavior. Match seed and hidden role state do not enter actor packets.
- Eight fixed private tasks; four seconds each, retained partial progress.
  Fake tasks are ambiguous animations and never count toward the quota.
- On hidden death, orphan work stays with its owner until a public meeting roster
  reveals absence. Then unfinished work is privately reassigned to active crew.
- Native geometry blocks 360-degree range-limited sight. Kills need visible close
  targets and ready cooldown. Direct witnesses must see both killer and victim.
- Reports reveal reporter/victim/region, never killer or death time. Meetings
  publish living participants, clear bodies, cancel intentions and freeze positions,
  interactions and cooldowns. No meeting teleport; fresh intentions resume play.
- Reports/emergency calls precede kills in simultaneous ticks; public ID breaks ties.
- One authenticated finite claim per speaking turn; claims can be false. DIRECT,
  PUBLIC and CLAIM provenance remain distinct. Claimed time is not delivery time.
- One public vote/skip per participant; self-vote allowed. A unique player plurality
  must beat skip; all ties and all-skip eject nobody. No role reveal on ejection.
- Central wins: no impostor, parity, all initial tasks complete, in that order.
  Deadline is a draw/TIMEOUT and fails the default acceptance gate.

## Final validation

Fresh baseline: 197 passed. Final suite: **291 passed**, one existing Gym spec
warning, 101.78 seconds during concurrent work. All original tests were retained.
The added tests include 93 Phase 3 checks plus one deterministic legacy viewer
regression. A diagnostic full run exposed the old training viewer entering an
iteration without a model after its initial render handled close; one guard in
`live_watch_training.py` fixes this. Indexed task PNG scaling was also found by
real visual QA and fixed in the optional loader. `pip check` passed.

Seeds 0–999: **1,000 complete + 1,000 identical full replays**. Zero crashes,
invalid states, timeouts, illegal scripted actions, navigation failures or replay
differences. Source hashes stayed unchanged throughout the run. Results:
**870 task wins, 41 impostor ejections, 89 parity wins** (911 crew / 89 impostor).
Mean duration 58.932 simulated seconds; 294.749 steps; mean concurrent worker
runtime 5.724 seconds. Total wall time 1,290.18 seconds with eight workers,
including every replay. See the benchmark report for cost breakdowns and caveats.

Only intentional reports and three privileged traces are committed. Local PNGs,
visual captures, audit downloads, progress journal and scratch test outputs stay
ignored. The mixed truth replay log must never become a learner's input. Use
`Phase3Observation` and its explicitly typed evidence instead.

## Limits and next phase

This is a controlled static-map research simulator. Exact structured localization,
no player-player collision, coarse fixed ticks and timed task substitutes remain
intentional simplifications. Scripts are validation fixtures and are not balanced
around 50/50. Commercial-client parity, networking, minigames, dynamic doors,
vents, sabotage, ghost play, CNN perception and LLM discussion are unimplemented.
The local artwork has documented redistribution limits; some colors have only a
single directional frame. Timings include concurrency, validation and hashing.

The next separately scoped phase is **Event Memory + Suspicion / Belief Model**.
It can consume stable actor packets, direct sightings and public claims/votes,
with role labels reserved for training/evaluation. Preserve the whole-input and
legal-candidate invariance tests when adding memory. No Phase 4 work has begun.
