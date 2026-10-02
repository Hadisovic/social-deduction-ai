# Continuation handoff

## Map implementation update (2026-10-02)

The current request authorized a separate, source-backed Skeld map without
removing legacy work. Run `python among_us_map_simulation.py`; import
`AmongUsMapEnv` from that module for the new sandbox. `assets/skeld/among_us_map.json`
is the combined blueprint; `tools/build_among_us_map.py` rebuilds it offline.
The original map/environment/navigation scripts remain unchanged, and
`skeld_config_legacy.py` is a byte-for-byte map snapshot. Existing training scripts
continue to use the legacy map until explicitly migrated.

Read `docs/among_us_map_simulation.md` and `assets/skeld/NOTICE.md` before integration.
Native coordinates and Gym observation/action shapes are documented there. This
is sourced historical geometry plus newer metadata, not a verified current-client
export. The map adds diagnostic A*, not a finished Phase 2 service. Phase 1 remains
frozen. No policy training, suspicion/kill heuristics, or social enum changes were
made. The older handoff below describes the preceding foundation.

## State and goal

- Repository: `Hadisovic/social-deduction-ai`, branch `main`.
- Starting revision for this work: `7511134` (clean working tree).
- Current work: Phase 1 information architecture complete; Phase 2 pending user review.
- Goal: a compact crewmate learns evidence-based strategy under legitimate partial
  information and improves team outcomes against held-out player policies.
- Canonical detailed project history and curriculum benchmarks: see `docs/PROJECT_PROGRESS.md`.
- See `git log -1` for the commit containing this handoff; do not embed a
  self-referential commit hash here.

## Completed foundation

README presents the core research mission, architecture, and current phase roadmap.
`docs/PROJECT_PROGRESS.md` contains the full historical navigation benchmarks (Stages 1, 2, 2.5),
the architectural pivot analysis, and settled design decisions.
`docs/project_architecture.md` owns module responsibilities; `docs/information_contract.md` owns
rules, invariants, and data structures.

`social_deduction/` contains immutable truth and actor snapshots, a crew-only
visibility projector, actor-derived action candidates, typed evidence provenance,
minimal observation history, independent identity/role setup, and separate role
labels. These are fixtures/interfaces, not a gameplay engine or full memory model.

## Settled decisions — preserve unless evidence requires change

- Main target is our controlled game; classical known-map navigation is allowed.
- Initial game: four crew, one impostor. One learned crewmate comes later.
- Exact own/visible position and public tick timestamps are accepted abstractions.
- Private tasks; no live task bar; ambiguous observed task animations.
- Death disclosure through sightings/reports/meeting roster, not live global flags.
- Public votes; no role reveal on ejection; no actor role roster even after the end.
- Claims retain source and asserted time separately from delivery time.
- Actor/memory receive no truth, training labels, debug dictionaries or truth helpers.
- The public log is a typed allowlist. Phase 3 must authorize publication triggers.
- No new training; legacy navigation source and behavior remain unchanged.

## Validation

Phase 1 is implemented and ready for review/freeze. Validation on Python 3.12.10:

| Command (using `.venv/Scripts/python.exe`) | Result |
|---|---|
| `-m pytest -q` | 139 passed: 42 new contract tests + 97 existing tests |
| `test_skeld_environment.py` | 34/34 passed |
| `test_stage1_randomization.py` | 17 checks passed, including nested legacy/infrastructure suites |
| `test_stage2_randomization.py` | 15 randomization + 9 reward checks passed |
| `test_stage2_blocked_and_timeout.py` | Blocked, timeout and reward-sanity suites passed |
| `test_rl_env.py` | 41 checks passed, including nested acceptance checks |
| `test_stealth_env.py` | 15 acceptance checks passed |
| `test_ppo_infrastructure.py` | 14 checks passed, including disposable 512-step smoke runs |
| Frozen Stage 1 `PPO.load` and `predict` | Passed without modifying checkpoint |
| `-m pip check`, `git diff --check` | Passed |

Final pytest warning: Gymnasium cannot check alternative render modes because the
Skeld environment has no registration spec. No failing tests. Standalone counts
include nested suites; do not add them to the pytest total as unique tests.

The local venv initially lacked pytest, Torch and SB3. The latest resolved Torch
2.14.1 failed `c10.dll` initialization. The Windows development requirements now
pin the validated pair Torch 2.6.0 / SB3 2.7.0; Gymnasium resolved to 1.1.1, pytest
to 9.1.1. SB3 2.6.0 was rejected during validation because the preserved SB3 2.9.0
checkpoint uses newer schedule objects; SB3 2.7.0 loads and predicts cleanly.
Use Python 3.12 and `pip install -r requirements-dev.txt` for this Windows test
stack. Runtime dependency ranges and all existing navigation source are unchanged.
No production training, model/log edits, or Phase 2 work were performed.

## Known limitations and next work

**Exact next phase: Phase 2 — Reliable Navigation Service, after user review.**
No architectural decision blocks Phase 2. Choose controller arrival tolerance and
benchmark timeout relative to physically executable route length during that work.

Read `skeld_navigation.py`, `skeld_environment.py`, `skeld_config.py`,
`test_skeld_environment.py`, and `inspect_skeld.py`. Address diagonal edge/endpoint
clearance, grid versus AABB movement consistency, floor versus sensor boundaries,
arrival/cancel/replan, and all 1,560 executed task routes plus random spawns.
The checked-in all-pairs test currently queries only 10 selected pairs. A connected
grid component is not a proof of controller reliability.

Phase 1 visibility uses independent rectangular fixtures; Skeld adaptation is not
implemented. This Python boundary prevents accidental coupling, not malicious code
inspection. History stores complete snapshots/public prefixes; Phase 4 must
deduplicate/compress evidence and implement beliefs. Identity/role RNG streams must
remain independent in the future match runner. Labels stay outside actor replay.

Phase 3 still needs transition ordering, concrete timers, task reassignment,
publication authentication, rule enforcement, match replay, and wrapper-level
leakage tests. These are not reasons to expand Phase 1 or delay Phase 2.

Do not redesign identity, provenance, observation/label separation, or the six-phase
roadmap merely to fit a new model. Do not launch training, build social gameplay,
or start Phase 2 automatically. Review each phase separately.
