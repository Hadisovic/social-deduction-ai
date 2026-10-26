# Crewmate environment V2: implementation design and context map

Independent, opt-in engineering environment based on verified research commit
`2baa5dbc1b36b047ed97934d69526573b510c280`. User-authorized implementation and bounded
testing only: no training, reward tuning, final-test execution, merges or deployment.
Phase 6.1 historical acceptance remains pending.

## Reviewed approach

Use an additive `crewmate_v2` package, leaving every historical compatibility-hashed
source unchanged. Editing the shared engine would invalidate selected policies; duplicating
all physics would risk drift. Instead subclass the engine at explicit rule hooks and reuse
its movement, collision, action validation, identities and trusted projection. The small
versioned V2 tick loop preserves action ordering and makes projection optimization testable.

| New files | Responsibility |
|---|---|
| `crewmate_v2/config.py` | Explicit lobby settings, reproducible hidden match-mode selection |
| `crewmate_v2/tasks.py` | Metadata-backed categories, authored stages/durations, permanent ownership |
| `crewmate_v2/ghosts.py` | Task-only deterministic ghost execution, no social/world/evidence inputs |
| `crewmate_v2/api.py` | Immutable new public evidence and actor-local task details |
| `crewmate_v2/engine.py` | Opt-in engine rules, authenticated public/direct publications |
| `crewmate_v2/runtime.py` | Independent actor memory, future-policy packet and unchanged reward adapter |
| `crewmate_v2/provenance.py` | Separate V2 source/config schema; historical compatibility remains strict |
| `tools/benchmark_crewmate_v2.py` | Bounded deterministic throughput/equivalence comparison |
| `tests/test_crewmate_v2*.py` | Mechanics, privacy, replay, historical-source integrity |

Read-only dependencies: `phase3_engine.py`, `phase3_runner.py`, `phase5_env.py`,
`phase5_features.py`, `phase6/env.py`, `phase6/config.py`, `among_us_map*.py`,
`navigation_service.py`, `social_deduction/phase3_{api,observation}.py`, `belief/*`,
`assets/skeld/among_us_map.json`. Reference tests: `test_phase3_engine.py`,
`test_phase3_boundary.py`, `tests/test_phase6_{vision,foundations,route_boundary}.py`.
No database or production migration. The new observation schema intentionally requires
a future V2 policy; an old policy must not silently receive a changed input contract.

## Rules and information boundaries

- Canonical settings: speed 2.5 (1× approximate), crew sight 4.5, impostor sight 6.75,
  kill distance 4 × collision radius (two diameters, 0.88 at default geometry), initial
  and subsequent kill cooldown 45 seconds, discussion 15, voting 120, one emergency,
  emergency cooldown zero, visible vote identities, confirmed ejections, always-updating
  aggregate completed-task bar and local visual-task completion evidence. Timeout is a
  separate explicit 900-second engineering cap, configurable; not a retail lobby setting.
- Task types/categories come from native metadata. Timed interactions are authored
  approximations, not retail minigames. Common type and its stages are common to the
  lobby; each crew owns a separate immutable plan and mutable progress. One common,
  one long, two distinct short types per crew. Supported multistage routes use existing
  native/supplemental consoles. Inspect Sample includes an explicit waiting stage.
- Dead/ejected crew keep all tasks. Their living motion/body position stays frozen;
  separate ghost state traverses straight segments within the map's world bounding box,
  ignoring obstacles. A deterministic own-task-only executor has no world, role, killer,
  observation, RNG, social-action or living-memory input. External ghost communication
  intentions are illegal. Ghosts never produce visual proof. The sole allowed living
  consequence is ordinary aggregate task completion; it has no ghost/owner annotation.
  Ghost work continues through meetings; ordinary task victory can consequently occur
  during a meeting. Living movement/interactions pause as before.
- Legacy actor packets remain exact immutable Phase 3 values. New public information
  lives in a separately validated V2 envelope: confirmed ejections, aggregate bar and
  directly witnessed living visual completions. Separate actor-owned memory retains
  only that actor's legitimate evidence; no ghost memory is shared or resumed.
- Ordinary/deduction-only mode is selected at initialization from a separate RNG
  domain with explicit configurable frequency (default 0; no final mixture chosen).
  It changes only task-victory eligibility; task activity/bar remain. The sampled bit
  and frequency never enter actor packets. Outcomes can legitimately reveal the mode
  retrospectively. Parity/ejection/timeout remain available; no impossible dummy task.
- Keep original +1 own completed task, -5 death (once), elapsed-time cost, terminal
  reward and physical-time discount. No reporting/voting override or reward shaping.

## Acceleration decision and validation

Headless execution already has no sleep/rendering. Compare full per-tick projection
against projecting only submitted non-WAIT actors; all submitted actors are still
observed before any same-tick action. Preserve every physics tick, decision interval,
sight opportunity, event timestamp and timer. This removes unused work without losing
body/kill observation opportunities. Canonical rules are independent of this execution flag.
Benchmark paired bounded engineering scenarios and hash actor inputs, actions and
semantic outcomes; exclude wall-clock diagnostic counters from equality checks.

Event jumps through meetings must consider speaker changes, votes, ghost task-bar updates
and deadlines. Prefer exact-tick optimization first. Uniformly scaling all simulated
quantities, including dt and decisions, preserves tick count and cannot alone improve
throughput; fewer/coarser ticks change observation opportunities and need a separate study.
Do not ship numerical timer compression without a measured, equivalent implementation.

Tests include controlled ghost-secret perturbations with identical living history,
task plans and ghost start state. Compare living V2 packets and task-bar schedules
through complete ghost execution, including different fake secrets and private ghost
histories. Test every rejected communication intention and cross-owner task submission.
Run relevant historical rule/boundary suites and verify source/checkpoint hashes. Use
engineering seeds outside registered research/final partitions. Publish aggregate
engineering results only; no diagnostic raw observations or role-label archives.

## Implementation sequence and review

1. Add settings, catalog, pure ghost executor and immutable actor contract.
2. Integrate engine hooks, public evidence and both exact-tick execution paths.
3. Add independent memory/future-policy interface and original reward semantics.
4. Test mechanics/noninterference/legacy compatibility; fix defects in V2 only.
5. Measure deterministic throughput, document fidelity limits and future fair study.
6. Open reviewable PR against the research branch; stop at the training/final-test gate.

Self-review: scope is one opt-in environment, not self-play or opponent overrides.
Approximation boundaries and information sources are explicit. Existing research remains
reproducible and acceptance is not advanced by these engineering tests.
