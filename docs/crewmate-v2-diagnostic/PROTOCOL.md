# V2 pre-training opportunity diagnostic (development only)

Registered before execution, 2026-10-26. Base: PR #9 commit
`e62ba0b6c33ffd0317f78e18b4289d778cd5af2b`; no changes to canonical sources,
neural models, rewards, lobby values or historical evidence. Phase 6.1 acceptance
remains pending. No final-test partition is executed or evaluated.

## Design and bounded sample

Primary sample: eight fresh scenario seeds **2,630,000–2,630,007**, each run with
the existing five scripted controllers under ordinary (A) and deduction-only (B)
rules. MatchMixture frequencies 0 and 1 select the endpoints, with the same
domain-separated actor RNG streams, assignments, roles and canonical lobby.
All five actors act; there is no learned focal actor. Use the optimized engine.
Alternate A/B execution order. Compare complete actor/action hashes through A's
terminal tick; only termination eligibility may differ. Outcomes after A stops
are a counterfactual continuation, not an independent scenario.

One deterministic rerun of the first ordinary scenario verifies full replay and
metric equality (wall times excluded). Seeds 2,630,100–2,630,109 are reserved only
for unit fixtures. These ranges are disjoint from the published Phase 6
train/validation/development/final partitions and V2 engineering ranges. Check
the registry before execution; never select or generate a final-test game.

If baseline reports are at least 90% from one role (or no reports), probe a
diagnostic-only **staggered-reporting** wrapper on the first two already registered
scenarios, in both A/B conditions: at most four additional executions, no new
scenarios. It gates the existing `_report` helper with an independent per-actor
0-or-18-second delay after the first legitimate sight of each body. Delay is
selected without reference to role. All movement, claiming, killing and voting
remain the original scripts. This is a small mechanism probe, not a balanced
training opponent population or a causal population-level estimate.

Maximum 21 executions (16 primary + 1 replay + 4 probe), **900 wall seconds total
in the diagnostic process**. Timer starts before constructing the first game and
includes initialization, instrumentation and saving. Reserve time for replay;
skip optional probes if the remaining budget cannot accommodate them based on
measured paired duration. A tick-level deadline interrupts incomplete matches;
save them separately and exclude them from completed-match ratios. Stop early
for illegal actions, navigation failures, source drift, mode leakage or replay
divergence. No adaptive new seeds or quiet budget extension.

## Observation boundaries and measurement

Controllers receive only the original immutable Phase3Observation. The wrapper
accepts the same actor packet and its private behavior seed; it never receives
world state, role rosters, evaluation counters or another actor's memory. It may
read its own role because this is legitimate, but reporter delay is role-agnostic.
Extra V2 evidence is recorded separately; existing scripts do not exploit the new
visual/confirmed-role channels. Privileged roles are used only by an offline
scorer. Raw event records and per-match role-attributed data remain local.

- Sample opportunity counters at the unchanged 0.6-second decision schedule.
  First legal kill time additionally examines the impostor's legal action packet
  every physics tick; distinguish physical availability from scheduled opportunity.
- Record actual kills, surviving crew witness–victim pairs (exclude killer and
  victim), crew body observer–victim pairs, and actor-body reporting opportunities.
  Report legal actor-windows and unique actor-body pairs separately. Count reports
  submitted, actually accepted, and same-tick arbitration losses separately.
- Voting denominator: unique actor-meeting opportunities. Report legal windows
  separately; they are not independent voting chances. Score crew correct/wrong
  accusations, explicit skips and automatic deadline skips separately from impostor
  votes. Report discussion claims and meetings reaching/resolving voting.
- Scheduled living actor-windows: no legal actions = forced waiting; legal but only
  WAIT = wait-only; non-WAIT legal actions = actionable. Also report selected WAIT
  despite actions and an approximate multi-choice count. These do not claim that
  every navigation or continuation choice is strategically meaningful.
- Record task completion at 30-second checkpoints while the game is alive, plus
  its terminal value. Split actual completed assignments by living/ghost execution.
  A time bin after termination is missing, not zero. Compute a fixed-trajectory
  no-ghost-progress bound by removing ghost completions, without pretending to run
  a causal ghost ablation.
- A deduction sequence requires a surviving crew witness or crew body discovery,
  a report meeting, and a legal crew ballot opportunity. Track each stage and
  whether task termination truncates discussion/voting/results. This is exposure,
  not proof of learned identification or justified reasoning.
- Measure wall time and simulated time by phase, including fully balloted voting
  windows. Quantify post-ballot forced waiting; do not modify timers or implement
  acceleration during this diagnostic.

Aggregate across paired scenarios; report censored first-event times as missing
with occurrence counts, not as zero. No significance/acceptance claim from eight
scenarios. Any reporting variant comparison is exploratory and only two scenarios.

## Context map and review

| New file | Responsibility |
|---|---|
| `tools/diagnose_crewmate_v2.py` | bounded orchestration, observation-only wrapper, offline opportunity scoring |
| `tests/test_crewmate_v2_diagnostic.py` | denominator fixtures, mode/prefix privacy, reporter legality, deterministic replay |
| `docs/crewmate-v2-diagnostic/PROTOCOL.md` | frozen design and seed/budget registration |
| `docs/crewmate-v2-diagnostic/REPORT.md` | findings, caveats and recommended next experiment |
| `docs/crewmate-v2-diagnostic/aggregate.json` | reproducible sanitized summary only |

Read-only dependencies: `crewmate_v2/{config,engine,runtime,provenance}.py`,
`phase3_{bots,runner,engine}.py`, `phase4_scripts.py`, actor API/geometry and
`phase6/config.py`. Reference patterns: V2 throughput tool and privacy tests.
No public API changes, migration, new lobby setting or source-guard relaxation.

Reviewed alternatives: a full eight-seed × two-mode × two-reporter factorial
would offer a cleaner reporter comparison but may exceed the budget. Reusing
only the old three benchmark seeds would not supply fresh evidence. The chosen
eight-pair baseline plus conditional two-scenario mechanism probe prioritizes
the A/B question and records its reporting limitations explicitly. The user's
request authorizes diagnostic design judgment and isolated scripts; no additional
implementation or compute approval is inferred beyond these bounds.

Self-review: no missing seed range, unspecified reward change, hidden policy
input, future-test use or automatic follow-on training. Stop after reporting.
