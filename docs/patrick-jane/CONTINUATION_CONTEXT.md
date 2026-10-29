# Approved continuation — 2026-10-28

The owner approved the design in `CONTINUATION_SPEC.md`: preserve the current
network, navigation decisions, reward coefficients and PPO algorithm; correct
confirmed Classic mechanics in the isolated Patrick Jane implementation; extend
the existing spectator; measure 8/16/32 workers, then run an approximately 250,000
individual-decision pilot after readiness passes. No historical model or partner
branch changes. This approval supersedes additional design-approval ceremonies.

## Context map

| Files | Purpose and intended changes |
|---|---|
| `patrick_jane/config.py`, `environment.py` | Versioned Classic lobby, emergency timers, sabotage and meeting corrections |
| `patrick_jane/api.py`, `features.py` | Legitimate own timer and repair-hold observations within existing tensor widths |
| `patrick_jane/worker.py`, `collector.py`, `training.py`, `telemetry.py` | Real instance monitoring, measured scheduling, bounded training and evaluation |
| `patrick_jane/ppo.py`, `network.py` | Profile existing recurrent updates; optimize only with equivalence evidence |
| `spectator/v2/*`, `run_spectator_v2.py` | Extend existing independent renderer, real streams and training monitor |
| `tests/patrick_jane/*` | Mechanics, privacy, source locks, likelihoods, save/reload, renderer compatibility |
| `docs/research/ISSUES.md`, `docs/patrick-jane/*` | Evidence, uncertainty, benchmarks and staged pilot registration |

Dependencies `crewmate_v2`, `phase3_engine`, `social_deduction`, real Skeld geometry
and navigation service remain unchanged. Their implementation patterns are reused
through PJ subclasses; old source/checkpoint compatibility remains a boundary.

## Design and sequence

1. Preserve the prototype in local commit `2e0991757f364d763741113eba6ce9286ce2055b`.
2. Keep the persistent desktop spectator open on genuine engineering recordings.
3. Separate confirmed setting corrections from unsupported retail approximations;
   test role-local observations and event ordering before new performance runs.
4. Add observer-only throttled snapshots and a real instance list to the existing
   viewer. Workers never consume spectator truth or depend on rendering.
5. Benchmark matched complete-game workloads at 8/16/32 workers; verify a full
   GPU update, save/reload and historical replay compatibility. Preserve failed
   benchmarks, negative results and their source locks.
6. Register seeds, compute ceiling, complete-match overshoot and held-out
   evaluations from measured costs. Launch only the approved pilot, preserving
   latest and champion separately. Stop on non-finite values, crashes, privacy
   violations, exploitation or serious regressions.

The narrow extension of existing components is preferred over a replacement
viewer or architecture redesign: it preserves compatibility and keeps performance
comparisons interpretable. More complex asynchronous scheduling is conditional
on a measured bottleneck and semantic-equivalence tests.

## Review

No architecture/reward change, production deployment, publication, final-test
access, or partner-branch modification is included. New mechanics need a new
source lock. Current 2026 numerical preset defaults remain uncertain; the owner
supplied historical baseline is explicit rather than represented as verified
retail defaults. Readiness was qualified with corrected matched benchmarks,
48 PJ correctness checks, 46 historical compatibility checks and complete-game
GPU/checkpoint verification. The bounded pilot is now complete; scientific
acceptance remains pending.

## Completed pilot and next review — 2026-10-29

Frozen core `dbbf9ae4073fd81a89725c8b4be4d25f21903eee`; local run
`local-runs/patrick-jane/pilot-250k-seed17-v3`. Six updates, 255,083 individual
decisions, 192 complete training matches, 400 full development validation
executions on 100 distinct paired scenarios. Thirty-two workers was fastest
in the matched 8/16/32 comparison; actual pilot training rate 76.026 decisions/s.

Final validation won 1/100 versus 0/100 initially; 98 critical sabotage losses,
zero critical repairs, only ten living-crew tasks. Voting became more selective,
reports increased, and accusations remained poor. This is partial behavioral
change with major competence limits, not Phase 6.1 acceptance.

[Measured report, paired uncertainty and provenance](results/pilot-250k-seed17-2026-10-29/REPORT.md).
All checkpoints and raw records remain local and unchanged. Latest and selected
champion are `update-000006.pt`, SHA256
`d8b81ef06788a0139822a5677801f34069aef116286cc079cb12de8c39538ab8`.
Keep the frozen source archive: strict loading intentionally rejects source
changes. R035/R036 preserve failed attempts; R037 records results; R038 explains
the replay collection-counter/empty-input-trace limitation.

Next proposed work is a small registered task-persistence, critical-repair and
witness-to-vote diagnostic. No implementation or new training has started.
One million extrapolates to 8.83 hours before contingency and requires owner
review/authorization. The larger stage, final partitions, production, partner
branch and existing PRs remain untouched.

## New owner-approved preparation — lights-only manual run

The next user instruction changes the future regime: keep only lights sabotage
and add a small unresolved-lights time cost. This overrides the attached
disable-all-sabotage / preserve-all-coefficients language for those two settings.
All original coefficient values, network/PPO design, normal kills and private
actor memories remain unchanged. New cost is -0.005 per living crew per simulated
dark second, including meetings; no ghost penalty.

[Manual launch and results guide](RUN_LIGHTS_TRAINING.md). The default command
opens the existing spectator, uses 32 workers and continues the preserved pilot
checkpoint for one million **additional** decisions. The source-verified bridge
records the regime change; it does not weaken the old strict loader. Eight
paired entry/final development games replace extensive per-checkpoint evaluation.
The new run has a 12-hour cumulative active ceiling and its own results folder.

Preparation was checked with unit/compatibility tests and isolated, short
32-worker interruption/resume/reporting runs. These are plumbing tests, not
evidence of improved learning. Historical pilot/champion and all raw data remain
unchanged. **Do not start the long run from the agent session.** The owner launches
it manually and later asks the agent to inspect `lights-learning-v1` reports.
See research R039/R040 for measured checks, limitations and negative findings.
