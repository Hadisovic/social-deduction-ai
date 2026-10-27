# Cleanup and performance preparation — 2026-10-27

Base `8e99383`, isolated branch `codex/crewmate-v2-cleanup-performance` in our existing
crewmate worktree. Remote references fetched before changes. No merge/rebase of `main`,
the impostor branch or the independent spectator branch. Their ancestry is not permission
to reorganize their files. Full training remains unapproved.

## Context map and dependency review

| Files | Responsibility / allowed change |
|---|---|
| Root README, `archive/`, `.gitignore` | Current entry point, preserved history, future output hygiene |
| `local-runs/engineering-tests/` | Move only our small generated JUnit XMLs; preserve bytes |
| `crewmate_v2/learning/` | Opt-in headless collectors, central inference/update, engineering benchmark |
| `docs/crewmate-v2-performance/`, new tests | Manifest, measurements, execution proposal, isolation/GAE/resume checks |
| Canonical V2 flat modules, `phase6/`, `phase5_*`, `belief/`, `social_deduction/` | Read-only simulation, PPO and privacy dependencies |
| Map/navigation/render modules, all assets, historical checkpoints/results | Read-only protected dependencies |

AST imports show `benchmark_navigation` is used by navigation tests; `rl_environment`
feeds the historical training/evaluation/watch suites; environment/geometry/player/observer
remain coupled. V2 engine imports `navigation_service`, which consumes the current map.
The renderer consumes the same map assets. Existing source locks include root navigation,
Phase 3–6 and canonical V2 modules. Moving those would invalidate historical loaders.
Spectator tree inspection is read-only: its app/recording/replay modules and existing
rendering/asset interfaces stay untouched. No compatibility with its unpublished future
changes is claimed. Retain uncertain files; no Python relocation or binary duplication.

## Design selected within the authorized preparation scope

Three options were considered: preserve serial collection; use process-owned independent
environments with central batched policy inference; replicate a policy in each process.
Benchmark the second against serial and retain serial if process overhead wins. Replicated
policies add synchronization and RNG complexity, so are unnecessary here.

Each worker holds one private environment, opponent controllers, memory and reward ledger.
The parent alone owns the PPO policy, Adam and update RNG. Barrier rounds use deterministic
worker order and independent action-sampling generators. No policy update until a complete
rollout is assembled. Compute GAE independently per environment segment, then concatenate;
never propagate advantages across worker trajectories. Preserve every real physics tick,
time discount, terminal flag, mask and reward event. No rendering is required.

Measure workers 1/2/4/8 including startup, inference, rollout collection, PPO update and
checkpoint serialization. Compare serial and batched inference where useful. Use only
fresh engineering seeds, equal total actionable choices and a bounded update, not the
512-choice pilot or validation partitions. Log memory/CPU, per-worker seeds and partial
episodes rather than silently treating budget cutoffs as game termination. Test deterministic
replay and collector snapshot/resume before proposing a parallel pilot.

The proposed 512-choice A/B pilot must retain four 128-transition updates and the same
20 validation contexts/model. Parallel rollouts change trajectory/update boundaries even
with equal budgets. Register that execution change explicitly and obtain owner approval;
do not silently substitute it for the existing serial proposal.

## Verification and risks

Relevant suites: V2 mechanics/privacy/integrity/rewards/pipeline/diagnostics; map, physical
navigation and render smoke tests; new parallel isolation/seeds/masks/termination/GAE/
checkpoint reproducibility tests. Preserve a local byte-hash snapshot before edits.
Cleanup and performance work receive separate commits. No database, gameplay, reward or
public rendering/API migration. Risks are multiprocessing startup/RAM cost, shallow
benchmarks and changed rollout trajectories; report limitations instead of claiming a
training-quality gain. Approval remains required after all preparation is complete.

Completed implementation and measurements: [report](REPORT.md),
[curated metrics](benchmark-results.json), [verification](validation.json) and
[spectator interface](SPECTATOR_INTERFACE.md). The original serial protocol is preserved;
the four-worker/eight-stream proposal remains unapproved.
