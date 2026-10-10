# Phase 6.1 seed-17 stage

> Publication scope (11 October 2026): this document records the approved plan and
> development-branch evidence. PRs #3/#4 remain unmerged. Phase 6 runtime,
> runners and checkpoints described below are **not included in main** by
> this documentation publication. Future capabilities remain planned.

The owner approved the runner and **three** 8,192-decision models only: full,
history_no_belief and current_only, all initialization seed 17. The remaining
six models and all final tests remain unapproved. PR #3 stays open/unmerged.
The recorded research branch started from verified foundation head
`16101131e9787c9bcbaebe250d25d6dd0d3574fb`; this main publication includes only its documentation.

The approved stage has now completed at source revision `cd48b65a8de9de0cab33894ddb2558057487847d`.
See the [actual results and revised estimate](https://github.com/Hadisovic/social-deduction-ai/blob/8187d51b6bbeb506fd2a59f25504adb368ab5006/docs/benchmarks/phase6/seed17-stage/README.md).
The protocol below was frozen before execution; no selection rule changed.

The [agreed continuation roadmap](phase6_roadmap.md) plans to finish the full
nine-model comparison with this protocol unchanged. This document remains the
record of the approved seed-17 stage; it does not authorize the remaining six or
final tests. New population/dashboard/concurrency plans do not modify this study.

## Context map and design review

| File | Purpose / change |
|---|---|
| phase6/config.py | Expand pinned sources to belief/actor/truth and analysis dependencies |
| phase6/study_support.py | Finite-value, source/frozen/deadline guards, packet trace, trainer-only metrics |
| phase6/study_training.py | Fixed seed-17 registered conditions; preserve existing smoke cap; immutable research runs |
| phase6/study_validation.py | Validation-only seeds, guarded match worker, frozen ranking |
| phase6/study.py | Preflight replay; serial three-model training; two-worker validation; stop before more models |
| phase6/study_report.py | Actual experience, tasks/voting/failures, curves and revised remaining budget |
| tests/test_phase6_study.py | Partition/selection/source/failure/reproducibility/orchestration checks |

Dependencies: existing Phase6Env; unchanged Phase5 scorer, PPO update and GAE;
existing full-match uncertainty summarizer; frozen belief/release models; exact
native map/navigation. No simulator, reward or neural architecture change.
The smoke CLI remains capped. Rather than exposing arbitrary long-run arguments,
the stage command fixes the approved conditions/count/seed. Reviewed risks are
checkpoint/source compatibility and selection leakage; historical artifacts are
preserved with their original source snapshots. No migration/UI change is needed.

## Frozen protocol

Each condition uses 24-second options, Euclidean distances, learning rate 0.0003,
rollout 256, entropy coefficient **0.02 throughout**, gamma 0.99, unchanged PPO
epochs/minibatches and crew/impostor vision 4.5/6.75. Fresh initialization, identical
initial state digest, same ordered training seeds 1200000–1219999; input condition
is the sole experimental difference. Training actions are sampled; validation
uses greedy policy actions. Physical time and completed/partial episodes are logged.

Save all four checkpoints at 2048/4096/6144/8192 and final weights in new directories.
Screen each on the **same 20 ID validation seeds 1220000–1220019**. Rank by total
crew wins, then total own tasks, then the earliest checkpoint. Record an exclusive
selection lock with file/source digests before extended validation. No reselection
based on the subsequent 200-match records.

Evaluate each selected model and idle/random/scripted/frozen Phase5 transfer on
all **200 ID validation seeds 1220000–1220199**: 240 screening executions + 1400
extended executions = **1640**, using at most two workers. Screening is reused
within extended validation, not independent evidence. Baselines use full legitimate
memory and identical new game rules. The patient final partition 1240000–1240499
and familiar final partition 1230000–1230499 remain closed. No patient validation
or final testing is silently added to this stage.

## Verification, stopping and preservation

Before full training: all relevant regressions pass; two 32-decision infrastructure
runs reproduce final tensor digests and complete transition traces; a saved policy
replays the same development match identically twice. These verification weights
are labeled smoke-only and cannot populate validation.
Two control matches on the first validation seed verify actual Windows two-worker
execution before expensive training. They are recorded as preflight probes, not
selection evidence; the frozen protocol is unchanged by their outcomes. Runtime
Python/Torch/NumPy/Shapely/Gymnasium versions are recorded and checked.
Source hashes normalize Python line endings and include belief/actor/truth code; metadata records Git SHA,
model digests and complete configuration. Hashes are trainer-only.

Stop the entire stage on nonfinite packets/rewards/losses/weights, illegal or masked
decisions, navigation failures, exhausted partitions, changed sources/frozen models,
checkpoint corruption or failed reproduction. A shared stop file halts validation
workers at decision boundaries; pending jobs cancel. A two-hour training limit per
model and six-hour total stage deadline prevent unexpected overruns. Previously
written checkpoints and failure records remain; there is no automatic retry or
resume, and output directories cannot be reused. Task/voting weakness is reported
as a scientific result rather than silently extending training.

```powershell
python -m phase6.study --output artifacts/phase6/runs/seed17-stage-a
```

The runner requires a clean committed source checkout. Source files stay fixed
while it runs; documentation/report curation can follow. All outputs are local
ignored files until curated evidence is committed. Report actual elapsed training,
simulated exposure, task contribution, correct/incorrect votes, skips, reports,
policy choices, uncertainty and failures. One initialization does not establish
replicated memory effects or final robustness. Report measured estimates before
requesting approval for seeds 29 and 43; do not merge automatically.
