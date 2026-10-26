# Approved Phase 6.1 seed-29/43 continuation — 26 October 2026

The owner approved exactly six remaining 8,192-decision models: full,
history_no_belief and current_only at initializations 29 and 43. The estimate is
4.28 additional hours with a hard planning ceiling of 6.42 hours. This extends
PR #4 head `8187d51b6bbeb506fd2a59f25504adb368ab5006`; do not integrate with main,
merge PRs #3/#4/#5, edit production or duplicate the published roadmap here.

The approved continuation is complete: see the [nine-model validation report](benchmarks/phase6/remaining-six/README.md).
Actual stage time was 3.644 hours; final partitions remain unopened and require
separate approval. The design/protocol below was fixed before training.

## Reviewed design and context map

| File | Minimal responsibility / change |
|---|---|
| phase6/study_training.py | Parameterize only the existing Torch/NumPy initialization and seed metadata; permit 17/29/43 |
| phase6/study_validation.py | Match checkpoint seed to the job's authorized initialization |
| phase6/study.py | Parameterize bounded preflight; retain historical seed-17 CLI behavior |
| phase6/study_continuation.py | Fix six approved runs, verify/reuse pilot evidence, source/dependency checks, budget forecasts/deadline, frozen screening/selection/validation |
| phase6/study_comparison.py | Nine-run summaries, voting/task mistakes, per-initialization spread and paired scenario uncertainty |
| tests/test_phase6_study.py, tests/test_phase6_continuation.py | Existing guards, initialization reproducibility, source/seed/partition/budget/preservation/orchestration checks |

Reviewed alternatives: changing global checkpoint compatibility would weaken
provenance; copying a second trainer would risk drift. The selected design keeps
the existing trainer and strict loader, with explicit initialization arguments.
Old checkpoint metadata is never rewritten. The original source commit, source
hashes, model hashes, runtime versions and all pilot rows remain pinned. Old
checkpoints retain their compatible historical source; new checkpoints pin the
extension sources. All sources outside the five extension files must match the
pilot exactly. Actor/physics/observations, PPO, rewards, network architecture and
selection functions remain unchanged. No database/UI migration is needed.

## Frozen scientific protocol

Use the recorded 24-second options, Euclidean descriptors, lr 0.0003, rollout
256, entropy 0.02, gamma 0.99 and unchanged PPO epochs/minibatches. Same train
seeds, paired ID validation seeds 1220000–1220199, opponents and role-specific
vision 4.5/6.75. Only input condition and registered initialization differ.
Independent initializations must differ, and conditions at one initialization
must share the same initial tensor digest. Training is serial; evaluation uses
at most two workers with one Torch thread each. No concurrency benchmark occurs.

Save every checkpoint at 2048/4096/6144/8192 and final weights to fresh directories.
Screen each on the same 20 validation seeds; rank wins, then own tasks, then
earliest step. Lock selections before 200-match validation. Reuse all original
controls; 480 new screening + 1200 extended = 1680 additional executions. The
total nine-model study has 3320 executions, with screening reused as a prefix.
Compare all 120 new selected screening rows to their repeated extended rows.
No final familiar/patient seeds or seed-17 research training are executed.

## Verification, budget and stop rules

Before long runs: targeted tests pass; each new initialization repeats bounded
smoke training and development inference exactly; pilot models, all source hashes,
runtime dependency versions, selection and match records verify unchanged.
Preflight smoke weights remain excluded from screening. Record and preserve both
source manifests; never silently relax compatibility or rewrite pilot metadata.

Use the existing per-model two-hour cap and fail-safe checks. Stop on nonfinite
values, illegal/nav failures, changed frozen/source/model hashes, failed replay,
partition mismatch or worker failure. Shared validation stop flags propagate.
Immutable output directories, selection locks and existing checkpoint preservation
remain mandatory. No automatic retry/resume or budget extension is allowed.
Record measured wall time and forecasts at model/phase boundaries; stop for
approval if projected total exceeds 6.42 hours. A shared 6.42-hour stage deadline
also interrupts decisions if the forecast proves optimistic. Verification within
the stage is included; separately record implementation-test time. No training
protocol changes are authorized to meet the ceiling.

## Reporting and gate

Report every initialization/condition and the reused controls: wins with Wilson
intervals, own tasks, vote participation, correct/incorrect cast votes, skipped
votes, reports and correct/innocent team ejections. Do not attribute team ejections
solely to the focal vote. Show mean/SD/range across three training seeds. Keep
paired scenario uncertainty separate from initialization spread; the same 200
scenarios across three policies are not 600 independent scenarios. Preserve all
new weights, raw records, model/source digests, exact configurations and timings.
Do not claim final robustness or Phase 6.1 acceptance from validation alone.
Stop and request separate approval before final familiar/patient evaluation.
