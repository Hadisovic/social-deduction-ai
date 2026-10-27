# Crewmate V2 preparation results — approval pending

2026-10-27. Cleanup commit `27ff525`, based on `8e99383`, isolated on
`codex/crewmate-v2-cleanup-performance`. Source hashes in [benchmark-results.json](benchmark-results.json)
identify the exact tested collection implementation. Nothing was pushed, merged or deployed.
Full seed-17 A/B training, seed-29 replication and final partitions remain unopened.
Phase 6.1 acceptance remains pending.

## Safe repository cleanup

- Archived the earlier README in `archive/research/README-before-v2-organization.md`,
  preserving its research findings and updating relative links. Root README now explains
  the current V2 work, protections, project structure and safe entry points.
- Reorganized **16 small local JUnit XML files, 115,660 bytes**, into ignored
  `local-runs/engineering-tests/`. Original/new paths and byte hashes are in the
  [cleanup manifest](cleanup-manifest.json). No raw diagnostic directory, checkpoint,
  model or large binary archive was moved or duplicated.
- **Files removed: none. Python entry points relocated: none.** Dependency inspection
  found navigation/geometry/environment/watch imports still used by existing tests and
  historical workflows. Unknown partner/spectator dependencies were retained explicitly
  in the [archive index](../../archive/README.md).
- Improved output-specific ignore rules without hiding all historical `artifacts/`.
  The original source/asset/model snapshot contains **234 protected files**. Cleanup
  verification and imports passed; [165 tests passed](cleanup-validation.json) before
  performance code was introduced. Existing curated benchmark/results documents stay intact.

The main checkout belongs to spectator development. It was not edited. The impostor
branch was neither checked out, edited, reset, merged nor pushed. Remote references were
fetched read-only before creating our isolated branch; no changes were pulled into other
working directories. Future integration of research branches remains a separate review.

## Performance measurements

All rows use **eight independently seeded logical game streams, 64 total actionable
choices, one disposable fresh policy and one four-epoch PPO update**. Each case starts
in a fresh interpreter and pays imports, worker/environment startup, neural inference,
physics, optimization, checkpoint save/load and process shutdown. Same initial weights,
scenario streams, opponent population and reward arm (original) in every case. These are
engineering checks, not an A/B reward study or learning evidence. Eight short cases total
**512 choices across eight disposable policies**; no model received the proposed full
512-choice pilot budget. Measured executions total **13.68 minutes**.
The original suite used 733.32s; a separately registered eight-worker confirmation used
87.19s. Combined measured execution stayed below 900s.

Hardware: 13th Gen Intel(R) Core(TM) i7-13700HX, 24 logical CPUs,
31.73 GiB RAM; Python 3.12.10, NumPy 2.5.3,
Torch 2.6.0+cpu. One Torch thread per process. No mandatory renderer or GPU path.

| Configuration | End-to-end s | Choices/s | Simulated s/real s | Average CPU cores | Peak summed working set MiB | Peak summed private commit MiB |
|---|---:|---:|---:|---:|---:|---:|
| Serial reference | 118.89 | 0.538 | 11.51 | 0.96 | 364 | 1214 |
| 1 / cached unbatched | 118.82 | 0.539 | 11.51 | 0.96 | 449 | 1213 |
| 1 / batched | 117.57 | 0.544 | 11.63 | 0.96 | 410 | 1209 |
| 2 / batched | 104.67 | 0.611 | 13.07 | 1.20 | 964 | 3188 |
| 4 / batched | 84.56 | 0.757 | 16.17 | 1.72 | 1580 | 5296 |
| 8 / batched | 90.21 | 0.709 | 15.16 | 2.35 | 2751 | 9488 |
| 4 / repeat | 98.47 | 0.650 | 13.89 | 1.69 | 1576 | 5293 |
| 8 / repeat | 87.19 | 0.734 | 15.69 | 2.45 | 2669 | 9485 |

Every case advanced **1,367.8 simulated seconds** and completed three episodes, with
ongoing streams retained as partial rather than terminal failures. Actor/action traces,
final actor packets, forced waits and simulated experience match across all configurations.
All batched configurations also have identical post-update tensors. Both four- and eight-
worker repetitions reproduce their weights and traces exactly. Checkpoint roundtrips passed.

**Recommend four worker processes, eight logical environments.** Four's first run was
fastest; repeated means are 91.52s for four and 88.70s for eight.
The latter is only 3.2% faster on these two-case means,
within the 10% practical tie rule registered before the confirmation. Four uses roughly
1.54 GiB summed working set / 5.17 GiB private commit versus eight's 2.69 / 9.27 GiB.
Summed working sets include shared pages repeatedly, and commit is not resident RAM.
Measurements sample our process tree every100ms and can miss brief peaks; they do not
measure unrelated applications or whole-machine load. Four averages about 1.7 CPU cores,
roughly7% of this machine's24 logical CPUs. Slow held options and barrier stragglers limit
utilization; increasing workers does not guarantee lower wall time.

Four's mean throughput is **0.699 choices/s and 14.95
simulated s/real s**, a **1.30×** speedup over the matched
serial reference (0.538, 11.51).
The observed four-worker speedup spans **1.21–1.41×**.
The cached-unbatched and batched-one-worker results differ by about1%, below the observed
repeat variation: no material batching-only speedup is established. Batched inference is
simple and remains useful as a central policy boundary; no GPU/microbatch queue was added.

This reference reproduces the original computational pattern (single-observation inference
plus separate next-value calls), but uses the same eight logical streams for fair comparison.
It is not the earlier one-game smoke. That historical pair achieved64 choices in85.90s
(excluding top-level overhead), approximately0.745 choices/s and12.30 simulated s/real s;
its different scenarios/experience prevent attributing its difference to optimization.
No original benchmark result was overwritten. Timing order, thermal/background variation,
short trajectories and one initialization limit performance generalization; no statistical
confidence interval or learned-policy improvement is claimed.

## Correctness and compatibility

Final regression: **174 passed in101.24s**, including all165 cleanup checks and nine new
collection tests. [Verification](validation.json) records the preserved234 source/asset/
checkpoint files, all nine selected checkpoint hashes, frozen releases, unchanged25+19
private diagnostic files, unchanged25 earlier smoke files and successful older V2 smoke
checkpoint loading. New tests cover process isolation, distinct seeds, legal masks,
termination, reward reconciliation, independent physical-time GAE, optimizer equivalence,
immutable checkpoints, live-prefix replay, exact resumed updates and the denied pilot gate.

Each game owns its seed stream, engine, controller objects, observation history, actor and
public memory, task/ghost state and reward ledger. Processes may reuse static map geometry,
a deterministic geometry-only route cache and a read-only frozen belief predictor; none
holds another game's evidence. The parent alone owns the trainable policy/Adam/update RNG.
Action sampling uses independent per-stream Torch generators and deterministic slot order.

Each128-choice rollout is collected with one fixed policy. Advantages are computed separately
for each environment's trajectory, using unchanged physical-time gamma/lambda and terminal
flags, then concatenated and normalized once. Nonterminal budget cutoffs bootstrap from the
critic; trajectories never carry credit into another game. Workers have equal choice quotas,
so slow games are not discarded to fill batches faster. Cached next values reuse the same
unchanged policy and packets within the rollout. No physics ticks or observation updates
are skipped. The opt-in reward ledger, legal-action masks, reporting/voting choices, lobby,
vision, speed, walls, cooldowns, tasks and meetings remain unchanged.

A separately locked immutable checkpoint records policy/Adam/update RNG, per-stream action
RNG and current-episode prefixes. Restore reconstructs actual seeded games and replays
recorded actions, then verifies packet/time/reward accounting before continuation. Tiny
fixture tests verify exact subsequent actions, returns and optimized weights. Reconstruction
can be slow and is not an automatic extension of a spent budget. Source/topology/lobby or
packet mismatches fail. Old learning source hashes stay unchanged because collection is a
separate subpackage, so historical and earlier V2 smoke checkpoint loaders remain strict.
No self-play, learned impostor training or strategic report/vote override was added.

See [spectator integration](SPECTATOR_INTERFACE.md): shared renderer/assets/interfaces stay
unchanged; optional JSONL metadata/checkpoint reads require no rendering in training.
Full privileged tick recording and live dashboard work remain outside this preparation.
An interrupted uncommitted rollout may have unwritten episode evidence; checkpoints preserve
the last complete update and permit audited replay. No claim of complete tick-history storage.

## Revised controlled pilot for owner review

[Execution proposal](../../configs/crewmate-v2-learning/parallel-execution-proposal.json) and
[unapproved record](../../configs/crewmate-v2-learning/parallel-approval-required.json) bind
this exact runtime and execution. **Approved remains false.** The earlier serial proposal,
configurations and smoke records remain unchanged and available for comparison.

- Original versus exact requested shaped rewards, fresh identical initialization17,
  same51,778-parameter network and frozen shared Phase4 predictor.
- **512 actionable choices/model**, four128-choice rollouts/model, four epochs/update,
  minibatch64, same learning rate/entropy/clip/value/gradient settings, physical-time
  gamma.99/lambda.95; immutable128/256/384/512 checkpoints, fixed512 selection only.
- Candidate execution: **four workers / eight logical streams**. Each update now has
  **16 choices per stream**, rather than a single128-step trajectory. This changes the
  GAE/bootstrap horizon and scenario scheduling. It is a scientifically matched A/B
  comparison, but not identical to the previous serial protocol and requires explicit
  approval. Increasing total choices is not necessary to preserve equal A/B budgets and
  is not authorized automatically. A later learning-quality comparison of horizons would
  be a separate experiment, not a performance conclusion from these timings.
- Fresh training seeds remain2,700,000–2,709,999. Stream slot s at local episode e uses
  start+8e+s, rotation(8e+s)%3: distinct seeds across streams. Both arms use the same
  assignment and legitimate varied opponents. This replaces serial reuse of each seed
  with three report rotations; used/partial contexts must be recorded. Episode counts and
  simulated experience can differ by learned behavior, and are reported beside choices.
- Unchanged20 validation games/model,40 model-games over four paired scenarios,
  original/varied reporter conditions and ordinary/deduction checks. Validation stays
  serial for this proposal. No validation or reserved final-test seeds were used in
  engineering. Seed29 remains separately gated.
- Conservative measured training projection **26.3min** for both models,
  plus retained serial validation estimate **58.8min** = **85.1min**.
  The2× planning allowance is **170.1min**, under the unchanged
  **180min hard stop** including training and validation. This extrapolates the slower
  four-worker repetition (including startup) and the existing smoke-to900s validation
  rate. Policy-dependent experience can differ; the forecast is not a guaranteed bound.
  Preflight recomputes it and refuses an over-ceiling estimate.
- Stop on wall/source/core-library/config drift, illegal/masked/navigation actions,
  frozen-belief mutation, nonfinite values, KL>.10, clipping>.80 or preclip gradient>100.
  Keep completed checkpoints/logs; no silent retries, earlier-checkpoint selection or
  mid-pilot reward/curriculum adjustments. Preserve reporter-only controls, rare-opportunity
  limitations and the +2/−1 guessing-vote incentive from the original proposal.

Owner approval of both preparation results and this execution revision is required before
actual crewmate training. No benchmark authorizes the full pilot, replication or acceptance.

## Reproduction entry points

Run from this research checkout using the validated environment and fresh output paths:

```powershell
python -m pytest tests/test_v2_collection.py -q --basetemp=local-runs/collection-tests-fresh
python -m crewmate_v2.collection.benchmark --output local-runs/collection-benchmark-fresh --wall-seconds 900
python -m crewmate_v2.collection.train --help
```

The benchmark command intentionally performs bounded updates on disposable policies; it is
not a training-launch command. Keep its records local. The current approved:false record
must not be changed until the owner approves the revised execution and compute allowance.
The complete first benchmark record and separate eight-worker confirmation remain unchanged
under ignored `local-runs/`; curated JSON contains aggregate engineering metrics and hashes,
without raw observation records or post-game role datasets.
