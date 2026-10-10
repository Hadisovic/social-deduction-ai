# Phase 6 roadmap — agreed continuation, 10 October 2026

> Publication scope (11 October 2026): this document records the approved plan and
> development-branch evidence. PRs #3/#4 remain unmerged. Phase 6 runtime,
> runners and checkpoints described below are **not included in main** by
> this documentation publication. Future capabilities remain planned.

This is the current future roadmap. It supersedes the earlier ordering that put
alternating adaptation in 6.3 and population expansion in 6.4. It does not change
the completed seed-17 pilot or the frozen nine-model crewmate protocol.
**This update authorizes documentation only, not implementation or compute.**

## Verified development state and continuation point

- Latest fetched main: `6b78b08e41d0791e49ff486b996d7490666db970`.
- Foundation [PR #3](https://github.com/Hadisovic/social-deduction-ai/pull/3):
  open, unmerged, head `16101131e9787c9bcbaebe250d25d6dd0d3574fb`.
- Pilot [PR #4](https://github.com/Hadisovic/social-deduction-ai/pull/4):
  open, unmerged, head `8187d51b6bbeb506fd2a59f25504adb368ab5006`.
- Phase 6.0 vision was verified on the foundation branch; Phase 6.1 has three
  completed seed-17 models on the pilot branch.
  [Pilot evidence](https://github.com/Hadisovic/social-deduction-ai/blob/8187d51b6bbeb506fd2a59f25504adb368ab5006/docs/benchmarks/phase6/seed17-stage/README.md) remains unchanged.
  No learned impostor, enlarged match, dashboard or self-play is operational.
- The executable simulator still uses four crewmates and one impostor. Larger
  populations and exactly two impostors require future reviewed support; they
  cannot be obtained simply by starting additional training processes.

All checkpoints, raw local runs, frozen historical releases and prior results
must remain available. PRs #3 and #4 are not merged or rewritten by this update.
The original documentation PR #5 is stacked on #4. For publication, a clean
documentation-only branch starts from latest main; it imports no development
commits, runtime files, checkpoints or benchmark records. PRs #3/#4/#5 stay intact.

## Milestones and gates

| Milestone | Agreed objective | Current status / next gate |
|---|---|---|
| 6.0 | Preserve baseline and verify role-specific vision | Verified on unmerged PR #3; implementation review pending; historical releases frozen |
| 6.1 | Finish the existing nine-model learned-crewmate study without protocol changes | Three of nine trained/validated; remaining six await execution approval; final tests remain closed |
| 6.2 | Train and validate a learned impostor against learned and scripted crewmates | Planned; requires accepted crewmate evaluation and a reviewed impostor observation/action contract |
| 6.3 | Gradually reach fully AI-controlled matches with exactly two impostors | Planned curriculum: 6 crew + 2 impostors, **8 crew + 2 impostors (main target)**, then 10 crew + 2 impostors |
| 6.4 | Alternating training, historical opponent pools, competitive adaptation and eventual multi-agent self-play | Planned after standalone policies and population support pass their gates |

### 6.1 — finish the registered study

Keep `full`, `history_no_belief` and `current_only`, each at initialization seeds
17, 29 and 43, each trained for 8,192 decisions: nine models / 73,728 decisions.
Seed 17 is complete; the six seed-29/43 models are the next intended compute step.
The runner on unmerged PR #4 deliberately accepts seed 17 only; any future extension must
preserve the recorded experimental conditions and compatible source provenance.
Do not overwrite checkpoint metadata to bypass source compatibility.

Preserve 24-second task/travel holds, Euclidean descriptors, learning rate 0.0003,
rollout 256, constant entropy coefficient 0.02, gamma 0.99, unchanged PPO epochs,
minibatches, rewards, opponents, action semantics and training-seed order. Input
condition and registered initialization seed remain the experimental factors.
Use the same actor/physics behavior, sampled training and greedy validation.

Save checkpoints at 2,048 / 4,096 / 6,144 / 8,192 plus final weights. Screen each
on the same 20 ID validation seeds, rank by crew wins then own tasks then earliest
checkpoint, and lock selection before the 200-match extended validation. Reuse
the already evaluated idle/random/scripted/frozen Phase 5 transfer controls under
identical rules. The complete study totals 3,320 validation executions: 1,640 done
and 1,680 remaining. The 20 screening seeds are a prefix of the 200, not additional
independent evidence. Keep every checkpoint and report all conditions/seeds.

Retain the fresh partitions, information-boundary checks, finite-value/source/
frozen-model guards, deadline rules and validation-only selection in the
[staged protocol](phase6_1_staged_protocol.md) and
[experiment plan](phase6_experiment_plan.md). Training remains serial for this
registered study; do not introduce concurrency tuning mid-study. Task/voting
mechanics, network architecture and reward shaping are not being changed.

Measured estimate for the remaining six: **4.28 additional hours**, provisional
**3.42–6.42 hours**, excluding final tests and new implementation/benchmarks.
Final familiar and patient partitions remain unopened pending separate approval
and an immutable release selection. Evaluate task contribution, reports, vote
coverage, mistakes, ejections, survival, seed spread and paired uncertainty.
The full pilot never skipped; ablations almost never voted. These are unresolved
behavioral findings, not a reason to silently tune the registered comparison.

### 6.2 — first learned impostor

Start with one learned impostor in the existing five-player setting against
frozen learned and scripted crewmates, with explicit opponent checkpoint hashes
and a registered mixture. Train and evaluate separately against both groups;
use fresh, reviewed partitions and keep model selection separate from final tests.
Training against fixed opponents comes before simultaneous adaptation.

Audit the role's legitimate observation and legal supported actions first. Reuse
existing movement, elimination, fake-task, report, structured claim and vote
mechanics where supported. Sabotage, vents and free-form dialogue are not current
research capabilities and are not added by this roadmap. Report impostor wins,
eliminations, survival, detection/ejection, illegal attempts, meeting behavior and
performance against held-out crew policies. Do not advance from one lucky run.

### 6.3 — population curriculum, exactly two impostors

Register and validate player-count/role-count support before enlarged training.
Progress through **6 crew + 2 impostors (8 total)**, **8 crew + 2 impostors
(10 total; main target)**, then **10 crew + 2 impostors (12 total)**. Each stage
always has exactly two impostors. Increase the number of learned participants
gradually until all players at that population size are AI-controlled.

Audit spawn/collision safety, variable-size observations and legal action masks,
identity-independent features, two-impostor belief labels, role-count-aware match
outcomes and replay compatibility. Do not assume the current five-player model
or one-impostor probability representation already supports these cases. Preserve
task ownership and existing task/vote action semantics; any necessary structural
generalization requires a separate review without retroactively changing 6.1.

Every agent receives its own projected observation and independent memory.
Model weights may be shared in a future reviewed design; observation/history
state must not be shared. Define any legitimate impostor teammate knowledge
explicitly rather than granting unrestricted truth. Freeze policies and opponent
mixtures while measuring each population stage. Gate expansion on reproducible
safe matches, useful crew task/vote behavior and useful impostor behavior; report
both roles, individual contribution and uncertainty, not just team win rates.

### 6.4 — adaptation and eventual self-play

Begin with alternating updates: freeze crew, train impostors, validate, retain
the checkpoint; then freeze those impostors and train crew. Keep historical
opponent pools and evaluate cross-play against older and new policies so gains
against one opponent do not hide forgetting. Record opponent versions, mixture
weights, learner role, sources and selection evidence for every cycle.

Only after alternating learning is stable should broader multi-agent self-play
be considered. Register evaluation populations and unseen behavioral families
before training. Measure cooperation, evidence use, deception, false accusations,
strategy diversity, adaptation and forgetting. Self-play remains a planned
capability, not an implemented result or a synonym for fully AI-controlled games.

## Planned local training dashboard and game viewer

Plan a local, read-only dashboard displaying run/condition/seed, decisions and
simulated experience, rollout/checkpoint progress, rewards, crew/impostor win
rates, own tasks, reports, correct/incorrect votes, cast counts, skips and failures.
Show reward, win/task/vote and PPO diagnostic graphs, including entropy, value
loss, KL and clipping. Clearly separate sampled training metrics, checkpoint
validation and eventual final-test results; display sample counts and missing
values rather than treating an absent vote denominator as perfect accuracy.

Show per-process and aggregate CPU/RAM usage, actual elapsed time and an ETA
derived from observed throughput and remaining work. Separate training from
validation estimates, refresh the estimate as rates change, and label it as
provisional. Concurrency may change throughput; decision progress alone is not
simulated experience or a reliable wall-time estimate.

Pair this with a separate checkpoint/replay-based Skeld viewer using the exact
recorded source/config/model version. Preserve the existing Phase 4/5 viewers.
Label checkpoint playback and replay explicitly; do not imply the viewer is
rendering every live training step. Planned spectator controls can select an
agent, show its legitimate observations/vision and policy options, and inspect
clearly labeled privileged overlays. Viewer/debug truth must never feed back
into any actor or training input. Poll telemetry without blocking training or
changing RNG consumption. Dashboard and viewer integration need a separate
implementation review and budget; neither is built in this documentation pass.

## Planned parallel-training benchmark

Benchmark **one versus two concurrent training processes**, optionally three
only if measured total throughput improves and reproducibility remains intact.
Use a separately approved, bounded benchmark outside the frozen 6.1 study with
the same work budgets, source/config hashes and thread settings across treatments.
Compare total batch wall time, decisions and simulated seconds per wall second,
CPU/RAM usage, failures and checkpoint/transition reproducibility. Report process
time separately from wall time; more workers do not guarantee faster training.

Use process isolation and independent, explicitly initialized Python/NumPy/Torch
RNG states and opponent streams; never share a mutable environment, agent memory
or optimizer across processes. Keep independent output directories, logs, locks
and checkpoints. Paired registered match seeds can be reused for comparisons
without sharing RNG objects. Record thread counts, dependencies and hardware;
verify each run's initialization and repeatable policy/transition digests against
its single-process counterpart. Keep existing stop/preservation rules and reject
a concurrency setting that introduces failures or unreproducible behavior.
Choose future concurrency from these measurements, not CPU core count alone.

## Research invariants

- Crew vision **4.5**, impostor vision **6.75**, with native wall/opaque-prop LOS.
- Private role-appropriate observations, independent per-agent memory, individual
  task ownership and no hidden positions, other players' private tasks or future
  events in policy inputs. Privileged metrics stay trainer/spectator-only.
- No task or voting mechanic changes now; the nine-model study stays frozen.
- Preserve old rules, released weights, results, replays and source/model hashes.
- Fresh partitions, paired comparisons where appropriate, immutable validation
  selection and separately approved final testing. Report failures and seed
  uncertainty; no premature claims of superiority, scalability or self-play.

## Reviewed documentation context map

| Files | Documentation change |
|---|---|
| `README.md`, `HANDOFF.md`, `docs/PROJECT_PROGRESS.md` | Current direction and next-session gate, retaining historical results |
| `docs/phase6_roadmap.md` | Canonical future stages, dashboard/viewer, concurrency and integrity plan |
| `docs/phase6_experiment_plan.md`, `docs/phase6_1_staged_protocol.md` | Link current roadmap and preserve frozen protocol; update later milestone ordering |
| `docs/project_architecture.md` | Distinguish executable five-player design from planned population expansion |
| `project-status.json`, Phase 6 copy in `site/app.js` and `site/index.html` | Mark Phase 6 in progress; future milestones planned and branch-only evidence qualified |
| `site/scripts/validate-site.mjs` | Validate published roadmap metadata and preserve historical Phase 5 facts |

Dependencies reviewed: main already supports an in-progress major-phase status.
The existing mission presents the future roadmap through text; no new dashboard
or checkpoint UI is imported. Static validation is updated for the planning
metadata: 6.0 implementation in review, 6.1 in progress, 6.2–6.4 planned. Validate
JSON, source/built site, links, browser behavior and the documentation-only diff;
compare protected files before/after. No simulator, training, model or benchmark
change is part of this publication.
