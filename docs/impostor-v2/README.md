# Impostor V2 engineering verification

Implementation is on `codex/impostor-environment-v2`, based on V2 commit
`e62ba0b6c33ffd0317f78e18b4289d778cd5af2b`. This is an independent development branch,
not the owner's crewmate training branch.

See the [collaborator handoff](../phase6_parallel_research_handoff.md) for the frozen
model adapter, information boundaries, safeguards, new observation/action interface,
reward proposal, source references and approval gates.

**Engineering implementation and verification complete, 27 October 2026.**
No RL training run, reserved final evaluation, merge or deployment has occurred.
The new 48,738-parameter impostor network is untrained.

## What passed

| Check | Result |
|---|---|
| New impostor tests | 27 passed |
| V2 mechanics, ghost privacy and integrity tests | 45 passed |
| Historical rules, boundaries, memory, vision, navigation and policy-interface regression | 167 passed |
| Final regression total | **239 passed**, 0 failures; 146.02 seconds |
| Frozen files | All nine selected checkpoints, two selection locks and Phase 4/5 releases preserved |
| Original strict checkpoint loading | All six continuation checkpoints loaded; seed-17 migration not claimed |
| Final engineering benchmark | **10/10 completed**, 374.30 seconds (6 min 14 sec) |
| Canonical benchmark illegal actions / navigation failures | **0 / 0** |
| Reference / optimized / repeat | Identical impostor packet stream, opponent/focal action trace and semantic event trace |
| Frozen weights after benchmark | File and tensor checks unchanged; no gradients |

Evidence: [test record](test-results.json), [full aggregate benchmark](benchmark-results.json),
[source/configuration manifest](manifest.json), [live session checklist](SESSION_CHECKLIST.md).
Implementation revision: `cb9b5f4fc3a606383f3b04815151602cb54f45ec`; the benchmark started
with a clean checkout. Later commits only document the results/checklist.
Benchmark duration covers its timed match loop, excluding initial Python/model setup.

The new tests cover actual full-29 proposals, unchanged legacy feature encoding,
all four safeguard conditions, body approach/report, witness memory and votes,
someone else reporting first, unavailable killers, deaths, walls, fake kill claims,
isolated memories, unseen-position noninterference, legal action candidates, deterministic
reset/inference, player-row permutation, terminal packets, seed separation and exact-tick
coalescing with reward/discount composition. Existing V2 tests retain the stronger
ghost-secret noninterference and permanent-task-ownership checks.

One old synthetic optimizer/serialization unit test passed during the first regression
attempt on a disposable fresh policy. It used eight synthetic samples, did not train a
gameplay agent and did not alter frozen weights. It was deselected in the final run.
No impostor optimizer/update or RL training run was executed. Initial fixture mistakes
and the corrected WAIT/no-submission audit mismatch are disclosed in the test record;
their local outputs were preserved, not substituted for final evidence.

## Natural match results — engineering controls, not learned performance

Seeds below are development IDs 3500000 and 3500001. Only **two distinct scenarios**
are used; control variants and repetitions are not independent evidence of win rate.
All keep the canonical 45-second cooldown and 15/120-second meeting durations.

| Impostor control / mode | Seed suffix | First legal kill / first actual kill (s) | Kills | Crew reports | End (game seconds) | Wall seconds |
|---|---:|---|---:|---:|---|---:|
| Hunter / ordinary | 0 | 59.4 / 59.4 | 1 | 0 | Tasks, 266.6 | 31.43 |
| Hunter / ordinary | 1 | 58.8 / 58.8 | 2 | 0 | Tasks, 408.0 | 40.64 |
| Hunter / deduction-only | 0 | 59.4 / 59.4 | 3 | 0 | Impostor parity, 486.0 | 55.46 |
| Hunter / deduction-only | 1 | 58.8 / 58.8 | 3 | 0 | Impostor parity, 445.8 | 52.57 |
| Idle / ordinary | 0 | 113.4 / none | 0 | 0 | Tasks, 136.2 | 31.07 |
| Untrained sampled network / ordinary | 0 | 90.0 / none | 0 | 0 | Tasks, 136.2 | 44.39 |
| Hunter without self-report / ordinary | 0 | 59.4 / 59.4 | 2 | 1 | Tasks, 266.6 | 22.28 |
| Hunter without self-report / ordinary | 1 | 58.8 / 58.8 | 1 | 1 | Tasks, 273.0 | 28.30 |

Two additional runs repeat the first ordinary hunter scenario for reference/optimized
equivalence. In total: 2,951.6 simulated seconds, 4,924 scheduled observation windows
and 2,482 meaningful choices (counts include those repetitions).
Here “meaningful” means more than the lone no-submission option, not necessarily a
kill opportunity or a strategically useful choice.

**Interpretation:** ordinary games permit kills before task victory: none of the six
ordinary control executions ended before a legal kill opportunity. But task victory
constrained every ordinary control. Deduction-only allowed three kills and parity in
the two paired scripted cases; that changes the objective and is not evidence of a
better curriculum. Keep ordinary mode for the first pilot and compare modes explicitly
only if approved. Do not change cooldowns or task speed to make wins easier.

The ordinary hunter self-reported every body it reported, reproducing the historical
reporter confound. The non-reporting control therefore matters: living crew reported in
both scenarios. In one, the baseline controller already chose the report (no changed
proposal); in the other, the guard changed the proposal. Body discovery did not force
an accusation: one incorrect crew vote remains in the non-witnessed control. This is
expected—the wrapper guarantees eyewitness voting, not omniscient deduction.

There were no naturally witnessed kills, ejections or timeouts in these ten runs.
Controlled tests, not these natural traces, establish the witnessed-kill reporting/voting
sequence. Ghost tasks completed after kills (one to three per active scripted-control
match), but these comparisons do **not** isolate their causal effect on victory time.
That needs a separately registered counterfactual study, not removal of canonical ghosts.

## Speed and compute

Single CPU process, one Torch thread, 32 logical CPUs available, Windows 11,
Python 3.12.3, Torch 2.9.1+cpu, NumPy 1.26.4. The manifest records the processor's
reported family/model string; a friendly processor-name lookup was unavailable.
No GPU, rendering, sleeping, numerical timer compression or parallel training was used.

The matched reference run took **37.11 s**; the optimized repeat took **26.58 s**:
**1.40× speed ratio**, one scenario, not a general PPO speedup. Meaningful choices
remained 223, while exposed Gym calls fell from 445 to 223 by coalescing forced empty
windows. Every physics/opponent/observation tick remained. Raw calls/second are not
comparable when one path includes forced empty actions. Repeated timings vary with
caches and machine load; no confidence interval is justified from this one pair.

The untrained network's complete rollout took **44.39 s for 227 decisions (5.11/s)**.
That is measured inference/environment throughput, **not optimizer throughput**.
The new driver intentionally does not invoke PPO; future trainer wiring must use
seconds-aware GAE, the new packet keys, private sampling RNG and an optimizer containing
only impostor parameters. Validate that wiring after approval before a learning run.

## Recommended first approval budget

Request **two ordinary-mode runs ×4,096 decisions**, seeds 71 and 83, plus **20 paired
validation matches per run** (40 total). Save intermediate checkpoints, but evaluate
the preselected final checkpoint in this pilot to avoid a small-data checkpoint search.
Use both opponent safeguards and the exact pinned full-29 model. No final tests.

At the untrained rollout rate, 8,192 decisions alone project to about **27 minutes**.
The 40 validation matches project to roughly **15–40 minutes** using these measured
control durations. Learning can change episode duration and meeting frequency; optimizer
overhead is unmeasured. Therefore plan **1–2.5 hours total, with a hard 3-hour ceiling**,
sequential local CPU runs. Stop and report rather than extend automatically. No exact
finish time or learned performance is promised by the engineering benchmark.

Review kills, witnessed kills, legal opportunities, reports and report sources,
win reasons, task completion, ejections, incorrect crew votes, skips, timeouts, invalid
actions, policy entropy and learning curves. Do not judge a policy only by kills or
win rate against one frozen opponent. The broader cross-play, ablation, multiple-seed
and final-test study requires a later budget and acceptance gate.

**Readiness:** the environment, frozen adapter, safeguards, actor interface and inference
path are ready for technical review and a small approved pilot. The crew transfer is
lossy, ordinary task pressure is strong, reporter bias remains with self-reporting scripts,
and no learned impostor quality has been established. A training runner and its optimizer
checks are the next implementation step after the design/budget is approved; large-scale
training is not justified yet. The friend's crewmate work remains independent.

## Remaining preparation

The shared chat has now been read through Edge. See
[TRAINING_NEXT_STEPS.md](TRAINING_NEXT_STEPS.md) for its project conclusions and the
remaining trainer/evaluator/resume checks. Allow 45–90 minutes for preparation and
about **2–4 hours to the first pilot results and graphs**, including the proposed
1–2.5-hour learning/validation stage. Training has not started.

## Reproduce engineering checks

From the repository root with its existing dependencies:

```powershell
python -m pytest tests/test_impostor_research.py tests/test_crewmate_v2.py tests/test_crewmate_v2_privacy.py tests/test_crewmate_v2_integrity.py -q --basetemp=temp_impostor_recheck
python tools/benchmark_impostor_v2.py --output temp_impostor_new_benchmark
```

Use a **new** output directory; the benchmark refuses overwriting an existing one.
It executes exactly ten bounded development matches, uses only registered development
IDs and writes aggregate results plus hashes. Raw actor trajectories and role-label
datasets are not published. The full final regression command additionally covered
Phase 3 engine/boundary/bots; Phase 4 memory/boundary; Phase 6 vision/foundations/route
boundary; Phase 5 boundary/core, with the synthetic optimizer test excluded.
