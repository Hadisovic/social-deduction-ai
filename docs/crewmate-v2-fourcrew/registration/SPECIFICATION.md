# Final four-crewmate experiment registration

**Execution update — 2026-10-28:** The owner subsequently authorized this exact registration. Its single execution is complete; [report](../results/registered-2048-seed17-v1/REPORT.md). The specification below is preserved as the pre-authorization snapshot. Phase6.1 acceptance and further work remain pending.

2026-10-27. Experiment **`fourcrew-reporting-2048-seed17-v1`**. **Registered for review; full training is not authorized or started.** Phase 6.1 acceptance remains pending.

The owner approved the overall design. This document and [plan.json](plan.json) resolve the remaining choices before final authorization. The local runner refuses to create a study without an approval record binding the exact plan and runtime hashes. Registration and benchmark work do not grant that approval.

## Fixed comparison

- Exactly four shared-policy crewmates and one scripted impostor. Each living crew independently observes, remembers and samples; weights are shared, knowledge is not.
- A: new four-crew original control. B: shaped-v2. Both start independently from fresh initialization **17**, with identical parameter hashes. A separate zero-update initialization-17 reference is frozen and evaluated without optimization.
- Rewards unchanged: B living own task +1, successful accepted report +1.5, correct/wrong ballot +2/-1, death -5, team win/loss +10/-10, draw -5, skip 0, automatic ghost-task credit 0; time penalty -0.005 per 0.6 simulated seconds. A disables report/vote bonuses and likewise has zero individual ghost-task credit. Historical focal profiles remain unchanged.
- Same `FourCrewPolicy` architecture and actor-local 42 / 4x29 / 4x8 / 96x23 tensors. Same original GAE/PPO: gamma .99, lambda .95, Adam .0003, four epochs, minibatch 64, clipping .2, entropy .02, value coefficient .5, gradient clip .5.
- Four headless simulation workers collect two matches per arm concurrently. Policies freeze during collection; each optimizer updates centrally only after the paired batch. Per-agent/per-match trajectory boundaries, real elapsed-time discounts and terminal settlement remain isolated.
- Lobby remains V2.2 with the canonical 900-second match timeout, 45-second kill cooldown, 15-second discussion, 120-second maximum voting, early all-voted resolution, crew vision 4.5 and impostor vision 6.75. No mechanics or reward retuning.

## Exact development partitions and exposure

| Purpose | Exact seeds | Use |
|---|---|---|
| Registered training | **2780000–2780063**, in ascending order | Up to 64 matched matches per trained arm; stop earlier only by the common registered rule |
| Registered validation | **2790000–2790007** | Eight paired seed scenarios; ordinary plus one controlled context per seed, for each of A/B/zero |
| Representative benchmark | **2760020–2760023** | Four complete engineering matches only; already measured, no study data |
| Prefix/correctness fixtures | **2760050–2760057** and existing engineering fixtures | Legality, privacy and coverage checks only |

All partitions are development-only and disjoint from historical V2 studies and reserved final-test partitions. Training/validation seeds above have **not been executed during readiness**. Seed registration is not evaluation access.

Training index `i=0..63` uses seed `2780000+i`: even indices are natural ordinary games; odd indices use the body-opportunity prefix with deduction-only termination. The exact context/opponent rotation is in plan.json. Each arm receives the same accepted seed/context/opponent schedule, starting at index zero, without cherry-picking after outcomes. Scripted hunter behavior and the seeded prompt/inspect/bypass reporter rotation are matched across arms; scripts receive only their own observations. Realized trajectories can diverge because policies act differently.

## Legal controlled-prefix construction

The prefix is an explicitly scripted **scenario setup before neural control**, not a learned action or a natural-game performance sample. All movement uses legal navigation intents from the normal spawn. There is no teleportation, cooldown editing, evidence injection, task ownership change or lobby change.

1. Public destination waypoints come from the existing map: the EmergencyConsole standing point is the common site; Weapons/Admin/MedBay task standing points provide parking. Parking rotates by case. A trusted scenario allocator identifies the one impostor and a cyclic crew victim; role assignments never enter the actor packet.
2. The attacker and victim legally navigate to the site. Other crew legally navigate to parking locations. At tick **225 / 45.0s**, after the canonical initial cooldown, the impostor submits a legal kill against the registered victim. A missing legal kill or missed route deadline fails the prefix; no state correction is substituted.
3. All four survivors legally navigate to the common site. At tick **375 / 75.0s**, each surviving crew has a real observed corpse and a legal report candidate. The impostor is also present, avoiding a deterministic “only missing player is impostor” position cue. The kill cooldown has not been reset artificially.
4. **Body context:** neural control begins at 75.0s, with no scripted report. The policy can report, wait, navigate or choose any other legal option. The impostor resumes its registered script. Competing report submissions are measured; a race-losing submission gets no accepted-report credit.
5. **Meeting context:** at 75.0s a predetermined innocent or impostor reporter submits a genuine legal report. Neural control begins at tick **376 / 75.2s**, during discussion, before any scripted claim or ballot. No learned actor is forced to report or vote. Prefix reports are labeled setup events and earn zero learning credit.

Every view updates only from that actor's real observations throughout the prefix. Dead memory freezes. At handoff the reward ledgers restart, so prefix kills/reports/task progress/time earn no learning credit, and a prefix death is not charged as a policy-caused death. Legitimate task-bar changes and observed history remain. In controlled cases one crew is already dead at handoff; the three surviving crew use shared PPO, while the ghost acts only through canonical automatic tasks. Ordinary games start all four alive. Victim assignment cycles across crew ranks; no fixed learned actor is deliberately sacrificed across the schedule.

If the model exploits lawful but artificial choreography, that is a limitation of controlled scenarios. Ordinary outcomes remain separate; no generalization claim is made from these prefixes.

## Exact 48-execution validation matrix

All three policies use **greedy** action selection and identical contexts. For seed `2790000+i`, each policy plays one unprefixed ordinary game and this controlled game:

| Index / seed suffix | Controlled context | Setup reporter |
|---|---|---|
| 0 / 0000 | Body opportunity | None forced |
| 1 / 0001 | Body opportunity | None forced |
| 2 / 0002 | Body opportunity | None forced |
| 3 / 0003 | Body opportunity | None forced |
| 4 / 0004 | Meeting | Innocent crew |
| 5 / 0005 | Meeting | Impostor |
| 6 / 0006 | Meeting | Innocent crew |
| 7 / 0007 | Meeting | Impostor |

Thus there are **16 games/policy, 48 executions, eight seed clusters**, not 48 independent scenarios. Within the controlled subset, reporting has four scenario clusters and the reporter-role challenge has four (two per role). Prefix-generated meetings do not count as reports learned by the policy. Ordinary, body-opportunity and meeting results are tabulated separately.

## Budget, overshoot and fixed checkpoint selection

The unit is an **individual non-forced actor decision**, never a joint tick or “512 decisions per crew.” Targets are **2,048 accepted actor decisions per trained arm**. Actor exposure can be unequal because of death and option duration; report actual counts per actor and match.

- Process pairs of training contexts per arm, preserving complete matches and isolated trajectories. The minimum is **16 matched matches/arm** (eight ordinary, eight body-prefix) to establish the reporting quota. The maximum schedule is **64 matched matches/arm**.
- Stop at the **first common completed paired batch** where both arms have at least 2,048 accepted decisions and at least 16 matches. There are no arm-specific extra seeds.
- Acceptable final totals are **2,048–3,072 per arm**. This is a prospectively declared complete-match overshoot allowance, not identical decision exposure. Equal scenario exposure is exact; equal post-policy decision counts are not.
- Before either optimizer updates, reject the entire collected paired batch if either cumulative count would exceed 3,072. Retain its raw artifacts, do not trim trajectories or train one arm alone, and mark the budget endpoint inconclusive. Already completed valid checkpoints stay unchanged.
- If both targets are not reached by 64 matches, stop as schedule-inconclusive. Do not add seeds, lower the target, choose a more favorable checkpoint or silently extend time.
- Select the last common accepted checkpoint satisfying this rule. All intermediate checkpoints are immutable. **Validation never selects or tunes checkpoints.** The zero reference has no updates.

The benchmark's 51 sampled decisions in one complete training-context match showed that a 32-match draft ceiling might exhaust too early. The prospective schedule was therefore expanded to 64 **before any study seed was used**, without changing the target or evaluation coverage.

## Coverage and interpretation rules

- Training: at least **24 legal actor/body pairs over at least eight distinct body-prefix matches per arm**. Three surviving crew see a real reporting choice in each such match. These correlated pairs are not 24 independent bodies or scenarios.
- Validation: at least **12 legal actor/body pairs over four body cases per policy**, plus **12 legal actor/meeting pairs over four setup meetings**, including two innocent and two impostor reporters. A missed construction/coverage quota makes the relevant endpoint inconclusive.
- At least **two actual accepted neural crew reports during training per arm** are required to interpret the reporting-learning endpoint. This is not guaranteed by setup and does not itself demonstrate learning. Fewer reports remain a meaningful negative/inconclusive exposure result.
- Reporting denominators distinguish visible bodies, range/legal candidates, opportunities at decisions versus held options, selections, submissions and accepted meeting-triggering reports. Prefix reports are excluded; zero denominators are unavailable, not zero-percent failures. Actor conversion and per-game report occurrence are both reported because actors compete for the same corpse.

The **primary endpoint** is the proportion of controlled body games with any accepted neural crew report. Report B-A, B-zero and A-zero paired differences with descriptive scenario-cluster bootstrap intervals; there are only four body clusters. Secondary outcomes include ordinary/controlled win reasons, individual/ghost task contribution, votes/skips/errors, claims, survival and the frozen predictor's top-1/NLL/Brier/calibration.

Reporter shortcut controls score accuse-reporter, avoid-reporter and uniform baselines from legitimate public reporter identity. The initial controlled-meeting comparison uses **nonreporter observers and per-game macro weighting**, separately for innocent/impostor reporters, so extra actors or later meetings do not overwhelm the role balance. Include direct-witness strata and unavailable-data indications. Action preferences are not calibrated identity beliefs. The Phase 4 predictor is one shared frozen model, not three newly trained belief models.

Report all effects, including task-dominant or harmful behavior. A higher reporter-conditioned vote accuracy alone is not evidence of deduction. Skipping remains legal; no blanket skip penalty, forced accusation or reward change is introduced. This one-initialization/eight-cluster study cannot complete Phase 6.1 acceptance or establish general strategic intelligence.

## Four-hour safety boundary

The **14,400-second total deadline** starts on authorized study launch and includes model setup, training, validation, checkpointing and result writing. A parent watchdog interrupts at the deadline; match workers check it cooperatively, and the executor terminates only its own unfinished workers. No unrelated viewer or partner process is killed.

Before another training batch, require at least **3,600 seconds reserved for validation plus 300 seconds for the next batch**. Otherwise stop as compute-inconclusive; do not silently reduce the 48 evaluations or pretend smaller training met the target. The reserve is a safeguard, not a guarantee of completion. Preserve partial checkpoints/logs/replays and mark incomplete games honestly. No automatic retries, continuation or ceiling extension.

Also fail closed on source drift, nonfinite/unstable PPO, illegal intentions, prefix/privacy failure, changed zero-reference hash, checkpoint mismatch or incorrect scenario exposure. Preserved historical checkpoints/results, final partitions, impostor branch and existing PRs are outside the experiment.

## Measured readiness cost

One bounded four-job benchmark used **engineering seeds 2760020–2760023**, canonical 900-second match limits and unchanged rewards. All matches reached real outcomes:

| Context / policy behavior | Simulated seconds | Actor choices | Worker wall seconds | Outcome |
|---|---:|---:|---:|---|
| Ordinary / frozen greedy reference | 84.8 | 42 | 37.17 | Impostor ejected |
| Body opportunity / sampled disposable training collection | 230.4 | 51 | 74.61 | Parity |
| Innocent-reporter meeting / frozen greedy reference | 106.8 | 10 | 37.69 | Parity |
| Impostor-reporter meeting / frozen greedy reference | 90.2 | 6 | 32.72 | Impostor ejected |

Four-job collection wall: **84.69s**. One disposable-clone PPO update: **0.1365s**. Total benchmark: **89.96s**; the zero reference remained unchanged. Legal reporting choices occurred, but **zero neural reports were accepted**. Setup meeting reports were real and excluded from learning. These are engineering observations, not a four-game learning evaluation.

Measured batch imbalance/startup factor: `84.69 / (sum(worker wall)/4) = 1.8594`. For planning, use the larger 74.61s measured cost as a conservative ordinary-training proxy (a sampled ordinary match was not separately benchmarked), actual measured evaluation-context costs, this batch factor and a **50% margin**. Approximate totals are **99 minutes near 42 matched training games/arm**, and **137 minutes at the 64-game schedule ceiling**, including small update allowance. Therefore plan **1.5–2.5 hours**, retaining the four-hour hard stop and all 48 evaluations.

This is one small benchmark, with no full 900-second timeout observed. Future policies can change decision density and match length; the estimate is not a guarantee or a proven four-worker speedup. CPU/RAM were not instrumented in this benchmark. If actual execution cannot fit, stop incomplete and request a newly registered smaller study; never silently cut contexts or labels.

Benchmark source is preserved at `0405076`. Later changes only finalize partitions, scheduling/approval/progress and post-game aggregate endpoints; simulation, prefix execution, inference, rewards and forecast scoring remain byte-identical. Benchmark checkpoint compatibility remains bound to that source. A fresh current-source readiness reference is separately hash-checked; no checkpoint is migrated around a strict loader.

## Spectator and activation

The viewer stays separate from headless workers. Prefix events are recorded with their scripted-control/handoff metadata; neural decisions begin only when PPO actually acts. Four PPO controller identities, real probabilities/value/suspicion, current-meeting hidden ballots, ghost freeze, actual outcomes and clickable speed controls are verified. A narrow progress-label correction uses registered 2,048/48 budgets rather than historical 512/20 defaults; old status files keep their defaults.

[VIEWER.md](VIEWER.md) gives the exact launch command and prepared watch directory. [readiness.json](readiness.json) records tests, source compatibility, cost calculation and integrity checks. [plan.json](plan.json) and [approval-template.json](approval-template.json) contain exact hashes; the template remains **approved=false**. Study output has not been created and study seeds have not been executed. Final owner authorization is the remaining activation requirement.
