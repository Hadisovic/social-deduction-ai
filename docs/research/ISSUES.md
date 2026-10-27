# Research issue register

Updated 2026-10-27. This complements the existing [project timeline](../PROJECT_PROGRESS.md). Each entry records phase/date, observation/confirmation, evidence, investigation/root cause, proposed and implemented changes, commits/tests, results/limitations and final status. Unknown historical dates use development phases. Unmeasured claims remain unmeasured.

## R001 — Navigation prototype to social-deduction RL

- **Phase / observation / evidence:** Original navigation/task-routing prototype evolved into typed observations, meetings, memory and strategic RL. Confirmed by the [timeline](../PROJECT_PROGRESS.md), [architecture](../project_architecture.md) and commits `7511134`, `8a0b4c9`, `9dcc0a2`, `c2c6ea2`.
- **Investigation / cause:** Navigation competence alone cannot establish strategy or deduction; these require different objectives and information contracts.
- **Proposal / implementation / tests:** Retain deterministic navigation as infrastructure; existing phase implementations/tests added private observations and strategic options. This task does not reimplement navigation.
- **Results / limitations / final status:** Historical measurements stay in their original reports. Navigation foundation retained; deduction research remains in progress.

## R002 — Task-dominant behavior

- **Phase / observation / evidence:** Confirmed limited deduction success alongside task wins in the [V2 diagnostic](../crewmate-v2-diagnostic/REPORT.md) and [pilot](../crewmate-v2-live/results/REVIEW.md).
- **Investigation / cause:** Task opportunities and rewards are frequent; evidence/reporting opportunities are scarce. This association does not isolate a causal coefficient.
- **Proposal / implementation / tests:** Separate task/ejection wins and individual contribution; four-crew support replaces scripted teammates. Existing task ownership and reward tests remain.
- **Results / limitations / final status:** Both focal models won 4/4 ordinary games, versus 1/16 and 0/16 deduction games. One initialization/four scenario clusters; task dominance remains an open learning limitation.

## R003 — Skipping to return to reliable task rewards

- **Phase / observation / evidence:** Owner-reported historical incentive concern. Precise older rates and causal explanation are unverified. [Pilot metrics](../crewmate-v2-live/results/metrics.json) show six training skips per arm, zero validation skips.
- **Investigation / cause:** Stochastic training and greedy validation differ. No experiment isolated task-credit-driven strategic skipping.
- **Proposal / implementation / tests:** Preserve strategic skips; log evidence-conditioned participation and win outcomes. No forced vote or blanket skip penalty was added; skip/reward regressions retained.
- **Results / limitations / final status:** Preserve the historical observation without attributing current behavior to it. Cause unresolved; future comparison required.

## R004 — Reporter identity exploitation and false self-report accusations

- **Phase / observation / evidence:** Confirmed confound in [reporting diversity](../crewmate-v2-reporting-diversity/REPORT.md) and the [focal pilot](../crewmate-v2-live/results/REVIEW.md).
- **Investigation / cause:** Reporter roles were predictable from scripts. Reporter-only accusation matched 88.89%/82.76% target accuracy. Rankings were correct for impostor reporters and wrong for crew reporters; aggregate equality does not identify every actual target.
- **Proposal / implementation / tests:** Four private PPO crew replace crew scripts; legitimate opportunity interruption and both-role engineering report fixtures added. Reporter roles remain offline labels.
- **Results / limitations / final status:** Legal reporting from either role is tested; no new learned reporter distribution is measured. Shortcut unresolved; reporter controls mandatory.

## R005 — Failure to report versus lack of opportunity

- **Phase / observation / evidence:** Zero focal reports confirmed in the [pilot metrics](../crewmate-v2-live/results/metrics.json), with only two legal episode/body pairs per arm in training.
- **Investigation / cause:** Held navigation can traverse reporting range without a fresh policy choice. Sparse exposure prevents calling all zero reports refusals.
- **Proposal / implementation / tests:** Per-owner visible/range/legal, choice/held, selected/submitted/accepted/meeting counters; newly legal reports interrupt a held option without forcing a report. Exact successful/illegal/dead/duplicate reward tests added.
- **Results / limitations / final status:** Engineering correctness established; learned reporting effectiveness remains untested.

## R006 — Validation with no legal reporting choices

- **Phase / observation / evidence:** All 40 focal validation executions had zero legal reporting pairs at decision boundaries. Conditional reporting rate is undefined. A had two physical opportunities inside held options; see [review](../crewmate-v2-live/results/REVIEW.md).
- **Investigation / cause:** Evidence exposure and scheduling interact; this is an evaluation coverage gap, not a measured zero-percent reporting rate.
- **Proposal / implementation / tests:** Separate physical, held and choice counters; require prospective evidence-opportunity and reporter-role coverage. Legal range fixtures test counters/mechanics.
- **Results / limitations / final status:** New random-policy engineering collection does not prove sufficient learning exposure. Experimental coverage gap remains open.

## R007 — Poor survival and early elimination

- **Date / observation / evidence:** 2026-10-27 owner observation: about 12 first deaths in 15 viewings. Preserved completed sample: 4 first kills and 9 total focal kills in 11 replays per arm; [post-pilot review](../crewmate-v2-fourcrew/REVIEW.md).
- **Investigation / cause:** All nine preceding snapshots lacked another crew within 4.5 line-of-sight. Last choices: five room, two emergency-travel, two task; none followed the eventual killer. Last-choice age at death: median 5.4s, max 18s. `phase3_bots.py` targets nearest visible players without a learned/focal flag.
- **Proposal / implementation / tests:** Retain exposure/survival metrics; independent options and death isolation tested. No privileged danger signal, forced fleeing or survival reward change.
- **Results / limitations / final status:** Owner's viewing set unavailable; paired arms are not independent samples. Unsafe routing, escape latency and scripted-teammate causality remain unresolved. Four learners are not assumed to fix survival.

## R008 — Ghosts appear to vote

- **Date / observation / evidence:** Viewer bug confirmed; sampled logs contain zero dead submissions among 70 submissions per arm.
- **Investigation / cause:** Spectator `6dd8a1b` accumulates historical vote submissions across meetings and displays private targets prematurely. Actual historical ghost ballots were not found.
- **Proposal / implementation / commits/tests:** New viewer scopes claims/submissions by current meeting, excludes ineligible players and hides targets until public resolution. V2.2 rejects direct dead/ejected/duplicate/invalid votes (`77f478c`). Meeting/privacy and gameplay regressions added.
- **Results / limitations / final status:** Repaired and engineering-tested. Historical replays/results are unchanged; no claim of a historical simulation ghost-vote bug.

## R009 — Voting waits after all eligible responses

- **Date / observation / evidence:** Fixed-deadline behavior confirmed in V2.1 `engine.py`.
- **Investigation / cause:** Only the timer triggered resolution.
- **Proposal / implementation / commits/tests:** V2.2 resolves in the submitting step when all eligible living actors respond, retains timeout/default skips and ejection/parity ordering (`77f478c`). Early, simultaneous, timeout and dead/ejected tests pass.
- **Results / limitations / final status:** Implemented. Disconnects have no supported game state; no disconnect support is claimed.

## R010 — Ghost tasks continue during meetings

- **Date / observation / evidence:** Confirmed simulation bug: 35 ghost task completions during meetings in each paired eleven-replay sample.
- **Investigation / cause:** V2.1 advanced ghosts unconditionally after each tick.
- **Proposal / implementation / commits/tests:** V2.2 gates living/ghost work to roaming and pauses incubation deadlines (`77f478c`). Position/partial work persist; interrupted living tasks can be legitimately reselected after meetings. Before/during/after, partial-stage, incubation and ordering tests added.
- **Results / limitations / final status:** Repaired in the new regime. The old pilot retains this limitation as historical evidence; outcomes were not recomputed.

## R011 — Suspicion appears to reset after votes

- **Date / observation / evidence:** Owner observation; no actual memory reset reproduced. Old replays lack numeric suspicion histories, so the exact displayed transition cannot be reconstructed.
- **Investigation / cause:** `ActorMemory` persists across meetings, freezes after death and is replaced on new-match reset. Agent switching, stale last-decision display and legitimate new evidence remain possible causes.
- **Proposal / implementation / tests:** Two-meeting persistence/new-match reset tests; per-owner suspicion serialization and explicit selected-actor private debug. Shared frozen Phase 4 predictor unchanged.
- **Results / limitations / final status:** Memory persistence tested. Numerical predictions may legitimately change; they are not forced constant. Original visual-reset cause unresolved; telemetry improved.

## R012 — One focal actor to four shared-policy crewmates

- **Phase / observation / evidence:** Historical focal pilot remains at `5b663f6`; new intended design has four PPO crew and one scripted impostor.
- **Investigation / cause:** Independent actors require private observations, memories, sampling and trajectories; sharing weights does not share knowledge.
- **Proposal / implementation / commits/tests:** `fourcrew/` uses owner-bound views, independent options/samplers, frozen complete-match policy versions and segmented GAE (`555e3d0`, `961aab9`). Individual rewards are not broadcast; shared outcomes intentionally pay once per actor.
- **Results / limitations / final status:** Packets/log probabilities, inline/two-worker determinism, death settlement, replay equality and strict checkpoints tested. Bounded central PPO update executed. Full learning study not run; acceptance pending.

## R013 — Reward shaping and revision

- **Phase / observation / evidence:** Historical +1.25 reports and +2/-1 ballots gave no demonstrated benefit: 5/20 versus 4/20 wins. [Pilot review](../crewmate-v2-live/results/REVIEW.md).
- **Investigation / cause:** Scarce reporting, four updates, return-scale differences and reporter confound prevent attributing outcomes to a coefficient.
- **Proposal / implementation / commits/tests:** `732b555` versions reports at +1.5 in `crewmate-fourcrew-rewards-v2`, preserving all other approved coefficients and historical `RewardConfig`. New four-crew controls give zero individual ghost-task credit. Owner/event, illegal/dead and exact-ratio tests pass.
- **Results / limitations / final status:** Versioned implementation complete; behavioral effect untested. No reward tuning or historical record change.

## R014 — Parallel training and changed budget units

- **Phase / observation / evidence:** `c18ff5f` and [performance report](../crewmate-v2-performance/REPORT.md): about 1.30x four-worker speedup over serial; eight workers offered little benefit.
- **Investigation / cause:** Independent simulation streams need separate sampling/GAE. Four actors in one match are not four independent scenarios.
- **Proposal / implementation / tests:** New match workers receive frozen copies of one shared policy; reuse original PPO/GAE kernels. Budget counts individual non-forced actor choices. Complete-match batches may exceed minimum targets; log actual counts.
- **Results / limitations / final status:** Inline/two-worker equality and bounded four-worker smoke tested. Focal 512-choice budgets are not directly comparable. Large-run four-crew throughput unmeasured.

## R015 — Spectator recovery, integration and UI

- **Phase / observation / evidence:** `3fd3601` recovered; `6dd8a1b` added real completed-PPO replay workflow. Draft PR #11 is preserved, unmerged. Owner confirmed the original visible replay before the pilot.
- **Investigation / cause:** Speed pills were drawn but had no click targets. Old telemetry lacked per-actor suspicion; private data was not explicitly gated.
- **Proposal / implementation / tests:** Isolated `codex/spectator-fourcrew`, commit `fa76497`: speed hitboxes/mouse handling/resizing, D private-debug toggle, four PPO controller labels, per-owner suspicion, current-meeting public projection. Original 31 tests plus new UI/privacy regressions; actual four-actor recording on/off equality.
- **Results / limitations / final status:** Repaired locally; completed-episode playback only. PR #11 unchanged; no unfinished-frame streaming or learned intelligence claim.

## R016 — Seed-17 pilot limitations and negative result

- **Date / evidence:** Completed 2026-10-27 at `5b663f6`: [report](../crewmate-v2-live/results/REVIEW.md), [metrics](../crewmate-v2-live/results/metrics.json), curves and immutable local artifacts.
- **Measured results:** 512 choices/20 validation games per arm, 48.41 minutes total; wins 5/4, tasks 80/80, correct votes 24/24, wrong 3/5, reports zero.
- **Investigation / proposals / implemented tests:** Four scenario clusters, one initialization, no zero-update evaluation and unresolved shortcuts. Preserve hashes and historical source locks; use a strict new checkpoint schema after repairs.
- **Limitations / final status:** Important exploratory baseline; no retraining, retroactive correction or claim of success. New mechanics invalidate direct comparison, not the record. Phase 6.1 acceptance pending.

## R017 — Legitimate evidence versus scripted behavior

- **Phase / observation / evidence:** Reporter-only prediction matches apparent competence; no direct witness does not imply no legitimate body/report/claim evidence. [Pilot interpretation](../crewmate-v2-live/results/REVIEW.md).
- **Investigation / cause:** Shared frozen belief predictor, trained action policy and observation trajectory are different components. Repeated predictor evaluations are not separately trained belief models.
- **Proposal / implementation / tests:** Independent knowledge/ghost isolation tests and real per-actor debug telemetry implemented. Proposed study adds zero-update reference, reporter-role/evidence strata, claim/vote targets, opportunity denominators and paired uncertainty.
- **Results / limitations / final status:** Evaluation requirements established; evidence-based four-crew generalization remains untested.

## R018 — Larger lobbies and competitive learning

- **Phase / observation / evidence:** Future only; [approved roadmap on main](https://github.com/Hadisovic/social-deduction-ai/blob/main/docs/phase6_roadmap.md). That published roadmap is not duplicated onto this research branch.
- **Investigation / cause:** Five-player inputs have four other-player rows. Parameter sharing preserves that shape; 10 crew plus 2 impostors needs a separate variable-population design/protocol.
- **Proposal / implementation / tests:** Retain planned 6+2, main 8+2 and eventual 10+2 goals, learned impostors, historical pools and self-play. None was implemented or launched here; current five-player capacity fails rather than truncating candidates.
- **Results / limitations / final status:** Planned future work, requiring separate design and approval.
