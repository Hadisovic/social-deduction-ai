# Research issue register

## R023 — Phase 2 opponent diversity and social reconstruction (2026-10-28)

- **Authorization / design:** Owner-approved implementation and short engineering diagnostics, isolated from research `fab5c70`. See [context and design](../phase2-opponents/DESIGN.md). Preserve A/B/zero, original scripts, rules, partner work and approved viewer. No training, rewards, actor features, final tests, push or merge.
- **Confirmed investigation:** Across 48 historical validation recordings, shaped B selected 45 defenses; 44 were followed by that same actor voting against the same target in the same meeting (34 impostor, 10 innocent). All 44 defenses were argmax choices at 6.7–7.2% probability; 39 targets already had at least 80% suspicion. Different-agent links and sampled training remain separate. A belief reversal, stochastic validation sampling and strategic deception are not adequate blanket explanations. Private historical observation packets are unavailable.
- **Implementation / measured progress:** New actor-local overlapping strategy mixture with independent RNG streams and legal, optional observational diagnostics. Initial 22 engineering tests passed in 181.23s. All 16 comparison games across eight engineering seeds completed in 827.68s, including a complete repeat with auditing/recording disabled: identical events, actions and decisions. Strict A/B/zero checkpoint loads/hashes passed; no updates. Both roles reported; all five modes occurred. Mixture also produced two timeouts/fewer meetings; diversity alone is not improvement or readiness for training.
- **Status:** Full aggregate/probe/classifier analysis and final preservation checks are being documented. Baseline remains installed; new mixture is opt-in. Phase 6.1 acceptance pending.

## R022 — Historical reward observability and same-step ballot accounting (2026-10-28)

- **Confirmed observation / evidence:** The Spectator V2 read-only audit checked all 48 registered validation games. Neural report totals and cumulative rewards matched the original ledger. Six matches had discrepancies between actual published crew ballots and ledger counters (six omitted ballots in aggregate: two correct, four wrong). Twenty-seven recorded component histories had unattributed starting balances (maximum absolute 0.095); 24 inactive-prefix actors had no reward record. Original results are preserved, not silently corrected.
- **Confirmed cause / tests:** `fourcrew/rewards.py:ActorLedger.consume` filters own report/vote events when post-step `active` is false. A legal ballot submitted while alive can therefore lose its count and reward if its owner is ejected in that same step. `tests/test_phase2_accounting.py` reproduces the missing -1 incorrect-ballot component while retaining the death penalty. This is a reward-attribution implementation bug, not an illegal-vote rule or coefficient change. No historical reward source was modified.
- **Telemetry limits / handling:** Viewer shows actual replay ballots separately from registered ledger counters, cumulative rewards separately from recorded interval components, and explicit unattributed balances. No invented decomposition or inactive-prefix personal reward. Components and full private memories cannot be reconstructed from absent records.
- **Status / proposal:** Confirmed, unresolved in frozen source. A separately versioned repair and regression review is recommended before another learning study; historical checkpoints/results must retain their original provenance. Research acceptance remains pending.

## R021 — Spectator V2 research instrument (2026-10-28)

- **Status / authorization:** Owner requests a full desktop spectator overhaul with unchanged map/characters, truthful neural/reward/task/meeting views and explicit spectator-only omniscience. Implementation, real-recording tests, visible GUI QA and local commits are authorized; training/gameplay changes, publication and merges are excluded.
- **Confirmed baseline:** Latest viewer `e8a815f` preserves repaired speed controls, meeting scope, four-crew telemetry and completed-recording playback. PR #11 remains draft/unmerged at `6dd8a1b`. The old fixed panels lack a settings UI, comprehensive roster/task bars, selectable ground-truth ballots and integrated research comparison tools. Owner-reported social inconsistency remains a question, not a confirmed cognition bug.
- **Design / boundaries:** [Design and context map](../spectator-v2/DESIGN.md). New isolated `spectator/v2/` shell, background reading, chronological owner-scoped inspection, explicit truth/public/private layers, no fabricated missing fields or sabotage. Preserve the original parser/recorder and pinned artwork. The R001–R020 history below is copied unchanged from `fab5c70`; original supporting research documents remain in that preserved branch/worktree.
- **Implementation / evidence:** The owner subsequently inspected and approved Spectator V2. Its final local test artifact records 66 passing viewer regressions. Phase 2 preserves its implementation byte-for-byte and uses it for specific behavioral inspection; general UI verification is not repeated. Remaining historical telemetry caveats are recorded in R022. No new scientific result is claimed by the UI redesign.

## R020 — Registered four-crewmate reward comparison (2026-10-28)

- **Authorization/design:** The owner approved the full exact R019 registration. [Activation](../crewmate-v2-fourcrew/registration/ACTIVATION.md), [report](../crewmate-v2-fourcrew/results/registered-2048-seed17-v1/REPORT.md), [aggregate metrics](../crewmate-v2-fourcrew/results/registered-2048-seed17-v1/metrics.json). Four private shared-policy crew, one scripted impostor, matched fresh initialization17 original/shaped-v2 and a frozen zero-update policy. Source723c8d4 and spectatore8a815f stayed unchanged; no retuning or mechanics changes.
- **Execution/budget:** One execution, exit0,34.54min of240min hard budget. Training2780000–2780045:46 complete matched games/arm (23 ordinary/23 body),2058/2092 accepted individual decisions,23 updates and immutable checkpoints/arm. First common stopping point; no rejection/repeat or validation checkpoint selection. Validation2790000–2790007: all48 executions across eight paired seed clusters; ordinary/body/meeting kept separate. Unused registered training slots remain unused.
- **Measured behavior:** Original won8/8 ordinary task games and0/8 controlled; shaped won4/8 ordinary (3 ejections/1 task) and8/8 controlled ejections; zero won2/8 ordinary and3/8 controlled. Training reports5/arm; controlled validation neural reporting0/4 original,4/4 shaped,0/4 zero. Shaped body12 selections/submissions produced4 accepted reports because crew compete for one body. All opportunity/reporter quotas passed; no registered inconclusive flag.
- **Investigation/interpretation:** Original skipped all42 recorded validation ballots despite often correct conditional target rankings. Shaped cast12 correct/8 wrong ordinary ballots and23 correct/1 wrong controlled ballots, with no skips. No selected false suspect claims, but nine wrong ballots remain. The shared frozen Phase4 predictor had five high-confidence wrong ordinary audits and one body audit on shaped trajectories. Identification quality, action selection and behavior-dependent evidence all matter; task success alone obscures deduction failures. Reporting/voting bonuses form one tested bundle, not an isolated +1.5 effect.
- **Shortcuts/uncertainty:** Initial controlled nonreporter per-game macro top-1 under shaped was100% for both reporter roles, exceeding accuse-reporter's balanced50%; only four constructed meetings, with no shaped direct-elimination witnesses there. Reporter/script/choreography confounds remain unresolved. Ordinary shaped reports arose from4 crew/3 impostors versus original0/6 and zero0/4; role event shares do not measure opportunity-adjusted propensity. Primary reporting difference+100points has a degenerate four-cluster bootstrap; ordinary shaped-minus-original wins-50points, descriptive95%[-87.5,-12.5]. One initialization, tiny correlated samples and greedy/stochastic differences prevent generalization or acceptance.
- **Correctness/provenance:** Post-run strict loads, unchanged zero tensor, finite diagnostics, exact aggregate reproduction, complete partitions, first common stop, zero illegal actions and zero personal ghost-task rewards verified. All eight historical/readiness preservation manifests passed, including the nine earlier selected checkpoints. All48 validation recordings match outcomes/decisions, legitimate handoffs, no dead actions and frozen meeting movement/tasks across3051 frame pairs. No full rerun or final-test access. Viewer remains independently open with96 completed replays; raw traces/role labels/checkpoints remain local, unchanged and unpublished.
- **Recommendation/final status:** Review ordinary wrong ballots and confident belief errors from existing recordings; separately preregister fresh-scenario/additional-initialization replication before scaling or changing rewards. No fix, further training, learned impostor, larger lobby, deployment, push or merge initiated. **Phase6.1 acceptance remains pending.**

## R019 — Final four-crewmate study registration and readiness (2026-10-27)

- **Execution update (2026-10-28):** The owner-authorized single execution completed normally; see R020 and the [results](../crewmate-v2-fourcrew/results/registered-2048-seed17-v1/REPORT.md). Exact registration and historical artifacts preserved; acceptance remains pending. The following bullets preserve the pre-authorization readiness evidence.
- **Observation/status:** Owner approved the overall design but requires exact registration and realistic cost evidence before final training authorization. The focal seed-17 pilot remains a negative/exploratory baseline; failed report exposure, reporter-pattern confound, voting incentive concerns and V2.2 repairs remain documented in R002–R017.
- **Investigation/design:** [Registration context map](../crewmate-v2-fourcrew/registration/CONTEXT.md). Matched fresh initialization, zero-update reference, independent memories, legal-prefix body and reporter-role cases, complete-match overshoot and a four-hour global deadline. No reward coefficient changes.
- **Implemented/tests:** [Final specification](../crewmate-v2-fourcrew/registration/SPECIFICATION.md), exact machine plan/runtime hashes and false approval template. Matched training seeds2780000–2780063; paired validation2790000–2790007; 2048 target /3072 admissible complete-match ceiling per arm, minimum16 / maximum64 matched games. Eight controlled-prefix engineering families, independent memory, matched/frozen initialization, source/checkpoint guards, GAE/budget/approval tests and prefix recording equality pass. Full-study seeds remain unused.
- **Measured results:** One bounded benchmark of four complete engineering matches used2760020–2760023 with canonical900s caps:89.96s total,84.69s parallel collection,0.1365s disposable update. Legal reporting choices occurred but zero neural reports were accepted; preserve that negative result. Frozen zero reference unchanged. Measured planning estimate99–137min with50% margin, rather than three-second extrapolation. The four-hour hard stop and all48 evaluations remain; reserve one hour for validation.
- **Reasoning/changes:** Low measured decision density motivated a prospective64-slot maximum instead of32 before any study seed was opened. Equal accepted scenario exposure is exact; decision totals can differ within the declared ceiling. Controlled evaluation separates four body opportunities from four meetings with two innocent/two impostor reporters. Prefix kills/reports are legal scripted setup, excluded from policy credit; observations remain private. No rewards or gameplay were retuned. Viewer progress denominators and scripted-prefix/inactive-ghost labels were corrected locally; PR#11 remains untouched.
- **Limits/expected outcomes/final status:** Reporting improvement over original and frozen initialization is a hypothesis, not an expectation of success. Fewer than two actual training reports makes that endpoint inconclusive. Four body/four meeting clusters, one initialization, lawful choreography cues, competing reporters and unmeasured900s-timeout costs limit conclusions. Reporter-role macro controls and ordinary results must be reported separately. Registration is ready for review; final authorization is still required. No full study, final-test access or Phase6.1 acceptance occurred.

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
