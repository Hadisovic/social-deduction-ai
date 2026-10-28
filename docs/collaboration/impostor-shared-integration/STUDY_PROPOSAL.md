# Proposed first shared-environment impostor pilot — NOT APPROVED / NOT RUNNING

Use the verified engineering adapter before scaling. The present transferred
impostor lost the paired A/B smoke by ejection; that is one scenario, not an
accuracy estimate. This proposal changes neither architecture nor reward weights.

## Initial design

| Item | Proposal |
|---|---|
| Opponents for initial adaptation | Four sampled frozen B actors; separate memory/RNG; no social guards |
| Initial weights | Explicitly transferred83/8192; preserve untouched copy as zero-update reference |
| Replicates | Two private training/sampling streams,101 and202; same initial tensors, **not** two independent parameter initializations |
| Initial learning budget | Eight complete training episodes per stream; about1k neural choices each if smoke length is representative, not a promised count |
| Training scenarios |3600000–3600007, paired across streams; private stream seed affects impostor sampling, never neural features |
| PPO | Existing separate actor/critic and12-family/candidate joint log-probability; Adam3e−4,eps1e−5;4 epochs,minibatch64,clip0.2,value0.5 |
| Entropy / gradients | Preserve original family entropy schedule0.02→0.005 and conditional0.001; actor/critic clipping0.5/1.0 |
| Temporal credit | Gamma1;lambda0.995 per real simulated second; full episodes only, terminal bootstrap0 |
| Candidate checkpoints | End of episode4 and8 for each stream; no silent extension |
| Selection | Eight fresh ordinary B scenarios3610000–3610007 per candidate; primary wins, then kills, fewer own ejections, earlier checkpoint, fixed stream tie-break |
| Lock | Immutable source/config/checkpoint SHA before confirmation; all-zero selection wins must be reported as insufficient |
| Fresh ordinary confirmation |3610020–3610035: locked model and untouched reference, each against four A and four B, paired scenarios |
| Controlled evidence probes | Separate new development fixtures with explicit body/report/witness opportunities; do not pool with ordinary win rates |
| Final tests |Remain sealed; these are development selection/confirmation partitions |

This is an incremental adaptation study from a previously learned policy. It is
not a from-scratch comparison or evidence that warm start is superior. A separate
future initialization/ablation study would answer those questions. The dedicated
PPO experiment runner/registration must be completed and checked before rollouts;
current `impostor_shared.run` intentionally performs no updates.

## Budget estimate to approve

Recorded engineering cases took roughly32–68s including loading/verification and
other concurrent work, with105 impostor choices in the paired learned case.
Sixteen training episodes,32 selection executions and64 confirmation executions
suggest **about60–90 minutes on this CPU with up to two isolated one-thread
workers**, with high uncertainty. Propose an absolute90-minute wall ceiling;
reserve time for selection/confirmation and stop with an incomplete report if the
budget cannot finish. Do not buy more time or tune from confirmation automatically.
No GPU/CUDA runtime migration is included. This budget is a proposal, not approval.

## What to measure and graph

- True wins and reasons, kills, survival, task losses, ejections and uncertainty.
- Own-kill self-report frequency and latency; distinguish accepted reports from
  selected actions and count legal opportunities/visible evidence exposure.
- Witnessed kills and post-kill travel. Define an escape metric before running
  (for example leaving the kill room and surviving a fixed roaming interval);
  report censored follow-up separately. It is retrospective analytics, not an
  oracle feature or new hidden-information reward.
- Correct/incorrect crew ballots/ejections, claim types and factual consistency.
  A false claim followed by an innocent vote is correlation, not proven successful
  deception. Use separate matched controlled probes for causal claims.
- Real reward components, ready-idle/faking time, fake completions/streaks,
  pursuit credit/correction, legal/masked choices and navigation outcomes.
- Family/candidate entropy, PPO KL/clip fraction, actor/critic gradients, value
  fit, decisions, complete episodes, simulated experience and CPU/wall time.
- Paired selected-versus-zero-update A/B outcomes. Do not call training wins test
  accuracy, pool controlled scenarios into ordinary wins or rank models globally.

## Independent recommendations

1. B-first is a convenient controlled pilot, not a recommendation for long-term
   exclusive B training. Its shaped objective may expose report/vote mistakes
   that an impostor exploits rather than learning general strategy.
2. Keep A outside initial adaptation and evaluate it as task-pressure control.
   One paired smoke produced identical A/B world traces despite different weights;
   more opportunity-stratified cases are needed to claim behavioral diversity.
3. After the pilot, consider a preregistered pool of all-A/all-B/2A+2B teams sampled
   per complete match. Implement/check the mixed-seat dispatcher before using it;
   do not silently switch opponents mid-episode or blend their tensors.
4. Retain a current-engine scripted-crew reference if needed. Historical mixed
   opponents had different guards/timing and are not a fair isolated control.
5. Improve post-kill/meeting choices through evidence-driven learning; do not ban
   self-reports or add a flat cost simply because this smoke self-reported/lost.
6. Add sabotage only as a separately versioned mechanics/both-role training task.
   The published handoff has none. Larger lobbies and self-play remain later work.

**Stop for owner review. No PPO study starts from this document.**
