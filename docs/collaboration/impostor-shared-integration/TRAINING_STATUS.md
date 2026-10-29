# Shared impostor pilot — 29 October 2026

The owner approved starting training. The fixed study in STUDY_PROPOSAL.md is
stopped with incomplete validation; this supersedes its earlier running status.
Training, selection, and confirmation use development partitions only. No merge,
deployment, historical checkpoint replacement, or reserved final-test access.

- [x] Owner approval: "start training".
- [x] Implement a dedicated complete-episode PPO runner.
- [x] Preserve the architecture, rewards, historical weights, frozen crewmates,
  actor-local information boundary, and private sampling state.
- [x] Pass 17 targeted PPO/integration/checkpoint tests (19.87 seconds).
- [x] Pass a separate real-match optimizer/checkpoint preflight: seed 2760010,
  88 impostor decisions, PARITY, eight finite minibatch updates, exact safe
  checkpoint/RNG reload. Preflight weights are discarded, not used by the pilot.
- [x] Freeze source/config/seed registration and launch both training streams.
- [x] Eight complete training episodes per stream; candidate checkpoints at 4/8.
- [ ] Finish eight B validation cases per candidate: 30/32 completed; one
  computational abort and one unstarted case. No immutable selection lock.
- [ ] Sixteen paired fresh cases per selected/reference model and A/B opponents:
  none started, because validation did not finish.
- [x] Produce four partial-results graphs, runtime/CPU details, candidate snapshots,
  preserved failure evidence and an honest learning assessment.

**Outcome:** [read the report and graphs](pilot-20261029/README.md). The pilot
ended after 39.17 minutes. Both streams won 7/8 stochastic training matches.
Validation: 101/episode 4 won 3 of 6 completed cases; 101/episode 8 won 8/8;
202/episode 4 won 3/8; 202/episode 8 won 5/8. Missing cases are not losses.
No model is selected, promoted, or shown to improve over the untouched reference.

Expected study duration: 60–90 minutes after launch, with a 90-minute wall cap.
Preparation/preflight time is separate. Training itself has a 45-minute cutoff
to reserve time for validation/confirmation. If a required stage cannot finish,
the run stops and preserves an incomplete report; it does not silently extend.

## Current execution

**Stopped at 03:03:43 Amman, 29 October. No active jobs.** Session 50872 exited
with code 1. Stream 101 / episode 4 exceeded the 600-second execution ceiling on
validation seed 3610006; seed 3610007 was not started. This is a computational
failure, not an in-game TIMEOUT. The other three blocks completed. No automatic
retry, extension, training restart, selection shortcut, or final-test access.
The monitor is being retired after publication of the incomplete report.

All 16 local checkpoints remain intact; four candidates passed strict reload and
are curated under artifacts/impostor-shared-pilot-20261029/. All 46 completed
study executions had zero illegal actions, private memories and consistent frozen
crew/belief hashes. Source modules remain at their registered hashes. An offline
reporting tool outside impostor_shared preserves those checkpoint source locks.

The slow-case root cause is not established. Next: separately document/profile
the evaluation path and its timing, then agree on an evaluation-only continuation
using these checkpoint bytes. Further PPO is not needed just to recover evaluation.

### Execution history

02:39 Amman update: both training streams finished. Stream 101 collected 1,066
decisions and 84 minibatch updates; stream 202 collected 770 decisions and 60
updates. All 16 episode-boundary checkpoints are preserved. Validation is running
under the original registration; no selection lock or confirmation results yet.

Started **29 October at 02:24:33 Asia/Amman**. Absolute deadline **03:54:33**.
Source commit: `7bb6616`. Parent exec session: `50872`, PID `33776`.
Initial worker PIDs: `38468` and `53400` (one CPU thread per worker).
Run directory: `local-runs/shared-pilot-20261029/`.
The immutable `registration.json` records exact module hashes, runtime, frozen
checkpoints, optimizer settings and scenario partitions. Each stream has already
completed real PPO updates and saved recoverable episode-boundary checkpoints.

The in-chat heartbeat `finish-shared-impostor-ppo-pilot` monitored at five-minute
intervals and stayed quiet during healthy work. Both worker pools have exited.
Evaluation used the historical
greedy family/candidate decoder; crew sampling uses fixed private initialization
17 for paired scenarios. Training uses stochastic impostor choices.

Both streams start from the same transferred V4.1 tensors. They differ in private
sampling/minibatch RNG, not parameter initialization. Four B crewmates train the
impostor; A remains outside adaptation. Crewmates/belief never receive gradients.
One update uses one complete episode, four PPO epochs, minibatches up to 64,
joint family/candidate log-probabilities, separate actor/critic clipping, and
elapsed-simulation-time GAE. The entropy coefficient decreases from .02 to .005
across the eight episode updates. Adam is reset for this explicitly versioned
warm-start study; historical optimizer state is not imported.

Run commands (isolated integration checkout):

```powershell
.\.venv\Scripts\python.exe -m impostor_shared.study --preflight --output local-runs/shared-pilot-preflight-20261029
.\.venv\Scripts\python.exe -u -m impostor_shared.study --output local-runs/shared-pilot-20261029
```

Use a fresh directory, never overwrite or restart a healthy run. Each stream
writes status.json, training.jsonl, complete-episode checkpoints and private
replays. The parent writes registration.json, status.json, selection-lock.json,
results.json, aggregate.json, REPORT.md and four graphs when all stages finish.
The default existing viewer still plays the earlier zero-update engineering
replay. Pass a new replay explicitly to watch the current pilot.

The preflight PARITY outcome happened during stochastic collection before its
optimizer update. It is a historical-policy compatibility win, not evidence that
new training improved play. Its post-update KL was .0259 and clip fraction .477;
the pilot records these diagnostics without silently changing the approved setup.

The proposal's additional causal evidence probes are not pooled into ordinary
results. Existing controlled privacy/action/reward regressions verify mechanics;
this pilot reports retrospective witness and self-report counts. It does not
claim that a false accusation caused a vote, or that escape/deception is learned.

No functional sabotage is added. A small warm-start pilot cannot establish the
best model or reliable large-lobby performance. All-zero validation wins must
be reported as insufficient, even if the fixed tie-break selects a checkpoint.
