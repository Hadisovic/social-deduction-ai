# Shared impostor pilot — 29 October 2026

The owner approved starting training. The fixed study in STUDY_PROPOSAL.md is
ready for launch; this authorization supersedes its earlier approval-pending text.
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
- [ ] Freeze source/config/seed registration, then launch the two training streams.
- [ ] Eight complete training episodes per stream; candidate checkpoints at 4/8.
- [ ] Eight B validation cases per candidate, then immutable selection lock.
- [ ] Sixteen paired fresh cases per selected/reference model and A/B opponents.
- [ ] Produce graphs, runtime/CPU details, and an honest learning assessment.

Expected study duration: 60–90 minutes after launch, with a 90-minute wall cap.
Preparation/preflight time is separate. Training itself has a 45-minute cutoff
to reserve time for validation/confirmation. If a required stage cannot finish,
the run stops and preserves an incomplete report; it does not silently extend.

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
