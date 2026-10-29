# Task Commitment Mode — manual training stage

Owner-approved temporary curriculum; Phase 6.1 acceptance remains pending.
No abandonment penalty and no reward coefficient or neural architecture change.
The agent prepares this stage; the owner starts long training manually.

## Manual command and budget

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Files\RL\_patrick_jane\Start-PatrickJaneCommitment.ps1"
```

Results: `C:\Files\RL\_patrick_jane\local-runs\patrick-jane\task-commitment-v1`.
Target: **250,000 additional individual living-crewmate neural decisions**.
Forced singleton actions count as actual policy calls but are reported separately
from decisions offering multiple legal choices. Complete-match/boundary batching
may overshoot; the absolute ceiling is 275,000. No automatic larger stage.

Previous Rewards V2 measured 82.07 decisions/s including PPO: 250,000 would take
about 51 minutes, plus roughly 18 minutes of previous validation/startup overhead.
Plan for approximately 70 minutes, with substantial uncertainty from changed
match lengths, masks and forced decisions. This is not a measured commitment-mode
speed prediction. Keep the existing **12-hour cumulative active stage ceiling**;
it is a fail-safe, not an estimate or authorization for an automatic extension.

The launcher opens/reuses the existing independent spectator with actual instance
activity, neural probabilities and completed replays. Terminal/viewer percent and
ETA use actual collected additional decisions. Ctrl+C once interrupts safely;
rerun the same command to restore the last complete checkpoint/optimizer/RNG.
Partial interrupted batches are withheld and may be recollected. The spectator
remains open after training. Success requires target, final validation and saved
outputs and displays `PATRICK JANE — TRAINING COMPLETE (100%)`.

## Inheritance, configuration and experimental boundaries

- Pinned Rewards V2 `attempt-0002-update-000016.pt`, SHA256
  `0696f9662e986bd8d10eb93c409e05b8122e99b5c59640df7337946b415c6387`.
  Base 1,266,883 lifetime decisions; 1,011,800 combined post-pilot decisions.
  Import all 52 policy/critic tensors, Adam moments and RNG unchanged. Rewards
  and PPO are unchanged, so optimizer continuity is retained explicitly.
- Verify the old checkpoint/source archive and immutable architecture, PPO,
  rewards, canonical game mechanics and opponent sources. A narrow versioned
  bridge permits the reviewed worker/navigation/mask/telemetry changes. Ordinary
  strict source loading is unchanged. Old runs, reports and models stay intact.
- `LightsExperiment.task_commitment_mode` defaults **False**. This new stage
  registers **True**; a changed value blocks same-run recovery. Turning it off in
  a future stage requires a separate registration and audited checkpoint
  transition. Off restores unrestricted task choices; the stale-Continue bug
  correction remains. Historical exact reproduction uses historical snapshots.
- Keep ten independently observed/memoried learning crew, two scripted impostors,
  32 workers, lights-only sabotage, normal kills and all Rewards V2 coefficients.
  Lights cost remains -0.005/living crew/simulated dark second. No rewards are
  awarded for starting, arriving, intermediate stages or abandonment. Full living
  assigned-task completion gives +1 once; ghost completion gives zero direct pay.
- Begin at the next unused training index 486 (seed 2900486) within the existing
  2900000–2900999 development partition. Entry/final evaluation remains eight
  paired development seeds 2910000–2910007. Final-test partitions remain sealed.
  Compare entry/final under the same restriction; no unlocked evaluation is
  silently added. Later unlocked transfer evaluation needs separate review.

## Commitment semantics

The network selects and legally starts an owned task without a pre-existing lock.
Only an actual `task_start` in executed gameplay activates commitment; sampling
an interaction that is pre-empted by a meeting/death does not. Stable task ID is
retained separately from `interaction_task`, which clears between stages.

Masks allow only work on that task, travel toward its current-stage console,
incubation waiting, and legal report/lights-repair exceptions. Detours, unrelated
goals/tasks and emergency meeting calls are blocked. The mode never advances task
progress, selects tasks or emits executable actions; every submitted action still
passes through neural sampling/decoding and canonical legality/physics. The
existing navigator supplies low-level paths, as before.

Report/repair pauses retain the original task. A body must be currently visible;
repair goals require actual unresolved lights. Routes to exceptions are scoped to
those targets. After a meeting, vanished body, completed repair or another actor's
repair, clear stale route memory and rebuild via a subsequent sampled action.
Social actions/voting remain unrestricted by the task lock during meetings.
Death/ejection releases the living commitment; the existing ghost executor is
unchanged. Mandatory incubation is not avoidable idle and is not charged the
existing idle cost. Darkness costs still apply. The idle coefficient is unchanged.

While actively working, Continue emits WAIT rather than restarting a retained
route. Work continues under the existing engine. No sampled action is replaced
after its probability is calculated.

## PPO, observation and failure handling

Store commitment-dependent masks in each pre-action model packet; PPO evaluates
the exact same outer and conditional masks. Neural hidden states continue across
stages and meetings and remain independent by actor. Singleton support gives
log-probability/entropy zero, not a fictitious actor learning signal. GAE, clipping,
epochs, batching, discounting and network layers/parameters remain unchanged.

Four private commitment fields occupy previously unused self-feature slots
188–191, with an overlap assertion. Input width stays 192. This changes input
semantics and the constrained behavior policy, so it is a new source-locked
curriculum, not same-protocol continuation. Forced successes do not establish
learned voluntary persistence.
The legacy registration key `reward_regime` labels this stage `commitment-v1`;
the registered reward coefficients remain exactly Rewards V2. Global singleton
counts include existing navigation cooldown constraints; commitment-specific
singleton/choice-bearing counts are recorded separately.

Three failed own navigation requests or 60 simulated seconds without task/route
progress cause an explicit technical TRUNCATED match, with living critic bootstrap
and no invented outcome reward. Genuine meetings/incubation pause the progress
watchdog; task work or at least 0.25 units of approach progress resets it. Unexpected
unmasked task abandonment also causes a technical failure. Existing 1,800-second
technical match bound and 300-second stationary guard remain. Three commitment
failure matches in one batch stop before PPO and preserve raw diagnostics.

## Results and verification

The normal results folder contains `TRAINING_REPORT.md`, `training_results.json`,
`completion-status.json`, `status.json`, appended `console.log`, learning curves,
checkpoints/optimizer/RNG, exact source/config snapshots, attempt journals and
selected replays. Metrics distinguish task starts/full living completions, ghost
completions, forced/choice-bearing decisions, commitment starts/end reasons,
interruptions by reason, exception pauses, incubation/meeting/total actor-seconds,
commitment durations and technical cuts. Replays annotate commitment state and
whether each sampled decision had singleton support.

See `TASK_COMMITMENT_VERIFICATION.json` and research issue R044 for actual tests,
bounded launcher checks, preserved failures, limitations and integrity results.
Keep historical negative results. No claim of improved learning is made from
engineering verification or from forced completion alone.
