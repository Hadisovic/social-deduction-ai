# Impostor reward/action improvements — session checklist

Status: implemented/tested; owner subsequently approved the bounded4h V4 study.
See [the training protocol](options-v4/TRAINING_PROTOCOL.md) for the new authorization.

The ongoing plan is now [IMPOSTOR_MASTER_CHECKLIST.md](IMPOSTOR_MASTER_CHECKLIST.md).
It includes learned skill acceptance, validation, paper evidence and future12-player work.

- [x] Read the previous session summary, result reports, source and current handoff.
- [x] Verify branch `codex/impostor-environment-v2`; checkout started clean.
- [x] Explain the current 75,075-parameter actor/critic architecture and observations.
- [x] Compare the new prompt with prior failures: fake-task restart is confirmed.
- [x] Implement versioned rewards and exact simulated-time cooldown accounting.
- [x] Implement finishable fake tasks with neural continue/cancel/leave choices.
- [x] Implement bounded, actor-observable pursuit shaping and exploit tests.
- [x] Preserve old V2/V3 source, strict checkpoint loaders and frozen opponents.
- [x] Verify reward boundaries, privacy, deterministic replay and checkpoint loading:32 tests passed.
- [x] Run two complete development matches: zero illegal actions/navigation failures;31/30 fake completions; both task losses.
- [x] Document old/new rewards, feature semantics, limitations and training proposal.
- [x] Open Antigravity and launch the real learned-policy game viewer; visible window inspected.
- [x] Stop for owner approval before any new full PPO study.

## Plan to finish the model after implementation

1. Approve a preregistered study with fresh scenarios and a fixed compute ceiling.
2. Compare unchanged V2 with the corrected action interface, then reward changes.
3. Measure greedy and sampled play: wins, kill opportunities, idle-ready seconds,
   fake starts/completions/interruptions, votes, ejections and reward components.
4. Select on validation only; lock the checkpoint before fresh confirmation.
5. Inspect failures before increasing training. Compare at least two initializations.
6. Pin the friend's compatible crewmate for paired cross-play. Keep its weights frozen.
7. Consider adapting opponents/self-play only after one useful learned impostor and
   one useful learned crewmate are demonstrated. Reserved final tests stay closed.

## Scope clarifications

The network keeps 12 action families; the prepared, untrained 13-family pursuit
experiment is separate. Adding rewards does not establish improved performance.
Fake-task progress and pursuit-credit state need explicit versioned observations.
The old viewer remains reproducible; viewing old weights under new rules must be
labelled as a transfer demo, not a newly trained model. No merge or deployment.

Read [the implementation and verification report](options-v4/README.md). Proposed
next study estimate:3–4h on this CPU, not yet authorized. The new smoke report and
unit collection checks performed zero optimizer steps. No new trained V4 model exists.
Current roster remains1 learned impostor,1 frozen neural crew and3 scripted crew.
The owner's10-crew/2-impostor question is answered in the report; no lobby expansion
or two-impostor training was silently started.

Latest owner steering: retain the architecture, document all failures/exploits/fixes
for the paper, plan eventual10crew/2impostors for both models, provide commands and
avoid further computer use. Old weights still choose new fake activities after
completion; new training is required to test a learned behavioral improvement.
