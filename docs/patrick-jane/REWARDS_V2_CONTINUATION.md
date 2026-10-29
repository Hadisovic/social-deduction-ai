# Rewards V2 — manual continuation from verified update 11

The owner intentionally changed the rewards. Keep those changes. This is a
new reward regime, not a same-protocol resume or evidence of learning failure.
The historical run stopped with `RuntimeError('Frozen training source changed')`.

## Verified inheritance

- Source run: `local-runs/patrick-jane/lights-learning-v1`, preserved unchanged.
- Checkpoint: `checkpoints/attempt-0001-update-000011.pt`.
- SHA256: `86eb2b968fd6453cf842c0c3fffd304f6224b76ea88ba0e3675f0e91f6e8e753`.
- Archive SHA256: `713ed713d4f9951bf712a59ff87375851a908a2c6cbc44f2f3d7661c36050563`.
- 764,660 lifetime decisions = original pilot 255,083 + earlier lights phase
  509,577. Five lights-phase updates, 160 training games. Update index 11.
- Fifty-two policy/critic tensors import exactly; architecture 1,031,070 parameters.
  Checkpoint/registration/archive provenance, runtime, finite policy and old Adam
  tensors, private-observation encoders, memory, PPO and all gameplay sources
  are checked. Only reward coefficients, audit, runner and reporting differ.
  The reward implementation is AST-equal outside its coefficient assignment.
- Reset Adam's moment estimates at this transition because reward/return targets
  changed materially. Keep every learned policy and critic weight and restore RNG.
  No fresh neural initialization replaces learned weights. Later resumes restore
  this new run's own Adam state normally; they do not reset it again.

## Registered new regime

```python
COEFFICIENTS = dict(
    win=10., loss=-10., task=1., ghost_task=0., repair=3.,
    impostor_ejection=4., innocent_ejection=-2., report=1.25,
    exploration=.02, new_player=.05, death=-2., idle=-.01,
    invalid=0., vote=0., accuse=0., defend=0.
)
```

Lights-unresolved cost remains -0.005 per living crew per simulated dark second,
including meetings. Ghosts retain owned task execution and team game progress but
earn zero direct neural task reward. Indirect effects on team victory still exist.
Death audits accept one -2 penalty, reject duplicates and incorrect magnitudes;
report bounds use 1.25; any nonzero direct ghost-task reward is rejected.

Ten independent-memory learning crew, two scripted impostors, lights only,
killing enabled, 32 workers, all architecture/PPO settings unchanged. Normal
duration is not a victory rule; technical cuts and living bootstrap remain as
registered in [the timeout investigation](TIMEOUT_INVESTIGATION.md).

Target **490,423 new-regime decisions**, starting at unused training index 352
(seed 2900352). Complete-match/boundary overshoot is recorded, with the existing
110% absolute guard. Combined post-pilot target is 509,577 + 490,423 = 1,000,000;
lifetime target 1,255,083. Neither inherited nor validation actions count toward
new-regime progress. Training partition stays 2900000–2900999; eight paired entry
and final development scenarios remain 2910000–2910007. No reserved final tests.

The earlier phase consumed 7,904.0767898 active seconds. The new command reserves
35,295.9232102 seconds (about 9 h 48 min 16 s) for this phase, preserving the
12-hour combined active ceiling. Pauses between attempts do not consume time.
Engineering checks use isolated folders, short technical guards and their own
small budgets; they never become the real warm start or competence evidence.

## One command, independent spectator and safe recovery

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Files\RL\_patrick_jane\Start-PatrickJaneRewardsV2.ps1"
```

Results: `C:\Files\RL\_patrick_jane\local-runs\patrick-jane\lights-rewards-v2`.
The registration/source snapshot and zero-update origin are prepared without
training. Running the command opens/reuses the existing spectator, verifies the
new registration, and starts training. Ctrl+C once interrupts; the same command
resumes from a committed checkpoint. Uncommitted partial batches are withheld.
The old run is never reused as an output directory or silently migrated.

Outputs include `TRAINING_REPORT.md`, `training_results.json`,
`completion-status.json`, `console.log`, `learning_curves.png` after the first
update, checkpoints/optimizer/RNG, selected replays, per-update metrics, live
status/percentage/ETA and entry/final evaluations. Completion remains contingent
on target, final checks and saved outputs. Spectator stays independent of training.

Reports distinguish this regime from the preserved old phase. Raw reward curves
across the boundary must not be pooled or interpreted as one homogeneous
learning experiment. Reporter/script confounds, scripted ghost work, repeated
development scenarios, one policy initialization and truncation censoring remain
limitations. Phase 6.1 acceptance remains pending. No long training is launched
by the preparation agent. See research issue R043 and `REWARDS_V2_VERIFICATION.json`.
