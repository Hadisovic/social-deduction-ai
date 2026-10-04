# Validation decision before opening final tests

Reviewed on 2026-10-04 (Asia/Amman). No final-test outcomes were available.

Select full-memory seed 17, checkpoint 8,192, using the declared validation ranking.
Stop this training batch and proceed to the independent final evaluation.

- On the same 20 diagnostic seeds, seed 17 reached 100% crew wins at both
  6,144 and 8,192 decisions. The other seeds were already near this ceiling.
- On the larger 100-match validation set, the selected policy won 96 matches,
  completed 1.69 own tasks per match and made 93 correct non-skip votes out of 93.
  There were no recorded illegal actions or navigation failures.
- Paired win differences were +46 percentage points versus idle (95% bootstrap
  interval +36 to +56), and +54 versus random (+44 to +64). This supports useful
  focal-player behavior, beyond scripted teammates carrying an idle player.
- The difference versus scripted was +4 points (0 to +9). Superiority over the
  scripted baseline is not established. Conditional voting accuracy does not
  measure all strategic skill, and its all-success interval is overconfident for
  unseen failure modes. Final evaluation must report the vote denominator.
- Other full-memory seeds and the one current-only ablation won 93/100. They
  completed tasks but cast no votes. One ablation seed cannot establish a
  replicated benefit from memory.
- Untrained controls completed zero own tasks on the 20 diagnostic seeds.
  Preserved V1 completed zero tasks and won 43/100 in larger validation.
- PPO diagnostics are finite; conservative policy updates and noisy critic loss
  do not demonstrate convergence. Greedy validation is much stronger than
  stochastic training performance. This decision applies to the deterministic
  deployment policy, not a claim that every sampled action is reliable.

More iterations are not automatically better. These results justify testing this
batch rather than tuning repeatedly against the same validation seeds. They do
not establish a globally best model or real-game transfer. Selection is immutable
once the final tests open. Any later research requires a new held-out protocol.

Final design: nine methods, 1,000 paired matches each; 500 ID and 500 patient
opponent matches. Include three full-memory seeds, one current-only seed, V1,
the selected seed's untrained initialization, and random/scripted/idle controls.
Phase 6 remains gated on independent evidence that one learned crewmate works.
