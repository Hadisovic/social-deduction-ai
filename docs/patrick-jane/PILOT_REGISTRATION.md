# Patrick Jane — bounded pilot registration

Owner authorization: approved staged learning in `CONTINUATION_SPEC.md`, reaffirmed
after the Escape interruption. One-million-decision training is **not authorized**
until the pilot is reviewed. This document records the rules before training;
the run's immutable `registration.json`, source snapshot and checkpoints record
the selected worker count, exact hashes, runtime and measured wall ceiling.

## Frozen design

Ten living AI crewmates initially, sharing one PPO policy but with independent
observations, event memories, GRU256 hidden states and actor/scenario RNG streams;
two legitimate-observation scripted impostors. No special roles or learned
impostors. The existing 1,031,070 parameters, attention, five heads and exact
`patrick_jane/rewards.py:COEFFICIENTS` remain unchanged.

The PJ Classic profile, individual task ownership, automatic ghosts, private
visibility and deterministic event ordering are described in
`GAME_SETTINGS_VERIFICATION.md` and `GHOST_MEETING_BEHAVIOR.md`. Retail omissions
and uncertain physical kill-distance calibration remain explicit limitations.

PPO: learning rate 0.0003, gamma 0.99 per 0.6 simulated seconds, elapsed-time GAE
lambda 0.95, clip 0.2, entropy coefficient 0.01, four epochs, gradient norm cap
0.5, sequence length 128, four independent sequences per optimizer minibatch.
Truncated recurrence starts each chunk from its collected hidden state; there is
no additional burn-in. Recurrent sequence batching changes implementation
efficiency, not the architecture or mathematical objective.

## Seeds, budget and evaluation

- Fresh policy/optimizer initialization: seed **17**, never an old checkpoint.
- Training: sequential unused PJ development seeds **2900000–2900999**.
- Fixed held-out development scenarios: **2910000–2910099**, using the registered
  held-out scripted-opponent mixture. Same scenarios at every evaluation.
- Engineering: **2920000–2920099**, separate from the training/validation ranges.
- Phase 6 familiar/patient final partitions remain unopened.
- Nominal target: **250,000 individual living-crewmate action decisions**. Shared
  agents and repeat evaluations do not multiply the number of distinct scenarios.
  This counts policy calls including choices constrained to one legal action;
  dead ghosts make no neural decisions.
- Sampling ignores batch-only target padding; simultaneous local memory events
  have stable public-ID tie ordering. Spawned workers use Python hash seed **0**.
- Complete matches only enter PPO. Start with the measured fastest stable worker
  configuration. Later batch size is the smaller of that capacity and
  `ceil(remaining target / observed mean decisions per completed training match)`,
  with a minimum of one. Stop after the first complete updated batch meeting the
  nominal target. Record actual overshoot; never overwrite checkpoints.
- A **275,000-decision absolute collection guard** prevents an extreme tail from
  silently expanding this pilot. An interrupted partial batch is preserved as a
  failure and is withheld from PPO, rather than truncated into an episode.
- Evaluate the frozen zero-update policy, each crossing of 100,000 decisions,
  and the final policy on **all 100** fixed development scenarios. No coverage
  reduction to fit time. Evaluations never update weights.
- Champion selection: held-out crew wins first; ties use task wins, then correct
  minus innocent ejections. Keep champion copies separate from latest resumable
  checkpoints. Selection on repeated development scenarios is exploratory,
  not a blind generalization claim.

## Readiness and automatic stopping

Require matching private-stream regression tests, full-game GPU PPO and identical
checkpoint reload, historical compatibility, source/checkpoint preservation,
ghost/sabotage correctness and the corrected matched 8/16/32 comparison. The first
comparison is retained as a negative result because padded categorical sampling
changed actor RNG streams across worker counts (research issue R033).

Register the wall ceiling from measured complete-game collection and update
costs, including validation and contingency, before launching. The runner stops
on deadline, process/RPC failure, non-finite loss/gradient/weights, changed source
hashes, legality errors, reward duplication/attribution errors, sustained severe
throughput regression, or severe paired validation regression. The last two
criteria are three full batches below 40% of the first-three median collection
rate, and a paired win-rate loss of at least 25 percentage points with one-sided
exact p<0.01 versus zero. Normal smaller final batches are excluded from the
throughput regression test. No automatic recovery changes the protocol.

The spectator remains an independent desktop process. A single selected worker
publishes throttled observer snapshots; complete selected matches have immutable
replays. Spectator role truth never enters neural tensors or agent memory.

## Interpretation

Report training and full-run throughput, win-rate uncertainty, sabotage repairs
and failures, living/ghost task contributions, legal report/repair opportunities,
accepted reports, policy vote/skip choices, correct/incorrect votes and
accusations, meetings, reward components, checkpoint integrity and actual budget.
Compare each learned state to zero on paired scenarios. Reporter-role imbalance,
scripted opponent patterns, shared-match correlation, one initialization, short
training and scripted ghost work remain limitations. Increasing reward without
gameplay improvement is a negative or inconclusive result, not proof of deduction.
