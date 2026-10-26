# Before the first impostor training run

**Pilot approved:** the owner's “go on” on 27 October authorizes the two 4,096-decision
runs and 40 validation matches below, with a 3-hour learning/validation ceiling.
Trainer verification and source locking precede those runs. No further training,
reserved final tests, merges or deployment are approved.

Updated 27 October 2026 after reading the owner's
[shared conversation](https://chatgpt.com/share/6adfb9c6-ffe8-83eb-973b-b12e0e27816f)
through its existing Edge tab. Earlier web/in-app attempts failed; it is now read.

## What the chat confirms

The conversation contains unrelated CarPlay troubleshooting and a final Among Us
section. Only the project section informs this plan:

- The owner develops the impostor; the friend independently improves the crewmate.
- Use the verified V2 baseline and an independent impostor branch.
- Freeze full-29 and the belief predictor; implement reporting/witness safeguards.
- Verify engineering behavior, then propose learning design and compute budget.
- Training, merges and deployment require explicit approval.

These requirements are covered by draft PR #10. The chat adds no architecture,
reward coefficients, run count or approved training budget. Unrelated troubleshooting
instructions are outside this project.

## Remaining work, in order

- [x] Read the shared chat and reconcile it with the attached specification.
- [x] Finish the environment, frozen adapter, safeguards and engineering checks.
- [ ] Implement the independent impostor trainer and evaluator.
- [ ] Wire the new packet keys and 256-candidate masks into PPO batching.
- [ ] Restrict the optimizer to impostor parameters; check frozen hashes throughout.
- [ ] Implement seconds-aware GAE; verify rewards across coalesced meeting windows.
- [ ] Add immutable checkpoint save/load/resume with source, lobby, rewards, opponent,
  safeguards, initialization and RNG provenance; reject mismatches/overwrites.
- [ ] Verify resume reproduces the next action and preserves private actor memory.
- [ ] After approval, check impostor gradients/updates while frozen models stay unchanged.
- [ ] Lock the pilot configuration, budget and validation protocol before learning.
- [ ] After explicit approval, run the two pilots below.
- [ ] Evaluate the preselected final checkpoints on 20 paired validation scenarios each.
- [ ] Produce learning curves, PPO diagnostics, seed outcomes, kills/witnesses,
  reports, ejections, timeouts, win reasons, legal-action checks and timing records.
- [ ] Use those results to propose any further training or controlled comparisons.

The learning driver and resume path are not implemented or verified yet. Passing
environment tests does not establish that a future optimizer works.

## Proposed first learning stage

| Item | Setting |
|---|---|
| Learner | New 48,738-parameter impostor network, initialized from scratch |
| Opponents | Pinned frozen full-29 plus three scripts; both safeguards enabled |
| Rules | Ordinary V2; canonical cooldown, meetings, ghosts and task victory |
| Runs | Two initializations, seeds 71 and 83 |
| Experience | 4,096 policy decisions per run; 8,192 total |
| Checkpoints | Save 0/1024/2048/4096; preselect step 4096 for pilot evaluation |
| Validation | Same 20 fresh paired scenarios per model; 40 executions total |
| Rewards | +10 win, -10 loss/timeout, -0.001 per simulated second; gamma=1 |
| PPO | Rollout 256; minibatch 64; four epochs; lr 3e-4; clip .2; entropy .02 to .005 |
| GAE | Proposed lambda .99 per simulated second; compose over elapsed time |
| Runtime cap | 3 hours for learning plus validation; no automatic extension |
| Final tests | Unopened |

Both initializations start at registered training seed 3100000 and advance by episode
index. Pilot validation uses 3200000–3200019 for both models, excluded from training.
Record independent RNG streams for action sampling and optimizer shuffling. Existing
Phase 6.1 final partitions remain unused. Settings are proposals, not proven optima.

## Time needed

| Stage | Planning estimate |
|---|---:|
| Trainer/evaluator, checkpointing and verification preparation | 45–90 minutes |
| Two pilots plus 40 validation matches | 1–2.5 hours; 3-hour hard ceiling |
| First results and graphs, including preparation | **About 2–4 hours total** |

The untrained network benchmark measured 227 decisions in 44.39 seconds (5.11/s).
At that rate, collection of 8,192 decisions alone projects to about 27 minutes;
40 validation matches project to 15–40 minutes. Optimizer/resume overhead is unmeasured,
and learned behavior, episode lengths, meetings and machine load can change throughput.
The wider planning estimate includes preparation, verification and reporting.

This is the first learning assessment, not a promise of a strong final model. There
is no justified fixed duration for **all** needed training before inspecting pilot
curves and behavior. If learning is useful, propose more replicated seeds and broader
opponents. If it fails, diagnose incentives and opportunities before adding steps.

Start only after configuration review, trainer checks and explicit approval. Stop for
budget exhaustion, changed frozen hashes, privacy/replay mismatch, nonfinite values or
illegal actions. Preserve partial outputs; do not silently retry or tune from final tests.

[Engineering results](README.md) · [session checklist](SESSION_CHECKLIST.md) ·
[collaborator handoff](../phase6_parallel_research_handoff.md).
