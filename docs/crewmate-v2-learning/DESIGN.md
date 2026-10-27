# Opt-in V2 rewards and PPO: implementation registration

2026-10-27. Base `7468ba8`, branch `codex/crewmate-v2-pretraining-diagnostic`.
Owner authorizes implementation/tests and a minimal optimizer smoke only. Full pilot,
historical checkpoint changes, reserved final tests, merges and deployment remain gated.
No additional standalone reporting diagnostic is requested or scheduled.

**Measured-cost amendment:** the four-model/1,024-choice proposal below is the original
design record, superseded by the current proposal awaiting approval. Smoke timing led to a smaller first stage
before any pilot ran. [PILOT.md](PILOT.md) and the versioned JSON specify the final
proposal: one seed-17 reward pair, 512 choices/model, 20 validation games/model and a
three-hour total ceiling. Seed29 is reserved for a separately approved replication.

## Context map / reviewed design

| New path | Responsibility |
|---|---|
| `crewmate_v2/learning/config.py` | versioned rewards, fixed pilot/seed registrations |
| `crewmate_v2/learning/rewards.py` | trainer-only per-physics-step event ledger |
| `crewmate_v2/learning/opponents.py` | actor-safe varied scripts, role-independent style rotations |
| `crewmate_v2/learning/env.py` | new training adapter, event tap, optional forced-WAIT coalescing |
| `crewmate_v2/learning/policy.py` | new masked V2 policy, strict checkpoint schema |
| `crewmate_v2/learning/provenance.py` | historical + V2 + learning source/config locks |
| `crewmate_v2/learning/train.py` | reuse PPO/physical-time GAE, smoke and approval-gated pilot |
| `crewmate_v2/learning/evaluate.py` | opportunity, belief, action-ranking and shortcut evaluation |
| `tests/test_v2_learning*.py` | exact rewards, privacy, replay, optimizer and guards |
| `configs/crewmate-v2-learning/*`, `docs/crewmate-v2-learning/*` | reviewable config, exact pilot and verification |

Read-only dependencies: original V2 flat modules; Phase 5/6 options, features and PPO;
Phase 4 belief; actor API; diagnostic reporting/shortcut patterns. No existing source
file is edited. A subpackage avoids changing the flat V2 provenance glob/digest.
No public-engine/API migration or production change.

### Rewards

Keep `rewards=None` behavior exactly reproducible with original opponents and forced
coalescing disabled. An independent event tap observes each completed physics step,
without changing the engine. A ledger attributes task completion according to the focal
actor's alive state at that event's position in event order; reports require an accepted
focal REPORT; votes require an actually submitted legal focal ballot, not a deadline
default. Deduplicate task IDs, reported bodies and meeting ballots. Reward only accepted
actions, never approach/navigation, repeated attempts or teammate actions.

The shaped profile is exactly the proposed +1 living task, +1.25 report, +2 correct vote,
-1 wrong vote, 0 skip/ghost task, -5 death, +10/-10 team outcome, -5 draw and original
-.005 per .6 simulated seconds. Draw is explicitly retained from the legacy reward.
Coefficients are versioned/configurable. No unapproved tuning follows smoke outcomes.

Per-tick frames preserve original gamma=.99 per .6 seconds, including terminal steps
without a clock increment. Outer options sum time-discounted frames from their own
start, while raw/component totals remain trainer-only. Hidden roles label votes in the
ledger but never enter observations, memory or policy tensors. Neither rewards nor
privileged info are fed back as model inputs. Frozen evaluation never updates weights.

### Policy / execution

Global42, candidate4x29 and action96x23 inputs retain the V2 actor projection. Encode
own-task4x8 with a 16-unit shared task encoder; add actor-only task row indices for task
choices. Candidate encoder29→32→32; task encoder8→16→16; pooled context90→128→128;
masked action scorer(128+32+16+23)→64→1; critic128→64→1. No new supervised identity
head or belief training. Frozen Phase 4 predictions and new public flags remain distinct.

Optional forced-WAIT coalescing executes the SAME physics ticks, observations, agent
memory updates and reward frames. It only avoids returning control to PPO when the
sole action is no submission. Never coalesce a state with a strategic choice. This
execution option applies identically to both pilot arms and has separate metadata;
canonical timing remains untouched. No report/vote override or coarse time step.

### Pilot to prepare, not run

Four fresh models: original versus shaped rewards × initialization seeds17/29. Each
gets 1,024 actionable PPO choices, rollout128, four update epochs, minibatch64, lr3e-4,
clip.2, value.5, max-grad.5, entropy.02, gamma.99, time-aware lambda.95. Final-budget
checkpoint only is the candidate; save intermediate256/512/768 without validation
selection or tuning. No historical checkpoint resume.

Training uses fixed deduction-only context (frequency1.0) in BOTH reward arms; no
mixture comparison in this experiment. A future mixed curriculum remains separate.
Opponent report styles rotate across episodes from private role-independent streams;
movement/meeting and hunter/patient/self-report styles also vary legally. Focal is
excluded from scripts. Validation uses four fresh scenarios, all three report rotations
and ordinary/deduction modes (24 games), plus original-opponent ordinary/deduction
controls (8 games): 32 games/model, 128 model-games, FOUR distinct validation scenarios.
This small pilot cannot establish acceptance or precise population effects.

Fresh match partitions: train2,700,000–2,709,999; validation2,710,000–2,710,003;
smoke2,720,000–2,720,003; unit fixtures2,730,000–2,730,099. Disjoint from historical,
diagnostic2,630,000 and engineering2,610,000 ranges. No new final-test partition is used.

Smoke is at most 32 transitions per arm, same initialization17 and same smoke stream,
one tiny PPO update per arm, with an overall 10-minute process bound. Record actual
time/experience and serialization, not learning success. Use its timing for a conservative
pilot estimate; the proposed total pilot ceiling is THREE wall hours including validation,
with preflight refusal if measured projection exceeds it. Owner approval must bind the
exact plan/config/source hashes before any full run. No parallel jobs in this pilot.

### Scientific limits / review

+2/-1 voting has positive expected immediate reward when p(correct)>1/3; two-candidate
guessing is therefore rewarded. Reporter-only accuracy81.6% makes this an exploit risk,
not proof of evidence learning. Implement the requested hypothesis, document a possible
wrong-vote -2 alternative before approval, but do not quietly change the locked pilot.
Report accepted-body farming, evidence-conditioned versus uninformed votes, skips,
survival/ghost/terminal components and team outcomes separately.

Role-independent style rotation balances style assignments, not accepted reporter roles.
Validation must measure reporter-only baselines and performance across reporting regimes.
The frozen belief model may retain its old reporter bias; score it separately from the
policy's conditional vote-ranking distribution. Pre-vote NLL/Brier/calibration use offline
truth labels only. No belief improvement claim from training four action policies.

Self-review: no historical mutations, no reward-to-actor path, event timing anchored to
actual physical calls, explicit empty-output/checkpoint guards and owner-approval gate.
This design follows the user's explicit implementation authorization; no extra diagnostic
round or intermediate permission gate is introduced.
