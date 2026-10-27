# Impostor track — implementation handoff

Owner: impostor research. Friend: independent crewmate improvements. Only the
impostor policy learns in this branch. Crew and belief weights stay frozen.

**Current, 27 October:** later owner approval supersedes the old training hold below.
The first two-run pilot completed without useful validation killing/wins;
[preserved results](impostor-v2/pilot-20261027/README.md). An approved improved
family-policy/reward study completed from `f96abbe`: two 8,192-decision runs,
80 selection and80 fresh confirmation/control matches in117.24 minutes. The selected
71/8192 model won2/20 selection games but0/20 fresh confirmation games. It stalls on
fake-task restarts; it is not a strong winning model. Read the [session summary](impostor-v2/SESSION_SUMMARY.md)
for all four runs, failures, mixed crew roles, measured causes and next fixes.
See [protocol](impostor-v2/strategic-v2-protocol.md) and
[live checklist](impostor-v2/SESSION_CHECKLIST.md). No merge, deploy or final-test opening.
The root `run_impostor_ai.py` now watches the evaluated release. Prepared pursuit
code passed tiny development checks but no registered training study ran. The owner
requested summary/push and an end to today's work; no training remains active.

## Sources and authority

- User's implementation request and attached **message (5).txt**, read in full on
  26 October 2026. That initial instruction held **all RL training**, including small
  optimizer smoke runs, until explicit approval. Later direct approvals are recorded above.
- [V2 PR #9](https://github.com/Hadisovic/social-deduction-ai/pull/9), pinned baseline
  `e62ba0b6c33ffd0317f78e18b4289d778cd5af2b`. Verified against the fetched remote.
- Research baseline `2baa5dbc1b36b047ed97934d69526573b510c280`, verified ancestor of V2
  and current fetched `codex/phase6-1-seeds29-43` head.
- [Original two-track handoff](https://github.com/Hadisovic/social-deduction-ai/blob/docs/phase6-two-track-handoff-20261026/docs/phase6_parallel_research_handoff.md),
  read from that branch. Its old instructions to use pre-V2 mechanics and eyewitness-only
  reporting are superseded by the owner's newer V2/body-discovery requirements.
- [Shared planning chat](https://chatgpt.com/share/6adfb9c6-ffe8-83eb-973b-b12e0e27816f),
  supplied by the owner as context. **Read on 27 October 2026 through the existing Edge
  tab**, after earlier web/in-app attempts failed. Its final Among Us section confirms
  the separate crewmate/impostor workstreams and repeats the V2 implementation and
  approval requirements. It adds no training settings or budget. Unrelated CarPlay
  troubleshooting in the conversation is outside this project.
- [V2 mechanics and limits](crewmate-v2/README.md) and [implementation design](crewmate-v2/DESIGN.md).

Branch: `codex/impostor-environment-v2`, created directly from the exact V2 commit,
not main. No historical runtime, checkpoint, selection record, shared V2 mechanic,
website or production setting is edited. No PR is merged or deployment performed.

## Frozen crew integration

`impostor_research/frozen.py` calls the **original strict loader** for
`artifacts/phase6/remaining-six/full-29/checkpoint-8192.pt`:

`238904b7fe303b77e885c4cd6fdf4d9d447a62eb5b463436a391c0b4d6e1cb43`

The model and Phase 4 belief predictor use evaluation mode, no gradients and tensor
hash checks. Each actor has separate memory; model weights may be shared read-only.
The original engineering driver constructs no optimizer; the approved standalone
trainers optimize only their new impostor networks. The owner/friend cannot silently swap
this pinned checkpoint during a future run.

The adapter projects the legitimate V2 envelope's **unchanged legacy actor member**
through the original encoder. It preserves 40 global features, 4×26 player features,
96×23 candidate features, original time normalization and option meanings. V2-only
task-bar, stage, readiness, confirmed-role and visual-proof fields are not fabricated
as neural features. Real V2 legal checks prevent interacting during incubation.

This is **lossy transfer**, not compatibility with the new V2 network schema or evidence
of equal performance. Four permanent multistage tasks, longer games and ghosts are a
distribution shift. The old clock saturates at 240 seconds; the belief predictor can
conflict with V2 proof. Safeguards provide behavior guarantees, not neural retraining.
Do not compare this assisted transfer's wins directly with historical validation wins.

Real deterministic probe: in a controlled V2 kill scene, full-29 had a legal REPORT
but chose GO_TO_TASK. The regression test preserves this observation. It is one fixture,
not an estimate of the model's overall report rate.

## Opponent safeguards

`opponents.py` applies to **all four living crew opponents**, one frozen learned and
three scripted. Default: both safeguards on. Four explicit settings remain available:
neither, body discovery only, eyewitness only, both.

1. A visible body is reported when legal; otherwise the crew approaches through A*.
   While approaching, competing task proposals are suppressed. Lost/resolved bodies
   cancel only the wrapper's pursuit. Meeting transitions clear pursuit.
2. Only validated direct `WitnessedElimination` events record a killer. That private
   memory survives someone else's report. A subsequent eligible ballot targets the
   witnessed killer; ineligible targets fall back to the ordinary proposal.

Seeing a body does not identify a killer. Claims and suspicion do not become eyewitness
facts. Dead crew never receive controller decisions; the existing task-only ghost
executor remains unchanged. Proposals, executed actions, reasons and whether they differ
are separately recorded in private audit data. These assists are **scripted opponent
behavior**, never claimed as learned reporting or voting. Existing scripts already have
some reporting/witness behavior; disabling wrappers does not disable those baseline scripts.
An original proposal may be a held option's mechanical continuation rather than a fresh
network call; the benchmark reports neural inference calls separately.

## Independent impostor interface

`env.py` owns the trusted world orchestration. `memory.py`, `features.py` and `policy.py`
use only the impostor's own validated V2 envelope. No crew memory, hidden position,
ghost position, hidden role labels or trainer metrics enter network packets.

- 35 globals: own position/movement/fake-task status, cooldown, time, public task bar,
  meeting allowance, phase/region, received public accusations/kill claims/votes.
- Four 21-value player rows: legitimate public/direct history, visibility, current
  observed and last-seen displacement, age, authenticated public/local proof.
- Up to 256 candidates ×48 values, target indices and a legal mask. Capacity overflow
  fails explicitly. Padding cannot be selected; there is always a no-submission action.
- Choices: room/console/observed-location travel, cancel, legal kill targets, fake tasks,
  body approach/report, emergency, vote/skip and finite public-vocabulary claims.
  Fabricated alibis/accusations are choices, not truth leaks. No vents or sabotage.
- Private impostor memory has no fake crewmate role and does not use the crew belief
  predictor. Terminal network packets are absorbing zeros, not post-game truth.

The new untrained network uses a shared 32-unit player encoder, pooled context with
two 128-unit layers, a 64-unit candidate scorer and a separate 64-unit value head.
Shared candidate scoring avoids fixed player-ID strategies. Explicit temporal summaries
make a small feed-forward first baseline reviewable; a recurrent model is a later controlled
comparison if history compression demonstrably limits learning.

The canonical lobby is loaded unchanged: 1× speed, 1×/1.5× vision, 45-second cooldown,
medium range, 15-second discussion, 120-second voting, one emergency, 1/1/2 tasks,
visible voting results, confirmed ejections and visual proofs. Timeout is 900 simulated
seconds. Ordinary games remain default; deduction-only is an explicit comparison arm.

Headless execution retains every .2-second physics tick and .6-second scheduled actor
observation. Forced empty meeting windows can be coalesced into a single Gym transition;
opponent actions, evidence, clocks and ghost tasks still advance on the original ticks.
Rewards and discounts compose over actual elapsed simulated seconds. This is not timer
compression, and long forced windows cannot create extra learning samples.

## Reward and experiment proposal — not approved or executed

Start with +10 impostor victory, −10 crew victory, −10 timeout, −0.001 per simulated
second. No per-kill, witness, fake-task, report, claim, vote, proximity or survival bonus.
The reward is computed in the trusted trainer boundary; its private inputs never enter
observations. This prioritizes winning without paying for reckless kills or deceptive
actions that have no effect. Use **undiscounted finite-horizon return (gamma=1)** for
the first pilot: discounting a negative terminal outcome can reward delaying a loss.
The 900-second cap bounds return, and the small time cost penalizes passive delays.
The environment still supports time-composed discounts for explicitly reviewed ablations.

Risks: sparse reward can stall; time cost can favor earlier defeat among losing paths; task
victories may dominate; forced safeguards define a narrower opponent distribution.
Track timeout/passivity, first kill opportunity/action, witnessed kills, task wins,
ejections, report sources, learned proposals versus overrides and generalization.
If learning stalls, propose a separately registered modest kill-shaping ablation;
never silently alter the canonical cooldown or pay for hidden witness information.

Proposed first approval: **two 4,096-decision ordinary-mode pilot runs**, initialization
seeds 71 and 83, both safeguards, pinned full-29 plus three scripts. Fixed 0.6-second
decision windows first; no speculative 24-second impostor movement holds that could
hide kill opportunities. PPO rollout 256, minibatch 64, four epochs, lr 3e-4, clip .2,
value coefficient .5, gradient norm .5, entropy .02 → .005. Use physical-time GAE with
an explicitly registered seconds-based lambda (proposed .99 per second); do not blindly
reuse a helper that assumes every transition spans .6 seconds.

Save checkpoints at 0/1024/2048/4096; evaluate the **preselected final 4096-step checkpoint**
on 20 paired validation scenarios per pilot (40 total), without searching checkpoints
in this first budget. A larger study can separately approve checkpoint selection and
100-match validation. Leave final tests unopened. Report Wilson intervals
and paired scenario differences; two training seeds cannot establish robust generality.
Evaluate safeguards off/body-only/eyewitness-only on the selected policies later, plus
a separately hash-pinned full-43 opponent and held-out scripted styles. Those adapters
and larger evaluation batches require separate review; current loader intentionally pins full-29.

New registered ranges: training 3100000–3199999, validation 3200000–3200199,
reserved impostor final 3300000–3300499, engineering 3500000–3500019. Existing Phase 6.1
final ranges 1230000–1230499 and 1240000–1240499 are untouched. This branch contains
only an engineering driver; it exposes no training/final-evaluation command.

Stop immediately for changed frozen hashes, illegal actions, nonfinite tensors,
privacy/replay mismatch, or budget exhaustion. Review after each pilot; no automatic
extension, curriculum switch, opponent replacement or use of final outcomes for tuning.
If ordinary matches offer too few legal kill windows, compare a separate deduction-only
pilot with identical seeds/configuration; do not mix modes without registration.

Methods references: [PPO](https://arxiv.org/abs/1707.06347) supports the clipped-policy
baseline; [RL statistical evaluation](https://arxiv.org/abs/2108.13264) motivates reporting
uncertainty across runs. Numerical hyperparameters above are project proposals, not
claims that those papers establish optimal settings for this game.

## Verification and compute

See [engineering report](impostor-v2/README.md) for measured results, reproducibility,
timing limits, test counts and the approval budget. Passing these checks demonstrates
engineering behavior only; no learned impostor performance exists yet.

Final engineering result: **239 tests passed; 10 bounded matches completed; exact
reference/optimized/repeat traces; frozen models unchanged.** Proposed first pilot:
two 4,096-decision runs plus 40 paired validation matches, **1–2.5 hours estimated,
3-hour hard ceiling**, requiring explicit approval. No training has begun.
The [remaining trainer work and timing](impostor-v2/TRAINING_NEXT_STEPS.md) adds an
estimated 45–90 minutes of preparation: about **2–4 hours to first pilot results and
graphs**, including preparation. Total training to a strong model remains outcome-dependent.

Next collaboration milestone: frozen-versus-frozen cross-play under a deliberately
shared ruleset. Only after independently useful agents exist should the team consider
alternating training, historical-opponent pools, larger lobbies or self-play.
