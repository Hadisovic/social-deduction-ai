# Impostor research record — failures, exploits and corrections

Purpose: preserve evidence for the final paper. Keep unsuccessful experiments;
distinguish engineering checks, training, selection, confirmation and reserved tests.
Updated27 October2026, Asia/Amman. Current architecture remains unchanged.

## Experiment ledger

| Version / initialization | Training | Training outcomes | Selection / confirmation | Interpretation |
|---|---|---|---|---|
| First trial V1 /71 | 4,096 decisions;18 complete games | 0wins,0kills | Preselected final checkpoint:0/20 validation wins,0kills | Failed to learn useful killing |
| First trial V1 /83 | 4,096;18 games | 0wins,1kill | 0/20 validation wins,0kills | One sampled training kill did not generalize |
| Improved V2 /71 | 8,192;82 games | 0wins,68kills | 4096:0/20;8192:2/20 selection wins | More kills; selected8192 before confirmation |
| Improved V2 /83 | 8,192;78 games | 1win,57kills | 4096:1/20;8192:0/20 selection wins | First real learned win; not reliable play |
| Locked V2 /71-8192 | No additional training | Same frozen checkpoint | Fresh confirmation0/20wins,4kills,20task losses,0ejections | Winning selection result did not establish fresh success |
| Prepared pursuit V3 | No registered study | Only tiny development gradient/resume checks | None | Not a fifth research training run; owner stopped before study |
| Reward/action V4 | **No PPO optimizer steps** | Implementation +32 new regression tests | Two complete development transfer smokes:0wins,0kills,31/30 fake completions | Fixed activity completion; no evidence of stronger learned strategy yet |

These are four registered trained runs, not six. Selection executions reuse paired
scenario sets; do not sum them as independent trials. V4's old-weight transfer is
not a newly trained model. Tiny V3 test updates are development checks, not a study.

The first learned win was seed83's training scenario3100168: two kills and an
innocent crew ejection produced parity. It is valid to report the first neural
impostor win; the logs do not establish that a learned deceptive claim caused it.

## Observed failures and mechanism

| Observation / exploit | Evidence | Correction or next test | Status |
|---|---|---|---|
| First trial barely chose useful combat | 36games,1total kill; both final models0/20wins and0kills | V2 balances action-family choice separately from target count | Implemented/trained; still weak |
| Many destinations can dominate exploration | First-trial action distribution about97.6% task/room travel | Hierarchical family/target sampling; separate actor/critic fitting | Implemented; not a clean isolated causal ablation |
| Greedy fake-task restart loop | Locked V2 confirmation:3,560 fake selections; engine restarts4s activity at each0.6s choice | V4 repeated same-console fake continues current activity; voluntary leave/cancel retained | Regression/replay tests pass; smoke completions31/30 |
| Fake preference persists after mechanical fix | Both V4 old-weight smoke games still lose by tasks,0kills | Retrain/evaluate under versioned interface after approval; inspect greedy vs sampled behavior | No new study yet |
| Too few actual kill opportunities | Only4/20 locked V2 fresh games offered legal kills; all4 were taken | Observe pursuit, ready-idle time and path choices; small bounded pursuit potential | Implemented, learning benefit untested |
| Weak independent confirmation | Selected2/20, then0/20 fresh; Wilson95% win interval0–16.1% | Larger preregistered fresh comparisons/replicates; lock before confirmation | No superiority claim |
| Potential pursuit reward farming | Design risk, **not an observed historical exploit** | Lifetime0.05 high-water credit; no payment for stationary actor/new target; terminal correction | Oscillation/switching/visibility tests pass |
| Cooldown time cost could bias risky kills or meeting abuse | Design risk, not yet an observed trained V4 behavior | Log ready seconds, meetings, kills/witnesses and reward components; select on wins | Future validation requirement |

The V2 mechanical loop is confirmed. Sparse credit, short experience and poor
pursuit are plausible contributing explanations, not isolated causal conclusions.
No ejections/witnessed kills in a passive policy is not proof of learned stealth.
More kills alone is not successful strategy. Reward return is not test accuracy.

Owner's live follow-up: the impostor repeatedly targets fake tasks in every game
they watched, including an empty room. The number of watched games was not stated;
keep this qualitative observation separate from the two recorded development
matches and previous20-case confirmation. The correction makes fakes finish, but
the frozen NN immediately selects new ones. Watching applies inference only, not
PPO updates. Mark learned fake-loop avoidance **failed/pending retraining** rather
than presenting completion of an activity as a solved strategy problem.

## Controls and opponent limitations

All current matches use1 learned impostor,1 frozen full-29 NN crewmate with its
frozen belief model, and3 scripted crew. All living crew have body-report and direct
witness voting overrides. Those safeguards are rules, not learned crew competence.
The full-29 adapter omits some new V2 task/proof fields. Team results therefore do
not isolate one learned crewmate's contribution. The friend's improved crew is not
loaded; neither side should claim an all-neural duel yet.

In20 paired fresh confirmation scenarios, the selected V2 won0, the older NN won0,
an untrained NN won0, and the handwritten attacker won1 (31kills). The scripted
attacker is a control in separate matches, not a fallback behind the neural viewer.
The sample is small. Do not claim statistically established script superiority.

## Evidence and reproducibility

- [Previous complete session summary](SESSION_SUMMARY.md): failed runs and actual logic.
- [V1 report/data/6 graphs](pilot-20261027/README.md):25.37min complete study.
- [V2 report/data/8 graphs](strategic-v2-20261027/README.md):117.24min complete study.
- [V4 exact changes/reward table](options-v4/README.md):3–4h proposed next budget, not approved.
- [V4 smoke JSON](options-v4/smoke-results.json):123.60s, fixed development cases, zero optimizer steps.
- [Exact current NN tables](NN_ARCHITECTURE.md): all dimensions, feature semantics, output families.

Preserve all16 historical policy snapshots, checkpoint hashes, frozen opponent
hashes, source manifests, raw ignored resume journals, seed lists and failed
results. Version new action/reward/observation contracts; never relabel old results
as V4. Keep terminal/task loss/vote/reward counts, real wall time and simulated
experience separately. No reserved final partition has been opened in this work.

## Roadmap for both models: configurable lobbies up to15 players

The owner requests custom settings up to15 total players and1–3 impostors, subject
to validated legal combinations for the targeted real-game version.10crew/2
impostors is one12-player preset. Keep the current5-player experiment and keep the
current architecture while fixing learning first; do not enlarge it without evidence.

Typically share **one crew policy and one impostor policy**, plus the existing crew
belief network. Each seat has independent actor-private memory/observations and
action RNG; sharing weights does not share hidden knowledge. It does not require
15 separately trained models. Additional checkpoint variants serve opponent
diversity, not a mandatory network per seat.

1. Demonstrate useful single learned impostor and learned crewmate behavior with
   frozen opponents, correct reporting/witness voting and private information.
2. Agree with the crew developer on a shared versioned variable-player observation/action
   contract. Each actor can have up to14 other-player slots rather than4; current hardcoded
   roster checks, masks, target indices, counts/normalizations and candidate capacity
   must be audited. Shared encoders/pooling may remain useful; transfer is not automatic.
3. Implement1–3 legal impostor seats, client-selectable validated lobby settings,
   legitimate teammate identity information,
   friendly-kill masks, simultaneous-action/kill resolution, separate cooldowns,
   parity/task victory, body evidence, meetings/voting and ghost isolation. Do not
   expose one impostor's unseen observations to its teammate by accident.
4. Adapt and test **both** policies against the same contract. Version checkpoints
   and retain the5-player baseline. Test player-ID permutations, roster sizes,
   dead actors, evidence provenance, targeting and deterministic replay.
5. Define mixed and fully neural rosters explicitly. Freeze specific compatible
   crew/impostor checkpoints for paired cross-play; report each opponent mixture.
6. Only after individual agents work, evaluate coordination with adapting opponents
   and self-play. Register training/validation/confirmation seeds and compute budgets
   before runs. Keep final tests separate and request training approval.

The owner's latest instruction is to record this roadmap, not start variable-lobby
implementation or training in the current reward/action task.

## V4 study approval

After the implementation report, the owner said “go on”: the proposed two-scratch-
initialization study is now approved with a4h ceiling. Read
[the frozen protocol](options-v4/TRAINING_PROTOCOL.md). Prelaunch genuine-gradient
checks are development tests, not extra research runs. Keep earlier zero-optimizer
implementation results as historical evidence; do not rewrite them as new learning.

## Interrupted V4: real-spawn integration flaw

The training audit found pursuit positive credit was zero in all75 completed71
games and15 durable83 games. Initial spawn distances of about0.7 map units were
inside kill range; they saturated the lifetime high-water mark during the initial
45s cooldown. Synthetic far-target tests had not covered this initial condition.
This is an implementation limitation, not evidence that active pursuit shaping
failed to improve learning. The kill/time/ejection/fake corrections were active.

Owner explicitly chose to stop, fix and restart within the ORIGINAL deadline.
V4 was interrupted before validation:71 finished8192/75games/68kills/0wins;83 is
partial1792/15games/16kills/0wins. Preserve the last durable counts, not in-flight
work. Six policy snapshots and compact logs are retained in
[the interrupted report](options-v4-interrupted-20261027/README.md).

V4.1 arms geometry only when cooldown is ready; clears geometry during cooldown
without refilling credit, and retains the0.05 lifetime bound and terminal correction.
Its actual45s clustered-spawn/visible-approach test proves nonzero legitimate credit
without forced kills. Five corrected preflight tests passed130.82s, including exact
PPO continuation and two isolated development workers. An earlier fixture stopped
at a room doorway and never observed/approached crew; the fixture was corrected to
use a public-console route. That was not a second production shaping defect.

Read [V4.1's separate protocol](options-v41/TRAINING_PROTOCOL.md). It uses new
scenarios and RNG streams, scratch weights and the unchanged75,075-parameter NN.
Two one-thread CPU training workers run independent seeds in parallel; no extra
runs/decisions. Absolute deadline stays01:03:50 Amman on28October; failed work and
repair consume the same original budget. No test outcomes were used for this repair
because original selection/confirmation had not opened.
