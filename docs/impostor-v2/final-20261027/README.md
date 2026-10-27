# Impostor AI — detailed teammate guide

Start with the shorter [session summary](../SESSION_SUMMARY.md).

```powershell
python run_impostor_ai.py
```

Run from the repository root on `codex/impostor-environment-v2`. Install the normal
requirements if needed: `python -m pip install -r requirements.txt`. The script loads
the explicit release checkpoint documented in the
[release manifest](../../../artifacts/impostor-ai/release/release.json); it does not
train or substitute a scripted impostor. The game contains
one learned impostor, frozen full-29 and three scripted crew, with report/witness
safeguards. Your friend's newer crewmate is not silently swapped in.

**Result:** 0/20 fresh confirmation wins; 0%
win rate (Wilson 95% interval 0.0–16.1%), 0.20
mean kills/game, 0 own ejections. **The model has not demonstrated a useful winning strategy on this fresh confirmation set.**

The release follows the immutable **V2** selection. Confirmation did
not change its identity. These are validation/confirmation results, not reserved
final-test accuracy. All old models and failed runs remain preserved.

## Controls

Space/P pause; Right single-step; +/- speed; R replay the same scenario; N new
scenario; 1 spectator roles; G routes; V sight; Esc quit. The orange outline marks
the learned impostor. The right panel shows legal neural action-family probabilities,
kill cooldown, last choice and outcome. `--stochastic` samples the learned policy;
the reported evaluation uses greedy family-then-target choices.

## What the NN decides

The released V2 impostor has **75,075 parameters and 12 action families**. It was
initialized from scratch, separately from the crewmate. Inputs are 35 globals plus
8 own extras, 4×21 player features and up to 256×51 legal candidate features. Its
actor and critic use separate encoders. PPO uses lr0.0003, rollout256, minibatch64,
four epochs, clip0.2 and physical-time GAE; full settings are in the V2 report.

- Movement, pursuit of visible player positions, cancelling/continuing and fake tasks.
- Which legal nearby player to kill and when. There is no forced-kill/kill-alone rule.
- Whether to report or call a meeting. No rule forbids self-reporting.
- Structured accusations, defence of other players, own location alibis and remembered
  sightings; voting or skipping. No free-text conversation or persuasion script.
- Vents and sabotage are outside this interface. A* executes navigation; it does
  not select the impostor's target or strategy.

Inputs include visible players/bodies, own cooldown/goals, the public task bar,
legitimate memory and public meeting evidence. There are no hidden positions,
private crew beliefs, ghost secrets or privileged actual-witness labels. Living crew
can report bodies and must vote for directly witnessed killers. Avoiding witnesses,
effective alibis and sensible voting are behaviors to learn, not established skills
merely because the action vocabulary includes them.

Rewards: +10 win, -10 loss, -12 timeout, +2 successful kill, additional -2 ejection,
-0.002 per simulated second. No movement/fake/report/claim reward. Kill shaping can
change risk preference; true wins assess strength. Frozen crew/belief never train.

## Exactly which crewmates it faces

| Slot | Controller | Learns during these runs? |
|---|---|---|
| 1 impostor | Our separate PPO NN | Yes during training; frozen during evaluation |
| 1 crewmate | Frozen `full-29/checkpoint-8192.pt` + Phase 4 belief NN | No |
| 3 crewmates | Scripted `StudyController` variants | No |

The learned crew seat is deterministically randomized per match. This is **not four
learned crewmates**. We did not train or improve the friend's crewmate in this session.
Five players, 45s kill cooldown, 15s discussion/120s voting, ordinary task victory
and permanent ghost tasks remain unchanged.

The one learned crewmate has two frozen neural components:

1. Its actor-safe memory records sightings, bodies, public claims/votes and direct
   witnessed kills. The Phase 4 belief NN estimates suspicion over other players.
2. The full-memory PPO policy uses that belief, task/progress/location features and
   memory to score legal task travel, room travel, follow/flee, reports, emergencies,
   claims, votes and skips. It chooses greedily in these tests. A* executes movement;
   the adapter keeps longer goals up to 24s (follow/flee up to 3s) with interruptions.

This is a **legacy-to-V2 projection**: 40 global, 4×26 player and 96×23 candidate
features. V2-only task-stage/readiness, confirmed-role and visual-proof details are
not new neural inputs. Legal readiness is still checked before executing an intent.
Thus its historical training win rate is not a measured V2 cross-play win rate.

All four living crew also receive explicit safeguards: approach/report visible
bodies and retain direct witnessed-kill evidence to vote for the killer when legal.
Those are overrides, not newly learned skills; body discovery alone never identifies
the killer. Dead crew have no social actions or shared private memory.

The other three crew use scripts: nearest/shuffled/social task ordering, A* movement,
body reporting, and evidence/cautious/skeptical meeting styles. Voting scores direct
witness evidence, nearby-body sightings and public accusations, then applies a
threshold and skips ambiguous ties. It is **not a neural vote model or simply a
uniform random vote**, but it can make mistakes. Some uncertain scripted claims use
seeded randomness. The mandatory direct-witness override remains above those rules.

## Every completed training run, including failures

| Run | Decisions | Completed training games | Training wins | Training kills | Training wall |
|---|---:|---:|---:|---:|---:|
| First-trial AI 71 | 4,096 | 18 | 0 | 0 | 7.73 min |
| First-trial AI 83 | 4,096 | 18 | 0 | 1 | 6.67 min |
| Improved V2 71 | 8,192 | 82 | 0 | 68 | 35.94 min |
| Improved V2 83 | 8,192 | 78 | 1 | 57 | 26.35 min |

Four registered runs, **24,576 decisions total**. These are training counts, not
test accuracy. The first pilot's two final checkpoints each scored **0/20 validation
wins and zero validation kills**. Those are different scenarios from V2; do not pool
them as one model's test sample. All failed checkpoints, logs and graphs remain.

V2 checkpoint selection used the same 20 fresh scenarios for each candidate:

| Initialization | Checkpoint decisions | Selection wins | Total kills |
|---|---:|---:|---:|
| 71 | 4,096 | 0/20 | 26 |
| 71 | 8,192 | 2/20 | 12 |
| 83 | 4,096 | 1/20 | 13 |
| 83 | 8,192 | 0/20 | 14 |

Initialization71/checkpoint8192 was locked before fresh confirmation. All methods
below then played the same 20 new scenarios; 80 executions are not 80 independent
cases. None of these are the reserved final tests.

| Impostor control | Fresh wins | Mean kills | Own ejections |
|---|---:|---:|---:|
| Selected trained V2 | 0/20 | 0.20 | 0 |
| Older trained AI, initialization83 | 0/20 | 0.00 | 0 |
| Scripted attacker, no NN | 1/20 | 1.55 | 2 |
| Randomly initialized V2 NN | 0/20 | 0.00 | 0 |

## Why the wins look like this

The measured failure is **stalling**, not frequent ejection. In V2's 20 selected-model
confirmation games it chose FAKE_TASK **3,560 times**,
GO_TO_LOCATION 209 times, and had legal kill opportunities
in only 4/20 games. It made 4 kill selections.
Every selected-model confirmation game ended by crew task completion; no own ejection
occurred. This does not prove that it learned clever witness avoidance.

The engine's FAKE_TASK action restarts a four-second activity. The current wrapper
re-decides non-travel actions every .6s, so repeated fake selections restart it before
completion. This is permitted game behavior, but a poor learned use of the interface.
The sampled training policy sometimes moves away from that loop; greedy evaluation
can keep choosing its highest-scoring fake action repeatedly. Its exact optimizer
cause has not been isolated.
No positive fake-task reward exists. Sparse delayed game outcomes, too little useful
pursuit experience and imperfect credit assignment are plausible contributors,
not proven single causes. More steps are not automatically the remedy.

The one training win involved two kills and a wrongly ejected crew member. That
does not prove a learned lie caused the mistake. Likewise the whole mixed crew
team winning does not prove that full-29 alone carried the match: the three task
scripts and ghost task completion contribute. Current aggregate logs do not isolate
the learned crewmate's own task contribution.

The selected V2's 2/20 selection wins fell to 0/20 fresh confirmation wins. Small
samples and selecting the maximum among candidates can produce that optimism;
the unchanged lock and fresh scenarios expose it. The model is **not yet a strong
winning impostor**, and advanced deception/sensible voting are not established.

## What changed

1. First pilot: travel destinations overwhelmed rare useful actions. Both seeds
   won 0/20 validation games with zero validation kills; **25.37 minutes** total.
   [Preserved results/six graphs](../pilot-20261027/README.md).
2. V2: learned action-family/target hierarchy, separate critic and clipping, kill
   shaping, legitimate visible geometry and interrupted three-second travel choices.
   Two 8,192-decision runs; 80 selection + 80 fresh control/confirmation matches.
   [Full settings, eight graphs, controls, processor and timings](../strategic-v2-20261027/README.md).
3. Pursuit correction: visible-player goals get their own learned family rather
   than competing with many room/task goals. Warm-start only our own impostor;
   retain identical PPO/reward/observation/rules. The pursuit family was prepared and passed tiny development gradient/resume checks, but its registered training study did not run. Owner requested summary and push after completed training; no further training.
   [Registered conditional protocol](../pursuit-conditional-protocol.md).

## Graphs and interpretation

V2's linked report explains all eight training, PPO, kill-exploration, selection,
paired-confirmation and compute graphs. “Old pilot83” means the older trained AI;
“Hunter script” is a handwritten attacker without a NN. They are separate impostor
controllers tested against the same crew, not extra roles in one game.

![Training outcomes](../strategic-v2-20261027/figures/learning_curves.png)

Training wins/returns are not test accuracy. Falling value loss alone does not prove
good strategy. The orange spike is the single sampled training win, not robust success.

![Fresh comparison](../strategic-v2-20261027/figures/confirmation_results.png)

Whiskers show uncertainty. The trained model's four total kills did not produce a
fresh win. The scripted attacker won once;20 games is too small to establish a broad
performance ranking. Architecture, rewards, option durations and experience changed
together, so the first/improved comparison is not a pure ablation.

![Verified default viewer](viewer.png)

Actual selected checkpoint on demo seed3600000; this frame is not winning evidence.

## Time, checks and remaining limitations

- V2 study: 117.24 minutes, including its two runs and 160 evaluations.
- No pursuit study: Owner requested summary and push after completed training; no further training.
- The original three-hour deadline is unchanged. Preparation, reports and UI checks
  are outside timed learning/evaluation execution; concurrent preparation is reflected
  in wall time. Single CPU worker, one Torch thread, 32 logical CPUs available,
  no GPU; processor/runtime details are in V2's study manifest.
- 78 improved preflight tests, two controlled win/loss end-screen tests and three
  pursuit/warm-start/resume/viewer tests passed. Earlier engineering passed 239
  regression tests and 10 deterministic/control matches. Suites overlap; do not sum
  these counts as unique repository tests.
- Every released/copied checkpoint strictly loaded and retained its byte hash.
  Frozen full-29/belief/history checks remain enabled. Final confirmation has zero
  illegal actions or navigation failures. A recorded win is not proof of deception.
- No reserved final tests, merged PRs, deployment or multi-agent self-play.
- Results concern one frozen legacy-to-V2 learned crew plus three scripts. They do
  not establish strength against human opponents or four improved learned crew.

## Next stage

1. Fix the repeated FAKE_TASK interface in a new version: test a finishable,
   interruptible option, or suppress only mechanically redundant restarts while the
   same fake is active. Keep the decision to fake/leave learned; do not force hunting
   or alter the frozen canonical engine. Verify effects before another training run.
2. Test the prepared pursuit family so visible-player goals do not compete with
   many room/task destinations. It still chooses family/target neurally; it does not
   auto-chase or auto-kill. No pursuit training study ran in this delivery.
3. Register fresh scenarios. Compare sampled and greedy behavior explicitly; inspect
   actor KL/entropy, useful opportunities and task-loss rate before extending training.
   Then tune learning rate/entropy or temporal memory one factor at a time.
4. If shaping is needed, test potential-based rewards from legitimate observations
   with terminal handling, rather than hidden witness labels or rewards for fake-task
   activity. Keep real wins primary and do not make cooldown/task rules easier.

Freeze the chosen impostor. When your friend's best compatible crewmate is ready,
pin its checkpoint/hash, verify its V2 inputs and ghost isolation, and run paired
cross-play against both old/new crew. Only after useful single learned agents work,
explore adapting opponents and self-play with frozen opponent pools and held-out
tests. This release is best within its registered selection set, not globally best.
