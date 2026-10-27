# Impostor V4 — reward/action correction

This report records implementation/development checks before research training.
The selected V2 checkpoint and all previous results remain unchanged. V4 keeps the
same 75,075-parameter network, 12 families, tensor dimensions and canonical lobby.

**Later owner approval:** “go on” authorizes the bounded4h study in
[TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md). The implementation evidence below
predates research training; see the master checklist/handoff for live execution.

## Old versus new rewards

| Event / interval | Historical V2 | New V4 |
|---|---:|---:|
| Impostor win | +10 | +10 |
| Crewmate win | -10 | -10 |
| Timeout | -12 | -12 |
| Successful kill | +2 | **+3** |
| Extra own-ejection penalty | -2 | **-6** |
| Unconditional time cost | -0.002/s | **0** |
| Roaming with own kill cooldown ready | No additional cost | **-0.02 per simulated second** |
| Meetings, inactive actor, after termination | Historical unconditional cost | **No cooldown-ready cost** |
| Pursuit progress | 0 | Bounded potential difference; at most **+0.05 total positive credit per match**, terminal correction |

The ejection terminal reward is -10 loss plus -6 extra = **-16**, before previous
time costs/potential correction. No ejection penalty is applied twice. A successful
kill resets the engine's original45s cooldown; its interval is not charged as idle
ready time. All penalties use actual engine elapsed time, not choice counts, GUI
speed or wall time. Crossing from0.1s cooldown through a0.2s roaming tick incurs
0.1 × -0.02 = -0.002. Crossing exactly at the end of the tick costs zero for that tick.

The new ready cost applies even when no crew is visible. Reporting enters a
meeting before physics and excludes that interval. A meeting's resolution interval
also stays excluded; subsequent roaming physics intervals count normally. An
immediate terminal kill that advances zero engine ticks consumes zero seconds.
The terminal physics interval is charged only for elapsed eligible roaming time;
nothing is charged after termination.

**Tradeoff:** 500 ready roaming seconds cost10 reward units. A slow win can now have
a negative shaped return; the previous report's claim that every win has positive
return does not apply to V4. This may encourage risky kills or excessive meetings.
Keep true wins, safe kills, meeting usage and each reward component separate in
future selection. The kill/time/ejection changes do not preserve the win-only
objective mathematically. Their effectiveness needs fresh experiments.

## Finishable fake task

- First FAKE_TASK starts the original4s engine activity.
- Repeating the same console's active FAKE_TASK submits continue, preserving the
  remaining time. The candidate remains available to the NN.
- The policy still chooses every0.6s: continue/repeat, cancel, move, kill or report.
- Cancel/travel/meeting interrupts via the existing engine. No automatic resume.
- Once the activity finishes, another explicit fake selection can start a new one.
  Repeated completed fakes are allowed; no heuristic forces pursuit or departure.

This fixes an unfinishable restart, not a guarantee that old weights stop faking.
No reward is paid for fake starts or completions. Opponent schedules, walls,
navigation, task victory, ghost tasks, map and cooldown code are untouched.

## Pursuit shaping and exploit resistance

The shaper receives the actor's own immutable observation, not the engine. It sees
only currently visible players and its own previous observations. It never selects
a target or changes an action.

Proximity is `clip(1 - (distance - kill_range)/(sight_range - kill_range), 0, 1)`.
A visible ID must persist across observations. Own movement toward its previously
visible position and net distance reduction must each reach0.05 map units. Paying
progress cannot exceed the actor's own contribution, so a player walking toward a
stationary impostor does not pay. New/reacquired targets establish a non-paying
global proximity high-water baseline. Oscillating and switching targets cannot
repay already covered proximity; meetings and kills never replenish the budget.

Let credited progress potential be `Phi ∈ [0, 0.05]`, initially0. Each step adds
`F = Phi(next) - Phi(current)` to reward, with discount gamma1. Terminal Phi is0:

`sum(F over the complete match) = Phi(terminal) - Phi(initial) = 0`.

Progress supplies a small early learning signal, not extra terminal reward. Its
terminal correction prevents profiting by endlessly avoiding an outcome. A flat
+0.05 every approach would be farmable and decision-frequency dependent, so0.05
is the scale of one full lifetime proximity budget. This conservative mechanism
can produce little or no shaping once any observed target established a close
baseline. That limitation is deliberate and must be measured before tuning it.

## Versioned observations without resizing the network

| Existing tensor slot, zero-based | V2 meaning | V4 meaning |
|---|---|---|
| `global_[3]` | Boolean own interaction | Remaining own fake-task time /4; zero if not faking |
| `extra[2]` | Own recorded kill count /3 | Lifetime observed proximity high-water mark |
| `extra[3]` | Age of last own kill | Current pursuit credit / configured0.05 scale |

All other slots retain their meanings. The own timer uses only the focal actor's
own activity; other players' task timers are never read. This is a new observation
contract, not a claim of unchanged feature semantics. Historical V2 environments
and strict loaders remain available. New checkpoint schema is
`impostor-options-v4`; reward/action/observation schemas are separately recorded.

The exact current V2 table is [NN_ARCHITECTURE.md](../NN_ARCHITECTURE.md). The new
namespace is `impostor_options_v4/`; no guarded old runtime file was edited. Training
adapter uses the original family-policy PPO and optional strict own-impostor V2
weight initialization. It never initializes from crew weights. Production session
construction requires explicit approval; no new study runner is launched here.
Resume reconstructs fake activity, reward ledger and shaping state through the
recorded action journal and checks packet, actor, opponent and summary hashes.

## Watch

```powershell
python run_impostor_ai.py
```

This remains the historical evaluated V2 model under V2 semantics.

```powershell
python run_impostor_options.py
```

This shows the correction using the old V2 model with a prominent **TRANSFER DEMO /
NOT TRAINED IN V4** label. It is a real neural controller, not a script. A future
V4 checkpoint can be passed with `--checkpoint`. Space/P pause, Right steps,
+/- changes speed, R replays, N changes scenario,1 roles, G routes, V sight, Esc quit.
No training occurs in either viewer. Reserved final-test seeds are rejected.

Lobby: **1 learned impostor + 1 frozen full-29 neural crewmate + 3 scripted crew**.
Ten crew/two impostors is unsupported: it needs variable roster features, two
impostor observations/coordination, voting changes and new tests/training. It is
separate from this reward/action task. Friend's current crew model stays separate.

## Verification and next approval

The new regression suite passed32 tests in56.59s. It covers ready/cooldown crossing,
meeting pause/resume, terminal accounting/ejection, repeat/continue/cancel/move/
meeting fake behavior, deterministic replay, bounded pursuit and hidden-position
isolation. It loads all eight historical V2 snapshots and tests the actual release
in the new headless viewer. Tiny collection/resume checks perform **zero optimizer
steps**. These tests verify implementation, not model skill.

The original impostor-core suite also passed27 tests in14.99s:59 tests across
these two distinct files. All8 V1 and8 V2 policy snapshots strict-loaded through
their original loaders; frozen models remain unchanged. See [test-results.json](test-results.json).

The bounded development smoke report is [smoke-results.json](smoke-results.json).
It uses existing V2 weights under V4, fixed development scenarios, a hard wall
ceiling, and no selection or optimizer. See its actual completion flags; incomplete
matches are not counted as wins/losses. Do not combine these outcomes with past
validation or call them accuracy.

| Development seed | Completed outcome | Decisions | Simulated seconds | Fake starts / completions | Kills | Ready seconds | Return |
|---|---|---:|---:|---:|---:|---:|---:|
| 3500000 | Crew tasks completed | 227 | 136.2 | 32 /31 | 0 | 91.2 | -11.824 |
| 3500001 | Crew tasks completed | 221 | 138.0 | 31 /30 | 0 | 93.0 | -11.860 |

Both matches had zero illegal actions/navigation failures and zero pursuit payment.
Their final active fake was interrupted by match termination. Total wall time was
123.60s, with the live viewer also in use. Existing NN weights and all checkpoint
bytes were unchanged. This confirms completion/accounting, **not improved strategy**.

Before any full PPO study, approve its protocol and budget. Suggested first study:
two initializations ×8192 decisions, four checkpoint candidates ×20 fresh paired
validation scenarios, selection lock, then20 fresh paired confirmation games for
the locked new model and unchanged V2 control. Compare greedy and sampled action
behavior without tuning on confirmation. Keep reserved final partitions closed.

The previous study took117.24min; these two smoke matches took about62s each with
the live viewer running. Budget approximately **3–4 hours on this CPU** for that
bounded study, with a proposed fixed4h ceiling and no automatic extension. This is
an estimate, not an approved allocation; fake persistence changes experience per
decision and two smoke matches give a noisy runtime estimate. Record
ready seconds, fake starts/completions/interruptions, pursuit positive credit and
terminal correction, kills/opportunities, votes/ejections, component returns,
entropy/KL and true wins. A reward-only/action-only ablation and more independent
initializations would require a separately approved larger budget.

Only after useful single agents work: pin the friend's compatible frozen best
crewmate, paired cross-play, then adapting opponents/self-play. No merge/deployment.
