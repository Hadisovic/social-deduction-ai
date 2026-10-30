# Lizbon: Jane impostor implementation and first training plan

30 October 2026. Own branch: `codex/jane-impostor-v1`, from reviewed shared
foundation3766cba. Crew developer keeps their independent Patrick Jane branch.

## Same game, independent models

Use the pinned A2 engine at d459f386 and the shared native12p settings. Ten
crewmates use the published frozen update102 policy. Two impostors share one
new randomly initialized recurrent policy, with independent histories, GRU
states and RNG. Patrick Jane is a design reference, not copied neural code or
transferred crew weights. The old75K impostor and all earlier results stay intact.

Keep 1 common/1 long/2 short tasks, lights, doors, vents, meetings, body reports,
structured claims/votes, role-specific vision,45s kill cooldown and10s initial
cooldown. Native crew task pursuit/commitment stays unchanged. A2's exclusion of
self-voting and its task/minigame/animation approximations remain explicit.

The new impostor chooses goals, kill targets/timing, fake/continue/abandon,
report/emergency, claims/defence/silence, vote/skip, lights, doors and vent actions.
No scripted hunting, forced kills, report ban or automatic teammate defence.
Arrival/continuation masks suppress redundant completed commands, not genuine
waiting. The engine still handles legal physical movement and interactions.

## First implementation gates

- [x] Native source and frozen crew weights verified in isolated Torch2.7.1 CPU runtime.
- [x] New actor-local schema and recurrent network; no hidden teammate state.
- [x] Persistent fake options, arrival masks, social actions and team ability contention.
- [x] Finite gradients, recurrent PPO accounting and strict new checkpoint loading.
- [x] Bounded smoke/replay and frozen crew hash checks.
- [x] Collection measured: 30 simulated seconds took 49.93 wall seconds.
- [x] Owner approved the proposed 90-minute pilot budget on 30 October.
- [ ] Freeze implementation commit; start registered pilot and live viewer.
- [ ] Complete up to four training matches within the 60-minute training allocation.
- [ ] Lock last complete optimizer boundary before paired development checks.
- [ ] Preserve failures/cuts; inspect graphs and gameplay; document sufficiency.
- [ ] Push independent impostor branch without merging or deploying.

## Initial reward proposal

Team win/loss +10/-10; each accepted kill +1 to each team's actor return from one
event ledger; -0.005 per actual simulated eligible second with cooldown ready.
Exclude meetings, elimination, underground/transition pauses and post-terminal
time. No reward for moving, faking, reporting, claims or sabotage presses. No
extra individual ejection penalty: two-impostor sacrifice has a team consequence.
No pursuit shaping in the initial study. All decisions remain neural.

Technical/wall cuts are not losses and are preserved but excluded from initial
complete-match PPO updates. Log this collection limitation. Do not open old or
new reserved final sets. Training wins are not accuracy or generalization.

## Approved bounded pilot

One fresh initialization, seed731. Up to four complete training matches against
the fixed update102 crew. Maximum total study time90 minutes; training allocation
60 minutes. Matches have a20-minute technical wall cap. The learner checks its
deadline between recurrent chunks and discards any unstepped gradients at cutoff.
Only completed optimizer passes survive. No automatic extension or restart.

Lock the last completed PPO boundary by checksum before development. Compare it
with the saved initial policy on the same two fresh development seeds9320001/2.
Both use stochastic inference and private per-actor streams. Allocate up to27
minutes to development plus graph/report time within the30-minute remainder.
Cuts remain missing outcomes, never losses. One initialization and two scenarios
cannot establish a strong general model; no final-test partition is opened.

## Learning and evaluation

Begin with bounded engineering segments and disposable optimizer checks. Then
use a measured, approved pilot against one fixed crew release. Save the initial
reference and complete-match checkpoints; record all failed/cut matches. Keep
policy versions fixed within each match. Use fresh development scenarios for
paired initial-versus-trained comparisons; lock before any later final study.

Graphs: outcomes/reasons, returns, kills, action distributions, idle/fake behavior,
PPO losses/entropy and timing. Keep gameplay and live monitor observational.
When runnable, launch the actual recorded gameplay and training view; label
playback and untrained demonstrations correctly.

## Deferred until the first version works

| Feature | Why later |
|---|---|
| Oxygen, reactor, communications | Not registered in current A2; crew counterplay must be verified first |
| Admin/Security observation abilities | Need legitimate device contracts and both-role tests |
| Clean Vent and exact minigames | Need occupancy/eviction and calibrated task semantics |
| Ghost impostor sabotage | Requires new lifecycle and capabilities; absent in current A2 |
| Scientist/Engineer/Shapeshifter/Phantom/etc. | Changes evidence, masks and learning problem |
| 15-player/three-impostor study | Capacity exists12+3; trained generalization is unproven |
| 13/14-crew lobbies | Current native Population rejects them |
| Adaptive opponents/self-play | First establish one competent frozen-opponent learned impostor |
| Larger architecture/extra shaping | Add only when diagnostics identify a need |

## Your next steps and your friend's

You: inspect the impostor's gameplay for repeated goals, pointless faking,
witnesses, self-reports and meeting choices; approve the measured pilot budget.
Keep observations distinct from conclusions about strategy.

Friend: continue crew A2 work, especially reports, witnessed evidence and ballots;
publish a frozen source/settings/checkpoint manifest for cross-play. Do not change
the opponent release during our registered study. Share proposed rule/event/action
changes before either side silently modifies the common contract.

Both: retain separate development branches; share reviewed rules/settings and
adapters through the foundation. Exchange frozen releases first; jointly study
adaptation/self-play only after both learned base roles have held-out evidence.
