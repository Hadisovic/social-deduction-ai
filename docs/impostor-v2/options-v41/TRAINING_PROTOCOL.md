# Corrected V4.1 study — original deadline, no extension

On27 October2026 the owner explicitly chose **“Fix and restart within original
deadline (Recommended)”** after the crowded-spawn shaping flaw was reported.
This is a new, separately recorded corrected experiment, not a continuation or
unreported retry. The original V4 study/checkpoints remain intact.

## Budget and interrupted evidence

Original start:2026-10-27 21:03:50 +03. **Absolute deadline:2026-10-28 01:03:50 +03.**
The4h envelope includes the interrupted run, repair/preflight and this replacement.
The runner reads and verifies the original study's exact deadline and SHA; it
records actual remaining time at replacement start. No fresh4h allocation.

Interrupted V4: initialization71 completed8192 decisions/75games/68kills/0wins;
83's last durable snapshot is1792 decisions/15games/16kills/0wins. Pursuit paid zero
in all90 durable games. No selection/confirmation case had opened. Partial rollout
in flight at interruption is not counted as a completed update/game. Preserve its
raw study.json, interruption.json, snapshots, episodes and updates for the paper.

## Narrow repair and fixed learning

Keep the75,075-parameter12-family network, PPO, fake-task fix, reward coefficients,
45s cooldown, frozen full-29/belief +3crew scripts, ordinary tasks, ghost isolation
and reporting/witness safeguards. The only repaired behavior is pursuit-frontier
arming: initial close players cannot saturate it during cooldown. While cooldown
is positive, clear only geometry history; preserve credited potential/budget. Once
ready, the first visible observation establishes a non-paying baseline. Subsequent
meaningful own progress can pay. A kill's cooldown starts another approach period
without refilling the0.05 lifetime budget. Meetings do not reset its frontier.

The same actor-observable approach tests, no-switch/oscillation farming, own-motion
requirement and terminal potential correction remain. No forced hunt/kill/target.
Expose the same existing high-water/credit slots; no new tensor/parameter sizes.
V4.1 uses a distinct checkpoint/shaping schema; all V4 source files remain byte
unchanged so interrupted models can still be strict-loaded and reproduced.

Two fresh scratch initializations71/83 ×8192 neural decisions; no reuse of the
interrupted weights. New private RNG streams. PPO: rollout256,4epochs,batch64,
lr0.0003,clip0.2,value coefficient0.5,actor/critic gradient clips0.5/1,
family entropy0.02→0.005,conditional entropy0.001,gamma1,GAE lambda0.995/s.
Two parallel CPU training workers, one Torch thread each; two isolated CPU method
workers for selection/confirmation, sequential sampled diagnostics. This changes
wall scheduling, not runs/decisions/architecture,
and helps preserve the original deadline. Save resume after each256-decision rollout;
policy-only snapshots0/2048/4096/8192. Freeze a clean source before launch.

## Fresh cases, selection and confirmation

| Stage | Scenarios | Executions |
|---|---|---:|
| Training, each initialization | 3111000 onward, maximum3199999 |8192 decisions |
| Greedy selection |3200140–3200159 |20 per candidate ×4 =80 |
| Sampled diagnostic after lock |3200140–3200144 |5, locked model only |
| Fresh greedy confirmation |3200160–3200179 |20 locked V4.1 +20 unchanged V2 =40 |

Four candidates:71/4096,71/8192,83/4096,83/8192. Rank by wins, kills, fewer timeouts,
earlier step, lower initialization. Complete all80 selection executions, then write
the immutable path/hash selection lock before diagnostics or confirmation. Never
change the selected identity or tune using confirmation. The5 sampled cases reuse
selection and are descriptive only. Total125 executions, not125 independent cases.

The control is the unchanged original V2 release/environment. Compare true wins
and behavior, not shaped returns across versions. No extra models, seed extensions,
reserved finals, new lobby sizes, merge or deployment. Stop/report on the original
deadline, nonfinite optimization, illegal actions/navigation failures or source/
checkpoint/replay changes. Retain partial results if time expires; no automatic
restart/extension. Unit gradients are development tests, not extra research runs.

Preflight must include the real clustered spawn, actual45s cooldown and a legitimate
visible approach after leaving spawn; demonstrate **nonzero** bounded credit, no
automatic kill, and terminal correction, plus replay/old-checkpoint compatibility.

Record each update/episode/case, fake streaks and actual faking/ready seconds,
visible-player absence at fake choices, pursuit credit/correction and the fraction
of games with positive credit, kills/opportunities, outcomes, votes/claims,
KL/entropy/value diagnostics, exact configs/hashes and wall/simulated time. Report
interrupted V4 as flawed/incomplete and V4.1 as a separate experiment. Do not claim
broad reliability or globally best play from20 fresh cases.
