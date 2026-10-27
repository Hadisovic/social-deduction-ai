# Approved V4 study —27 October2026

Owner approved the proposed study with “go on” after the3–4h estimate. Hard ceiling:
**14,400 wall seconds (4h)** including learning, selection, diagnostics and confirmation.
No automatic extension, no reserved finals, no merge/deployment or new lobby sizes.

## Fixed experiment

- Same75,075-parameter,12-family actor/critic; same PPO settings as V2.
- V4 rewards/options/observations from the versioned correction, unchanged.
- **From scratch**, independent initializations71 and83. No warm start from old
  feature semantics or crew weights. Preserve all old policies/results.
- Each initialization:8192 sampled neural decisions, rollout256, four epochs,
  minibatch64, lr0.0003, ratio clip0.2, value coefficient0.5, separate actor/critic
  clipping0.5/1.0, family entropy0.02→0.005, conditional entropy0.001, gamma1,
  physical-time GAE lambda0.995/s. No scripted strategic action override.
- Frozen full-29 + frozen belief +3crew scripts; ordinary task victory, living-crew
  report/direct-witness safeguards, ghosts and canonical45s cooldown remain.
- One CPU worker, one Torch thread. Snapshot/checkpoint every256 updates; retain
  policy-only snapshots at0,2048,4096,8192. No source edits while healthy workers run.

## Fresh partitions and locked selection

| Stage | Scenarios | Executions |
|---|---|---:|
| Training, each initialization | 3110000 onward; maximum3199999 | Until8192 decisions |
| Greedy checkpoint selection | 3200100–3200119 | 20 per candidate ×4 =80 |
| Sampled diagnostic, after lock | First5 selection scenarios,3200100–3200104 | 5, locked model only |
| Fresh greedy confirmation | 3200120–3200139 | 20 locked V4 +20 unchanged V2 =40 |

Total evaluation executions125, not125 independent scenarios. Training scenarios
are shared across initializations; policies use separate private RNG streams.
The sampled diagnostic reuses selection cases, cannot change the locked identity
and is not fresh confirmation. Its small sample is descriptive only.

Four candidates are71/4096,71/8192,83/4096,83/8192. Rank by true wins, then kills,
then fewer timeouts, then earlier checkpoint, then lower initialization. All80
selection executions must finish before writing an immutable selection-lock.json.
Lock the method/path/hash before sampled diagnostics or fresh confirmation.
Never tune or change the choice using confirmation results.

The unchanged V2 control uses its original environment/actions/rewards and exact
released weights. Compare true outcomes/behavior; shaped returns differ in units
and objective and must not be treated as directly comparable scores. No script or
untrained control is added to this approved small study. Previous controls stay
documented; stronger claims require additional separately registered comparisons.

## Measurements and stop conditions

Record every episode/update and evaluation case, all seeds/source/config/frozen
hashes, checkpoint parent (none), optimizer/private RNG states, action journal,
actor/opponent/reward replay checks, wall time and simulated experience.

Track wins, kills/legal opportunities, witnesses, task/ejection/timeout outcomes,
fake starts/completions/interruptions, longest consecutive fake selections,
fake selections with no visible player, actual fake-active/ready-faking seconds,
ready seconds, pursuit positive credit/terminal correction, all reward components,
claims/votes and entropy/KL/clip/value diagnostics. Policy telemetry never enters
observations or chooses actions. The same telemetry wrapper leaves V2 control
physics/observations unchanged.

Before launch, pass a tiny real gradient/exact-resume check and verify frozen
opponents. Commit a clean source baseline. Production study refuses a dirty start
and source/checkpoint changes. It uses an exclusive worker lock and explicit
resume checkpoints; no automatic scientific retry or extra training.

Stop/report on deadline, nonfinite optimization, illegal action/navigation failure,
hash/replay mismatch or exception. Preserve partial evidence and do not call an
unfinished study complete. On completion generate graphs/report, assess the fake
loop and genuine wins, and provide an explicit checkpoint command; do not overwrite
the historical release or silently declare the model best. One or two small
confirmation sets cannot establish broad reliability.

## Future roster scope

Future configurable lobbies target up to15 total players and1–3 impostors, with
legal combinations checked against the targeted real-game version.12players with
10crew/2impostors is one preset. Default design shares one policy per role and the
existing crew belief model, with separate actor-private memories, not15 separately
trained models. No such lobby/interface change is part of this study.
