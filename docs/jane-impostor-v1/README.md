# Lizbon V1: learned impostor

Independent learned impostors on the reviewed Jane A2 game. Branch:
`codex/jane-impostor-v1`, based on shared foundation3766cba. The shared foundation
contains settings/contracts; each developer keeps their own model branch.

The owner named this impostor model **Lizbon**. Jane remains the shared game
foundation; Patrick Jane is the frozen crewmate reference.

**Latest result:** the first pilot stopped on a reproduced native door invariant
after one complete game and eight PPO passes. No development comparison ran.
See the [results, graphs and next steps](pilot-20261030/README.md). The recovered
weights are research artifacts, not a validated playing-strength release.

## What is playing

| Side | Controller | Status |
|---|---|---|
| 10 crewmates | Frozen Patrick Jane task-pursuit-v1, update102 | Published learned weights; native engineered task pursuit/commitment retained |
| 2 impostors | One new recurrent policy, separate private memory and RNG per player | Fresh initialization; trained only by the new pilot |

The crew checkpoint was published for task-pursuit training, **not trained on A2**.
It is a fixed cross-environment opponent, not a proven strong A2 crew. No crew or
old impostor weights were transferred into the new impostor. No scripted hunting,
automatic kills, report ban or forced teammate defence controls its actions.

## Network and decisions

| Component | Shape / operation |
|---|---|
| Own observation | 192 values; position, phase, own timers/route, task bar, host settings, last action |
| Other players | 14 slots x24; visible or remembered positions, age, public/direct evidence, known allies |
| Event history | Last64 private/direct/public/claim events x16 |
| Candidate actions | Dynamic number x48; type, target, position, timers, claim and ally information |
| Encoders | Own192->256->128; player24->128->128; event16->128->128; Tanh |
| Attention | 128-wide, four heads for players/events/candidates |
| Recurrent memory | Fused384->256, GRU256; private state per impostor |
| Actor / critic | Shared recurrent backbone, separate256->256->128 trunks |
| Outputs | 16 action families plus conditional candidate score; one value estimate |
| Size | 1,036,561 trainable parameters |

Families: continue, cancel, travel, fake task, kill, report, emergency, claim,
vote, skip, lights, repair, doors, enter vent, move within vents, exit vent.
Travel can target a public console/room, visible player, remembered sighting or
visible corpse. Claims support accusations, defence, location claims, alleged
kills and sharing remembered evidence. Truth does not filter strategic claims.

The actor receives no unseen live positions, teammate memory, hidden ballots,
opponent hidden state or training reward. The critic uses the same local packet.
Ground truth is used by the trusted reward ledger and omniscient spectator only.

## Rules and learning

Same A2 profile as the crew: 12 players (10+2), 1 common/1 long/2 short tasks,
45-second kill cooldown, 10-second initial cooldown, lights/doors/vents, reports,
emergencies, structured discussion and public ballots. Native legal rules and
collision/navigation handle execution; the network chooses strategic actions.
This is a versioned simulation approximation, not an exact complete real client.

Arrived static goals and the same active route are omitted. Genuine waiting and
later revisits remain available. Fake tasks finish in four simulated seconds;
continue preserves them, cancel or another activity abandons them. Simultaneous
requests for a consumed team ability become logged no-effects, with the original
sample/log-probability retained. No alternate strategy is inserted.

| Reward | Amount per impostor trajectory |
|---|---|
| Team win / loss | +10 / -10 |
| Each accepted team kill | +1 |
| Eligible cooldown-ready time | -0.005 per simulated second |
| Movement, faking, claims, reports, sabotage presses | 0 |
| Extra individual ejection penalty / pursuit shaping | 0 |

Meetings, elimination, vent/transition pauses and post-terminal intervals are
excluded from ready-time costs. Technical cuts are preserved and excluded from
complete-match PPO; they are not losses. This design is an initial hypothesis,
not a claim that camping or all reward exploits have been solved.

PPO: Adam3e-4, clip0.2, four epochs, recurrent chunks128, per-second lambda0.995,
gamma1, entropy0.01, value coefficient0.5 on half-MSE, gradient norm limit0.5.
Weights stay fixed through a whole game. Every recurrent pass reconstructs
history from match start; no optimizer step occurs between its chunks.

## Approved pilot and evidence

See [STUDY_PLAN](../../jane/impostor/STUDY_PLAN.md) and
[ENGINEERING](ENGINEERING.md). Owner approved90 minutes: up to60 for four training
matches, then up to30 for paired development and figures. Initial seed731 is one
initialization. The last completed update is locked before comparing two fresh
scenarios with the initial network. Reserved final partitions stay sealed.

Training wins are training outcomes. Development outcomes measure this fixed
small comparison. Neither is test accuracy or evidence of a globally best model.

## Watch the game

The door renderer now uses original Skeld sprites. See [sources, animation preview and tests](DOOR_VIEWER.md).

The native spectator is observation-only. Select an impostor in the roster for
its panel. Omniscient positions in the viewer are never neural inputs.

Live observer, from this repository in PowerShell:

```powershell
.\local-runs\jane-impostor-v1\runtime\Scripts\python.exe .\run_lizbon.py --watch .\local-runs\jane-impostor-v1\pilot-20261030-seed731\train
```

Completed initial-policy baseline playback (before the first PPO update):

```powershell
.\local-runs\jane-impostor-v1\runtime\Scripts\python.exe .\run_lizbon.py --replay .\local-runs\jane-impostor-v1\pilot-20261030-seed731\train\match-001.jsonl
```

This machine's ignored source.json points to the verified external upstream
snapshot. On another machine, pass `--source-root` with an independent checkout
of commitd459f38674ebc537acde20647cbd07700c494533. Install the dedicated requirements
in a new environment; do not replace historical runtimes. The adapter verifies
source and published weights before running. Runtime/source/schema/settings
mismatches reject checkpoints instead of silently adapting old models.

The failed post-update recording has no match outcome. The completed playback
above does not demonstrate the recovered updated checkpoint. Training is stopped;
the --watch command is useful during a future authorized run, not a live current job.

## Work split and deferred features

You/impostor developer: inspect search, faking, witnesses, reports, votes and
repeated goals; diagnose failures before approving larger replicated training.

Crew developer: improve A2 reporting, evidence, ballots and sabotage counterplay.
Publish a frozen source/settings/checkpoint manifest for the next cross-play.
Do not silently replace our active opponent release.

Both: propose shared rules/schema changes together; keep separate model branches.
Cross-play frozen releases before adapting opponents or self-play.

Later: oxygen/reactor/comms, Admin/Security devices, exact minigames/Clean Vent,
ghost-impostor sabotage, optional roles, 15-player/three-impostor training,
larger architecture and additional shaping. Native capacity does not prove learned
generalization. Current native Population rejects13/14-crew lobbies; do not
advertise unrestricted real-game lobby settings yet.
