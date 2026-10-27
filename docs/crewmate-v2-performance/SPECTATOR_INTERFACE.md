# Optional spectator integration contract

This preparation does not edit, merge, import or launch the independent spectator app.
Its inspected renderer imports `among_us_renderer.MapRenderer`, `phase3_assets.character_image`
and the Phase 3 sprite/color helpers. Those modules and all map/art assets stay byte-identical.
Canonical V2 match, observation, task, movement and replay APIs also stay unchanged.

## Available without rendering in training

The optional collection runner writes these files inside a fresh local run directory:

| File | Fields / timing |
|---|---|
| `registration.json` | Full immutable runtime and execution proposal, approved budget and forecast |
| `<arm>-17/config.json` | Reward arm, source locks, initial weights hash, optional resume checkpoint name |
| `<arm>-17/updates.jsonl` | Decisions, accumulated simulated seconds, wall seconds, PPO loss/KL/clip/entropy/gradient diagnostics; one line per update |
| `<arm>-17/episodes.jsonl` | Slot, scenario seed, rotation, outcome, living/ghost tasks, reward counts/components, focal opportunity denominators; completed episodes only |
| `<arm>-17/checkpoint-128.pt` etc. | Existing new V2 policy schema plus additional strict collection source lock and private continuation state |
| `<arm>-17/validation.jsonl` | Existing development-only evaluation result schema, one finished game at a time |
| `failure.json` | Stop reason and number of fully completed arms; earlier checkpoint/evidence files retained |

Readers can tail JSONL and ignore an incomplete final line. Wall seconds are durations,
not simulation time. An update contains 128 actionable choices across eight logical games;
physics ticks and forced waits are recorded separately. Reports/votes remain policy actions.
Episode/update lines are flushed after a collected rollout and optimizer update. An interrupted
uncommitted rollout can contain completed games that were not yet written; resume uses the
last complete checkpoint, never treats a cutoff as a terminal loss, and must not silently
extend an approved wall budget. A spectator must label partial evidence accordingly.

CPU/memory measurements are supplied by the engineering benchmark, using native Windows
process-tree counters. Live resource monitoring can be an external reader. No renderer,
mandatory tick recorder, IPC subscription server or spectator dependency is added to PPO.

## Checkpoint and replay boundary

`V2Policy.load` can load these fresh V2 policy tensors while enforcing the unchanged
learning-runtime lock. The additional collection lock must also match for training resume.
Older smoke checkpoints retain their original source compatibility because collection
code is in a separate subpackage. Historical Phase 6 models remain a different schema.

Continuation metadata is private engineering state: each stream records its episode index,
current episode action-index history, packet hash, reward accounting and RNG state. Seed
and rotation derive from the registered partition/slot/episode formula. Resume reconstructs
the real game from its seed and replays the prefix to restore observations, controllers,
cooldowns, memory, tasks and ledger; mismatched packets/accounting fail. This can be slower
than loading tensors alone. Complete historical trajectories are not stored by this format.

For a visual demo, a future spectator adapter can run a **separate fresh development game**
with the saved frozen policy, preserving the V2 input shapes and legitimate actor packet.
It may attach its existing `MatchRecorder` to that separate game if supported by its adapter.
It must not claim this regenerated game is an exact training replay. Exact current-prefix
reconstruction requires matching sources and the recorded continuation data. No viewer
compatibility claim is made for unreviewed spectator or impostor changes.

If a full privileged per-tick replay is added later, keep it local and separate from actor
observations. Spectator truth/role overlays must never feed policy inference, opponent
controllers or rewards. Publishing raw traces or post-game role datasets still requires
separate owner authorization. Training never waits for a renderer or UI consumer.
