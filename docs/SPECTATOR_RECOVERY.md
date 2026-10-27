# Recovery and real PPO integration

Original `3fd3601` (parent `a463a42`) contains all16 reported files. Its25 tests passed
independently. The original root checkout and untracked artwork stay intact. Follow-ups
use `codex/spectator-live-integration`; no spectator commits enter the crewmate branch.

## Reviewed context map

| Files | Work |
|---|---|
| Replay format / watcher | Safe completed-file parsing/publication/discovery; persistent favourites |
| Recorder | Optional real PPO labels, truthful task progress, actual positions/events |
| Playback / UI / app | Fix selection/queue and show recorded model, worker and training progress |
| New tests / notes | Partial/corrupt handling, controls and recording invariance |
| Existing maps/assets/render helpers | Read-only; local optional artwork referenced without publication |

The separate crewmate integration observes real headless PPO games and publishes completed
JSONL episodes using this schema. It samples streams before outcomes and records actual
central policy probabilities/value, never fabricated data. Spectator truth never feeds
actor packets. No frame streaming from unfinished matches is claimed.

Before full training: real PPO replay, noninterference/overhead checks, auto-discovery and
a verified visible desktop window. The owner then authorizes the fixed seed17 pair without
another approval:512choices+20validation/model, four workers, eight streams, three-hour
total stop. No seed29/final tests, impostor changes, crewmate push or merge.


## Verified recovery

31 spectator tests passed (original25 plus six recovery checks). The scripted fixture
still has269 ticks/153 events and contains no neural trace. A60-frame app smoke and a
separate real PPO-rendered frame succeeded. Real PPO on/off engineering:87.28s versus
87.61s (one pair; about0.4%, within timing noise), four completed episodes and exact
weights/packets/actions/rewards/episode-boundary equality. Automatic discovery/queueing
of an authentic completed PPO recording was verified while the app ran. The owner
confirmed the visible replay and neural panel after native capture access timed out.
The persistent viewer runs without `--capture`, which exits after saving a frame.

The crewmate regression passed178 tests. Its code and all private replay/results data
remain on a separate local branch; this spectator PR includes no simulator/training
implementation, research checkpoints or actual training recordings. No merges.
