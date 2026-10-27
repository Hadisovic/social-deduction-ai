# Live spectator / approved pilot handoff

Owner attachment `d42ec8ae-3a1a-49b5-b136-bc91c368320e` explicitly authorizes the seed17
pair after real replay integration and a verified running desktop GUI. Do not ask again
once prerequisites pass. Stop before training if the native GUI cannot be verified.

Crew workspace: `_v2_pretraining_diagnostic`, branch `codex/crewmate-v2-live-pilot`,
base `c18ff5f`, preserving cleanup `27ff525` and performance work. Do not push this branch.
Spectator workspace: `_spectator_live`, branch `codex/spectator-live-integration`, based
on original `3fd3601` (parent main `a463a42`). Original root/branch untouched. Publish
spectator only and create draft PR; never merge. Impostor branch remains untouched.

Original spectator25 tests passed. Follow-up parser/atomic writer/selection/progress
fixes are being verified. Existing pinned artwork is referenced by a local ignored
junction, not copied or published. Windows desktop API initializes and reports unlocked;
no actual spectator window has yet been verified. Training has NOT started.

## Context / implementation direction

Keep canonical V2, old learning and collection modules byte-identical. Add `crewmate_v2/live/`
as a separate observer/driver, reusing central collection, policy, GAE and PPO kernels.
Use actual spectator recorder from its separate source path. Completed episode publication
only; no invented frames/probabilities, no privileged spectator fields entering actor packets.
Register forward hook for real probabilities/value and route them to selected stream recorders.
Workers remain headless. Select streams0â€“3 before outcomes; engineering smoke may record all.

Retain four workers/eight streams and16-step trajectory slices; validated128-choice updates,
four updates/model, seed17 independent A/B initialization,512choices/model,20matched validation
games/model over four scenarios. Original/shaped coefficients unchanged, five players only.
Train+validate total hard stop10800s. No seed29 or final partitions; acceptance pending.

Next: finish spectator tests; implement/export real PPO smoke, compare recording on/off
outcomes/packets/weights and overhead; launch/inspect native GUI on a real PPO replay and
watch pilot output directory; freeze source and record prerequisites; run authorized pilot;
monitor viewer and preserve partial results; report actual outcomes and next experiment.


## Preflight complete

Spectator recovery commit6dd8a1b is published with draft PR#11 (unmerged). Original
feature branch3fd3601 is also published.31 spectator tests and178 crewmate regression
checks passed. Paired72-choice engineering runs recorded four real PPO episodes and
preserved exact actions/rewards/episode boundaries/final packets/updated weights; off/on
87.28/87.61s, ratio1.00387 (one pair, timing noise applies). All8 streams were recorded
for this conservative smoke; the pilot samples streams0–3. Source/asset/checkpoint
preservation preflight passed, including the nine historical selected models.

Persistent GUI session18955 is running in the spectator worktree. Native window462292
was discovered; its actual app-loop heartbeat is fresh and the owner explicitly confirmed
seeing the Skeld replay and neural panel. Native capture access timed out; this was resolved
by owner verification, not by bypassing permissions. A real rendered frame is saved locally.
Discovery of an authentic completed replay was verified (library3→4); source bytes/timestamps
were restored. User inspection switched the viewer to manual mode; M selects auto-next.

Authorization/prerequisite record is private `local-runs/live-pilot-authorization.json`.
Training has not started at this checkpoint. Next command from the crewmate worktree:

```powershell
python -m crewmate_v2.live.train --output local-runs/seed17-live-pilot-20261027 --spectator-source ../_spectator_live --replay-dir local-runs/live-watch/pilot --status-file local-runs/live-watch/training-status.json --approval local-runs/live-pilot-authorization.json --proof local-runs/live-watch/engineering-smoke/proof.json --gui-state ../_spectator_live/local-runs/pilot-gui-state.json --old-smoke temp_v2_learning_smoke_verified/smoke.json
```

Use the one fresh output, preserve all partial results, never silently rerun after a stop.
The owner has authorized this command after the now-complete prerequisites; do not ask again.
The total10800s hard stop begins inside this command, including both training and validation.
Do not push the crewmate branch or private recordings/results. Keep source immutable during
execution; document updates/analysis outside the locked live Python subpackage.
