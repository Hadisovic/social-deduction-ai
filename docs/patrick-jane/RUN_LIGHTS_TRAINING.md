# Launch Patrick Jane lights-only training

Prepared on `codex/patrick-jane-recurrent-ppo`. **The long run has not been
started by the agent.** The owner starts it manually with this one command:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Files\RL\_patrick_jane\Start-PatrickJaneLights.ps1"
```

The command works from any current directory. Leave that PowerShell window open;
Codex does not need to remain active. It starts/reuses the existing spectator
in its own process. Rendering speed does not control simulation time or PPO.

## Registered defaults

- Ten shared-policy learning crewmates, each with independent observations,
  memories, hidden states and sampling RNG, against two scripted impostors.
- **Only lights sabotage** is available. Killing remains enabled, including
  the existing initial 10-second and subsequent 45-second kill cooldowns.
  Tasks, reports, meetings, voting, ghosts and private vision remain operational.
- **-0.005 reward per living crewmate per simulated second** while lights are
  unresolved. A minute costs 0.30; five minutes costs 1.50. Darkness during
  meetings also counts. Dead/ghost actors receive no darkness cost. Render FPS
  and wall-clock delays do not affect it. All other coefficients are unchanged.
- **32 workers**, same architecture and PPO hyperparameters/batching. No new
  worker comparison or curriculum.
- Continue the original pilot's `update-000006.pt`: 255,083 individual decisions,
  SHA256 `d8b81ef06788a0139822a5677801f34069aef116286cc079cb12de8c39538ab8`.
  The policy, Adam state and RNG are imported after verifying the historical
  archive and checkpoint. This is an explicit new training regime. The original
  model and champion remain untouched; engineering-test models are not used.
- **1,000,000 additional individual living-crewmate decisions**, giving at
  least 1,255,083 lifetime decisions. Complete-match batches can overshoot;
  the absolute guard is 1,100,000 additional decisions. No ghost neural calls.
- **Eight paired ordinary development games at entry and eight at the end**,
  using 2910000–2910007. Training continues at the next unused development
  seed 2900192. No large per-checkpoint evaluations or final-test access.
- **12 hours cumulative active run time** is the technical fail-safe ceiling,
  including resumed attempts. This is a ceiling, not a measured duration
  prediction for the changed lights-only workload. Time paused between attempts
  is excluded; an unclean exit reserves 60 seconds beyond its last progress.

The launcher parameters `AdditionalDecisions` and `RunDirectory` are configurable
for a fresh study. Registered settings/source cannot be changed during an
incomplete run. Use a different run folder for a different study; a completed
run is displayed and never silently extended. `EngineeringCheck` is for bounded
plumbing verification only and requires a separate folder.

## Where everything is saved

Default results folder:

`C:\Files\RL\_patrick_jane\local-runs\patrick-jane\lights-learning-v1`

| File/folder | Contents |
|---|---|
| `TRAINING_REPORT.md` | Readable outcome, task/navigation issues, rewards, social behavior and limits |
| `training_results.json` | Aggregate and per-checkpoint metrics, exact counters, hashes and learning diagnostics |
| `learning_curves.png` | On-policy task, ghost, reward, win, interruption and darkness curves |
| `console.log` | Appended trainer stdout/stderr, checkpoints, interruptions and completion banner |
| `completion-status.json` | `complete`, `interrupted` or `failed`; success is true only after final checks |
| `status.json` | Real progress, workers, rewards, completed games, CPU/RAM, throughput, ETA and checkpoint |
| `registration.json`, `source-snapshot.zip`, `launchers/` | Frozen settings, source hashes, original import provenance, launcher/report sources |
| `checkpoints/`, `latest.json`, `origin.json`, `champions/` | Exclusive complete checkpoints and verified pointers; old pilot remains separate |
| `history.json`, `evaluations/` | Full completed-batch summaries and the small paired entry/final comparison |
| `attempts/attempt-XXXX/` | Per-attempt journal, errors, raw selected replays, bounded live snapshots and recovery records |

Task starts and interruptions are tracked separately from completed owned
tasks. Movement/task/meeting actor-time, navigation arrivals/failures, alive and
ghost completions, kills, reports, votes/skips, accusations, legal opportunities,
darkness exposure and exact penalty totals are retained. On-policy curves mix
changing scenarios/policies; increasing reward alone does not establish learned
competence. The paired comparison has only eight scenarios and is exploratory.

## Completion and safe resume

Successful completion prints the large terminal message:

**PATRICK JANE — TRAINING COMPLETE**

It then prints additional decisions, matches, training time, PPO updates,
the final checkpoint and the report path. The status JSON has `state: complete`
and `success: true`; the spectator also displays completion. The spectator stays
available for the newly recorded matches. No further stage starts automatically.

For a safe stop, press **Ctrl+C once** and let shutdown finish. It prints
**TRAINING INTERRUPTED**, preserves partial logs/replays and the latest committed
complete checkpoint. Failures print **TRAINING FAILED**. Neither is success.

To resume, run **the same command above**. It acquires an exclusive writer lock,
verifies checkpoint hashes/source/runtime/settings, restores optimizer/RNG and
starts a new attempt folder. It does not repeat already checkpointed updates.
An interrupted partial batch is withheld from PPO and may be collected again
from the last complete boundary. Corrupt pointers are rejected; registered
history/origin candidates are checked before any fallback. It never silently
starts from fresh random weights. Do not edit source/rewards while it is running.

If the cumulative ceiling is exhausted or no checkpoint passes validation,
preserved results remain available for review; no automatic budget extension.

When it finishes, tell the agent **“the lights-only run finished; inspect the
default results folder.”** No upload is needed on this machine. Start with
`completion-status.json`, `TRAINING_REPORT.md`, `training_results.json` and
`console.log`. All raw data and checkpoint history remain local.
