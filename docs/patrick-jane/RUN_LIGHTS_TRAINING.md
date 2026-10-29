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
- New profile **`patrick-jane-lights-learning-v2`** disables the former normal
  900-second game-ending rule. Legitimate crew/impostor victory conditions remain.
  The legacy lobby `timeout: 900` field is retained for source compatibility but
  is inactive in this profile; preflight prints the effective normal timeout as
  `null`. Historical profiles/replays keep their original semantics.
- Technical safeguards cut a simulation at 1,800 simulated seconds or after
  300 seconds stationary without progress. Bounded meeting phases finish before
  a cut; a genuine victory takes precedence. A cut is **TRUNCATED**, with no
  fabricated win/loss reward, and is excluded from win/loss/draw rates. Its cause,
  progress diagnostics and final critic values are saved. Living actors bootstrap
  from legitimate final roaming observations and private recurrent states;
  genuine terminal outcomes bootstrap zero. Dead/ejected actors stay terminated:
  actually observed delayed credits are retained, but future ghost work and team
  outcomes beyond a cut are censored. Frequent truncations would limit conclusions.
  A validation checkpoint is eligible for the existing champion rule only if
  every registered evaluation outcome is legitimate; cuts withhold selection.
  Latest checkpoints and raw evaluations remain available in either case.
- Continue the original pilot's `update-000006.pt`: 255,083 individual decisions,
  SHA256 `d8b81ef06788a0139822a5677801f34069aef116286cc079cb12de8c39538ab8`.
  The policy, Adam state and RNG are imported after verifying the historical
  archive and checkpoint. This is an explicit new training regime. The original
  model and champion remain untouched; engineering-test models are not used.
- **1,000,000 additional individual living-crewmate decisions**, giving at
  least 1,255,083 lifetime decisions. Batches finish through genuine outcomes or
  explicit bootstrapped technical boundaries and can overshoot;
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

The terminal and viewer display actual additional learning-agent decisions and
their percentage of the registered target, capped at 100%. Validation actions,
scripted impostors, ghosts, ticks and bootstrap-only critic calls do not advance
this counter. Periodic updates include measured decisions/second, ETA, cumulative
active elapsed time, legitimate games, separate truncations, PPO updates, latest
checkpoint and training state. ETA initially says `measuring`. Resume restores
the saved count/time; a withheld partial batch can roll back to the last committed
checkpoint. Reaching 100% during collection is not completion until the update,
final validation and all outputs have successfully saved.

Successful completion prints the large terminal message:

**PATRICK JANE — TRAINING COMPLETE (100%)**

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
An operator-interrupted partial batch is withheld from PPO and may be collected again
from the last complete boundary. Corrupt pointers are rejected; registered
history/origin candidates are checked before any fallback. It never silently
starts from fresh random weights. Do not edit source/rewards while it is running.

If the cumulative ceiling is exhausted or no checkpoint passes validation,
preserved results remain available for review; no automatic budget extension.

When it finishes, tell the agent **“the lights-only run finished; inspect the
default results folder.”** No upload is needed on this machine. Start with
`completion-status.json`, `TRAINING_REPORT.md`, `training_results.json` and
`console.log`. All raw data and checkpoint history remain local.
