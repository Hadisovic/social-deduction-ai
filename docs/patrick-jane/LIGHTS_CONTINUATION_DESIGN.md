# Lights-only manual training launch — 2026-10-29

The owner's typed instruction supersedes the attachment's disable-all-sabotage
and preserve-all-reward clauses: lights remain enabled; oxygen/reactor are
disabled by a run profile. All sabotage implementations remain available in
the original profile. Killing, 10 crew + 2 scripted impostors, tasks, social
actions, ghost ownership and approved neural/PPO design are unchanged.

The only new reward is -0.005 per living crewmate per simulated second with
unresolved lights, including meeting time while lights persist. Sixty seconds
costs 0.30; five minutes costs 1.50. No dead/ghost penalty, no wall-clock or
render-speed dependence, no charge after repair, no duplicated accounting.
All existing coefficient values remain unchanged in `rewards.py`.

Use a new isolated run directory, 32 workers, 1,000,000 **additional** decisions
after the preserved 255,083-decision checkpoint. Import its policy, Adam state
and RNG only after exact checkpoint/archive hashes, historical source locks,
unchanged network/PPO/action/memory source, runtime and architecture validate.
This explicitly records a changed training regime, not a claim of strict
same-protocol resume. Subsequent resumes require the new run's exact sources
and configuration. Never weaken the original strict checkpoint loader.

Reuse the current collector, independent actor RNG/memory, complete-match PPO,
finite/legality/reward/source/deadline guards, source archives, exclusive
checkpoints and independent spectator. Do not repeat worker benchmarks. The
new runner uses eight paired development entry/final games, rather than 400
per pilot. No reserved partition access; on-policy curves are descriptive.

One PowerShell launcher starts/reuses the existing spectator and runs a
lightweight guarded Python entry point in the foreground. Default 12-hour
cumulative active execution ceiling; interrupted batches never enter PPO.
Rerunning the same command resumes the latest hash-verified complete checkpoint
in a new attempt folder. A run lock prevents concurrent writers. A completed
run is displayed, not silently extended. Each update journals full game
summaries and diagnostics; reports and machine-readable results are regenerated
after checkpoints and at every normal/error/interruption exit.

The run folder contains registration/source archive/import provenance, stdout
and stderr log, attempt journals, checkpoints/latest pointer, selected replays,
status/completion files, `TRAINING_REPORT.md`, `training_results.json` and curves.
Report task starts, new-intention/navigation interruptions, alive/ghost work,
movement/task/meeting actor-time, navigation arrival/failure, deaths, wins,
reports, social choices, reward components, light exposure/penalties and hashes.
Only success after the target and final checks prints TRAINING COMPLETE;
exceptions/interruption retain partial results and explicitly report failure.

Verification: unit-test capabilities, exact simulated-time penalty and
ownership, import integrity, safe pointers, reports and resume. Exercise the
actual launcher with separate, bounded engineering outputs, 32 workers and
short matches; check interrupted/resumed completion and failure labeling.
These plumbing checks do not establish learned improvement. Do not launch the
long run or keep an agent waiting for it.

## Approved timeout and progress correction — 2026-10-29

The owner's subsequent review approves the existing design but removes the
normal duration-based game ending before manual launch. New profile v2 disables
the inherited 900-second rule; historical profiles/source archives remain intact.
Technical bounds (1,800 simulated seconds or 300 stationary seconds without
progress) produce TRUNCATED with no winner/outcome reward. A bounded meeting
finishes before the final roaming observation; genuine victory takes precedence.
Living actors bootstrap from that private observation and their independent
GRU state. Dead/ejected actors retain only observed delayed credits; future ghost
credit is censored. The PPO change is exactly the terminal-or-truncated assertion;
GAE, architecture, coefficients and all hyperparameters are unchanged.

Terminal/viewer percentages use additional actual neural actor decisions, exclude
validation and critic-only calls, and restore saved counters/active time on resume.
They are capped at 100%; successful completion also requires final evaluation and
all outputs. Truncation diagnostics and counts remain separate from wins/losses.
No worker-count benchmarks or long training are authorized in this preparation
pass. [Investigation and limits](TIMEOUT_INVESTIGATION.md); R041/R042 retain the
measured verification and negative findings. Default command/folder are unchanged.
