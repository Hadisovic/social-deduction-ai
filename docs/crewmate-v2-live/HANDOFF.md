# Completed live spectator / seed-17 pilot handoff

Status: **completed, development only, Phase 6.1 acceptance pending**. Do not rerun the pilot or start seed 29. No further training, evaluation, fixes or crewmate publication is authorized by this completed request.

## Preserved branches

- Crewmate worktree `_v2_pretraining_diagnostic`, branch `codex/crewmate-v2-live-pilot`. Runtime source frozen at `38bab0b`, based on performance `c18ff5f`, cleanup `27ff525` and reward/PPO work `8e99383`. Results are a subsequent local-only commit. **Do not push.**
- Spectator worktree `_spectator_live`, branch `codex/spectator-live-integration`, published `6dd8a1b`. Original spectator `3fd3601` remains preserved and published on `feat/crewmate-first-training-spectator`.
- Draft [PR #11](https://github.com/Hadisovic/social-deduction-ai/pull/11) contains spectator work only. It remains unmerged. Main and the impostor reference were unchanged.

## Completed verification and experiment

31 spectator tests and 178 crewmate tests passed. A paired 72-choice engineering check (four workers, eight streams) exported four real PPO episodes and preserved recording-off/on actions, rewards, outcomes, final packets and updated weights. Off/on wall times were 87.28/87.61 seconds; one pair is only a bounded overhead check.

Before training, native window 462292 was discovered, a real PPO frame rendered, the app heartbeat was live, and the owner explicitly confirmed the visible Skeld replay and neural panel. Native screenshot access timed out; owner confirmation resolved that gate. The visualization remained independently open during training. It discovered 22 completed actual pilot replays without a restart.

The authorized pilot ran **once**, session 12599, exit 0. Both original and shaped arms completed 512 choices, four updates, 20 completed training episodes plus eight partial streams, and 20 validation games. Each accumulated 11,632.4 simulated seconds. Four workers/eight streams/16-choice segments were retained. Initialization seed 17 matched; no weights were transferred between arms. Total wall time: **2,904.55 seconds / 48.41 minutes**, below the three-hour cap.

A won 5/20; B won 4/20. Both had four ordinary task wins. Correct/incorrect validation ballots: 24/3 versus 24/5. Neither made a report, and neither had a legal report opportunity at a validation policy-choice boundary. The report incentive was scarcely exposed in training. Reporter-only identification matches aggregate target accuracy, so independent deduction is not established. See [the full review](results/REVIEW.md), [aggregate metrics](results/metrics.json) and [preservation verification](results/verification.json).

## Viewer status and relaunch

Persistent viewer session 18955 / PID 64300 was still alive at final verification, with a 26-file library (four engineering plus 22 pilot recordings). The user selected manual inspection; automatic discovery continues. Press **M** for auto-next, or use the library to choose Model A / Model B. Do not force away from the owner's selected replay.

The viewer plays **completed training episodes** with real recorded neural probabilities, values and rewards. Partial `.writing` files remain excluded. No true unfinished-frame streaming or validation replays are claimed. The historical `--capture` option intentionally exits; do not use it for a persistent viewer.

To reopen if needed, from `C:/Files/RL/_spectator_live`:

```powershell
& 'C:/Files/RL/.venv/Scripts/python.exe' './run_training_spectator.py' --watch 'C:/Files/RL/_v2_pretraining_diagnostic/local-runs/live-watch/pilot' --mode auto_next --speed 8 --status-file 'C:/Files/RL/_v2_pretraining_diagnostic/local-runs/live-watch/training-status.json'
```

Do not launch a duplicate while the existing viewer is open. The local ignored artwork junction remains intact. A private rendered real-PPO frame is at `_spectator_live/local-runs/real-ppo-desktop.png`.

## Artifacts and resumption boundary

- `local-runs/seed17-live-pilot-20261027/`: frozen registration, `pilot.json`, `comparison.json`, and each arm's configs, logs, validation records and immutable checkpoints at 128/256/384/512 choices.
- `local-runs/live-watch/pilot/{original,shaped}/`: eleven completed training replays per arm, with partial streams preserved separately.
- `local-runs/live-watch/engineering-smoke/proof.json`: engineering-only noninterference/overhead proof.
- `local-runs/live-pilot-session.json`: completed session record; the old training command must not be repeated.
- `local-runs/seed17-live-pilot-artifact-hashes.json`: private integrity manifest for 53 pilot artifacts.
- `tools/summarize_v2_live_pilot.py`: offline aggregate/figure reproduction into a **fresh** output directory. It does not run games or train models. `results/REVIEW.md` is the separately authored scientific interpretation.

Final checks preserved 234 protected files, 25 previous pre-training files, 19 reporting-private files, 25 earlier learning-smoke artifacts and all nine historical selected checkpoints. All eight new checkpoints load; final selected tensor hashes match. The live runtime digest remains `886d22405554adf830d859dd83d592c474d5906e3bb274c6b2f4521f0da950b3`.

Raw replays contain privileged spectator state and remain local. Do not publish raw observations, role-label datasets or checkpoints. No reserved final partitions were opened, no reward/gameplay/architecture changes occurred, and no impostor work, merge or deployment occurred.

The recommended next step is a separately approved small learning pilot with legitimate evidence exposure, reporter-diversity controls and a zero-update reference. **Stop here for experimental review.**
