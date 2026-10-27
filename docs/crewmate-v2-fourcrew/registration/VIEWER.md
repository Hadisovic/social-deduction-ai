# Registered-study spectator launch

The viewer remains an independent Pygame process; headless workers never render. The repaired viewer worktree is `C:/Files/RL/_spectator_fourcrew`, now on isolated branch `codex/spectator-fourcrew-registration`. Original PR #11 and its `_spectator_live` worktree are unchanged.

Prepared next-study watch directory:

```text
C:/Files/RL/_v2_fourcrew_registration/local-runs/live-watch/fourcrew-reporting-2048-seed17-v1/replays
```

It is empty because the study has **not started**. The adjacent status.json says registered / awaiting final owner authorization, with zero accepted training decisions and zero validation games. Opening this viewer does not start training.

From PowerShell, paste:

```powershell
Set-Location 'C:/Files/RL/_spectator_fourcrew'
& 'C:/Files/RL/.venv/Scripts/python.exe' './run_training_spectator.py' --replay 'C:/Files/RL/_v2_fourcrew_registration/local-runs/representative-benchmark/replays/benchmark-ordinary/ordinary-2760020.jsonl' --watch 'C:/Files/RL/_v2_fourcrew_registration/local-runs/live-watch/fourcrew-reporting-2048-seed17-v1/replays' --status-file 'C:/Files/RL/_v2_fourcrew_registration/local-runs/live-watch/fourcrew-reporting-2048-seed17-v1/status.json' --mode auto_next --speed 4
```

The same command is saved in `local-runs/live-watch/fourcrew-reporting-2048-seed17-v1/launch-spectator.ps1` in the registration worktree. It initially displays a **completed engineering benchmark**, not a trained study model, and waits for future completed recordings. Use M for auto-next, select a crew, then D for that actor's private neural/suspicion debug. Speed pills are clickable; keyboard speed, pause, seeking and resize remain functional.

The registered study will label actual recordings `original`, `shaped_v2` or `zero`, with real scenario seed/context, runtime hash, policy version and prefix handoff time. Scripted prefixes are explicitly marked **no learning**, and dead actors are labeled automatic ghosts with policy inactive. Prefix motion is not fabricated neural behavior. Missing historical suspicion records remain unavailable. Public meeting ballots remain hidden until their legitimate publication; private debugging is selected-actor only.

Readiness verification parsed four actual complete engineering replays: ordinary contains neural records for all four PPO crew; controlled cases contain three living PPO actors and the dead prefix victim. Result records match simulator outcomes. Across 372 consecutive meeting frame pairs, living/ghost positions and completed-task counts remained frozen. No dead neural decisions, dead ballots, dead claims or task completions during meetings were found. Current-meeting projections hide unpublished targets. Real distributions and suspicion records were checked, and the selected-actor voting frame rendered headlessly with native artwork.

The original 34 spectator checks and the focused progress/prefix checks pass. The new viewer uses actual registered 2,048/48 progress denominators while retaining 512/20 defaults for historical status files. This readiness check does **not** claim a new visible desktop window was launched; it verifies actual recorded playback/rendering and interactions before the owner launches it.

Raw benchmark replays, private debug image and role-bearing traces remain local. Only completed atomic `.jsonl` recordings enter the library; `.writing` files stay incomplete.
