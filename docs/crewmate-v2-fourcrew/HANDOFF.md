# Post-pilot implementation handoff

Completed 2026-10-27. **No full new experiment is approved or running. Stop for owner review.**

Crew worktree `C:/Files/RL/_v2_fourcrew`, branch `codex/crewmate-v2-fourcrew`. Mechanics `77f478c`, architecture `555e3d0` / `961aab9`, reward revision `732b555`, stronger regressions `f8f9ff5`. Source is frozen; final engineering checkpoint source lock matches the current code. Documentation is a later separate commit. Do not push or merge without separate authorization.

Viewer worktree `C:/Files/RL/_spectator_fourcrew`, branch `codex/spectator-fourcrew`, local `fa76497`. Original `_spectator_live` stays at `6dd8a1b`; draft PR #11 is unchanged. Old persistent viewer, if still open, is the original app, not automatically upgraded by these isolated changes.

The source architecture has four independent actor views/memories/samplers/trajectories and one shared actor network with central PPO updates. One scripted impostor remains. Complete-match batches retain dead actors' final transition through discounted team settlement; no policy action or memory update follows death. Current executable CLI runs engineering seeds only. Exact partial-match decision-capped continuation and the next study's evaluation/prefix registration remain outside this engineering implementation.

Validation: 134 main V2 regressions; strengthened overlapping subset 17; spectator 34. Final four-worker smoke: four 3-second matches, 23 actor choices, one central PPO update, strict checkpoint round trip. Sixty-second engineering capture: 32 choices, zero updates, exact packets/actions/rewards/outcome off/on. The real replay rendered headlessly with private-debug p0; no new visible desktop launch claimed. Do not interpret these as learned reporting/deduction results.

Private outputs:

- `local-runs/fourcrew-final-smoke/`: current-source engineering checkpoint and summary.
- `local-runs/fourcrew-final-capture/`: complete actual four-actor replay and proof/source lock.
- `local-runs/postpilot-artifact-hashes.json`: nine private-file hashes.
- `local-runs/postpilot-replay-audit.json`: aggregate offline review of the historical completed sampled replays.
- `_spectator_fourcrew/local-runs/fourcrew-engineering.png`: private headless rendered frame.

All 234 historical protected files, 53 actual pilot artifacts, prior pre-training/reporting/learning-smoke archives and nine selected checkpoints passed final integrity checks. Original pilot worktree remains `5b663f6`, canonical V2 branch remains `e62ba0b`, main/impostor references remain unchanged. No reserved final partitions, seed 29, learned impostor integration, population expansion, publication, merge or deployment occurred.

Read [REVIEW.md](REVIEW.md), [validation.json](validation.json), [NEXT_EXPERIMENT.md](NEXT_EXPERIMENT.md) and [research register](../research/ISSUES.md). Owner observations not verified from preserved data remain unresolved. The new schema deliberately rejects historical checkpoints; use the preserved original branch for their strict loaders. No old checkpoint or report was rewritten.

Viewer launch from `C:/Files/RL/_spectator_fourcrew`:

```powershell
& 'C:/Files/RL/.venv/Scripts/python.exe' './run_training_spectator.py' --replay 'C:/Files/RL/_v2_fourcrew/local-runs/fourcrew-final-capture/fourcrew-engineering.jsonl' --watch 'C:/Files/RL/_v2_fourcrew/local-runs/fourcrew-final-capture' --mode auto_next --speed 4
```

Click a crew, press **D** for that actor's private neural/suspicion debug; speed buttons and keyboard controls work. This replay is zero-update engineering, not the old trained focal policy. The original trained replays remain in `_v2_pretraining_diagnostic/local-runs/live-watch/pilot`.
