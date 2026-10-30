# Lizbon dashboard lobby

Owner request: choose players, impostors and Common/Long/Short task counts in the existing Python dashboard. Implement locally; keep the pinned native source and all neural/checkpoint code unchanged.

- [x] Test and implement a validated lobby draft, persistence and exclusive profile creation.
- [x] Test and implement one owned, bounded inference process; weights-only transfer, no PPO, unique output, visible failure/cancellation.
- [x] Add a Lobby tab, count controls, model/budget selection, Save / Start / Stop / Watch and live snapshots.
- [x] Allow `python run_lizbon.py` to open the Lobby directly; preserve explicit replay/watch behavior.
- [x] Exercise actual Pygame controls and one bounded custom game; inspect profile, replay, zero updates and screenshots.
- [x] Review, document and push only this branch.

Design: two columns (population and tasks), then model, viewing mode and wall-time budget. Changing controls affects a new game only. Invalid population combinations block Save/Start with a reason. All-zero tasks show the engine's one-short fallback. Config and raw games stay under ignored local-runs; registered profiles are never overwritten.

Review focus: invalid 4-player/3-impostor combination; corrupt saved draft; missing/stale checkpoint; duplicate Start and closing a running game; live-to-completed playback and failure with partial artifacts.

Implementation ruling: existing session authorization covers the UI and inference smoke verification; no new training study. Use a local adapter rather than editing the pinned dashboard.

Verification:22 focused checks passed. Actual custom UI game and full playback verified; first Windows reader-lock failure retained. Reviewer lifecycle findings fixed with regressions. NN/engine sources unchanged. Publication follows the owner-authorized own-branch flow.
