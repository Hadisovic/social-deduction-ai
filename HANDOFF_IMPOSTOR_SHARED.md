# CURRENT — shared pilot stopped at incomplete validation

Read `docs/collaboration/impostor-shared-integration/TRAINING_STATUS.md` first.
Read its linked pilot-20261029/README.md for graphs, timing and results. The owner
approved training on 29 October. Both eight-episode streams completed, 14/16
training wins total. Validation stopped at 30/32 after a 600-second computational
abort on stream 101/episode 4/seed 3610006; next seed was not started. Stream
101/episode 8 won 8/8 B validation cases, but no selection lock, promotion or
paired A/B confirmation occurred. Improvement over the reference is unestablished.
The run ended after 39.17 minutes; parent session 50872 exited 1; no active jobs.
All 16 local checkpoints and four curated strict-load candidates are preserved.
Do not restart this study or alter its frozen source/registration/checkpoints.
Next requires a documented evaluation-only continuation/profiling plan, not
automatic extra PPO. Keep historical final sets sealed. No merge or deployment.
The engineering history below predates the training approval.

# Engineering handoff — 28 October 2026

Read `docs/collaboration/impostor-shared-integration/README.md`, CHECKLIST.md and
EXECUTION_STATUS.md. Those describe this task; inherited root HANDOFF.md is
historical and left unchanged to preserve the published handoff.

Own local branch `codex/impostor-shared-integration`, isolated managed worktree,
based on developer-pinned c3320583f7d562bc69ea527580a91ddc5db54f18. Owner explicitly
requested "finish and push" on28October; publication is authorized. This does NOT
approve merging, full PPO training or reserved final access. Stop for review of
STUDY_PROPOSAL.md. Current engineering CLI performs no PPO updates.

Five neural actors: four sampled frozen A/B crew (B default), transferred V4.1
impostor83/8192, private per-actor state. A/B strict loads and282 published files
pass. Original impostor strict loader passed before explicit tensor transfer;
original model unchanged, cross-runtime synthetic inference matched exactly.

All completed matches had0 illegal actions and frozen weights. Learned B/repeat
matched exactly; both ejected after1kill/1report at84.2s. Paired A has the same
world trace despite different weights. Scripted B reproduction matched developer's
parity event hash. Only2scenario seeds, not a study or win-rate estimate.

86 distinct tests pass,5 expected private-fixture skips. Final11 focused tests
include terminal idempotence and source-locked episode-boundary optimizer/RNG
restore. Checkpoint tests use synthetic gradients; no game-based RL updates.
No active jobs. Actual Spectator preview is local/private at
`local-runs/shared-integration/preview.png`; `run_impostor_shared.py` opens the
completed B replay and automatically selects its learned impostor.

Published map/assets/viewer/source unchanged. Functional sabotage is NOT included.
Opponent-pool assessment, proposed60–90min/90min-cap pilot and limitations are in
the report. Budget is NOT approved. Full training requires owner approval and a
frozen dedicated PPO experiment runner/registration preflight. Publish only the
integration branch; keep raw replays, private preferences and local runtimes out.
