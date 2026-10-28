# Current execution status

29 October: owner approved starting the bounded shared pilot. Follow
TRAINING_STATUS.md and the private run status/journals. The engineering-only
status below is historical. Never restart a healthy job or change loaded source.

Engineering COMPLETE; no active jobs. Owner now requested "finish and push":
publish this integration branch only. Full PPO and merging remain unapproved.
Read README.md, CHECKLIST.md, STUDY_PROPOSAL.md,
verification.json and publication-verification.json.

Worktree: managed `impostor-shared-integration/stealth-rl-navigation`.
Branch: `codex/impostor-shared-integration`, based exactly on developer-pinned
publication c3320583f7d562bc69ea527580a91ddc5db54f18 (draft PR12).
Published branch and original `codex/impostor-environment-v2` untouched.

Exact Python3.12.10/NumPy2.5.3/Torch2.6.0+cpu installed in local `.venv`, with
private Python in `.runtime/`. Both installation jobs completed successfully.
Unchanged `frozen_handoff.verify` passed282 files, A/B strict loads and artwork.

Original V4.1 strict loader passed in original Python3.12.3/NumPy1.26.4/Torch2.9.1.
Exact inference tensors exported explicitly, no optimizer. Corrected transfer:
`impostor_shared/assets-v1/`, SHA48255932c6a154397efb2785569da5af7a215d1292489033ec33d9529f41eb80.
Original checkpoint SHA69965afd8fab28e6ca5bdda46d9f8efdd8d6036ce57a05b86ff89fe22cf17ca9.
Tensor hash ebb59ec4cb1589a5b3e96d805085d2bec7ea35c7619be96c64a81c0ce1c94287.
Three synthetic packets passed cross-runtime parity with0 maximum probability
error. Failed first metadata export preserved in assets/; see FAILURES.md.

Four frozen B versus original scripted stand-in completed as published:168sim s,
PARITY,49crew decisions,33.69wall s,0illegal actions, frozen weights/private
memories intact. Event SHA b935b8b8d18be5b38166474b3dec86d38531b2a6351cad3c6ab9abffbb9555bd.
Local replay/summary: `local-runs/shared-integration/scripted-b-2760000.*`.

New namespace `impostor_shared/`: pure historical architecture/features/memory,
explicit transfer loader, actor-only V2+own-fake-timer projection, travel/fake
options, real neural recording, trusted per-physics reward/terminal collector.
Shared published sources/assets unchanged.86 distinct tests pass,5 disclosed
private-fixture skips:combined84 plus2 checkpoint tests. Final11 focused checks
overlap9 and add2; do not sum84+11. New source-locked snapshots restore optimizer/
private RNG at episode boundaries; arbitrary mid-game resume is rejected.
Synthetic optimizer unit updates are not game-based PPO training.

All tests/smoke/render jobs exited. Learned B2760002 and recording-off repeat:
1kill/1report, ejected84.2sim seconds,105 transitions,return-13.444,0illegal
actions, identical event/action/credit data. Wall32.00/31.62s. Paired A has same
world trace at68.03wall s overlapping other work; not a relative speed/strength
comparison. A/B all20tensors differ. Failed-start folders retained locally.

Watch: `.venv/Scripts/python.exe run_impostor_shared.py`. Default complete replay:
local-runs/shared-integration/learned-b-2760002-v2/replay.jsonl. Actual default
selected p2 and rendered preview.png through unchanged Spectator V2. All5 actors
have neural/reward records; completed result footer checked. This is replay,
not a live training loop. Fresh generation commands and proposed60–90min pilot
are in the report. Budget NOT approved; preserve untouched zero-update reference.

WRITE PERMISSIONS:managed worktree is outside this turn's writable roots.
Author new files in ignored `temp_shared_integration_stage/files/` under the
original workspace, then approved helper `python temp_shared_integration_stage/sync.py`
copies ONLY new integration namespace/docs/test/root wrapper. It now skips
unchanged bytes so documentation progress does not rewrite running source modules.
New runtime artifact writers need exec escalation; read-only loads can run default.
Do not weaken published locks. Do not use desktop computer-use; supply Antigravity
terminal commands and a verified local capture. No subagents authorized.
