# CURRENT: configurable lobby and full gameplay checklist,31October2026

Local Lizbon viewer now opens a Lobby tab with total4–15, impostors1–3 and
Common/Long/Short counts; Save creates exclusive profiles; Start launches one
owned bounded inference-only smoke game with existing weights. No PPO study,
selection, final tests, merge or deployment. Native pinned source and all
jane/impostor and shared engine/settings sources remain unchanged by this UI work.

22 focused tests passed, including real Pygame controls, validation, failure/cancel,
partial/full playback, stale async loads, doors and offline comparator. First
UI-started 4p game failed WinError5 replacing status.json; preserved under
local-runs/lizbon-dashboard-check-20261031. Viewer no longer opens critical producer
status/results during processing. Repeat under -retry completed naturally:4total,
1imp,0/0/1tasks,35.4simsec,29.74collectionsec,30.29runnersec. Crew task win,
zero kills/illegal/navigation failures, zero PPO/model changes. Full completed
replay separately confirmed parsed and paused at t0. These are engineering checks,
not performance evaluation. Histories/runtimes stay private/ignored.

Docs: docs/jane-impostor-v2/DASHBOARD_LOBBY.md (pictures/receipt) and
LIZBON_GAMEPLAY_CHECKLIST.md (all rules/actions/learning status, 1/2/3 allies,
current rewards, historical rates, deferred features, owner/friend next steps).
Current latest checkpoint still unselected/unvalidated; Classic prior training
0/3 must not be called its test rate. Old shared14/16 is a different system.

Restart viewer with python run_lizbon.py to get Lobby. Existing --replay/--watch
still work. Close only terminates this dashboard's own new inference game.
Publish own codex/lizbon-classic-v2 branch only. Preserve unrelated pilot README,
NAVIGATION_V2_PLAN, replays and protected docs/research/ISSUES.md.

---

# Previous: cache-coherence fix implemented31October2026

Shared adapter revision `paired-map-planner-v1` is implemented. After native door
updates and meeting resets, collision/projection/player motion maps bind to the
selected planner map; observations invalidate on rebinding. True door-revision
mismatches and the original projection guard still raise. Pinned source and all
historical V1/shared/NN code remain unchanged. Full details: docs/jane-impostor-v2/
CACHE_PAIR_FIX.md. No new training study, merge or deployment.

Verification:87isolated Jane tests passed; migration rechecked2passed after
verified-bytes hardening;3comparator cases passed. Maintained root295passed,
13skipped with the prior frozen-publication hash failure. Bare pytest has6prior
stable_baselines3 collection errors. See verification JSON for exact evidence.

Paired engineering seed9700032 runs stopped at180s/110and112decisions. The common
110decision prefix matches exactly across neural inputs/actions and physics,
including post-eviction observations. Policy/crew tensors unchanged; zero illegal
or navigation errors. Full120-decision target was NOT reached; original raw check
is preserved and the offline common-prefix assessment is explicit. No NN rerun.

Strict receipts now include geometry revision. Pre-fix saved checkpoints require
explicit migration (tools/migrate_lizbon_geometry_checkpoint.py), or historical
source checkout for exact reproduction. Migration permits only the3registered
old file hashes, loads verified bytes, preserves identical tensors, creates a new
receipt with parent ancestry and no optimizer/new PPO passes. Normal strict
loader is unchanged. A migrated checkpoint003 is preserved locally under
local-runs/lizbon-classic-v2/fix-20261031; it is still an unselected model.

No training is running and the failed37-minute pilot is NOT resumed. Old final
partitions stay sealed. Future work: shared crew/impostor use make_match factory,
profile full sampled match runtime, then obtain a new explicit study approval.
Fix is prepared for publication on codex/lizbon-classic-v2; no merge/deploy.
Future approved studies must retain the scoped Windows process-tree guard; this
geometry patch does not change the supervisor. Preserve unrelated pilot
README, NAVIGATION_V2_PLAN, untracked replays and protected research/ISSUES.md.

---

# Historical pre-fix status (no longer current)

# CURRENT: Lizbon Classic pilot failed; postmortem completed

No training is running. The bounded study stopped on30October2026 about20:05:49UTC
(23:05:49Amman), after2,232.5s /37m12.5s. No retry/extension, merge or deployment.
Frozen source037899471a62502a25b1a9351cc91379bfc6423b; branch codex/lizbon-classic-v2.
Historical supervisor25216 and workers9200/83260 are gone. Guard87104 exited on
fatal stop; no orphan workers remained to terminate.

Stream101:2crew-task losses,1wall cut, game4cancelled;16new PPO passes/1,830samples.
Stream202:1crew-task loss,1wall cut, game3geometry failure;8new passes/1,086samples.
Three complete games total, zero impostor wins. All24new optimizer passes came
from complete games. Three development checkpoints strict-loaded and changed.
No final.pt, selection lock, validation or confirmation; no selected model release.
All final partitions remain sealed. Final in-memory crew hash checks were not
reached; the frozen crew artifact file SHA is still unchanged.

All4,938readable choices analyzed, including56 recovered from an interrupted gzip.
Raw files/development weights remain ignored under local-runs/lizbon-classic-v2/
study-20261030. Five graphs, actual gameplay, hashes, timings, outcomes and concrete
next steps are in docs/jane-impostor-v2/study-20261030/README.md.

Controlled geometry-only cache eviction reproduces the mismatch with an equivalent
map and an older cached planner. Pairing map with planner.map passes in memory.
The upstream patch compiles/applies-check cleanly but IS NOT APPLIED to the pinned
source. Diagnostic regression1passed in13.12s. Exact crashed-process cache contents
were not recorded. No network architecture/reward tuning or further training.

Next: review the proposed cache-pair fix with crew developer, version/test the
shared engine under cache pressure/doors/meeting/vent scenarios, profile full
sampled matches, then obtain explicit new study approval. Keep frozen cross-play
before population/task curricula or adaptation/self-play. Current model competence
or improvement is not established. See main checklist and result report.

Postmortem/code pushed on codex/lizbon-classic-v2; no merge/deploy.
Completed trained-policy playback opened (launcher86888/actual88128), stream101
game3 at107s, selectedp3, paused. Completion heartbeat was deleted after verified push; no further study checks are scheduled. Preserve unrelated pilot README,
NAVIGATION_V2_PLAN, replays/demo_seed0007.jsonl and protected research/ISSUES.md.

---

# Historical in-flight handoff (no longer current)

# Lizbon Classic V2: current execution status

30 October 2026. **PILOT RUNNING. Preserve healthy jobs.** Branch
`codex/lizbon-classic-v2`, source freeze037899471a62502a25b1a9351cc91379bfc6423b.
Registered start19:28:36.513UTC /22:28:36Amman. Hard deadline22:28:36.513UTC /
31October01:28:36Amman. Training cutoff20:58:36UTC. Owner approved three hours.
Parent exec session1793; launcher89980/actual supervisor25216. Training stream101
launcher89196/actual9200; stream202 launcher71656/actual83260. Windows process-tree
guard87104 enforces the same failure/deadline without changing NN sources.
Update this block from persisted status, not assumptions. No merge/deployment.

## Approved implementation

- New shared Classic engine/profile/task allocator in `jane/classic_v2` supports
 4ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“15 total and accepted1ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“3 impostor combinations, including14 crew+1.
- Editable Common/Long/Short counts; unique task families/owner; common plans
 match-wide. Historical V1 and shared source bytes untouched.
- Inactive/underground stored-position invariant fixed in an opt-in local adapter;
 living surface collisions remain enforced. External pinned source is unchanged.
- New V2 runner/receipts, explicit V1 weights-only migration, optional explicit
 V2 cross-profile transfer and same-profile optimizer resume. NN architecture
 and V1 rewards/PPO preserved. Every impostor decision is audited.
- Isolated native tests74 passed. Maintained root tests291 passed/10 skipped with
 one pre-existing frozen publication contract failure; bare pytest has six root
 legacy-script import errors (stable_baselines3 missing). See checklist.
- Engineering seed9700001 completed naturally: crew task win, zero kills,
 zero illegal/nav failures,173.2 simulated seconds,142.67 collection wall seconds,
143.18 total. Greedy policy idled284/289 choices per impostor, selected lights
five times each (shared contention suppressed one request), never travelled.
 This diagnoses weak greedy behavior; no new PPO occurred in that engineering run.
 Its exact source is preserved in the ignored engineering-source.zip.

## Run location and frozen dependency

Active checkout:
`C:/Users/masaa/.codex/worktrees/impostor-shared-integration/stealth-rl-navigation`

Python:
`local-runs/jane-impostor-v1/runtime/Scripts/python.exe`
Python3.12.10, Torch2.7.1+cpu, NumPy2.5.3. Intel i9-14900HX24cores/32threads,
one Torch thread/worker, two concurrent workers, GPU not used.

Source root:
`C:/Users/masaa/Desktop/UNI/summer_phase/stealth-rl-navigation/temp_jane_foundation/upstream-review/d459f38674ebc537acde20647cbd07700c494533`

Verified frozen crew SHA256:
`da1d43d73c140ac39bf44834746dbae605b9b36d3c6c7af8a4f2be975e535e3c`
Published update102 is learned task-pursuit V1, **not A2-trained**.

## Study protocol and finish work

Read `docs/jane-impostor-v2/STUDY_PLAN.md`. Supervisor:
`tools/run_lizbon_classic_study.py`. Fixed two same-start sampling streams, four
matches each;90-minute training cutoff; fresh validation; immutable selection
lock before eight paired confirmation matches, sampled and greedy. Three-hour
hard total deadline; no automatic retries/extensions. Old/fresh final sets sealed.

Output: `local-runs/lizbon-classic-v2/study-20261030`.
Inspect registration/status, per-run status/matches/PPO and decision journals;
preserve healthy jobs and never edit loaded source. After completion inspect
results/failure, selection-lock and aggregate/REPORT plus five graphs. Regenerate
reports only if necessary; never rerun training/selection from confirmation.

Curate concise report, graphs, actual times/processor, every failure/cut and honest
learning sufficiency in docs/jane-impostor-v2. Release selected checkpoint with
receipt/ancestry if selection completed. Update checklist/handoff, verify/open
actual completed playback with run_lizbon_viewer.py, push this own branch only.
Keep unmodified unrelated pilot README/NAVIGATION_V2_PLAN work outside the commit.
Protected docs/research/ISSUES.md remains untouched.

Technical addendum after launch: Windows venv redirectors create actual runtime
children. The source coordinator's fail-fast guard terminates tracked Popen
processes but does not itself guarantee those children stop. Separate
tools/watch_lizbon_classic_study.ps1 matches this study's absolute output path and
exact worker script, stops its workers on a registered fatal-stop/deadline, and
leaves all other jobs/reporting alone. Its disposable runtime-tree deadline test
passed. See process-tree-guard.json and stderr log; keep it healthy. This changes
neither weights, rewards, sampling, selection nor the approved budget. Do not
patch the loaded supervisor during this study; fix tree termination in a future
source version before another run. No automatic retry on failure.

Next work for both developers is frozen cross-play and shared settings/engine
compatibility. Establish one competent policy before adding population/task
curricula, stronger adaptive opponents or self-play. Deferred features are listed
in RULES_AND_SOURCES.md. Never call training wins accuracy or claim globally best.

Progress19:48UTC: first match in each stream hit the900s wall cap (actual901.73/918.86s), with2/3 kills and no new PPO updates. Both healthy streams advanced to second registered matches. See docs/jane-impostor-v2/RUN_PROGRESS.md. Do not change caps/source or retry cuts.

Progress20:03UTC: each stream completed episode2 (crew task win), saved checkpoint002 and8 new PPO passes. Each has1complete/1cut; episode3 is advancing with updated weights. No validation/confirmation opened. Latest truth is per-stream status/matches/PPO. Preserve healthy jobs.
