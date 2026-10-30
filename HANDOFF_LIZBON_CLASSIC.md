# Lizbon Classic V2: current execution status

30 October 2026. Branch `codex/lizbon-classic-v2`, based on c9542c6. Owner approved
the new **three-hour** pilot after the task-count clarification. Training is not
started at this document revision. Update this top block from persisted status,
not assumptions about worker activity. No merge/deployment is authorized.

## Approved implementation

- New shared Classic engine/profile/task allocator in `jane/classic_v2` supports
 4–15 total and accepted1–3 impostor combinations, including14 crew+1.
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

Intended output: `local-runs/lizbon-classic-v2/study-20261030`.
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

Next work for both developers is frozen cross-play and shared settings/engine
compatibility. Establish one competent policy before adding population/task
curricula, stronger adaptive opponents or self-play. Deferred features are listed
in RULES_AND_SOURCES.md. Never call training wins accuracy or claim globally best.
