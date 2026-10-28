# Phase 2 verification and handoff

2026-10-28. Dedicated worktree `_phase2_opponents`, branch `codex/phase2-opponent-diversity`, base `fab5c70`. Spectator V2 remains in its existing worktree. No general UI rebuild/verification was repeated.

## Actual checks

The final suite passed **38 tests in 171.01s**, zero failures/skips:

```powershell
& ../.venv/Scripts/python.exe -m pytest `
  tests/test_phase2_opponents.py tests/test_phase2_social.py `
  tests/test_phase2_shortcuts.py tests/test_phase2_accounting.py `
  tests/test_v2_mechanics_repairs.py tests/test_v2_fourcrew_rewards.py `
  -q --basetemp local-runs/new-test-output `
  --junitxml local-runs/new-test-results.xml --tb=short
```

Always use a new workspace-local temporary directory. Final local evidence: `local-runs/phase2-complete-02.xml`. Earlier evidence is preserved: 22 tests/181.23s, 15/5.12s, and a targeted revised-fixture suite of 10/20.91s. These counts overlap and must not be added as independent tests. Tests cover local-only controller inputs, owner binding, independent streams, reproducibility, legal actions, stochastic reporting, twelve lawful scenario families, later hearsay versus direct witness, meeting/ghost/task mechanics, frozen coefficients, exact actor/meeting reconstruction and whole-scenario classifier holdout. The accounting test deliberately **reproduces an unresolved historical bug**; a passing test does not mean that bug was fixed.

The eight-scenario comparison completed all **16 planned games** and one additional complete reproducibility execution within **827.68s**, below its 900s wall ceiling. The repeat used the same frozen policy and scenario with extra auditing and recording disabled: exact events, actions and decision counts matched. This is not a new training experiment, independent replication or clean throughput benchmark. Two additional short inference probes took 34.06s and 33.40s. Engineering tests/read-only analysis are separate costs.

The read-only replay verifier passed all 16 completed files: 12,211 living owner-bound neural decisions, 56 eligible ballots, 1,041 adjacent meeting-frame pairs with frozen positions/ghost positions/task progress, zero illegal actions. Replay outcomes and actual published ballots remain separate from reward-ledger counters. The original parser/recorder and approved viewer source remain unchanged.

Specific historical defense/vote/Belief moments and one new mixture replay were rendered in the actual approved Spectator V2 code. Four local images are under `local-runs/behavior-views-v1/`; these are actual application renders from real completed recordings. The matched defense/vote action kinds and actor are checked by the inspection tool. No desktop/window-access or general UI test claim is inferred from those renders. The owner already approved the interface.

## Preservation and privacy

`local-runs/preservation-final-01.json` verifies:

- 1,031 original tracked files unchanged (all original files except the intentionally updated permanent register).
- 155 historical artifact hashes unchanged, including A/B/zero checkpoints and existing results/replays.
- 145 pinned viewer/source/artwork baseline hashes unchanged.
- 13 approved V2 source/test files unchanged.
- Other local branch refs unchanged; no partner-branch edits.
- R001–R020 preserved verbatim. R021 continuity retained; R022–R024 record new findings/limitations.
- All comparison-package source hashes recoverable from local implementation snapshot `91f7692`.

Strict `FourCrewPolicy.load` succeeded for A/B/zero with their original runtime/source contracts; no migration/bypass. Their file and tensor hashes remained unchanged after execution. No existing `crewmate_v2/fourcrew`, `learning`, engine, actor API, belief model, reward, map or historical spectator file was modified. The new package is opt-in and not installed into any training runner. No final-test partition was read, no PPO update ran, no PR was merged, and nothing was pushed.

Curated material contains aggregate counts, figures and sanitized representative prose. Raw observation/memory snapshots, individual role labels, private reconstructions, checkpoints, replay files, screenshots and validation logs stay in ignored local outputs. The metric curation tool removes per-actor/role-record datasets, local paths, target IDs and raw navigation/victim-ID tables. Launch commands use relative workspace paths. Verify the staged file list and path/raw-record scan before committing.

## Analysis corrections retained

1. Initial historical splitter used filename patterns, but both training and validation use `context-seed` names. `historical-social-v1` is preserved as a superseded analysis; v2/v3 use registered seed partitions. No result was silently overwritten.
2. Historical prefix reports are separated in v3. Accepted post-handoff reports, physical range/legality, decision opportunities, race-losing submissions and accepted reports are distinct.
3. The first no-witness probe was not actually weak: its belief was already 99.983%. Its output remains unchanged. A design clarification preceded v2, adding an empty emergency with no kill/body/report/claim. No scenario was selected by confidence or outcome.
4. The original phase2 reporter helper could carry a previous report into a later emergency. A narrow observer-only fix now matches the report tick to the current public meeting; a focused test reproduces the distinction. Every meeting in the completed comparison came from a report, so this did not change its results. Public classifier extraction also clears reporter state on emergency/exploration transitions.
5. No private historical evidence was invented. Claims heard later remain hearsay. No direct-causation claim follows from temporal correlation.

## Changed files

| Files | Purpose |
|---|---|
| `crewmate_v2/phase2/__init__.py`, `opponents.py` | Versioned opt-in mixture and independent local RNG streams |
| `crewmate_v2/phase2/audit.py`, `run.py` | Safe observations, opportunity/accepted-event accounting, strict frozen-policy comparison and stop rules |
| `crewmate_v2/phase2/scenarios.py`, `probe.py` | Legal engineering evidence histories and frozen distribution probes |
| `crewmate_v2/phase2/social.py`, `shortcuts.py` | Exact social reconstruction and public-behavior shortcut diagnostics |
| `tests/test_phase2_opponents.py`, `test_phase2_social.py`, `test_phase2_accounting.py`, `test_phase2_shortcuts.py` | New correctness/negative-result regressions |
| `tools/phase2_behavior_views.py`, `verify_phase2_recordings.py`, `summarize_phase2.py` | Approved-viewer behavioral inspection, read-only replay audit and aggregate-only curation |
| `docs/phase2-opponents/{DESIGN,README,REPORT,VALIDATION}.md` | Design, reproduction, findings, limits and handoff |
| `docs/phase2-opponents/results/{metrics.json,comparison.png,comparison.svg}` | Aggregate-only scientific artifacts |
| `docs/research/ISSUES.md` | Permanent research continuity; preserved prior history and new R022–R024 |

Local commits: `2017419` records the design; `91f7692` preserves the benchmark implementation. The final findings/observational-corrections commit follows these. No remote publication is implied.

## Stop boundary

Review the [recommendation](REPORT.md) before any further experiment. The frozen policies, historical benchmark and registered protocol remain research baselines. No combined model, new reward/architecture, learned impostor, sabotage, larger lobby or self-play work began. **Phase 6.1 acceptance remains pending.**
