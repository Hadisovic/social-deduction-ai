# Phase 2.1 validation and handoff

Isolated branch `codex/phase21-meetings-evidence`, base `8c6592f`, worktree `_phase21_meetings`. No production integration, training, push or merge.

## Test and execution evidence

Final suite: **41 passed in 137.51s**, zero failures/skips. Local JUnit: `local-runs/phase21-final-tests-01.xml`.

```powershell
& ../.venv/Scripts/python.exe -m pytest `
  tests/test_phase21_rewards.py tests/test_phase21_social.py `
  tests/test_phase21_priority_edges.py tests/test_phase21_runtime.py `
  tests/test_v2_mechanics_repairs.py tests/test_v2_fourcrew_rewards.py `
  -q --basetemp local-runs/new-phase21-test-temp `
  --junitxml local-runs/new-phase21-tests.xml --tb=short
```

Coverage: same-step ejection/terminal ballot credit, explicit/automatic skips, invalid/dead/duplicate events, unchanged coefficients, listener ownership, independent memories, exact single-claim percentage shift, clipping/normalization, repeated/conflicting claims, ghost/own-speaker exclusion, direct-evidence protection, manipulation threshold crossing, deterministic/frozen order and its private-score boundary, legal visibility geometry, witness reporting/approach/races/death, raw versus effective distributions, goal retention, disabled-bridge equivalence and unchanged policy tensors/actor packet keys. Existing meeting/ghost/task/reward regressions remain covered. Ten-second unit matches are correctness fixtures, not runtime extrapolations.

Main `local-runs/phase21-comparison-v1/execution.json` records all 36 social segments, 14 witness segments and 12 complete opponent matches, no failures, **949.68s**, below registered overall/section ceilings. A complete opponent-v2 repeat without extra opportunity auditing/recording took **41.37s** and reproduced exact events, actor summaries and decision counts. Policy hashes stayed unchanged. These are engineering simulations/inference, not PPO updates.

All **62** result summaries agree with actual correct/wrong/explicit+automatic skip ballots. The unchanged parser and V2 library accepted **47 genuinely completed recordings**; **15 partial `.writing` files** remain hidden. Complete recordings contain 10,729 living neural decisions, 180 eligible submitted ballots and 2,972 adjacent meeting-frame pairs with frozen positions, ghost positions and task progress. Illegal action counts are zero. Partial segments are not counted as wins/losses or complete replays.

Two specific actual hybrid behavioral moments were rendered using unchanged approved V2 code and inspected: `local-runs/phase21-behavior-views/{hybrid-report,hybrid-meeting}.png`. This checks new data/behavior, not the general UI again. The 100% report probability is an explicit forced legal-report distribution; raw NN probabilities remain separately logged. The Skeld map, sprites and viewer source are unchanged.

## Versioning, failures and corrections

- `1af7769`: design and bounded comparison registration.
- `0cae29e`: exact main comparison implementation, version `crewmate-phase21-v1`; local protocol records every package hash and actor runtime/checkpoint hashes.
- Final code is `crewmate-phase21-v1.1`. A priority edge test initially failed because an erroneous frozen posterior could mask a true directly witnessed killer despite the bonus. The precedence repair admits authenticated supported targets, rejects contradictory votes and preserves abstention. All **278 actual recorded meeting masks/bonuses** remain identical under this fix; the newly exercised adversarial path is separately tested. Original v1 data is not overwritten or recomputed as a different experiment.
- Naive prefix deepcopy failed on immutable mapping proxies and hash objects. The final explicit clone shares only immutable geometry/ownership/frozen models and copies hash accumulators; game state, RNG, private memories and ledgers are separate. Prefix event/memory hashes remain unchanged after every mode.
- The first wall fixture failed: intended destinations were occluded but navigation arrival tolerance made actual positions visible. Robust endpoint selection and actual arrived geometry checks repair the fixture using ordinary legal movement. The failed output remains local.
- One incomplete test mock lacked the standard `participants` field; it was corrected to match the immutable actor API. No gameplay rule was changed for the mock.
- Initial engineering clustering aggregation merged identical model labels across opponent arms. The corrected analysis uses explicit opponent keys. Initial outputs remain under `historical-analysis-v1`; final aggregates use v2. Registered A/B/zero aggregates were not affected.

No parameter was selected by seeking better outcomes.10-point influence's zero posterior/NLL failure, confidence-order leakage, inconclusive small classifier results and fewer v2 meetings remain reported.

## Preservation

`local-runs/phase21-preservation.json` verifies zero mismatches:

| Scope | Files |
|---|---:|
| Existing tracked files excluding intentionally updated register | 1,053 |
| Historical artifacts/checkpoints | 155 |
| Pinned viewer/source/artwork baseline | 145 |
| Approved Spectator V2 source/tests | 13 |

Other branch/tag/remote refs are unchanged. R001–R024 are preserved verbatim; R025–R029 add this phase's history. Main comparison source hashes recover exactly at `0cae29e`. Old canonical engine/API, Phase 2 scripts, actor architecture, frozen belief model, reward coefficients and source-compatible checkpoint loaders are unchanged. Strict A/B/zero loads, file/tensor hashes and the full repeat passed. No reserved final partition was used.

Raw replays/observations/role labels, private decision vectors, per-participant confidences, logs, initial failed analyses and screenshots remain in ignored local outputs. Curated JSON/figures contain aggregate measures, not raw observation or individual role-label datasets. New documentation uses relative reproduction paths. A final privacy/path scan and staged-scope check exclude local home paths, checkpoints, `.jsonl` and `local-runs` from the commit.

## Exact changed files

| Files | Purpose |
|---|---|
| `crewmate_v2/phase21/__init__.py`, `config.py` | Explicit runtime version and validated configurable features |
| `crewmate_v2/phase21/rewards.py` | Separate eligible ballot ledger, old coefficients unchanged |
| `crewmate_v2/phase21/social.py` | Private influence state, probability normalization, admissibility/priority and confidence ordering |
| `crewmate_v2/phase21/runtime.py` | Frozen-policy evaluation bridge, held-goal interrupts, ordered meetings, segments and raw/effective decision logs |
| `crewmate_v2/phase21/opponents.py` | Opt-in coherent mixture v2 and labelled script confidence proxy |
| `crewmate_v2/phase21/scenarios.py` | Legal visibility/evidence histories and safe matched clones |
| `crewmate_v2/phase21/diagnostics.py` | Optional private recorder events, ballot/claim/calibration/influence summaries |
| `crewmate_v2/phase21/analysis.py` | Read-only historical witness, proximity, action and travel proxies |
| `crewmate_v2/phase21/run.py` | Exclusive-output, bounded engineering comparison entry point; no training |
| `tests/test_phase21_rewards.py`, `test_phase21_social.py`, `test_phase21_runtime.py`, `test_phase21_priority_edges.py` | New unit/integration/regression tests |
| `tools/summarize_phase21.py`, `phase21_behavior_view.py` | Aggregate-only figures/data and narrow approved-viewer behavioral inspection |
| `docs/phase21-meetings/{DESIGN,README,REPORT,VALIDATION}.md` | Design, reproduction, findings and verification |
| `docs/phase21-meetings/results/{metrics.json,comparison.png,comparison.svg}` | Aggregate-only research artifacts |
| `docs/research/ISSUES.md` | Permanent history; R001–R024 preserved and R025–R029 added |

No existing training entry point was modified and no new checkpoint was saved. New learning would require separate owner approval, a frozen new source/checkpoint contract and registered development partitions/exposure/cost/selection rules. The old training baseline deliberately retains its historical accounting implementation. Stop for owner review; **Phase 6.1 acceptance remains pending**.
