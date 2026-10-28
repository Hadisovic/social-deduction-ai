# Phase 2.1 experiment design and context map — 2026-10-28

Owner-authorized implementation and bounded engineering diagnostics. Base `8c6592f`; branch `codex/phase21-meetings-evidence`. Original A/B/zero, Phase 2 scripts, registered source locks, historical results, approved Spectator V2 and partner work stay unchanged. New code is opt-in under `crewmate_v2/phase21`; no training or reward-coefficient tuning.

## Design and alternatives

Use a versioned pre-step ballot ledger, a frozen-policy decision adapter, private per-listener social states, an explicit meeting-order experiment, witness-priority option and goal-retaining opponent v2. A core rewrite would invalidate historical loaders. Retraining or a new architecture is not authorized. Pure threshold rules replacing the policy would falsely look like learned improvement. The adapter instead records unrestricted neural probabilities, then applies explicit experimental admissibility/priority changes; skip/withhold remain available. Reporting is independent of suspicion thresholds. Forced witness reports and any priority bonuses are labelled interventions, never learned competence.

Raw frozen belief, eligible-candidate normalization, social adjustments and actual action distribution remain separate. Defense≤.30, accusation/vote≥.70, strong≥.85, minimum target margin .10 are configurable hypotheses. No action is forced merely for exceeding a threshold. Own direct evidence blocks contradictory support; legitimate evidence changes can change admissibility. Frozen overconfidence can make filtering worse and is an endpoint, not a reason to tune thresholds during diagnostics.

Social influence magnitudes0/.03/.10 are compared. Only delivered typed claims to a living non-speaker listener count. Per-meeting speaker/target contributions are bounded, duplicates ignored, contradictory repeats cancel and lock; conflicting speakers net out. A single55% target can move to 65% with .10 influence; remaining mass is proportionally redistributed, with clipping and exact normalization. Per-target requested shift is capped .20. Direct-witness evidence is protected by default. Base belief already processes claims, so extra influence risks double-counting; this is explicitly tested/documented. No hidden label determines trust.

Suspicion order ranks each living participant by its own highest estimate over other eligible identities, ties by ID, frozen at meeting start. Exact scores stay private. The actual public speaking order intentionally discloses relative private confidence; this is an experimental rule, not an isolation-preserving default. The Phase 4 memory rejects impostor inputs: do not bypass that contract. Scripted controllers therefore supply a labelled local-evidence proxy; heterogeneous confidence estimators can create an additional role signature. Test and report this limitation. No ordering logic sees a hidden role table.

## Context map

| New file | Responsibility |
|---|---|
| `config.py`, `rewards.py` | Validated feature flags/version identities and pre-step eligible ballot accounting |
| `social.py` | Independent listener state, bounded influence, normalization, evidence-aware admissibility and priority |
| `runtime.py` | Explicit evaluation bridge using strictly loaded frozen weights; same architecture/tensors, new rules/ledger, provenance and raw/effective telemetry |
| `opponents.py` | Goal-retaining mixture v2, public speech-format overlap, actor-local pursuit/report decisions |
| `scenarios.py` | Legal engineering evidence/visibility prefixes and matched cloning with independent private state |
| `diagnostics.py`, `run.py` | Bounded social/witness/opponent comparisons; light observational logging, exclusive outputs |
| `analysis.py` | Historical witness/legality investigation, roaming proximity/exploration and aggregate curation |
| `tests/test_phase21_*.py` | Exact rewards, isolation, influence attacks, ordering, witness geometry, cloning, deterministic execution and mechanics regressions |
| `docs/phase21-meetings/`, `docs/research/ISSUES.md` | Specification, measured results, limitations and continuity |

Unchanged dependencies: FourCrewMatch/ActorView/ScenarioMatch, existing rewards/configs, engine, typed actor API, actor memory/frozen belief model, strict policy loader, registered checkpoints and approved recorder/viewer. No public website/main changes.

## Registered development comparisons

Engineering seeds 2760090–2760099 only. No registered train/validation or reserved final partition is opened for new play. Existing recordings are read offline. Scenario labels are constructed evidence contexts, not role inputs.

1. Social matrix: six legal contexts (empty emergency, innocent report, impostor report, direct witness, false/conflicting claims, legitimate changed evidence); unrestricted, guard, guard+.03, guard+.10, guard+.10+confidence order, and guard+witness priority. Compare **one complete meeting** per cell; unresolved matches are explicitly unfinished, not fake wins/losses or completed replays. Prefix state is matched across modes; raw/effective probabilities, belief/calibration, claims/votes, outcomes if resolved, influence and privacy consequences are retained. At most600s wall execution.
2. Witness assay: immediately reportable, outside range, wall-obstructed, spectator-only, multiple/racing witnesses, witness killed before reporting, later unknown-killer discovery. Compare unrestricted and optional witness priority from the same legal prefix. Observe at most one meeting/report or bounded roaming interval; latency and opportunities are primary. At most300s.
3. Opponents: four paired fresh engineering scenarios2760096–2760099 (two ordinary, two deduction natural), original/v1/v2 against frozen B:12 complete matches, unchanged canonical 900s simulator cap, at most900s total wall execution. No context/posterior-based scenario rejection or retries. Preserve partial outputs if the ceiling is reached; do not silently cut coverage. No promotion to default follows automatically. One repeat/noninterference check if time allows.

Overall comparisons have a hard1800s wall ceiling, with independent shorter section ceilings. Tests and read-only historical analysis are separate costs. Prefix cloning is allowed only after proving independent private memories/game state, identical initial event/memory hashes and no stale closures; immutable geometry, role ownership maps and frozen model weights may be shared. Initial naive deepcopy failed on mappingproxy and is preserved as an engineering observation; no historical state is changed.

## Verification and stop

First prove the disabled adapter matches baseline actions/events and reward repair changes only accounting. Then run correctness tests and bounded comparisons. Replays use the approved unchanged recorder; incomplete segments remain incomplete. Optional private sidecars explain raw versus effective decisions without modifying actor inputs. Read-only historical analysis separates actual direct witnesses from spectator proximity, choices from physical opportunities, races/deaths from refusals, roaming from meetings and task totals from efficiency. Update stable research IDs, hash-preserve sources/artifacts/branches, commit locally and stop for owner review. No full PPO training, reward tuning, merges, pushes, sabotage or partner changes.

Self-review: experimental versus learned behavior, new accounting provenance, ordering leakage, normalization, RNG independence, complete-match/segment units and bounded costs are explicit. The brief authorizes execution after correctness checks; no extra design approval is needed.

## Readiness clarification before execution

The real frozen-B empty-meeting smoke completed one discussion/vote/resolution in1.90s and correctly returned an unfinished match segment. No training throughput extrapolation is made. Witness contexts have a prospective25 simulated-second roaming observation window if no meeting occurs; expiry is a partial segment, never a timeout draw or completed match. Social scenarios end after one meeting; natural opponent games retain the canonical 900s simulator cap. Prefix cloning now explicitly shares immutable geometry/ownership/frozen models and copies hash accumulators, while game state, RNG, private memories, ledgers and mutable actor options are independent. The first wall fixture failed because endpoint arrival tolerance changed actual visibility; robust wall selection and actual arrived visibility checks now pass. Both failed readiness approaches remain documented; no model outcome was used to select a fixture.
