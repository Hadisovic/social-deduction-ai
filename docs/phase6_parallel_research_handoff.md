# Phase 6 two-track research handoff — 26 October 2026

**Status:** design/handoff only. **No PR merged, no models retrained, no mechanics changed.** This document is a reviewable preparation for separate crewmate and impostor work. It is **not** authorization to begin training, run final evaluations, or change historical models.

## Source of truth / checkout

- Repository: `Hadisovic/social-deduction-ai`
- Research base branch: `codex/phase6-1-seeds29-43` (PR #7)
- **Exact immutable scientific starting commit:** `2baa5dbc1b36b047ed97934d69526573b510c280`
- This handoff document is on a **separate docs-only branch** based on that commit. Neither `main` nor PR #3/#4/#5/#7/#8 is merged or altered by this documentation.
- `main` has the previously published roadmap but **does not yet include** the full Phase 6 experiment runtime/checkpoints. Never use `main` as the frozen opponent's scientific baseline.
- Base ancestry: foundation PR #3 -> seed-17 pilot PR #4 -> seeds-29/43 continuation PR #7; PR #6 (clean roadmap) was merged independently to main. PR #8 is a separate draft for sanitized belief-diagnostic publication.
- A collaborator may start a **new independent feature branch** from the pinned scientific commit. Do not commit experimental modifications onto the preserved PR #7 branch. Use separate folders/branches and review integration later.

## What is actually complete

- Phase 6.0 foundation: role-dependent crewmate/impostor sight (4.5 / 6.75 simulator units), wall-occluded LOS, event-time witnessed-elimination knowledge, map/navigation and original behavior protections, implemented on research branches.
- Phase 6.1: nine selected crewmate policies from three conditions (`full`, `history_no_belief`, `current_only`) × initialization seeds 17/29/43; 8,192 training decisions per run; matched 200-scenario validation per selected policy. **Phase 6.1 is not accepted.**
- Frozen shared Phase 4 belief predictor: `artifacts/phase4/belief.pt`; SHA256 `e931b1fa8a421d9a329a29280d676bf52f793cc8f07c02b8da61f2b3e5e537d1`. Frozen Phase 5 release: `artifacts/phase5/release/policy.pt`; SHA256 `1038054357cf1c2e0372de7e6c4c226aaa73162d667372e1cf876de82aa867b1`.
- Phase 6.1 results and selection locks: `docs/benchmarks/phase6/seed17-stage/selection.json`, `docs/benchmarks/phase6/remaining-six/selection.json`, and the respective `README.md` reports.
- Development-only belief diagnostic: 180 executions on 20 development scenarios; high meeting accuracy is **confounded** by all 173 observed body reports coming from impostors, and learned crew reporting is absent. The documented diagnostic exists in separate PR #8, not in this PR #7-derived checkout. There is no evidence of robust reasoning against reporter-balanced opponents. Voting sometimes ignores near-certain correct belief.
- Final familiar/patient partitions remain unopened. No learned impostor or fully learned team currently exists. The simulator is five players, four crew and one impostor.

## Initial frozen learned-crewmate opponent for impostor research

Use a copied, immutable **validation-selected** checkpoint, not a randomly chosen `policy.pt`, not a smoke checkpoint, and not a newly trained model.

**Default, source-compatible with the continuation:** 
`artifacts/phase6/remaining-six/full-29/checkpoint-8192.pt`
- Selection SHA256: `238904b7fe303b77e885c4cd6fdf4d9d447a62eb5b463436a391c0b4d6e1cb43`
- Condition: full history + frozen belief channels, initialization seed 29, selected at 8,192 decisions.
- Dev/validation performance: 90.0% crew wins across 200 validation matches; 149 correct of 167 votes cast; **zero own body reports**. This is a frozen study model, **not an accepted strong adversary**.
- Secondary varied frozen policy: `artifacts/phase6/remaining-six/full-43/checkpoint-4096.pt`, SHA256 `d3abbdff88743f8b105f630e11c0e50963f38f7b1e992634dd55999087e9c0e3`. Full-43 almost always skips; do not treat that weakness as the full opponent distribution.
- Other selected policies are enumerated in the two selection lock JSON documents. Mix in scripted crews and, later, historical learned crews; do not train only against a single deterministic opponent.

**Compatibility trap:** Seed-17 checkpoints were trained with a preserved pilot-compatible source revision, while seeds 29/43 used the continuation source. The original strict loader checks complete source hashes and environment/runtime versions; do **not** rewrite a checkpoint manifest, disable compatibility guards, or assume seed-17 weights load directly with the seed-29/43 checkout. For seed-17 comparisons, use the preserved matching scientific source or a separately reviewed compatibility migration.

Before any training, verify frozen checkpoint SHA256, load it with the existing `phase6.train.load_policy` guard and a matching dependency environment (Python, Torch, NumPy, Shapely, Gymnasium and thread setting), obtain a legal actor-only packet, and reproduce a deterministic development replay. **These are proposed preflight steps; they have not been newly run by this handoff.**

The legacy Phase 6 source fingerprint includes every `phase6/*.py`, `belief/*.py`, `social_deduction/*.py` and specific core Python/map files. Even **adding a new Python module inside `phase6/`** changes this fingerprint and may invalidate strict loading. Put experimental impostor implementation under a new, separately named package such as `impostor_research/` until an explicitly versioned compatibility strategy is approved. Do not edit hashed core sources solely to make a loader pass. A redesigned runner must preserve actor-only observations.

## Track B — teammate: learned impostor (separate workstream)

- **Only the impostor policy learns.** Crewmate neural parameters (and its frozen belief predictor) are fixed; no optimizer updates to the crewmate weights. Keep per-player memory independent.
- Start in the existing five-player/one-impostor setting against a registered mixture of frozen learned and varied scripted crews. Initially one or more explicitly designated learned crewmates may be used with other scripted actors; existing `phase5_runner.StrategicMatch` supplies only **one learned focal crewmate**, so multiple learned opponents need a new reviewed orchestration layer.
- Audit available role-local impostor observation and action primitives (movement, kill, fake task, report, discussion claims, vote/skip) before designing model inputs and rewards. No invented sabotage, vents, second impostor, unfiltered spectator truth, or free-form human conversation.
- Keep training seed partitions, selection and future final-test gates **separate** from existing Phase 6 crew study, with immutable opponent identifiers/checkpoint digests and independent training outputs. Report impostor eliminations, wins, survival, detection/ejection, reporting, illegal actions, generalization and exploit strategies—not win rate alone.
- The friend's initial opponent is the **current frozen baseline**, without incorporating the new ghost or eyewitness-action changes until later explicitly registered cross-play tests.
- No expensive training or final evaluation is authorized by this handoff.

## Track A — owner: improved crewmate (separate workstream)

These are design decisions/proposals for a **new opt-in ruleset/version**. **None is yet implemented**, and none retroactively changes frozen 6.1 experiments or Track B's initial opponent.

1. **Immutable task ownership:** Each crewmate owns its own task IDs; nobody completes another crewmate's tasks. In the existing simulator, owned interactions are enforced, **but unfinished tasks of dead/ejected players are currently reassigned after publicly established absence** (`phase3_engine._reassign_publicly_absent_tasks`). New rules must suppress this reassignment.
2. **Ghost task participation:** Dead/ejected crewmates keep their own unfinished tasks, can move **through obstacles in a ghost-only movement mode**, and finish their own tasks. Ghosts must be excluded from living players' vision, meetings, accusations, kills and reports; no ghost observation can reveal privileged live positions or grant information back to living agents. Living physics/pathing stay unchanged. A penalty for death already exists in the Phase 5 reward system; do not increase it before controlled ablation.
3. **Optional deduction-only matches:** Instead of an unassigned impossible task, disable the task-win condition for some **randomly designated training episodes** while preserving genuine task assignments and actions. The sampling rate is **not yet agreed** and must be an explicit, versioned research config. Normal matches retain both task- and ejection-based win conditions. Validate that disabling task victory does not merely increase timeouts/task farming.
4. **Direct eyewitness rule:** If a living crewmate **legitimately sees a killing event at event time** (existing `WitnessedElimination` direct evidence), preserve the witnessed killer ID in its private memory and, when legal, prioritize approaching/reporting that victim's body and voting for that killer at the next meeting **if the killer remains a legal target**. This is a proposed **explicit action override/guard**, not a learned deduction result; tag its actions and report assisted-versus-unassisted performance separately. No automatic role detection from hidden truth, last-known positions alone or mere body proximity; reporting on indirect evidence remains a freely learned choice. An immediate REPORT is only possible at legal range; avoid impossible/illegal forced actions.
5. Investigate reward attribution separately. Report and vote choices should remain meaningful; do not assume automatic witness reports prove learned reporting behavior. Correct/incorrect votes, skips, body reports, evidence quality and pre-vote probability calibration should remain primary metrics.
6. Opponent behavior diversity matters. The 180-game diagnostic had a near-perfect shortcut where reports pointed to the impostor. A reporter-balanced development study remains needed.

## Isolation and integration contract

- Branch A and Branch B must not share a mutable checkout, training state, RNG stream, replay file, output directory, or optimizer. Pin the initial opponent by full commit/checkpoint SHA, and keep copies read-only.
- Record branch, commit, source manifest, dependency versions, model hashes, experimental rule version, train/validation seed partitions, opponent mixture, budget, and results for every experiment.
- Keep privileged post-game labels strictly offline for scoring. Do not publish raw diagnostic traces/private role-label bundles without owner authorization.
- Frozen main/Phase 4/Phase 5/Phase 6.1 results must remain reproducible. Never open Phase 6.1 reserved final partitions or use them to tune new rules.
- When each track is ready, run **cross-play** against held-out opponent checkpoints and varied scripts. Only later consider alternating updates, archived opponent pools, and population expansion.
- No automatic merge, production deploy, model rewrite, expensive experiment, or modification to existing research PRs is authorized by this document.

## Entry points / key sources

- `phase3_engine.py`: game mechanics, original task reassignment, termination and event generation.
- `social_deduction/phase3_observation.py`, `social_deduction/phase3_api.py`: legitimate role-local observation, legal action filtering and directly witnessed events.
- `phase3_bots.py`, `phase4_scripts.py`: existing scripted bot behavior and opponent styles.
- `phase5_runner.py`, `phase5_env.py`, `phase5_features.py`, `phase5_policy.py`: single-learned-focal crew orchestration, rewards, action options and network.
- `phase6/config.py`, `phase6/env.py`, `phase6/train.py`: Phase 6 experiment schema, fixed simulator vision, strict checkpoint loader.
- `docs/benchmarks/phase6/remaining-six/README.md`: last complete nine-model validation results.
- `docs/benchmarks/phase6/seed17-stage/README.md`: original pilot and source provenance.
- Draft PR #8: sanitized development belief diagnostic. Its published docs are **not assumed present** in this PR #7-derived checkout.

**Approval gate:** Review these decisions and the teammate's environment specification before implementing modifications or authorizing compute.