# Phase 6 experiment plan — foundations, 10 October 2026

Phase 6 is **in progress**, not complete. This branch implements Phase 6.0 and
prepares Phase 6.1. No learned impostor, alternating learning or self-play exists.
Longer training and opening reserved validation/final partitions require the
owner's next compute approval. Historical Phase 5 final outcomes cannot select
a Phase 6 model.

## Baseline and compatibility

Baseline: `6b78b08e41d0791e49ff486b996d7490666db970`, `origin/main`.
Development branch: `feat/phase6-vision-crewmate-foundation`.
The original checkout's untracked materials are preserved; development uses a
clean managed worktree. The frozen Phase 4 and Phase 5 releases are verified by
SHA-256 in `phase6/config.py` and `docs/benchmarks/phase6/foundation/`.

Three approaches were considered: change global defaults (invalidates prior
evidence), duplicate the simulator (divergent rules), or opt into new rules in
the existing trusted engine. We use the last approach. `GameConfig()` remains
legacy version 1, with 4.5 sight for both roles. Legacy config serialization and
seed-7 actor/truth trajectory hashes remain exact. New rules version 6 carries
both explicit ranges and has replay schema `phase6-replay-v1`. Old config
dictionaries remain accepted by `GameConfig(**record)`.

## Context map and reviewed dependencies

| File | Responsibility / change |
|---|---|
| `phase3_engine.py` | Versioned config, effective role range, projector dispatch, event-time witnesses |
| `social_deduction/phase3_api.py` | Document existing settings as actor-effective distances; packet shape preserved |
| `social_deduction/phase3_observation.py` | Copy effective range into the detached public map view |
| `phase3_runner.py` | Legacy serialization adapter and distinguish new replay schema |
| `phase3_renderer.py` | Role range resolver and wall-clipped spectator outline |
| `phase6/config.py`, `audit.py` | Fresh partitions, frozen models, source hashes and identity diagnostics |
| `phase6/env.py` | Reuse Phase 5 physics; separate controlled option lengths, memory/belief conditions and public route descriptors |
| `phase6/train.py`, `evaluate.py` | Bounded smoke PPO and paired development-only reports |
| `run_phase6.py`, `run_phase5.py` | Reuse existing viewer with an optional environment factory; legacy default preserved |
| `tests/test_phase6_*.py` | Visibility, temporal, counterfactual, compatibility and option composition checks |
| `README.md`, `HANDOFF.md`, `docs/project_architecture.md` | Accurate current milestone and continuation gate |
| `project-status.json`, `site/app.js`, site validator | Existing mission only; verified foundation and planned later checkpoints |

Dependencies reviewed: Phase 1 actor/truth allowlists, Phase 3 bots/observations,
Phase 4 memory/current-memory/predictor, Phase 5 candidate scorer, time-aware GAE,
PPO update and option protocol, static navigation service. Frozen model files,
historical results, map geometry and version-checked Phase 5 sources are not
edited. The new option wrapper composes original base transitions; it does not
duplicate physics or select strategic actions. No database migration is needed.

## Rules and information boundary

Reference values: crew **4.5**, impostor **6.75** native simulator distance units.
These are a hypothesis (1.5× range), not an official game-client conversion.
Both roles use exact native barrier LOS, including walls, opaque props and map
boundary contact. The longer range grants no unseen positions or private tasks.
Kill range remains 1.1, report range 1.3, physical radius 0.22. See
[vision and rules](phase6_vision_and_rules.md).

Identity, role, task and color streams remain domain-separated. The audit checks
1,000 fresh setup draws for public-ID, color and spawn-slot coverage; these counts
are diagnostics, not a statistical proof of independence. Spawn slots are fixed
locations but role selection is independent of slot/identity/color. Opponent
behavior is seeded separately and never passed to the neural packet. Actor
features contain no seed, role labels, truth dictionaries or future outcomes.

## Registered partitions

| Partition | Inclusive seeds | Use |
|---|---|---|
| Training | 1200000–1219999 | New approved Phase 6 learning only |
| Validation | 1220000–1220199 | Screening and checkpoint choice; not opened here |
| Final familiar | 1230000–1230499 | Locked independent evaluation; not opened here |
| Final patient | 1240000–1240499 | Locked held-out opponent family; not opened here |
| Development | 1250000–1250019 | Small reusable debugging diagnostics, never selection |
| Smoke training | 1260000–1260099 | Infrastructure only, never candidate checkpoints |

All ranges are disjoint from recorded Phase 4 and Phase 5 study seeds, including
Phase 5 quick runs. Independent neural initialization seeds are separate from
match seeds. A given comparison pairs the same match seeds, assignments and
opponent streams across methods. Familiar and patient *development* cases reuse
seeds to isolate opponent-family changes; final families have separate ranges.

## Available Phase 6.1 foundations

The shared masked candidate scorer is reused without an architecture change.
Conditions are fully recorded by `phase6.config.Experiment`:

| Factor | Registered alternatives |
|---|---|
| Memory / beliefs | `full`; `history_no_belief`; `current_only` |
| Task/travel hold | 24 or 12 simulated seconds; follow/flee remain 3 |
| Distance descriptor | Euclidean or public static route length |
| Learning rate | 0.0003 or 0.0001 |
| Rollout | 256 or 512 decisions |

`history_no_belief` preserves historical policy features but sets belief
probability/logit channels to uniform/zero. `current_only` also removes historical
events, retaining current public meeting context and newly delivered legitimate
events. Thus history and learned suspicion can be studied separately. Candidate
generation still derives only from the selected legitimate memory condition;
removing history necessarily removes historical-sighting claim options.

Route descriptors query a trusted **static** service with own/public/visible
coordinates. Failed public routes encode maximum distance without changing masks.
No unobserved player target or dynamic hidden truth is queried. The slot's meaning
is recorded in the experiment, and checkpoints require matching source hashes.

Options stop on completion, phase change, newly visible bodies or witnessed
eliminations; no heuristic vote/report threshold is added. Every neural goal is
selected by the policy. Eliminated focal actors settle the team outcome without
ghost observations. Time-aware discounts and GAE preserve simulated durations.

## Development commands

Use the existing environment's Python from the repo root:

```powershell
python -m phase6.audit
python run_phase6.py
python -m phase6.train --config artifacts/phase6/configs/reference.json --steps 256 --seed 17 --output artifacts/phase6/smoke/reference-17
python -m phase6.evaluate --config artifacts/phase6/configs/reference.json --matches 2 --output artifacts/phase6/smoke/development
```

Use a new output directory each time. Configs can be written from
`dataclasses.replace(Experiment(), memory='history_no_belief').to_dict()` or the
other registered alternatives. The smoke trainer rejects more than 512 decisions.
The evaluator exposes only the development partition (maximum 20 per family).
Neither command can open reserved validation/final seeds. A later reviewed batch
runner is required after explicit approval; this limitation is intentional.

Outputs include complete config, baseline SHA, normalized source hashes,
checkpoint/frozen-release digests, match seeds, decisions, simulated time, option
termination reasons, PPO losses, entropy, KL, clip fraction, gradient norm and
explained variance. Evaluations report task contribution, survival, reports,
correct/incorrect votes, skips, ejections, navigation/illegal failures and match
outcomes, with paired whole-match bootstrap and Wilson intervals reused from
Phase 5. Development intervals are diagnostics; tiny checks do not establish
strategic improvement. Scripted impostor wins/eliminations are logged, not
misrepresented as learned-impostor results.

## Proposed first approved batch — isolate memory and learned suspicion

Nine fresh models: three conditions (`full`, `history_no_belief`, `current_only`)
× initialization seeds 17, 29, 43. Each: 8,192 decisions, Euclidean descriptor,
24-second hold, lr 0.0003, rollout 256, identical asymmetric rules/opponents.
Total 73,728 decisions. Change one major factor at a time; option length, route
distance and PPO tuning follow only if diagnostics warrant them.

Screen four saved checkpoints/run on 20 reserved validation matches (720 matches).
Evaluate each run's validation-selected finalist on all 200 validation matches
(1,800 matches), plus idle/random/scripted/frozen Phase 5 transfer controls on
those same 200 seeds (800 matches). Total validation work: 3,320 matches. Screening
reuses a prefix of validation; it is not independent evidence. Report seed spread,
vote participation and individual tasks, not just the best network's win rate.

Provisional budget based on Phase 5 and new smoke checks: about 30–50 minutes/run, or 4.5–7.5 hours
serial training. Training uses one process initially; validation uses at most two
processes with disjoint output jobs. At the recorded Phase 5 evaluation throughput
(9,000 matches / ~185 min using four workers), 3,320 matches with two workers may
take roughly 2.3 hours, with allowance for different hardware/load. Allow
**8–11 hours total**, subject to new-rule timings.
The development host is an i7-13700HX (16 physical cores / 24 logical processors,
about 32 GB RAM), different from the historical Phase 5 host. New smoke timings
include concurrent regression workload and are provisional, not a throughput guarantee.
Do not interpret CPU-worker time as elapsed time. Simulated exposure must be
measured; provisionally allow 4–12 simulated hours/run based on Phase 5 match
durations, then revise this estimate from the development smoke timings. The
nominal cap sums to 54.6 hours/run (8,192 × 24 seconds); it is not a hard bound,
because dead-actor outcome settlement can extend an option. Log actual exposure
and durations; decision counts alone do not equal an experience budget.
Compact checkpoint weights are about 0.2 MB each; four checkpoints plus final
weights × nine models are roughly 9 MB, plus logs/results/figures (budget 100 MB).
Outputs: `artifacts/phase6/runs/memory-<condition>-<seed>/` and
`docs/benchmarks/phase6/phase6_1/` for curated reports.

Stop on nonfinite optimization, frozen-model change, runtime hash mismatch,
illegal actions, unexplained navigation failures or partition overlap. Inspect
learning plateaus/entropy before extending budgets. Preserve checkpoints if
evaluation/plotting fails. This proposal is **not authorization to execute**.
Final evaluation gets a separate approval with a locked selection manifest
(checkpoint SHA, complete config, opponent hashes, partitions, selection criteria
and source hashes). Lock before opening tests; no reselection from final outcomes.

## Acceptance gates and remaining sequence

6.0: role-specific LOS and event-time evidence tests pass; legacy seed-7 replay
hashes match; historical models load; all prior regressions pass; frozen files and
historical outputs are unchanged; diagnostics produce reproducible safe records.

6.1 proposed acceptance: useful own-task contribution (>0.5 tasks/match), positive
paired win differences over idle/random on fresh final seeds, competitive scripted
performance (predeclared noninferiority margin −5 percentage points; paired 95%
lower bound above that margin), zero illegal/navigation failures in required
evaluation, and no leakage regression. Report all three seeds and uncertainty;
large seed sensitivity or unreliable voting requires diagnosis. Near-zero vote
coverage cannot support a voting-accuracy claim. Evidence availability, mistaken
votes/ejections and unfamiliar behavior robustness are required behavioral
diagnostics. The patient family is an existing unseen-to-training script family,
not novel adversarial generalization. Before broader robustness claims, register
additional held-out behavior scenarios without reusing historical final tests.
These proposed thresholds need approval with the batch protocol.

6.2: one learned impostor against frozen crew, with a separately reviewed actor
contract and legal supported kill/fake-task/claim choices. No sabotage/vent claims.
6.3: alternate frozen-role updates, retain historical opponents, evaluate cross-play
and forgetting. 6.4: increase learned crew count gradually and then investigate
multi-agent adaptation/self-play. Each milestone requires its own review and
compute gate; no arbitrary 100% win requirement applies.
