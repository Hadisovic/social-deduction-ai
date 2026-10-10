# Continuation handoff

## CURRENT — clean Phase 6 documentation publication (2026-10-11)

Read `docs/phase6_roadmap.md`, `docs/phase6_experiment_plan.md` and the frozen
`docs/phase6_1_staged_protocol.md`. Phase 6 is in progress, not complete.
This publication branch `codex/phase6-docs-main` starts from fetched main
`6b78b08e41d0791e49ff486b996d7490666db970`; it selectively applies documentation
and website status/copy. It does not import PR #3/#4 development commits.

1. 6.1: finish full / history_no_belief / current_only at seeds 17/29/43,
   8192 decisions each, with the registered protocol unchanged. Three seed-17
   runs are complete on PR #4; six remain gated. Final partitions remain closed.
2. 6.2: train/validate a learned impostor against frozen learned/scripted crew.
3. 6.3: expand gradually to fully AI-controlled matches with exactly two
   impostors: 6 crew + 2, **8 crew + 2 (10 players; main target)**, then 10 crew + 2.
4. 6.4: alternating training, historical opponent pools, adaptation and eventual
   self-play after policy/population gates.

Plan the local live dashboard (progress/rewards/wins/tasks/votes/graphs/CPU/RAM/
ETA), compatible checkpoint/replay Skeld viewer and isolated one/two-process
benchmarks, optionally three if faster/reproducible. None is implemented here.
Keep the current study serial. Preserve private observations, independent agent
memory, individual task ownership, task/vote mechanics, historical releases and
the branch study's crew/impostor vision 4.5/6.75. Main still uses legacy vision.

[PR #3](https://github.com/Hadisovic/social-deduction-ai/pull/3) remains open at
`16101131e9787c9bcbaebe250d25d6dd0d3574fb`; [PR #4](https://github.com/Hadisovic/social-deduction-ai/pull/4)
remains open at `8187d51b6bbeb506fd2a59f25504adb368ab5006`. Do not merge them under
this documentation authorization. Original stacked [PR #5](https://github.com/Hadisovic/social-deduction-ai/pull/5)
remains preserved at `85d4b1e0983ccc61a8c8808939b8b1d13cd762c5`; merging it directly
to main would inherit 151 development/artifact file changes. Use the clean main
review instead. Pilot evidence/checkpoints remain on #4 and the original worktree;
do not rerun seed 17 or rewrite model/source metadata. The main test badge and
historical benchmark files retain their existing Phase 5 meaning. Entries below
are historical; their older current/future headings do not supersede this plan.

Publication checks passed locally: JSON/Markdown links, documentation-only diff,
source/built site, JavaScript syntax, sprite palettes and desktop/mobile browser
checks. All 419 protected files in the original research worktree remain
byte-identical. No training, model inference matches or final tests ran.

## CURRENT documentation update (2026-10-04)

The owner now explicitly wants the teammate document in the repository,
superseding earlier external-only instructions. The canonical concise guide is
docs/phase5_teammate_guide.md, linked from README, with repo-relative pictures and
graph links. It explains the Phase5 fixed focal belief panel (no switch key;
F is Phase4 only), results/limits, experiment settings/timings and a proposed
Phase6 tuning and multi-agent sequence. No policy, training or viewer behavior
is changed by this documentation update. The external download mirrors the guide.

## CURRENT: Phase 5 completed and published (2026-10-04)

All approved Phase5 work is complete. PR #2 merged at commit
4212d05e4b282cc07d0546f8508ba1066eb1cb7e. GitHub Pages build/deploy run
37165652825 succeeded. Live site: https://hadisovic.github.io/social-deduction-ai/.
The published mission shows88.4%/90.4% crew wins with twelve evidence figures;
Phase6 remains future work. Red/black collaborator corrections are preserved.
Local main contains the merged release.372 tests passed; desktop/mobile browser,
palette, source/build validators and release viewer launch were checked.

The external Phase_5_Teammate_Handoff.md and Phase_5_Teammate_Package.zip remain
outside this repository. They include all experiment stages, counts, settings,
timings with caveats, Intel i9-14900HX hardware and worker details, result tables,
twelve graphs, learning assessment and the gated Phase6 sequence. Evaluation took
about3h05m elapsed. No further training or final-test selection changes occurred.
The completion heartbeat is paused after live verification. Historical entries
below describe earlier states and do not authorize restarting finished workers.

## CURRENT: final results complete; publication being verified (2026-10-04)

Session65107 exited successfully: 9,000 final matches, 90/90 chunks, nine methods.
Locked seed17/8192 wins 88.4% ID and 90.4% patient; own tasks1.67/1.68; vote
accuracy94.2%/95.7%, cast486/484. Clear gains versus idle/random, no established
superiority over scripted. No final-test selection changes. Phase5 simulator
study is accepted; real-game transfer and replicated memory benefit are unproven.
Release copied byte-for-byte to artifacts/phase5/release/policy.pt with SHA256
metadata. run_phase5.py defaults to it; dummy-video launch passed.

372 Python tests passed (203.73s, three warnings). Twelve report/public graphs
generated; Node source/build validators, palette tests and syntax passed.
External teammate document now has full ledger/timings/hardware and all result
tables. Phase_5_Teammate_Package.zip beside it includes twelve graph PNGs and
execution-metadata.json. Training sum129.4min (overlapping runs); final evaluation
about185.0min elapsed,721.8 summed worker-job minutes. Overall substantive study
span about4.67hours including reviews/gaps. Stage spans are filesystem estimates.

Remaining: finish browser verification at desktop/mobile, review/stage/commit
authorized changes, PR checks/merge and verify live GitHub Pages deployment.
No live publication claimed yet. Preserve red/black palette/animation fixes.
Automation must remain active until publication and deliverables are verified.

## CURRENT: locked final evaluation running (2026-10-04)

**Latest owner documentation requirement:** after completion, expand the external
teammate document with ALL training, validation, testing and evaluation details.
Include seeds, steps, checkpoint selection, data partitions, methods/controls,
metrics and uncertainty, measured duration per run/stage and total elapsed time,
plus processor/hardware and process/worker setup. Distinguish summed worker/run
time from elapsed wall time when jobs overlap. Recover timings from persisted
records where possible; label estimates and unavailable timings explicitly.
Explain the experimental process and the final learning-sufficiency decision.

Validation session22518 completed successfully. `phase5_analysis.py` was run
without --lock, the curves/paired comparisons/tasks/votes were inspected, and
`selection-decision.md` records the decision before final testing. No additional
training is justified in this batch. Selected full-memory seed17 checkpoint8192:
96/100 validation wins, 1.69 own tasks, 93/93 non-skip votes correct. Scripted92%,
idle50%, random42%. Paired versus scripted +4 points [0,+9]: no clear superiority.
Three full-memory seeds demonstrate useful task behavior; one ablation is exploratory.

`python phase5_analysis.py --lock` created the immutable selection lock. DO NOT
rerun selection, train/tune using final outcomes, or change the selected checkpoint.
The checkpoint curve plot now uses the SAME20 diagnostic seeds for every point;
the larger-sample comparison remains separate. No training/evaluation code changed.

**RUNNING session65107:** `python -u tools/evaluate_phase5_locked.py`, escalated
after Windows blocked process-pool pipes. Four local CPU workers evaluate NINE
methods x1,000 matches =9,000 matches, half ID/half patient. Preserve this healthy
job and inspect persisted `expanded/final-*/matches.jsonl` progress. Do not modify
source modules loaded by its workers or start duplicate evaluation jobs. Selection
is locked and the final test partition is now open. Partial outcomes are not final.

After completion inspect `final-results.json`, refresh report/figures and the
external teammate file, and assess sufficiency honestly. The report's pre-final
status detection only notices completed chunk metrics; if refreshing it mid-run,
correct any stale "unopened" wording in the documents (do not alter worker code).
After workers exit, improve that detection if needed. Update the teammate status
header and any stale pending-validation prose as well as its measured block.

Then export final graphs/story with `tools/export_phase5_site.py`, verify the
website and finish the existing GitHub Pages publication flow. Also integrate
the locked selected checkpoint as the viewer's default release after final review;
it currently defaults to the earlier smoke policy. Preserve all other artifacts.
The teammate document stays OUTSIDE the repository at the path below and includes
the gated Phase6 list. No final results or live publication have been claimed.
Keep the heartbeat active until all requested work is actually done.

## Expanded training authorization (2026-10-03, CURRENT)

**Current execution status:** all five substantive runs are complete: preserved
v1 seed 17 at 32,768 decisions, and V2 seeds 17/29/43 plus current-only seed 17 at
8,192 decisions each. V2 runs took 22.7–23.4 minutes apiece (replicates overlapped).
All four V2 checkpoint series have 20-match diagnostic validation. Seed 17 reached
100% wins at 6,144 and 8,192 decisions; other full seeds reached 95%. These are
reused small validation sets, not final evidence. Untrained initialization controls
for seeds 17/29/43 scored 50%/35%/50% wins and zero own tasks on those 20 seeds.

**RUNNING:** `tools/validate_phase5_replicates.py` session 22518 (escalated because
Windows denied process-pool pipes in the sandbox). It now evaluates 100 validation
matches for each screened full/current checkpoint, v1, and random/scripted/idle.
Its earlier checkpoint jobs are finished. Do not start another process against
the same evaluation directories. Training sessions and diagnostic/initial-control
workers have completed successfully. Preview server remains session 88477, port4175.

**Next sequence:** after session22518 completes, run `python phase5_analysis.py`
and inspect `validation-review.json`, curves and behavior. Decide from validation
whether a further training experiment is justified; do not blindly add compute
after a plateau. If selection is ready, run `python phase5_analysis.py --lock`.
Then `python -u tools/evaluate_phase5_locked.py` runs 1,000 final matches per method
(500 ID/500 patient), using four workers and disjoint 100-match chunks. This new
script may also need escalation for Windows process-pool pipes. It refuses missing
selection locks or changed checkpoint digests and checks the complete seed manifest.
It does NOT automatically start further learning. Final outcomes cannot tune selection.

After final results, regenerate `python tools/update_phase5_teammate.py "<external
teammate document path shown below>"`, inspect
the figures and write an honest learning-sufficiency assessment. The current-only
ablation has ONE seed; do not claim replicated memory benefit. Run
`python tools/export_phase5_site.py` to copy sanitized final graphs and populate
`project-status.json.phase5Study`. This exporter refuses unfinished final testing.
It does not publish. Website runtime/validator already support Phase 5 in-progress
and an expandable evidence gallery; the draft contains no old UI repair history.
Build/validate, check the browser, then finish the existing GitHub publication flow.

Current report: `docs/benchmarks/phase5/expanded/README.md` (nine figures before
final tests, twelve afterward). Full Python suite passed365 tests; seven additional
chunk/whole-match-statistics checks passed afterward. Node status/assets/palette
checks passed. The external teammate document is refreshed by the helper above;
it contains the requested gated Phase 6 list and current measured results.

The teammate helper requires the existing external document as a positional
argument; no machine-specific output path is embedded in the helper source.

The owner's latest request supersedes the earlier one-run limit: train further
as evidence warrants, create meaningful NN evaluation graphs, update the external
teammate document, and add a suggested Phase 6 list. Multi-agent learning/self-play
must wait until one learned crewmate works. No multi-agent implementation is authorized.

Preserve baseline `artifacts/phase5/runs/full-17-approved` (32,768 decisions).
The automatic `tools/complete_phase5_single.py` evaluator was STOPPED before any
validation/final matches; do not restart it while model selection is ongoing.
It would prematurely reveal the final holdout.

Baseline training showed no own tasks through 269 completed matches. A separate
`phase5_options.py` experiment preserves v1 and runs chosen travel/task actions
up to 24s (follow/flee up to 3s), interrupting on actor-visible events or completion.
It composes original physics/rewards/discounts and never chooses a social action.
An equivalence test passed. Seed-17 options training is running with 8,192 decisions
in `artifacts/phase5/runs/options-17-v2`, checkpoints every 2,048 decisions.
`phase5_options_training.py` records policy/value losses, entropy, KL, clipping,
explained variance, decisions and simulated seconds. Different option lengths
mean equal decision counts are not matched simulation budgets.

Next: validate before extending, replicate a useful approach on seeds 29/43,
compare a separately trained current-only ablation, lock selection on validation,
then open final ID/patient tests. Do not tune using final outcomes. Baselines
include idle to expose wins carried by scripted teammates. The updated evaluator
and viewer dispatch by checkpoint execution protocol and check its source hash.

The downloadable handoff remains outside the repository at
`C:\Users\masaa\.codex\visualizations\2026\10\02\01a0fc9e-88fa-7ce2-aa36-8902b0b5fd96\Phase_5_Teammate_Handoff.md`.
Its Phase 6 suggestions are explicitly gated. Update measured results after runs.
Heartbeat `finish-phase-5-training-results-and-website` has the expanded prompt.
Keep the public Phase 5 story about learned strategy/results only, not earlier
UI/map/player repairs. Final website implementation/publication is still pending.

## Authorized single-run experiment (2026-10-03, running)

The owner approved **one full-memory run, seed 17, 32,768 transitions**, followed
by graphs, outcome documentation, a judgment about further learning, and a website
Phase 5 update. Do not start the six-run protocol or another training run.
The requested website Phase 5 story must discuss strategy/learning/results only;
keep earlier map/UI/avatar repairs out of that phase's content. The website's two
collaborator characters should use the corrected palettes and walking animation.

Training is `train_phase5.py --steps 32768 --seed 17 --output
artifacts/phase5/runs/full-17-approved`. Progress is persisted in `updates.jsonl`
and `episodes.jsonl`; intermediate weights are saved every 8,192 transitions.
`tools/complete_phase5_single.py` waits for completion, evaluates 100 validation
matches per method, checks three intermediate checkpoints on the same 20 validation
seeds, records the learning assessment, then evaluates 200 final matches per method
(100 ID/100 patient). Methods are learned, original scripted and random-legal.
It generates six scientific figures and `docs/benchmarks/phase5/single-seed-17/README.md`.
Only a single initialization seed is evaluated; do not claim a multi-seed result.

The website character code now decodes the existing RGB frames at runtime and
animates both collaborators. Original PNGs are untouched. Node palette tests and
a local browser travel check passed. The public Phase 5 story/status/validator still
need updating from actual measured outcomes after evaluation completes.

Never rerun training automatically if a later plotting, evaluation or website step
fails. Preserve the checkpoint and repair the later stage. Final website publication
must be verified; do not claim the local preview is the live GitHub Pages deployment.

## Phase 5 implementation update (2026-10-03)

The owner subsequently authorized starting Phase 5 and then requested approval
before any longer training run, with an estimated duration. Two 512-transition
smoke models already exist under `artifacts/phase5/quick/`; no extended training
has run. `python run_phase5.py` loads the smoke focal-crewmate policy.

New Phase 5 modules provide actor-only feature/candidate construction, a masked
shared-candidate actor-critic, frozen Phase 4 inference, time-aware PPO, original
scripted opponents, paired evaluation and a spectator policy panel. The full
runner trains three full-memory and three current-only models, ranks full models
on reserved validation seeds, and only then evaluates final ID/patient seeds.
This is a serial reference implementation; strategic improvement is unproven.

Conservative collision buffers now cover the true radius-0.22 circle. The source
blueprint is byte-identical. Both 4,914-route benchmarks pass at dt=1/30 and .2,
with zero collisions, blocked steps or replans; eight targeted corridor tests pass.
`clearance_policy='legacy_polygon'` preserves historical replay geometry.

The red sprite source was an RGB material mask. Five other colors aliased every
walk frame to a still pose; they now use recolored complete mask animations.
Meeting banners show the public reporter/victim/room or emergency caller.
Phase 5 display interpolation uses recorded physical steps without changing rules.

The original smoke checkpoint predates the final room-action descriptor correction.
It remains a UI/pipeline demo only; fresh training uses public room coordinates and
records the feature-source hash. Fourteen rooms are strategic room destinations;
seven hallway regions remain traversable. Preserve historical smoke reports.
The owner's teammate handoff is outside the repository, as requested.

Final verification: **362 tests passed**, three dependency/Gym warnings;
`docs/benchmarks/phase5/pytest.xml`. The reviewed 16-match smoke evaluation in
`docs/benchmarks/phase5/smoke-reviewed/` recorded zero illegal actions and zero
navigation failures. Headless runs use SDL video/audio dummy drivers.

## Historical rendering repair and Phase 5 specification review (2026-10-03)

Starting review HEAD: `172f648` on `main`. The owner authorized rendering repairs
and an expanded specification, not Phase 5 training. Current UI changes improve
offline half-body sprites, restrict body circles to debug mode, increase small
avatar visibility, fix selected-monitor sizing and add an actor-only bottom HUD.
The missing pinned local artwork pack was installed (119 verified PNGs, still
ignored). Run `python run_phase4.py`; optional `--display 0 --window-scale 1`.

Validation: **334 tests passed**, including seven new UI checks; three existing
dependency/Gym warnings. Source geometry and Phase 4 weights are unchanged.
`docs/phase5_strategic_policy_specification.md` preserves the pasted original and
appends the corrected implementation checklist. Do not implement the original
43-feature/12-action draft without reading the amendment: live team task totals,
hidden live-player counts, candidate coverage and belief-threshold masks conflict
with the existing contract or learned-decision objective.

`tools/audit_phase5_corridors.py` / `docs/phase5_corridor_audit.json` show eight
successful collision-free runs, but exact circular clearance is slightly violated
by polygon-buffer corner approximation on one O2 route (~0.00018831 units).
The stricter audit exits 1 deliberately. This remains a Phase 5.0 prerequisite;
no corridor widening, player-radius reduction or map edits were made.

## Current state (2026-10-03)

- Repository: `Hadisovic/social-deduction-ai`, branch `main`.
- Phase 1/2/3 remain complete and preserved. Phase 3 baseline revision:
  `e8c5d664a3f17684c3fd844a4ef27ad089b65b12`; Phase 4 milestone commits:
  `68780df` (memory/data) and `e16d8ea` (training/evaluation/observer).
- **Phase 4 is complete** at implementation/results revision
  `fdc9f4401b781f64e3e79a2cd0bb3a80bcb515e8`. This handoff is a documentation-only
  follow-up; resolve its exact revision with `git log -1 --format=%H`.
- Final checks: full pytest suite (327 tests), five independent dataset/feature
  replays, selected-model exact retraining, final evaluation, and live rendered
  matches. The final push was to `origin/main`; see the repository history.

## Phase 4 outcome

The phase adds bounded typed event memory, permutation-safe 22-feature candidate
vectors, a calibrated set-based impostor-belief model, seeded scripted study data,
leakage/identity audits, retrained ablations, evaluation, and an interactive
belief observer. It does not add a strategic policy: task selection, reporting,
accusations and voting remain scripted.

The experiment contains 1,400 complete matches and 64,097 samples, with match-level
train/validation/ID-test/patient-family splits. Final calibrated model accuracy is
70.54% (NLL .5955) on ID and 67.41% (NLL .6375) on the held-out patient family.
These results establish performance only on the controlled five-player scripted
distribution. The current-only ablation was corrected to retain the public live
meeting roster; its NLL is 1.2455 / 1.2546. See
`docs/benchmarks/phase4/README.md` and its linked machine-readable records.

The release checkpoint is `artifacts/phase4/belief.pt`. Run the UI with
`python run_phase4.py`, regenerate data/train with `python train_phase4.py`, and
evaluate with `python evaluate_phase4.py`. Generated shards and run intermediates
are ignored; a fresh training run regenerates them. The small selected checkpoints,
selection record, benchmark report and demo captures are committed.

## Architecture and preserved contracts

`ActorMemory` accepts only typed actor observations and maintains bounded causal
events. `belief/features.py` derives candidate-relative features. `BeliefModel`
returns a distribution over legal crew candidates; it receives no role labels,
truth snapshots, match seed or hidden state. Identity/color permutation and temporal
boundary tests protect the interface. The observer reads terminal own-alive status
only for the UI and does not update model memory from it.

Phase 1/2/3 gameplay, scripts, navigation and map assets were not modified by the
Phase 4 implementation. Phase 3's 291-test contract remains in force. See
`docs/PROJECT_PROGRESS.md`, `docs/project_architecture.md`, `docs/belief_model.md`,
and `docs/phase4_design.md` for project state and technical details.

## Limits and next phase

No Phase 5 policy or win-rate claim is implemented. The next research step should
connect the belief interface to a policy, define a policy-level baseline and
pre-register match-level evaluation before tuning. Current limits include five
players, one impostor, structured claims, a static map, no sabotage/vents/free-form
language, and narrow scripted opponents. Near-perfect late meeting accuracy is a
property of this controlled script distribution, not evidence of robust human
deception reasoning.
