# Continuation handoff

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
