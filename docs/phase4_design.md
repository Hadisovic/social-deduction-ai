# Phase 4 design and experiment registration

Recorded before implementation/final-test generation, 2026-10-02. Baseline:
`e8c5d664a3f17684c3fd844a4ef27ad089b65b12`, main, clean, origin synchronized;
291 tests passed (one existing Gym spec warning). The user explicitly delegates
design and implementation decisions while offline. Phase 5 is excluded.

## Boundary and architecture

Actor-facing memory, features and inference import only the canonical Phase 3
actor API, never engine/truth/training/replay code. Trusted experiment orchestration
joins labels from `social_deduction.training.role_labels` AFTER encoding evidence.
The target is the historical impostor among all four non-self public identities;
ejection never masks a candidate. A directly witnessed elimination is legitimate
DIRECT evidence under the existing API; hidden killer metadata is never admitted.

Memory retains canonical events, last sightings, exact observed room/time pairs,
public status, claims and conservative contradictions. Routine sightings become
keyframes; exact room/time pairs support only exact-time contradictions. No hidden
trajectory or travel-time estimation. A conflicting claim implicates its speaker,
not its subject. Hearsay conflict remains weaker than direct observation conflict.
Snapshots are versioned JSON, checkpoints contain tensor state/config only.

Compare a shared linear scorer with a small DeepSets scorer (shared 32-wide
encoder, mean set context, shared head). Both are equivariant to candidate order;
neither has ID/color/slot embeddings. Explicit memory supplies temporal evidence,
so recurrence is unnecessary initially. Attention has no demonstrated advantage
with four candidates; a large sequence/pixel model would obscure the boundary.

## Registered experiment

- Train: 800 matches, seeds 10000..10799.
- Validation: 200 matches, seeds 20000..20199.
- Final ID test: 200 matches, seeds 30000..30199.
- Final strategy test: 200 matches, seeds 40000..40199.
- Train/validation/ID impostor styles: hunter and self_report. Entire patient
  family reserved for strategy test. All three crew and meeting styles mixed.
- Per-controller style draws independent of roles. Small actor-safe scripted claim
  variants may diversify formats/alibis without altering any Phase 3 game rule.
- All four focal crew histories per match stay in the same partition. Samples only
  while focal alive and match nonterminal: t=0, new meaningful events, and 6-second
  exploration checkpoints. No winner, postmortem or outcome-conditioned selection.
- Match-balanced loss/primary metrics prevent long matches dominating. Also retain
  sample counts/stage/identity analyses; uncertainty uses match-group bootstrap.
- Adam cross entropy; learning rate .003, weight decay .001, batch 512, maximum
  160 epochs, early stopping patience 20. CPU deterministic seeds 17,29,43.
- Select architecture by mean validation NLL, then lowest-validation-NLL seed.
  Training-only feature scaling pooled across all candidate rows. Positive scalar
  temperature selected on validation NLL only. Test sets read only after selection.
- Validation learning curves at 200,400,800 training matches, seed 17.
- Baselines: uniform prior and small documented evidence-weight softmax. Report
  both raw and validation-temperature-calibrated rule/model predictions.
- Retrain selected architecture for current-observation-only, no-claims and
  provenance-collapsed feature ablations; same partition/optimization budget.
- Accuracy (including fractional credit for exact ties), categorical NLL,
  multiclass Brier, 10-bin ECE, reliability/confidence/entropy and stage plots.
- Record label/identity randomization, source/config hashes, generation failures,
  family counts, runtime, disk size, sample events and reproducibility metadata.

Final metrics cannot guide tuning. Any follow-up experimental change after viewing
them must be explicitly labeled exploratory and require fresh confirmatory seeds.
Quick smoke runs use separate seed ranges and output directories.

## Implementation and acceptance sequence

1. Memory/features/baselines, JSON roundtrip, temporal/provenance/contradiction tests.
2. Trusted resumable match-sharded generator; smoke check, then full generation.
3. Train comparisons, validation selection/calibration, retrained ablations.
4. Lock selection, evaluate both final sets, scientific plots and evidence probes.
5. Observer-only live demo reusing Phase 3 renderer/scripts, independent truth toggle.
6. Paired hidden-world audit, equivariance, full regressions, actual desktop demo,
   documentation, review, commit/push, clean tree. No learned action selection.

Design references: [Deep Sets](https://arxiv.org/abs/1703.06114) motivates shared
encoding and invariant aggregation; [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html)
motivates held-out scalar temperature calibration. Neither guarantees generalization
to new social strategies. The measured experiment determines the conclusion.
