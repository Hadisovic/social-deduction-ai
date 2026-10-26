# Development-only exploratory impostor identification diagnostic

User-authorized observational diagnostic, 2026-10-26. Phase 6.1 acceptance remains pending.

Exactly nine validation-selected policies, each on development seeds 1250000–1250019:
180 executions, 20 paired scenarios. No training, checkpoint selection, final familiar/patient
evaluation, gameplay/reward change, or corrective intervention. The frozen Phase 4 predictor
is ONE shared model, not nine independently learned belief models.

## Implementation and publication boundary

The recorder and offline scoring tools were implemented outside the compatibility-hashed
scientific packages. They, their tests and the detailed recordings remain in the unchanged
local research archive. This sanitized publication includes only this protocol, the report,
aggregate metrics/tables, aggregate figures and sparse anonymized examples. It does not
include complete raw observations, per-game role-label datasets, detailed vote/event records
or local path/runtime manifests. No scientific runtime source, public API or configuration
was changed for this diagnostic or publication.

Read-only scientific dependencies include the original strict Phase 6 loader/environment,
actor-only belief memory/features/model, frozen Phase 4 metric definitions, locked policy
selection manifests and nine selected checkpoints. Results are exploratory development
evidence only; this is not a final-test study or Phase 6.1 acceptance decision.

Seed 17 uses the preserved compatible pilot runtime; seeds 29/43 use the continuation
runtime. The existing strict loader, frozen-file hashes, dependency versions, condition,
initialization seed and checkpoint hash must all pass before ANY of the 180 games starts.
No mismatch bypass or metadata rewrite. Output files are exclusive; reruns do not overwrite.

Record every live, nonterminal focal observation (0.2 simulated seconds), including four
candidate probabilities and the exact candidate feature matrices. Record action-boundary
snapshots separately. Death/terminal observations are excluded. Four original candidates
remain in the distribution, including publicly dead/ejected players, as in Phase 4.

Primary view: predictor on the policy's existing native memory. Secondary view: the SAME
predictor on an independent persistent actor-only memory, observed on the SAME trajectory.
The latter never feeds the policy. Full/history native and observer features must be equal.
Only full policies receive belief probability/logit channels; both ablations receive uniform
probabilities and zero logits. Current-only memory drops history; its observer is not a fix.

All actor-side recording accepts only immutable focal observations and memory. Hidden roles
are exported AFTER each game into a separate truth file and joined only by offline analysis.
The recorder does not receive a game/world reference. No post-death or terminal evidence is
used. Record direct/public/claim provenance; claims are not presumed true.

## Prespecified summaries

- Main sample: one snapshot immediately before each live policy decision, equal weight
  per model-game, then equal weight across models. Tick trajectories are secondary.
- Top-1 accuracy shares credit among exact ties (atol 1e-12), plus deterministic argmax
  sensitivity; NLL uses natural logs and 1e-15 floor; multiclass Brier sums four squared errors.
  Uniform references: 25%, ln(4), 0.75. Scores reuse frozen Phase 4 metric definitions.
- Ten fixed top-confidence calibration bins; report ECE and reliability counts. No fitting
  or recalibration. High-confidence incorrect = max probability >=0.8 and true candidate
  outside the tied top set. Report >=0.9 separately and distinct games as well as snapshots.
- Roaming versus discussion/voting; initial and last-live snapshots; voting opportunities
  versus actual cast/skip events; voted-target probability and agreement with top candidate.
- Meaningful evidence is an operational exposure marker, not proof: first body sighting,
  public body report, witnessed elimination, a claim about elimination/near-body, or a
  directly contradicted claim. Distinguish these types, including direct identity-bearing
  evidence, and compare before versus after first exposure only in games with both states.
  Also report immediate before/after event changes. These are descriptive, not causal.
- Resample 20 paired scenario IDs with replacement (5000 draws, seed 606), keeping all
  nine trajectories together. Never treat 180 executions or many ticks as independent
  scenarios. Condition contrasts average the three initialization runs within each seed.
- Representative examples chosen after scoring for clear success, failure/overconfidence,
  confident skip, and memory loss; examples are illustrative, not additional experiments.

## Integrity and limits

Use deterministic CPU inference, one torch thread, independent match RNGs, unchanged greedy
actions and existing transition guards. Check packet immutability around recording; exact
predictor replay from stored feature matrices and immutable network states; input-channel
checks; checkpoint/source hashes before/after. Tests use synthetic actor observations and
do not execute extra development/final games. Log action trace hashes for all 180 games.
Stop on invalid partition, source/checkpoint/runtime mismatch, nonfinite value, illegal action,
navigation failure, mutated weights, duplicate seed, existing output or execution failure.
An operational two-hour deadline prevents runaway diagnostic work (not a new training budget).

This small development set can localize failure modes, not accept Phase 6.1 or estimate
final generalization. Policies change the observations encountered; condition differences
therefore mix action trajectories and memory availability. No fixes are implemented.

Analysis clarification added before inspecting diagnostic scores: also compare against a
uniform distribution over candidates not identified as body victims by legitimate retained
memory. Only `body_known` features enter this context baseline; hidden roles never define
its candidate set. This distinguishes identifying the impostor from merely excluding known
victims. It is a secondary baseline, not a candidate mask or modification to the predictor.

Execution audit clarification: the existing simulator also records automatic skip ballots
when a participant submits nothing by the deadline. Offline scoring distinguishes explicit
policy skips from these timeout defaults, using the last live voting opportunity for the
latter's suspicion snapshot. Cast votes must match an explicit preceding policy selection.
Self-votes are legal in the unchanged simulator; their target is outside the four non-self
belief candidates, so the focal actor's legitimately known crew role gives target probability
zero. No voting rules are changed.
