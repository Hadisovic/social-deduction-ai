# Phase 4: legitimate event memory and calibrated role beliefs

The question is whether a compact model can estimate the hidden impostor better
than a uniform prior and explicit evidence rules, using one crewmate's legitimate
history. This phase predicts roles; the Phase 3 scripts still choose every action.
The predeclared experiment is in [phase4_design.md](phase4_design.md). Measured
results and plots are in [benchmarks/phase4](benchmarks/phase4/README.md).

## Run and reproduce

```powershell
python run_phase4.py
python train_phase4.py
python evaluate_phase4.py
python train_phase4.py --quick
```

The launchers use the existing project `.venv` when present. Full training generates
or resumes the registered dataset, compares architectures/seeds on validation,
calibrates, retrains ablations, writes compact checkpoints and evaluates final
sets. Quick mode uses separate seed ranges and an ignored output directory; it
cannot overwrite the release model. `--workers 4` limits generation concurrency.
`--data-dir PATH` selects a fresh cache. A mismatched source/schema/config cache
fails clearly. No data download, GPU or scipy is required. Install dependencies
from `requirements.txt` in a fresh environment; the measured runtime is recorded
in `artifacts/phase4/selection.json`.

`evaluate_phase4.py` needs the generated test shards. On a fresh clone run the
training command to regenerate them. The desktop demo needs only the committed
model; it does not load training data. Missing weights produce the exact training
command, without substituting an untrained network.

## Module and information boundary

| Responsibility | Files |
|---|---|
| Actor memory and schema | `belief/memory.py` |
| Identity-free features and explicit rules | `belief/features.py` |
| Shared scorer, safe checkpoint, prediction API | `belief/model.py` |
| Match-balanced scores and calibration | `belief/metrics.py` |
| Trusted generation and separate label join | `phase4_data.py` |
| Actor-safe scripted communication variety | `phase4_scripts.py` |
| Optimization/validation selection | `phase4_training.py` |
| Final evaluation, audits and plots | `phase4_evaluation.py` |
| Observer orchestration / actor-only panel | `phase4_observer.py`, `phase4_renderer.py` |

Actor modules import no simulator, truth, training, controller or replay module.
`game.observe(pid)` supplies a validated immutable `Phase3Observation` to memory.
Only the trusted generator calls `role_labels(game.snapshot())`; its labels join
the completed feature rows to form loss targets. IDs, colors, seeds, initial spawn
coordinates, styles and winners are separate audit metadata, never input columns.
Python import separation is an engineering boundary, not a hostile-code sandbox.

An authenticated `DIRECT WitnessedElimination` is legitimate observed evidence.
Its explicitly witnessed perpetrator differs from the hidden `BodyTruth.killer_id`,
which is never admitted. Claims of witnessing remain `CLAIM StructuredClaim` values.
No winner, future delivery, hidden cooldown, unseen current position or private
task belonging to another player can enter memory. Dead focal actors and finished
matches freeze evidence; evaluation excludes their later packets entirely.

## Persistent memory

Memory reuses canonical `EventView` payloads: player/body sightings, witnessed
eliminations, reports, meetings, structured claims, public votes and ejections.
Repeated public prefixes are deduplicated. Routine player sightings become
keyframes on room/interaction changes or after two seconds. Last-seen snapshots
still update on every legitimate decision observation, then remain frozen while
unseen. Exact `(player, observed_tick) -> room` pairs are retained separately for
claim checking; unobserved intermediate positions are never interpolated.

Memory persists across meetings and records actor-local event references.
Claims retain speaker, subject, region, delivery time and asserted time. Only
location claims with an **exact matching observed timestamp** can conflict with
direct room evidence. The signal belongs to the claim's speaker. Two conflicting
claims about the same subject/time create symmetric weak inconsistencies: neither
speaker is declared a liar. No travel-time inference is included. Compression can
miss weak body-proximity associations; it cannot invent new sightings.

`ActorMemory.to_json()` / `from_json()` use versioned explicit payload allowlists,
not simulator pickles. They preserve continued-update behavior. The Phase 1
`Knowledge` fixture API remains unchanged.

## Evidence representation and candidates

Each prediction has exactly four candidate rows: every public identity except
the focal crew member. Ejected candidates remain in the historical role target;
there is no role reveal and no hidden living-player mask. Public body evidence and
absence are features rather than hard probability masks.

The 22 columns in `FEATURE_NAMES` contain seen/age/visibility summaries, compressed
sighting count and visible-interaction fraction, known bodies/public absence/
ejection, direct elimination/near-body evidence, recent same-room report evidence,
reports, votes cast/received/skipped, hearsay elimination/near-body allegations,
accusations/defenses, self-location claims and the two contradiction sources.
Near-body uses only preceding sightings within three seconds/two native map units;
near-report uses preceding same-room sightings within three seconds. Those are
circumstantial features, never claims of guilt. Age is capped at 60 seconds, count
at eight, and sighting count uses `log(1+n)/4`. Repeated report/witness/body evidence
can each increment body knowledge; the model learns their weights.

No absolute room coordinate, color, ID, candidate index, spawn index or seed is a
feature. IDs only join evidence to rows and map output back to public players.
Training normalization pools all candidates. Shared scoring and mean pooling
guarantee candidate-order equivariance to floating-point tolerance. Four identical
rows give exactly uniform probabilities, even when an arbitrary ID happens to be
more frequent in finite training data.

## Baselines and neural alternatives

The prior assigns 0.25 to each candidate. The rule baseline adds these coefficients
to candidate scores and applies a softmax; all unlisted features have zero weight:

| Evidence count | Score increment |
|---|---:|
| Known body | -4 |
| Direct elimination | +6 |
| Direct near-body | +2 |
| Near report | +0.5 |
| Reporter | +0.25 |
| Claimed elimination | +0.7 |
| Claimed near-body | +0.4 |
| Accusation / defense | +0.1 / -0.2 |
| Direct contradiction | +2.5 |
| Hearsay inconsistency | +0.25 |

Both raw rules and rules with a validation-fitted temperature are evaluated, so a
learned model cannot win solely by being compared against uncalibrated rules.

Candidate architectures are a shared linear scalar scorer (23 parameters) and a
small set network: 22→32→32 encoder, mean embedding concatenated to each embedding,
64→16→1 shared score head (2,849 parameters). Explicit memory supplies temporal
state. No recurrent, attention, pixel or language model is justified at this stage.
Architecture is selected by mean validation NLL across seeds 17/29/43, then the
best validation seed is released. Cross entropy uses Adam, learning rate .003,
weight decay .001, batch 512, at most 160 epochs, patience 20. Training-only feature
scaling and deterministic CPU execution are saved. The state dict includes scaling
buffers; schema, feature order, architecture, evidence view and positive temperature
are checked when loading with `weights_only=True`.

## Dataset and experimental discipline

Registered ranges: 800 training matches (10000–10799), 200 validation (20000–20199),
200 final ID test (30000–30199), 200 patient-family test (40000–40199). Hunter and
self-report impostors are mixed in the first three partitions; the entire patient
family is withheld. Nearest/shuffled/social crew and evidence/cautious/skeptical
meeting styles are drawn independently of true roles for every controller.

Small communication variants preserve strong crew witness/body reports and vary
ordinary sightings, truthful self-alibis and uncertain accusations. Impostors can
fabricate public-room alibis/sightings and occasionally a witness accusation.
They use their own role and legitimately available identities/timestamps only.
These variants occur in every split; held-out differences remain patience and
voting behavior. The original Phase 3 bots and rules are unchanged. This is a
limited distribution shift, not evidence of generalization to human opponents.

Each living focal crew is updated at decision ticks. Samples are taken at start,
new meaningful events and six-second checkpoints. They are not every physics frame.
All four focal histories from a match remain together. Dead actors/terminal states
are excluded without selecting episodes by winner. Each match receives equal
weight in training/metrics; samples within a match share its weight. This avoids
long matches dominating, but the observer population is still conditioned on
being alive and its event density. Independent per-match NPZ/JSON shards support
resume; generation failures are raised, never quietly filtered. Source hashes
normalize LF to avoid Windows checkout artifacts.

Validation alone selects architecture, checkpoint and scalar temperature. Final
tests are evaluated only after writing selection metadata. Reported NLL-difference
intervals bootstrap whole matches. Accuracy grants fractional credit to exact
maximum ties; conventional argmax accuracy is also stored. The four-class Brier
score is a sum, so the prior scores .75. ECE uses ten equal-width confidence bins;
it depends on binning and is complemented by NLL/Brier/reliability/entropy. Temperature
scaling need not help under an unseen strategy shift.

## Ablations

All use identical matches/sampling/labels and retrain the selected architecture
with seed 17 and the same optimizer budget. Each has its own validation temperature.

- Current-only: fresh memory from current sightings and newly delivered events;
  the current meeting's public participant list is retained as current context.
  The repeated historical public prefix is otherwise removed explicitly. A public claim
  learned earlier is unavailable even if the engine repeats it in a later packet.
- No claims: all explicit claim-derived features and contradiction signals removed.
  Public votes remain legitimate downstream behavior; this is an input ablation,
  not a counterfactual match with communication disabled.
- Collapsed provenance: direct/hearsay elimination and near-body channels merged;
  direct conflicts merged with hearsay conflicts. This specifically removes those
  source distinctions, not every possible semantic clue carried by event types.

## Inference contract and desktop controls

```python
from belief.memory import ActorMemory
from belief.model import BeliefModel
memory = ActorMemory()
memory.update(legitimate_phase3_observation)
belief = BeliefModel.load().predict(memory)
print(belief.by_player, belief.entropy)  # logits and stable player_ids also available
```

The default seed 15 selects the first initially known crewmate and tracks all crew
observers independently from match start. F changes observer; an eliminated focal
actor's memory/probabilities freeze and are labeled. M toggles memory inspection.
PgUp/PgDn browse earlier evidence. The display fits the connected desktops while
preserving the original renderer's canvas. The right panel takes only memory, beliefs and own active status. It shows four
probabilities, entropy, last sightings, source-tagged evidence and conflict counts.
The original map/game panel is explicitly a spectator view; `1` separately reveals
debug roles and never changes model inputs. Space pauses, Right steps while paused,
`+/-` changes speed, R repeats seed, N advances, Escape/Q exits. Original G/V/T/L/C/E/B
overlays remain. `--seed`, `--speed`, `--capture`, `--exit-on-finish` support inspection.
No controller receives a belief yet: the observer-versus-original trajectory test
requires identical scripted match hashes.

## Scope and limitations

Five players, one impostor, one frozen historical Skeld geometry, simple scripts,
finite structured claims, no vents/sabotage/natural language and no learned strategy.
Learned probabilities may exploit real but brittle script habits (interactions,
reporting, voting); architectural leakage resistance does not eliminate distribution
shift. A single held-out family and 200 matches per final set are modest evidence.
Repeated focal histories are correlated; match grouping handles split/interval
leakage but does not manufacture independent evidence. Per-feature synthetic probes
check direction/uncertainty; they do not prove causal social reasoning. Optional
commercial artwork stays in the existing ignored local pack.

References: [Deep Sets](https://arxiv.org/abs/1703.06114) and
[temperature calibration](https://proceedings.mlr.press/v70/guo17a.html).
