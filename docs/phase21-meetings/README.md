# Phase 2.1: experimental meetings and evidence

**Engineering only; no neural training. Phase 6.1 acceptance pending.** Read the [results](REPORT.md), [verification and exact changed files](VALIDATION.md), [design](DESIGN.md) and permanent [research register](../research/ISSUES.md).

The original environment/scripts, opt-in mixture v1, A/B/zero checkpoints, historical results and approved spectator stay unchanged. New mechanisms are opt-in in `crewmate_v2/phase21/`. Default flags retain unrestricted decisions, zero social influence, original speaker order and no witness priority. The new eligible ledger is used only by this experimental bridge.

## Mechanics and interpretation

- `EligibleLedger`, schema `crewmate-phase21-accounting-v1`, retains pre-step eligibility through same-step ejection, with existing coefficients and per-meeting deduplication. Dead-before-step actors receive no vote/report credit.
- Configurable social admissibility restricts defense/accusation/vote options using the same private eligible-normalized suspicion state. Defaults are defense≤.30, accusation/vote≥.70, strong-candidate label≥.85, margin≥.10. Skip/withhold remain possible; a high score alone does not force an accusation. Reporting is never suspicion-threshold gated.
- Each listener has its own influence overlay. Own claims, unreceived claims and ghost communication do not count. Repeats are bounded; opposite repeats cancel/lock; conflicting speakers net out.0/.03/.10 are experimental magnitudes, requested per-target shift capped .20, with clipping and proportional redistribution. Raw frozen estimates and adjusted values stay separate. The base predictor already sees claims: double counting is a known limitation.
- Confidence speaking order is an explicit public-information rule: highest own confidence first, ID tie-break, frozen per meeting. Exact scores remain private, but the order reveals their rank. Script confidence uses a labelled local-evidence proxy because the frozen model is crew-only. This heterogeneous-provider variant leaked role identity in the small assay; keep it disabled by default.
- Optional witness priority uses only authenticated own direct evidence and legal candidates. It interrupts held goals, forces an available legal witnessed-body report, or a fresh feasible body approach; meeting priority supports the witnessed killer and preserves abstention. These are interventions, not learned improvements. V1.1 fixes direct-evidence precedence over erroneous model thresholds.
- `CoherentMixture` v2 retains patrol goals, bounds legitimate visible-target pursuit updates, and overlaps accusation/defense/neutral formats. It preserves the v1 report/kill preferences and legal local inputs. It remains opt-in; no sabotage was added.

## Reproduce bounded diagnostics

From this worktree, use a **new output directory**:

```powershell
& ../.venv/Scripts/python.exe -m crewmate_v2.phase21.run `
  --study ../_v2_fourcrew_registration/local-runs/fourcrew-reporting-2048-seed17-v1 `
  --spectator-source ../_spectator_v2 `
  --output local-runs/new-phase21-comparison
```

The comparison performs frozen B inference only and strictly checks A/B/zero checkpoints. Seeds 2760090–2760099 are engineering seeds. Social contexts share four seeds; seven visibility conditions share one seed; natural opponents use four paired scenarios. The protocol records feature/source/runtime/checkpoint hashes, a 30-minute overall execution ceiling and shorter section limits. It does not save/update model weights. Fresh runs use final v1.1; the completed main comparison used the preserved v1 code snapshot `0cae29e`. All 278 recorded meeting decisions had identical masks/bonuses under the v1.1 fix; an adversarial edge test covers the repaired path separately.

Social/witness rows can end after one meeting or 25 simulated roaming seconds. Unfinished matches are marked as segments, not completed wins/losses. Their recordings remain `.writing` and are hidden from the viewer library. Natural matches keep the canonical 900s simulator timeout. Do not sum correlated segments as independent full games.

## Open real completed diagnostic replays

Example: the explicit witness-priority report and subsequent meeting:

```powershell
& ../.venv/Scripts/python.exe ../_spectator_v2/run_spectator_v2.py `
  --replay local-runs/phase21-comparison-v1/replays/witness/witness_priority/immediate.jsonl `
  --watch local-runs/phase21-comparison-v1/replays `
  --preferences local-runs/phase21-viewer-preferences.json `
  --paused --omniscient --reveal-ballots --seek 45.2 --selected p1
```

Other complete samples are under `replays/opponents/{baseline,v1,v2}/2760096.jsonl`. The approved viewer source/artwork was not redesigned. Only specific data compatibility and behavioral moments were checked.

**Interpret probabilities correctly:** replay `action_probabilities` are the actual hybrid distribution used for execution. A forced legal report can therefore show 100%. That is not raw neural confidence or learned competence. Header metadata labels the adapter; `masked_actions` records experimental admissibility. Raw neural probabilities, base/adjusted suspicion and intervention reasons are in truth-only optional diagnostic events and ignored local decision sidecars. They never enter actor observations. Prefix event history is retained, with snapshots from handoff onward for constructed contexts.

## Review boundary

No feature has been promoted to the training default. Review the results before another registered study. In particular, do not enable10-point influence or heterogeneous confidence ordering on the strength of these fixtures. Rewards, architecture, historical checkpoints and canonical rules remain frozen; applying the repaired accounting to future training requires a new source/protocol lock and explicit approval. No push or merge is authorized by this work.
