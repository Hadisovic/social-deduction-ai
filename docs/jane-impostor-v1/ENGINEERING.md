# Engineering record: Jane impostor V1

30 October2026. No reserved final sets opened. Historical models preserved.

| Check / failure | Outcome and fix |
|---|---|
| Native stream seed exceeds Torch range | smoke-001 failed before gameplay; map private generator seed modulo2^63-1 |
| Progress dict duplicated wall_seconds | smoke-002 failed; separate match and total elapsed fields |
| Duplicate simultaneous shared-door requests | smoke-003 recorded one illegal request; neutral contention resolution preserves native ordering and sampled request |
| Invalid far-position test fixture | First privacy test used a point inside an obstacle; fixture now uses verified public standing points |
| Meeting generic validator rejected WAIT | Test assumption was wrong: native Jane removes WAIT before validation; direct discussion/voting execution tests preserve silence/no-submission |
| Base emergency family ignored Jane restrictions | Mask emergency while sabotage active or own public emergency cooldown remains |
| Wall-clock budget during PPO | Check deadline between recurrent chunks and before optimizer step; discard unstepped gradients |
| Strict frozen crew loading | Published Torch2.7.1 CPU weights-only inference; original2.6 environment unchanged |

## Preserved engineering segments

These runs have random initial impostor weights and **zero PPO updates**. They
are implementation checks, not evidence of learning or measured win rate.

| Run | Simulated time | Collection wall time | Result |
|---|---:|---:|---|
| smoke-001 | 0 | startup failure | Seed overflow; initial artifacts retained |
| smoke-002 | first step | startup/progress failure | Progress dict issue retained |
| smoke-003 | 24s | 45.07s | 80 impostor observations; one shared-door conflict |
| smoke-004 | 30s | 49.93s | 100 impostor observations; zero illegal actions, zero navigation failures; one neutral contention |

smoke-004 produced a verified actual GUI capture. Its two impostors were
untrained and made no kills in the short segment. Frozen crew tensors were
unchanged. Match truncation was deliberate, not a loss.

Tests cover actor privacy/noninterference, private RNG/memory, native legal
capabilities, arrived goals/revisits, fake-task continue/cancel/completion,
finite attention/gradients, recurrent PPO, strict checkpoints, real bounded
replay, frozen crew tensors, simulated cooldown overlap, meeting/vent/elimination
exclusions, native silence and shared ability contention.

Full regression result and the frozen implementation commit are recorded in
HANDOFF_JANE_IMPOSTOR.md when the pilot launches. Raw engineering runs, private
replays, runtimes and pilot files remain under ignored local-runs. Curated
reports/figures may be published; no historical artifact is overwritten.

## Remaining limitations

Only one initial seed and a fixed opponent release are in this pilot. Crew
update102 was not A2-trained. No additional learned teammate coordination or
ghost-impostor ability is assumed. Complete-only PPO discards technical cuts;
report this selection limitation. A short pilot may produce no complete updates.
Its reward scale and active-search behavior require evidence, not a forced hunt.

The protected docs/research/ISSUES.md was not edited. New implementation and
failed-engineering records are kept here instead.
