# Verification record — initial foundation and upstream correction

30 October 2026. Source base `eb5d122b2c338af4bea65967f0f01004258d0f22`;
branch `codex/jane-shared-foundation`. See `verification.json` for exact profile
and shared-bundle fingerprints. No match, PPO, benchmark or reserved evaluation
was run. No checkpoint was loaded or changed.

## Initial foundation tests (commit7994108, preserved history)

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_jane_foundation.py -q --disable-warnings -p no:cacheprovider
```

**87 passed in 3.75 seconds** on the existing Python 3.12.10 runtime. An earlier
86-test pass was followed by one additional clean-shell JSON-output regression.

Coverage: legal/invalid populations, real-style setting ranges, nonfinite and
boolean-as-number rejection, unknown/duplicate keys, immutable settings,
fingerprints and definition changes, disabled future roles/features, extension
dependencies, unchanged historical defaults, unsupported historical options,
missing-Jane-source blockers, role declaration version/hash mismatches, complete
asset task inventory and model/trainer-free configuration imports.

Clean-shell verification also caught the historical pygame support banner on
stdout. The new adapter suppresses that import banner locally so the inspection
CLI stays valid JSON; no historical module or global environment is modified.

The historical adapter result matches `crewmate_v2.config.LobbySettings()`
exactly. That verifies constructor translation; it is not a new match replay or
proof that a frozen model generalizes to changed host settings.

## Initial CLI checks (historical7994108 receipt)

| Command | Expected result |
|---|---|
| `python -m jane jane/configs/jane_12p_base.json` | Exit 0; valid 10+2 settings |
| `python -m jane jane/configs/jane_15p_base.json` | Exit 0; valid 13+2 settings |
| `python -m jane jane/configs/historical_5p.json --engine historical-v2.2` | Exit 0; configuration translation accepted |
| `python -m jane jane/configs/jane_12p_base.json --engine patrick-jane --check-pair` | Exit 2; missing source/adapters/verified registration |
| `python -m jane jane/configs/future_15p_full_classic.json --engine historical-v2.2` | Exit 2; unavailable mechanics/population/ruleset |

All supplied profiles parse, every new local Markdown link resolves, and the
task catalog preserves all 19 original asset task IDs/categories and map hash.
There are 18 canonical families when the two garbage variants are grouped.

## Preservation and limits

New files live under `jane/`, `docs/jane/`, the root entry document and
one focused test file. Scoped `.gitattributes` additions keep only these new paths
at LF for portable fingerprints. Historical engine/model/viewer/assets/checkpoints are
untouched. The earlier pilot README modification and untracked Navigation V2
plan remain local and are excluded from this commit. No merge or deployment.

At the initial7994108 delivery, Patrick Jane was absent from the fetched branches.
That historical absence is superseded by the source review below. This delivery
cannot verify 15-player neural gameplay, import its checkpoint, or certify both
Jane adapters. Those remain unchecked items, not claimed successes.

The owner subsequently confirmed the push. It was fetched/read before this
correction. The branch's settings, folders and model plans remain provisional.

## Upstream source-backed correction

Review: d459f386 implementation/model, c7c0ef0 guide. Five artifact byte hashes
verified; source/import hashes recorded. Isolated native configuration constructors
accept10+2 and12+3, reject13+2/14+1. New native profile translation is checked
against the actual constructor, without model/optimizer/game construction.

Latest correction tests and CLI checks: see `upstream-verification.json`.
The original `verification.json` is the immutable earlier draft-v1 receipt; its
profile hashes do not describe draft-v2. Do not overwrite historical receipts.

No teammate neural code/checkpoint was vendored, no source lock was relaxed, and
no PPO, final-test access, merge or deployment occurred. Current model runtime
requires a separately verified2.7.1 build; the configuration-only probe used the
existing environment. Actual paired learned execution remains pending.

### Completed correction checks

**114 passed, zero skipped, in6.52seconds**, including two real isolated upstream
constructor checks. The12p and15p native profiles round-trip to actual A2Experiment
values. Each check verifies36 source files and all five published artifact byte
hashes first. No model/game/optimizer is constructed. Six profiles, five CLI
outcomes and seven local Markdown links were also verified.

The first serialized-kwargs probe failed because JSON turned allowed_sabotages
from a tuple into a list; the locked A2 registration correctly rejected it. The
probe now restores the native tuple, with source-backed regressions for both
populations. This is a configuration repair, not learned behavior or performance.

The initial87-test foundation result remains historical. Read the new
upstream-verification.json for the current draft-v2 receipts and limitations.
