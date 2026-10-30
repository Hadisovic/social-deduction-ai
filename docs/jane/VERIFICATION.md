# Verification record

30 October 2026. Source base `eb5d122b2c338af4bea65967f0f01004258d0f22`;
branch `codex/jane-shared-foundation`. See `verification.json` for exact profile
and shared-bundle fingerprints. No match, PPO, benchmark or reserved evaluation
was run. No checkpoint was loaded or changed.

## Focused tests

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

## CLI checks

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

The source location for current Patrick Jane was requested from the owner. It
is absent from fetched remote branches and inspected worktrees. This delivery
cannot verify 15-player neural gameplay, import its checkpoint, or certify both
Jane adapters. Those remain unchecked items, not claimed successes.

Owner clarification: the teammate model is unfinished and will be pushed later.
Patrick Jane is reference only, not a copied base. After the owner's ready notice,
pull and review the progress before revising shared compatibility for both models.
The branch's settings, folders and contracts are explicitly provisional.
