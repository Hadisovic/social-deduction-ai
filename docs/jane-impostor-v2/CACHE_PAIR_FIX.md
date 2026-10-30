# Geometry cache fix — 31 October 2026

The shared Classic adapter now repairs the cache-coherence fault that stopped the
30 October pilot. Geometry revision: **`paired-map-planner-v1`**. No new training
study was started; the failed pilot and all its results remain unchanged.

## What changed

The pinned engine independently caches door maps and planners. Evicting only a
map can produce an equivalent replacement while the planner still owns the older
instance. Projection requires the exact same map object as interaction geometry.

After each native door update, the adapter now:

1. Checks that map and planner describe the same closed-door revision.
2. Takes the canonical map from the selected planner.
3. Gives that map to every player's collision/motion object.
4. Invalidates cached observations when any binding changed.

Meeting reset also performs this binding. The visibility/interaction identity
guard remains enabled. Navigation knowledge, neural inputs, strategies, rewards,
network architecture, task allocation and game settings are unchanged.

Implementation: `jane/classic_v2/engine.py`. The fix is in our shared adapter;
the external pinned engine and historical V1/shared bundles remain untouched.

## Both developers must use the shared adapter

Lizbon already calls this factory. The crew developer can use the same path:

```python
from jane.classic_v2.settings import load_profile
from jane.classic_v2.engine import experiment, make_match

profile = load_profile("jane/configs/classic-v2/default_12p.json")
match = make_match(seed, experiment(profile))
```

Directly constructing the unchanged external `A2Game` bypasses this fix. The
historical reproduction tool intentionally uses that old class so the original
fault remains reproducible. The proposed upstream patch in the failed-study
report is still useful if the crew developer releases a new upstream engine.

## Checkpoints stay versioned

The current receipt includes the geometry revision and changed source hashes.
The normal strict loader rejects pre-fix receipts. Original model files and their
frozen source commit0378994 remain available for historical reproduction.

An explicit offline migration tool supports the three registered development
checkpoints from the failed pilot:

```powershell
$py = '.\local-runs\jane-impostor-v1\runtime\Scripts\python.exe'
& $py tools/migrate_lizbon_geometry_checkpoint.py --parent local-runs/lizbon-classic-v2/study-20261030/train101/checkpoint-003.pt --output local-runs/lizbon-classic-v2/fix-20261031/checkpoint-003-migrated.pt
```

Choose a new output path if that file already exists. Migration:

- Accepts only exact SHA256 values from the previously published checkpoint
  manifest; loads the same verified bytes with `weights_only=True`.
- Verifies architecture, runtime, host profile, counters, crew revision and reward
  regime. Unknown or altered files are rejected.
- Preserves every tensor exactly, records parent hash/receipt/commit and inherited
  passes, and writes a new geometry receipt.
- Copies no optimizer state, records zero new training passes, and never overwrites
  the parent or an existing output. This is transfer, not exact study continuation.

The latest stream101 checkpoint retains its24inherited passes (8fromV1 plus16from
the stopped pilot). It remains an **unselected development model**. Migration is
not learning, evaluation or evidence of better gameplay.

## Verification

The new cache test first failed on the old adapter with different map identities.
After the fix, the checks cover geometry-only eviction, actual17-state LRU
pressure, planner-only/both-cache eviction, door expiry/reopening, meetings,
inactive/vented actors and the unchanged projection guard. A deterministic
door/movement trace matches the pre-fix coherent-cache baseline.

The isolated Jane regression run passed **87checks**. The migration tests were
repeated after review hardened the hash/load path. The normal repository tests
returned **295passed,13skipped and one pre-existing failure**:

`tests/test_impostor_shared_checkpoints.py::CheckpointTests::test_optimizer_rng_and_next_update_roundtrip`

That older frozen-publication contract rejects `.gitattributes` and the previously
changed `spectator/v2/map_view.py`. Its manifest was not rewritten to hide this.
The bare root test command also encounters the previously documented missing
`stable_baselines3` imports in legacy scripts.

`tools/verify_lizbon_geometry_fix.py` performs paired bounded engineering inference
with identical weights/scenario, forcing geometry-cache eviction in one copy.
It checks actor-trace equality, unchanged policy/crew tensors, action legality and
navigation errors over a common logical prefix. This uses an engineering seed, not validation/final partitions.
See [the verification evidence](cache-fix-20261031/verification.json) for the actual
measured result. Raw replays and migrated weights remain in ignored local-runs.

The measured runs hit their180-second caps at110and112impostor decisions, short
of the120target. Their unequal full-run hashes were not called equal. Offline
comparison of the common110decisions (through32.4simulated seconds) found exact
equality of impostor inputs/actions, all recorded neural decisions and physics.
That prefix includes observations after a forced eviction at30.6s. Both policy
and frozen crew tensors stayed unchanged; illegal actions/navigation errors were
zero. The original budget-limited check is retained as
[raw-inference-check.json](cache-fix-20261031/raw-inference-check.json), alongside
the explicit prefix assessment. No inference was rerun to obtain that assessment.

Three supplementary comparator tests verify that differing budget tails are
handled explicitly and that changed neural commands or physics are rejected.
This is a bounded correctness check, not a full-match reliability or model-quality
evaluation. Runtime remains a separate issue to profile before a new study.

## Next step

Review the fixed shared branch with the crew developer, then register a new
training/evaluation study and budget. The previous study remains failed and was
not resumed. Model competence, stronger crew cross-play and 4–15-player learning
generalization still require their own evidence.
