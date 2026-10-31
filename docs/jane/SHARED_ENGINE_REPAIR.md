# A2 shared-engine repair V1 — development readiness

2026-10-31. Research acceptance remains pending. This is a mechanics consistency
repair and continuation registration, not evidence of improved deduction.

## Shared source and supported adapters

The common source is `jane/classic_v2`, promoted from partner Lizbon revision
`0e7f8f5`, including the reviewed `c091badb53952580e9152694b201e825ed016144`
geometry repair. `jane/shared/runtime.py:engine_class` selects explicit allocation:

- `classic-unique-family-v2`: preserve the partner's Classic host profile and
  unique-family allocator. `classic_v2.engine.make_match` injects its constructor
  explicitly. No temporary global `A2Game` substitution remains.
- `patrick-jane-native-v1`: reuse native A2 reset, task assignment streams,
  ownership, stages, workload, task-bar observations and meeting behavior. Only
  state-aware containment and planner-owned geometry binding are inherited.

Patrick's opt-in `A2Experiment.shared_engine_profile` is the plain JSON object
`{"engine":"paired-map-planner-v1","allocation":"patrick-jane-native-v1"}`.
Its historical default is `None`. The unchanged collector sends the top-level
native experiment; each spawned CPU worker constructs the helper runtime locally.
Do not send Classic's function-local configuration classes through that pipe.
The shared runtime requires the repaired native adapter API, notably
`Match(..., game_factory=...)`; it is not a standalone simulator installation or
a promise that every old adapter/model is interchangeable. The partner learned
impostor branch, models and optimizer migration have not been changed or imported.

The normalized shared engine/configuration fingerprint on both branches is
`551e713a1283c2540a3b91c0660ee5dc3f133a758840092d1632b05f404f21fe`.
`fingerprint()` covers the explicitly shared runtime, settings and Classic
profiles, excluding role projects, model files and branch-specific documentation.
Registrations also freeze complete native source hashes, independent of this
shared fingerprint. The older `upstream/patrick_jane_review.json` remains a
historical review of c7c0ef0, not a claim that it reviewed this extension.

## Confirmed failure and bounded verification

The preserved failed A2 run ended before any PPO update: zero newly learned
decisions, with original update102 still intact. A first serial frozen diagnostic
timed out at 150.31 seconds without reproducing the fault; this negative attempt
is retained locally. A subsequent six-spawn diagnostic reproduced the original
assertion in 101.25 seconds: environment 0, seed 3200000, tick 370 (74 simulated
seconds), dead p1 at approximately (-5.34, -14.34), closed leaves 11:8 and 11:6.
That position was valid permanent-map floor but invalid against dynamic doors.
All living positions in that captured failure were valid. This identifies this
failure; it does not prove that every possible crash has the same cause.

Inactive and underground stored positions now use permanent-map validation.
Living surface actors still use the closed-door collision map. Cache binding
requires equal door revisions, selects the planner-owned map, updates motion
maps and invalidates cached observations. It does not move a body, bypass report
occlusion, suppress living-position failures or weaken projection guards.

Focused native tests: 24 passed / one real-PPO engineering test intentionally
excluded. Shared geometry and adapter checks: 12 passed, including the partner's
eight cache/meeting/expiry/vent/projection regressions, native allocation/packet
equivalence, invalid living positions, zero ghost reward, reporting occlusion and
meeting/vent transitions. A separate synthetic orchestration check passed for
new seed reservation, frozen profile, duration extension, recovery and export;
it performs no neural updates. Shared-foundation verification also passed 120 tests
(two historical pinned-upstream probes intentionally deselected); an initial
temporary-directory permission error was resolved by using a workspace-local
pytest base directory. Independent read-only implementation and launcher
reviews found no material defects.

The repaired production collector then completed all six original development
scenarios, 3200000–3200005, with frozen original update102:

| Measure | Result |
|---|---:|
| Spawned workers / complete natural games | 6 / 6 |
| Actor decisions | 17,198 |
| Wall time including pool cleanup | 380.95 seconds |
| Task victories / technical cuts | 6 / 0 |
| Full living / ghost task completions | 191 / 49 |
| Policy tensors unchanged / PPO updates | 52 / 0 |

This is an engineering regression on already allocated development scenarios,
not independent validation, a training run or evidence of better social play.
All selected replays and detailed diagnostic records stay local. No validation
or reserved final partition was opened.

## Preserved model and restart

Original update102 SHA256:
`6cf7048310e889967b01dd785317d313ddc71f310cd9faedcf346b950b8fd01b`.
It is Task Pursuit V1 trained, with 2,813,879 lifetime decisions. The explicit
hash-verified bridge retains all 52 policy and 156 Adam tensors, including saved
steps and moments. The added checkpoint-source enumeration is declared in that
bridge; ordinary checkpoint/source/runtime loading stays strict. No partner
optimizer-reset migration is used.

The separate runnable folder is `dist/PatrickJane-A2-SharedRepair-V1`. It reuses
the existing offline runtime/tools and Spectator V2, with copied sources and a
new manifest. Historical deliveries, source registrations, failed run, original
checkpoints and returned ZIP are preserved. No new installer ZIP is required for
this main-machine restart; the existing result-export ZIP remains automatic.

New continuation: three cumulative active collection/PPO hours, six complete
matches per wave and then unchanged PPO. First new training seed is 3200006;
previously allocated 3200000–3200005 are excluded from new learning. The original
3200000–3299999 development range, empty automatic validation, independent actor
memories/RNG, architecture, rewards, action dimensions, task commitment and all
PPO settings remain unchanged. In particular this is A2: full living task +0.5,
ghost zero and darkness -0.005/living simulated second, not the prior +2 regime.

Launch inside the new folder (also the safe-resume command):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Start-PatrickJaneA2.ps1 -Hours 3 -Workers 6
```

Results are package-relative `results/investigation-a2-v1`; transferable ZIPs are
in `exports`. Ctrl+C requests completion of the current wave/PPO checkpoint;
second interruption abandons unsaved work. Source, runtime, worker and numerical
checks remain active. A three-hour boundary finishes its current wave/update,
so wall-clock time can exceed three hours. Setup/report/export/downtime are not
charged to the training budget. The viewer runs independently at 30 FPS.

The owner subsequently authorized the coordinator to start this three-hour run
after readiness checks, then stop monitoring once real progress and saved output
are verified. The original plan's manual-only boundary is superseded by that
explicit authorization. Do not merge main or existing PRs.

## Retail evidence and remaining limits

Closed doors obstruct reporting, as described by
[Innersloth's reporting fix](https://www.innersloth.com/bug-fixes-for-among-us-2018-10-31/).
Innersloth also documented historical doorway collision problems in
[its meeting improvements](https://www.innersloth.com/improving-meetings-in-among-us-2019-5-7/).
These sources do not specify today's exact collision algorithm. We cannot
certify retail-identical behavior. The existing occupied-door closure delay is
an explicit approximation; no new pushing physics is introduced. Any future
change needs separate evidence/versioning. Historical task dominance, weak
social evidence, reporter-pattern shortcuts, shaped-reward exploitation and
on-policy evaluation limitations remain unresolved; Phase 6 stays in progress.
