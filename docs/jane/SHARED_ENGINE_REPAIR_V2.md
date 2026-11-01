# Shared Repair V2 — same-batch doorway navigation

2026-11-01. Development-only engineering correction. Acceptance remains pending.

## Confirmed failure and preserved learning

SharedRepair V1 completed one real PPO update: update103 at2,829,668 lifetime
decisions,15,789 newly checkpointed decisions,6natural task victories,0losses and
0technical cuts. Living/ghost task completions were181/59 of240assigned tasks;
accepted correct/wrong/skip crew ballots18/18/102. PPO loss0.4164751518 and
approximate KL0.0070732836 were finite. These on-policy outcomes do not establish
generalization or evidence-grounded deduction.

The next six-match wave was rejected before PPO because seed3200016 recorded one
illegal action. Displayed decisions before failure included unsaved in-flight
collection; only15,789 remained checkpointed afterward. This is the intended
fail-safe accounting. Update103, its Adam/RNG state, all historical checkpoints,
registrations, failed-wave data and exports remain unchanged.

Frozen six-worker reproduction identified the exact fault in68.67seconds with
all52policy tensors unchanged: tick231 /46.2simulated seconds. Crewmate p0's
parameterized navigation goal(-9.72,-13.68) was valid floor when submitted. Later
in the same sorted batch, scripted impostor p1 closed Electrical leaf11:5. The
original loop had already classified p0's goal against open-door geometry; later
ordinary validation then rejected the goal against the closed-door geometry.
This confirms an engine ordering defect, not a failed PPO probability or mask.

## Narrow repair and compatibility

The common `jane/classic_v2/engine.py` now classifies ordinary location goals in
a second pass after processing door/vent extensions. A temporarily door-covered
floor goal enters the existing deferred navigation route, preserving real door
blocking and local route recovery. Permanently invalid coordinates still fail.
The engine does not move bodies, teleport actors, bypass reporting occlusion,
ignore illegal actions or replace neural decisions. Task ownership/stages,
ghosts, action sampling, recurrent state, PPO, rewards and lobby settings stay
unchanged. Native and partner Classic allocation profiles remain separate.

Both actor orderings have deterministic regression coverage; the original p0
ordering failed before the repair. The existing collision/meeting/vent/ghost/
reporting guards remain covered. Independent read-only review found no material
defects in the repair or explicit recovery bridge.

Shared normalized runtime fingerprint:
`4ae8a8bf2a2b012b31916e19f6ba01d280ce441156bd3df9dfc3902083bb224b`.
V1 source snapshots and its earlier fingerprint are historical records, not
rewritten to describe this patch. Retail-identical doorway physics is not claimed;
the occupied-door closure delay remains an explicitly documented approximation.

## Recovery under the original budget

The separate runnable directory is `dist/PatrickJane-A2-SharedRepair-V2`.
`tools/prepare_pj_a2_repair.py` verifies original update103 SHA256
`2839a413e3b4d537a8ad8170b763642c1f07550a7370981ccee99890f331bc68`
and its source archive SHA256
`9899e08042b42e008d4352af14c3a4448fa95c08a78259e3601540a102755827`.
It verifies runtime, architecture, experiment, complete-match boundary, counters
and every archived source; the only permitted actor-source differences are the
common engine correction and report provenance. All52policy and156Adam tensors
and saved RNG states are retained exactly. Standard checkpoint loading remains
strict; the original update102 bridge is unchanged.

The fresh registration carries the successful first-wave history and
578.8447452active seconds toward the same10,800-second /three-hour target.
It does not grant another three hours. Rejected allocations3200012–3200017 are
never reused for learning; first fresh seed is3200018. The copied historical
update retains its old source revision; the bridge and new source revision are
explicit in registration, JSON and Markdown reports. The original failed V1
folder stays untouched. Runtime/payload binary hardlinks reuse immutable
installation files only; sources, configurations, models and results are
separate copies.

Start/resume from the V2 folder:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Start-PatrickJaneA2.ps1 -Hours 3 -Workers 6
```

Startup verifies the prepared update103 continuation and refuses to fall back to
update102 if that registration is missing. Existing spectator, strict recovery,
percentage/ETA, checkpoints, logs, reports, curves, selected replays and automatic
transferable export are reused. No validation/final-test partition, neural
retraining during verification, reward change, architecture change or PR merge.
The already approved three-hour continuation may resume only after complete
readiness coverage; finish an active wave/PPO checkpoint at the time boundary.

## Verification record

- Focused shared checks:15passed, including the former failing actor ordering.
- Real bridge regression:52policy/156Adam tensors exact; corrupted checkpoint
  rejected; preserved counters and next seed18 verified without PPO.
- First full failed-wave diagnostic reached its420-second bound without complete
  coverage; retained as inconclusive, not a pass.
- A later check saved82.66seconds of progress before a long interruption; its
  final reported elapsed time was60,003.97seconds with an RPC deadline timeout.
  The cause of that gap is not established. It is neither passing coverage nor
  evidence of a neural failure.
- A fresh full six-scenario check is required before restart. Raw traces and
  role-labelled diagnostic records remain local; only engineering conclusions
  and aggregate verification results belong in public documentation.


- **Completed fresh readiness check:**6/6 natural task victories,16,969actor decisions,350.392seconds,0illegal actions,0technical truncations,52policy tensors unchanged,0PPO. The exact failed-wave seeds3200012–3200017 were reused for frozen engineering only; training starts fresh at3200018. The ordinary training reward/legality audit passed. These are regression outcomes, not independent learning/generalization evidence.

- **Final focused checks:**21 native adapter/provenance/publication tests and10 shared-engine regressions passed. The full six-match frozen check passed the same reward/legality audit used before PPO; no optimizer updates were performed during verification.
