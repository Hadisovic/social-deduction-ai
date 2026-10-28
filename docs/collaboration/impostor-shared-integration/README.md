# Shared crewmate / impostor integration — review summary

28 October 2026. **Engineering integration works. No full PPO study is running.**
The existing impostor can play through an explicit, checked transfer adapter.
It is not a drop-in historical checkpoint load, and competent play is not proven.

Start with [the live checklist](CHECKLIST.md), [compatibility audit](COMPATIBILITY.md),
[preserved failures](FAILURES.md) and [proposed study](STUDY_PROPOSAL.md).
Machine-readable [verification](verification.json) and the unchanged final
[publication check](publication-verification.json) accompany the review.

## Watch the actual learned-impostor match

In Antigravity's PowerShell terminal, use the **isolated integration checkout**:

```powershell
.\.venv\Scripts\python.exe .\run_impostor_shared.py
```

This opens unchanged Spectator V2 on the completed neural match. Press Play or
Space; select a player in the roster and use Mind, Rewards or Evidence. The
replay contains real recorded decisions for all five neural actors. It does not
train or run a fresh simulation while playing back.

The saved demo is `local-runs/shared-integration/learned-b-2760002-v2/replay.jsonl`.
It is local/private and is not part of Git. To generate a different engineering
match and then watch it:

```powershell
.\.venv\Scripts\python.exe -m impostor_shared.run --seed 2760007 --crew B --output local-runs/shared-integration/demo-2760007
.\.venv\Scripts\python.exe run_impostor_shared.py --replay local-runs/shared-integration/demo-2760007/replay.jsonl
```

For a **fresh clone**, first follow the pinned-runtime setup in the
[published handoff](../impostor-training-handoff/README.md), then generate the
default local demo once before launching the viewer:

```powershell
.\.venv\Scripts\python.exe -m impostor_shared.run --seed 2760002 --crew B --output local-runs/shared-integration/learned-b-2760002-v2
.\.venv\Scripts\python.exe run_impostor_shared.py
```

Use a new output directory each time. The runner rejects existing output folders.
Allowed engineering seeds are 2760000–2760099. Do not use reserved old final sets.
The actor is the transferred learned model; this command never starts PPO.

Actual Spectator capture: `local-runs/shared-integration/spectator-learned.png`.
It shows p2's recorded self-report at67.8s, candidate probability47.5%, actual
critic value, cumulative reward and the ensuing meeting. Five actors' neural and
reward records and a completed result footer were checked. No missing telemetry
was fabricated. Map/characters/viewer source remain unchanged.

## Exact versions

| Component | Identity |
|---|---|
| Published branch | `codex/crew-impostor-training-handoff` |
| Developer-pinned publication, confirmed in draft PR12 and Git | `c3320583f7d562bc69ea527580a91ddc5db54f18` |
| Integration branch | `codex/impostor-shared-integration`; owner authorized publication; no merge |
| Shared engine | `crewmate-environment-v2.2`; compatible source `4bac4cdb099e81a5c6c7de982f423136f24213c5` |
| Crew model registration | `fab5c70e63e54d4d071d3343b1e0463fa4fbb6b6` |
| Frozen A | `artifacts/crew-handoff/original/checkpoint-batch-23-choices-2058.pt` |
| A SHA256 | `3b843996c5b9189da06888c89b42bc03c40479c629a01d060ad8ff4099162f7c` |
| Frozen B | `artifacts/crew-handoff/shaped_v2/checkpoint-batch-23-choices-2092.pt` |
| B SHA256 | `88103ee9bab248c7be4bbc64d39173300ffc7c8cbbbc40c7e95182a2c38355aa` |
| Frozen belief | Phase4, SHA `e931b1fa8a421d9a329a29280d676bf52f793cc8f07c02b8da61f2b3e5e537d1` |
| Original impostor source | `eee7ee608edb952c4b17c8d8d1b0f6303431c1a0` |
| Original V4.1 checkpoint | 83/8192; SHA `69965afd8fab28e6ca5bdda46d9f8efdd8d6036ce57a05b86ff89fe22cf17ca9` |
| Explicit transfer file | `impostor_shared/assets-v1/policy-transfer.pt` |
| Transfer SHA256 | `48255932c6a154397efb2785569da5af7a215d1292489033ec33d9529f41eb80` |
| Unchanged impostor tensor hash | `ebb59ec4cb1589a5b3e96d805085d2bec7ea35c7619be96c64a81c0ce1c94287` |
| Isolated runtime | Python3.12.10, NumPy2.5.3, Torch2.6.0+cpu |

The old checkpoint's strict runtime was Python3.12.3/NumPy1.26.4/Torch2.9.1+cpu.
Its unchanged original loader succeeded there before exporting exact tensors.
The new safe loader verifies schema, bytes, tensor identity and architecture.
Three synthetic forward checks produced identical probabilities and greedy choices
across the two runtimes. This is explicit migration, not weakening an old lock or
proof that changed-world gameplay is equivalent. The original weights/results
remain on their original branch and checkout.

## Architecture and information boundary

| Network/interface | Shape / behavior |
|---|---|
| Impostor | Same75,075 parameters; separate actor/critic,12 action families then candidate choice |
| Impostor inputs | Global35 +extra8; four other-player rows ×21;256 candidate rows ×51; target indices and legal mask |
| Crew A/B | Historical51,778-parameter FourCrewPolicy; no new recurrent architecture |
| Crew inputs | Global42; players4×29; own tasks4×8;96 candidates ×23 plus indices/masks |
| Opponent population | Four shared-weight frozen B actors by default; `--crew A` selects four A actors |
| Private state | Fresh per-actor memory, evidence, task options and sampling RNG; one shared frozen belief predictor |

The adapter reuses pure historical encoding/model components, not the old match
wrapper. A trusted gate projects only the legitimate V2 actor envelope plus that
actor's own fake timer. Public task progress/confirmed ejections/visible proofs are
legal; crew private beliefs/memories, hidden positions, replay truth and ghost
secrets never enter the impostor's neural packet. The controller holds no game or
recorder reference. Candidate indices decode to checked typed Intent values.

Travel leases last at most3s with visibility/phase/kill-opportunity interrupts.
Repeated active same-console fakes continue rather than restart; the NN still
chooses start, continue, abandon, move, kill, report, claim and vote. No forced hunt
or report ban was added. The shared callback's large private seed is explicitly
hashed into a63-bit Torch seed, never placed in neural features.

Physics remain five players, Skeld,45s kill cooldown,0.2s physics and0.6s decisions.
The handoff fixes living/ghost task and incubation pauses in meetings, legal voter
eligibility and early resolution when all living ballots arrive. These differ
from the old impostor study. Its map is still a documented approximation, not an
independent claim of exact commercial-game geometry. Functional sabotage is absent.
Old report/witness social overrides are not silently attached to the new crew.

## Rewards, credit and checkpoints

Crew ledgers use A's original or B's shaped coefficients. Learned engineering
matches explicitly use `eligible-v1` accounting; the scripted publication
reproduction uses its historical default. This accounting distinction changes
crew credit, not the common physics or frozen action probabilities.

The impostor has its own versioned ledger: win+10,loss−10,timeout−12,kill+3,
extra own-ejection−6, no unconditional time penalty, and−0.02 per simulated roaming
second while alive and cooldown ready. Meetings/death/after-termination are
excluded. Observable pursuit potential has at most+0.05 lifetime credit and is
repaid at terminal. No hidden-witness reward or arbitrary reporting cost was added.

The trusted per-physics hook records actual elapsed time; zero-time terminal kills
are legal. Complete-episode transitions have zero terminal bootstrap and finite
GAE using gamma1 and lambda0.995 per simulated second. A wall abort is an error/
incomplete recording, not a terminal loss or usable complete rollout.

`impostor_shared.checkpoints` provides a **new** source/runtime-locked training
snapshot contract, exclusive writes, actor/critic optimizer groups, private RNG
restore and next training seed. It rejects mid-episode snapshots and an optimizer
containing foreign parameters. The proposed first study uses complete episodes;
arbitrary mid-game resume/horizon truncation is not claimed. Historical optimizers
are not migrated. Checkpoint unit tests use synthetic gradients only; recorded
matches had zero optimizer updates.

## Completed engineering matches

| Configuration | Seed | Actual outcome | Impostor kills/reports | Simulated /wall seconds |
|---|---:|---|---|---|
| Four B, published scripted stand-in |2760000 |Impostor parity |Scripted control; not an NN win |168.0 /33.69 |
| Four B, transferred learned impostor |2760002 |Impostor ejected |1 /1 |84.2 /32.00 |
| Same learned B case, recording off |2760002 |Identical trace/outcome |1 /1 |84.2 /31.62 |
| Four A, transferred learned impostor |2760002 |Impostor ejected |1 /1 |84.2 /68.03 |

Four executions contain only **two scenario seeds**, and one execution is a
determinism repeat. They are engineering demonstrations, not win-rate estimates.
Timings include model checks and overlapped other jobs; do not infer that A is
twice as slow or claim a recording speed effect from these numbers.

The B learned match and repeat have identical semantic event SHA:
`8d854b3ed28fdb41d00051c44dc0f495474ec5a6a043ee8bd5f61c3e14e59bce`.
A produced the same world trace on this one seed. A/B are nevertheless different:
all20 state tensors differ, maximum parameter difference0.05537. Both caused an
ejection after the impostor's self-report here; the cause needs controlled tests.
The impostor made101 room choices, one kill/report/location claim/vote each.
More strategy learning is needed; no learned deception success is claimed.

Learned reward−13.444 =loss−10 +kill3 +ejection−6 −ready-cost0.444. Its105
transitions span84.2s, with22.2 cooldown-ready seconds, zero terminal bootstrap,
zero net pursuit reward and finite advantages. All matches had0 illegal actions,
unchanged policies/belief and independent crew memories.

## Tests and reproducibility

- Unchanged publication verifier:282 files, strict A/B loads,119 unique artworks
  and safe viewer defaults pass.
- Combined existing mechanics/privacy/viewer/handoff plus new integration suite:
  **84 passed,5 expected private-recording skips,54.81s**.
- New checkpoint roundtrip/boundary/optimizer/RNG checks: **2 passed,4.60s**.
  These add two distinct tests; they do not count as trained models or matches.
- Final focused run after the idempotent already-terminal reward guard:
  **11 passed,12.14s**, including9 integration and2 checkpoint tests. Thus86
  distinct passes overall, not84+11. Completed normal matches never take the
  already-terminal step path; their raw source fingerprints remain preserved.
- Hidden-player position noninterference, stale/wrong context rejection, private
  sampling, fake persistence/completion, reward boundaries, zero-time terminal
  credit, frozen gradients and cross-runtime parity are covered.
- Actual unchanged Spectator V2 rendered the completed replay and selected
  impostor panel. Neural/reward records exist for all five players; result footer
  and deterministic record-on/off credit/actions/event hashes were verified.

Raw local evidence is under `local-runs/shared-integration/`; it is ignored/private.
The combined JUnit result is `regressions.xml`. See FAILURES.md for the failed first
metadata export, seed overflow, test-fixture correction and sandbox temp errors.
No published source, map or artwork was edited to make a check pass.

## Independent assessment and remaining review gates

B is a useful first integration opponent because its training objective includes
reporting and ballots. It is not established as the strongest crew: its study was
small, its wrong votes/claims can be exploited, and A/B behaved identically in the
single paired smoke. Keep A as a task-pressure/generalization control. A later
preregistered per-match A/B pool can reduce specialization;2A+2B means separate
frozen weights per seat, never averaging their networks. Mixed/scripted-crew
compositions are proposed follow-ups, not tested by this runner.

Do not compare the old one-NN/three-script/guarded study directly with this four-NN
population as if only the impostor changed. Review reporter bias, evidence exposure,
task pressure, wrong crew ejections and post-kill choices separately. Missing
sabotage may change the problem but is not identified as the cause of losses.

The proposed bounded adaptation study needs owner approval, a frozen experiment
registration and a dedicated PPO loop using this collector/checkpoint interface.
The engineering CLI is inference-only. The owner subsequently requested publishing
the completed integration branch. No full study, reserved final access, merge or
deployment is authorized by that instruction. Published PR12 and the original
impostor branch remain untouched. Raw recordings/preferences/runtimes stay local.
