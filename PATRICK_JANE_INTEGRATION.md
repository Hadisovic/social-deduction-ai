# Patrick Jane: collaborator integration and A2 operation

Branch: [`codex/patrick-jane-recurrent-ppo`](https://github.com/Hadisovic/social-deduction-ai/tree/codex/patrick-jane-recurrent-ppo).
Pinned **implementation and model revision**: `d459f38674ebc537acde20647cbd07700c494533`.
The subsequent documentation commit adds this guide without changing that implementation.
Portable deliveries identify their exact build commit and normalized actor-source hashes in `SOURCE_VERSION.json`; registrations and returned reports retain that identity.

Update 102 is **task-pursuit-v1 trained**, at 2,813,879 lifetime individual actor decisions.
Its continuation collected 1,295,093 decisions in 84 PPO waves / 504 training matches: 482 task victories and 22 parity losses.
These are on-policy training outcomes, not independent validation. A2 mechanics/rewards are implemented; **no new A2 long training has been performed**. Acceptance remains pending.

## Retrieve the actual implementation and model

```powershell
git clone --branch codex/patrick-jane-recurrent-ppo https://github.com/Hadisovic/social-deduction-ai.git
Set-Location social-deduction-ai
git checkout d459f38674ebc537acde20647cbd07700c494533
```

The artifacts are ordinary Git files, not LFS pointers or external Windows paths. Both neural source and weights are available in a normal clone. Under `artifacts/patrick-jane/task-pursuit-v1-update102/`:

| File | Bytes | SHA256 |
|---|---:|---|
| `inference.pt` | 4,140,551 | `da1d43d73c140ac39bf44834746dbae605b9b36d3c6c7af8a4f2be975e535e3c` |
| `checkpoint.pt` | 12,453,153 | `6cf7048310e889967b01dd785317d313ddc71f310cd9faedcf346b950b8fd01b` |
| `source-snapshot.zip` | 190,601 | `2ee74da35096493975e8bfb8ef2d536704623c0fca588be14d524b8738cb42e3` |
| `architecture.json` | 1,112 | `de87c7a014a809b5219e6e833bdc88c70c23c6472e0430eaf9a35c32540fed23` |
| `training-config.json` | 1,529 | `09d7377d5bbb229a7b2cddb54d30f230fe636bfa9ced315f2ea9577c0358ee1d` |

`artifact-manifest.json` describes provenance and `SHA256SUMS` covers the published files. The original full checkpoint is copied byte-for-byte, including Adam moments, step counts, RNG states, counters, registered configuration and historical source hashes. It contains 52 neural parameter tensors, 52 Adam entries / 156 optimizer tensors, with Adam step 19,852. The historical source ZIP contains code and its source lock, not game traces. The 1.115 GB historical result export, raw traces/role-label datasets, local virtual environment and multi-GB offline installer are excluded from Git publication.

## Runtime and reproducible smoke test

The matching checkpoint runtime is Python **3.12.10**, NumPy **2.5.3**, PyTorch **2.7.1+cu128**. `requirements-patrick-jane.txt` pins the direct dependencies; `requirements-patrick-jane-lock.txt` lists the hashed Windows offline wheel set. The portable package bundles that runtime, its wheelhouse, Tk support and the signed Visual C++ installer. A compatible NVIDIA driver is a machine prerequisite; no system Python is changed.

For a source checkout with Python 3.12.10:

```powershell
python -m venv .venv-pj
.\.venv-pj\Scripts\python.exe -m pip install -r requirements-patrick-jane.txt
.\.venv-pj\Scripts\python.exe -B -m tools.smoke_pj_published
```

The smoke verifies both model SHA256s, strictly loads the published architecture, compares all 52 tensors, and samples one legal CPU inference decision using engineering seed 2920006 and sampler seed 17. It advances **zero game steps** and performs **zero updates**. Expected structural output: 1,031,070 parameters, update 102, 2,813,879 decisions, `exact_tensor_matches: 52`, `hidden_shape: [1,256]`. Numerical bitwise identity across devices is not claimed.

Minimal weights-only loading (verify SHA256 first):

```python
import torch
from patrick_jane.network import PatrickJane

weights = torch.load("artifacts/patrick-jane/task-pursuit-v1-update102/inference.pt",
                     map_location="cpu", weights_only=True)
policy = PatrickJane()
assert policy.architecture() == weights["architecture"]
policy.load_state_dict(weights["state_dict"], strict=True)
policy.eval()
hidden = torch.zeros(1, 256)  # one independent state per actor
```

`patrick_jane/checkpoint.py:load` deliberately rejects archived training checkpoints under changed sources. Do not disable that protection. A2 initialization uses the explicit `a2_transition.import_update102` bridge: it checks the full checkpoint/archive hashes, runtime, counters, all historical source entries and the declared source differences, then loads **all policy tensors and original Adam state**. Publication adds only observer-side terminal metrics to the bridge's declared source differences. The builder puts the original full file at `origin/update-000102.pt`; it never substitutes update 18.

## Neural architecture and tensor contract

Authoritative implementation: [`patrick_jane/network.py`](https://github.com/Hadisovic/social-deduction-ai/blob/d459f38674ebc537acde20647cbd07700c494533/patrick_jane/network.py).
Exactly **1,031,070 parameters / 52 tensors**:

| Component | Structure |
|---|---|
| Self encoder | 192 -> 256 -> 128, Tanh |
| Player encoder | 16 -> 128 -> 128, Tanh; four-head 128-wide self-attention and masked mean pool |
| Event encoder | 16 -> 128 -> 128, Tanh; four-head 128-wide attention queried by self encoding |
| Fusion | concatenated 384 -> 256, Tanh |
| Recurrent memory | one shared GRU layer, input/hidden 256 |
| Actor / critic | separate 256 -> 256 -> 128 Tanh MLPs |
| Actor heads | navigation 3, interaction 5, communication 6, voting 15, dynamic target selector |
| Target head | 16 -> 128 target encoder, four-head attention, projected query and scaled target dot products |
| Value | critic 128 -> 1 |

`network.collate` batches packets, padding only dynamic targets. With batch size B and maximum target count N:

| Input / output | Shape |
|---|---|
| self_features | B x 192 |
| players / player_mask | B x 14 x 16 / B x 14 |
| events / event_mask | B x 64 x 16 / B x 64 |
| targets / target_mask | B x N x 16 / B x N |
| input and next hidden | B x 256 |
| navigation / interaction / communication / voting logits | B x 3 / B x 5 / B x 6 / B x 15 |
| target logits / value | B x N / B |

`features.py` is the exact encoding specification. Self fields include own position, phase, own tasks/stages/incubation, public task bar, legal sabotage context, current navigation and exploration coverage. Player rows contain public roster identities, current legitimate sightings, actor-local last-seen information, public meeting participation and legitimately confirmed roles. They do not contain a hidden role roster or hidden current positions. Event rows retain observed/delivered times, subject/source, location and direct/public/claim provenance. A2 appends vent-enter, vent-exit and door-obstruction codes while preserving previous normalized event codes and all tensor widths; unchanged shape does not imply unchanged training distribution.

`memory.py`, `features.py`, `api.py`, `navigation.py`, `commitment.py` and `decoder.py` must be read together. Each crewmate owns a separate event memory, last-seen cache, navigation/commitment state and GRU hidden vector. Weights are shared; memories are not. Reset actor state at a new match. Preserve an actor's recurrent continuity across its own consecutive decisions, including meetings, while it remains active. Ghost task progression is engine-side and receives zero direct task reward; it does not become another living policy trajectory.

## Actions, masks and PPO

`actions.choose` samples conditional heads; `decoder.execute` translates the resulting `Choice` into legal simulator actions:

| Head | Values |
|---|---|
| navigation | Continue 0, temporary detour 1, new goal 2 |
| interaction | travel/navigation 0, owned task 1, report 2, lights repair 3, emergency 4 |
| communication | silent 0, share evidence 1, accuse 2, defend 3, question 4, strategic claim 5 |
| voting | packet roster slots 0-13; skip 14; legal mask excludes unavailable/self targets |
| target | packet-specific index into `Packet.candidates`; never a globally fixed player/task index |

Roaming samples interaction and then navigation or an interaction target. Discussion samples communication and its target when required. Voting samples the voting head. Unused `Choice` fields are -1. `target_mask` applies the exact conditional candidate restrictions. `model_packet` preserves the sampling masks for PPO; recomputation includes each sampled factor exactly once. A forced singleton has zero log probability and zero actor entropy. Never replace a masked neural decision afterward with an unrelated scripted action.

With pursuit enabled, the existing task head can select an unfinished owned task remotely and submit legal travel. Commitment starts only when actual task interaction begins. It persists through later stages, travel, incubation and meetings; reports and lights repair are legitimate pauses. The neural policy still selects tasks. Impossible-route and technical truncation counters remain explicit. See the existing commitment tests and `ppo.py` for recurrent sequence boundaries, GAE and minibatch reuse; this publication does not alter their settings.

## Simulator interface and learned-impostor integration

Key modules:

- `worker.py:Match` coordinates actor packets, independent memory and legal decisions; `collector.py` runs the existing complete-match collection waves.
- `environment.py:PatrickJaneGame` preserves the historical environment; `a2_environment.py` adds the opt-in A2 mechanics.
- `game.observe_pj(player_id)` creates a detached owner-specific `api.Observation`. `Match.prepare` and `Match.advance` implement the current crewmate policy boundary.
- `a2_opponents.py:TacticalOpponent.decide(obs)` and historical `opponents.py` expose the scripted-impostor boundary. Controllers receive **their own legitimate observations** and return `social_deduction.phase3_api.Intent` or PJ `api.Action` for supported extensions.
- Door geometry and the verified 14-vent graph live in `a2_geometry.py` and the source-backed Skeld assets. Navigation checks the current geometry; vent traversal and firsthand witness events are engine-validated.
- Existing `spectator/` and `run_spectator_v2.py` read live/replay records independently; spectator truth and retrospective reward diagnostics are never neural inputs.

The current environment is **10 neural crewmates + 2 scripted impostors**, not the older five-player frozen handoff. It uses shared crewmate weights, actor-local state, owned multi-stage tasks, ghost progress, lights-only non-door sabotage, seven door groups, graph-constrained vents, voting/communication phases and a frozen per-match 50/50 historical/upgraded opponent mixture. Optional modern roles are disabled. Existing historical conditions remain selectable controls. Rule abstractions and limitations are listed in `docs/patrick-jane/A2_RULES.md` and `A2_REGISTRATION.md`; do not treat this as a complete retail-game implementation.

For a separate learned impostor, first map its observation fields, own-role/teammate knowledge, action semantics, legal masks, target identities, decision timing and hidden-state resets explicitly to this interface. Keep one independent RNG stream and memory per impostor; no global hidden crew state may enter inputs. Own impostor role and teammate identity are legitimate. Door/vent extension actions require engine checks, not direct truth-state mutation. Freeze source/schema versions and validate team size, meetings, deaths, terminal handling and replay provenance before conducting a joint study. Neither checkpoint compatibility nor balanced competitive play is assumed. This task does not modify the collaborator's branch/model.

## Portable A2 launch, status and return package

The portable installer is delivered locally, separately from GitHub. Extract the entire versioned ZIP to a writable Windows x64 folder; open PowerShell in that folder. Start the existing training engine, independent Spectator V2 and terminal monitor with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Start-PatrickJaneA2.ps1 -Hours 12
```

The first invocation installs bundled dependencies offline and verifies CUDA, hashes and update102. It immediately trains after setup succeeds. On the target six-core / 16 GB laptop the existing hardware-aware cap normally selects six workers, then freezes it in registration. Each wave consists of one complete match per worker followed by PPO. `-Hours 24` extends the cumulative active-time target; it does not reset prior execution. Do not edit frozen source/configuration during a run.

Open the existing viewer without training (after setup):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Show-PatrickJaneA2.ps1
```

Closing the viewer does not stop training. It follows throttled snapshots from a selected environment and newly completed replays, with real concurrent-instance status; it is not a full-rate remote display of every concurrent game. Completed replay playback is labeled as playback. Progress is primarily refreshed during collection and at phase/update boundaries; a synchronous PPO update may have a quiet terminal interval. Playback speed affects the viewer only. The portable package retains the existing original map/sprites; the optional downloaded `assets/phase3_local` artwork cache is not newly published in Git. See asset provenance/notices; source-only rendering may use fallbacks until the optional cache is installed.

All run outputs are package-relative:

| Contents | Location |
|---|---|
| registration, source provenance, seeds, active time, outcomes, PPO history | `results/investigation-a2-v1/` |
| all resumable update checkpoints / final inference weights | `results/investigation-a2-v1/checkpoints/` / `weights/` |
| terminal / viewer logs | `results/investigation-a2-v1/console.log` / `viewer-*.log` |
| detailed batches and selected spectator recordings | `results/investigation-a2-v1/attempts/*/batches/` / `replays/` |
| Markdown/JSON report and curves | `results/investigation-a2-v1/TRAINING_REPORT.md`, `training_results.json`, `learning_curves.png` |
| automatic checksummed return ZIP | `exports/investigation-a2-v1-update-XXXXXX-export-NNN.zip` |

Ctrl+C requests a safe wave/update stop; alternatively run `Stop-PatrickJaneA2.ps1`. A second interruption can discard incomplete work. Rerun the **same launch command in the same package** to resume the latest verified checkpoint. Crashes do not count incomplete waves as learned decisions; their allocated seeds are not reused. Registration, duration extensions, interruption records, model/source identities, complete match records (including losses/cuts), recorded task/commitment/vote/door/vent/opponent statistics, PPO diagnostics, logs, selected replays and all checkpoints are retained. Missing measurements must remain unavailable, not invented.

`TRAINING COMPLETED` is printed only after a successful boundary, final report/weights and required export. Export failure now produces a failed status and nonzero exit while preserving checkpoints and reports. The exporter retains raw run evidence locally/in the owner's return ZIP; that evidence is **not** automatically published to GitHub. Copy the completed ZIP and its `.sha256` file to the primary machine. `Import-PatrickJaneResults.ps1 -Archive "path-to-return.zip"` verifies every member and imports into a fresh directory without replacing existing models. The return includes the source version and source files except installer payloads; the portable installer remains separately recoverable.

The A2 Windows active clock uses `QueryUnbiasedInterruptTime[Precise]`, excluding suspension. Setup, reporting/export, downtime and suspension do not consume collection/PPO allocation. Mocked clock/resume tests verify accounting; physical laptop sleep/resume has not been exercised. Historical Task Pursuit timing uncertainty remains unchanged. A hard crash may lose the last unsaved active-time sampling interval.

## Verification scope and remaining limits

Focused publication checks cover model identity, legal inference, recorded terminal counters, synthetic completion/resume, real export/import integrity and visible export failure; synthetic orchestration tests perform no PPO steps. A2 regression checks cover private ballot credit, full-task/zero-ghost accounting, doors, vents, witness privacy, masks and update102 transfer. No learning pilot, held-out comparison or reserved final-test game is part of this publication.

The bundled CUDA build includes Turing support, but physical GTX 1650 execution, thermal endurance and 16 GB memory headroom still require the secondary laptop. Development-host setup is not a measured performance guarantee for that GPU. A compatible driver is mandatory and startup has no silent CPU fallback. A2 training performance, improved deduction and generalization remain unverified. Preserve negative outcomes and uncertainty in the research journal before changing another regime.
