# Frozen four-crewmate environment for impostor training

This handoff publishes the **historical trained Model A and Model B**, their exact
five-player environment, and Spectator V2. It contains no Patrick Jane code or
weights. Four learning-era crewmates now run frozen inference against one externally
controlled impostor. No model is retrained by these commands.

Branch: [`codex/crew-impostor-training-handoff`](https://github.com/Hadisovic/social-deduction-ai/tree/codex/crew-impostor-training-handoff).
Pin the full 40-character publication SHA supplied in the draft PR/final handoff
before starting research; record `git rev-parse HEAD` with every run. Source
provenance and every required source/asset/checkpoint hash are in [manifest.json](manifest.json).
The base is main `a463a427`; compatible environment is `4bac4cdb`; selected models
come from registration `fab5c70e`. The viewer's owner-approved uncommitted V2 files
are individually hashed, with base `2e13afc`. No complete research branch was merged.

## Windows setup

Install **CPython 3.12.10 x64**. Its exact patch version is part of the unchanged
checkpoint contract. Check `py -3.12 --version`; a different patch version, NumPy
version or CPU/CUDA PyTorch build will correctly fail strict loading.

```powershell
git clone --branch codex/crew-impostor-training-handoff https://github.com/Hadisovic/social-deduction-ai.git
cd social-deduction-ai
# To pin the reviewed publication: git checkout --detach <40-character-SHA-from-PR>
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-handoff-lock.txt
.\.venv\Scripts\python.exe -m frozen_handoff.verify
```

The full dependency lock was generated from a new isolated environment. The shorter
`requirements-handoff.txt` lists the six direct dependencies; the lock pins their
transitive dependencies too. No neighboring worktree, private preferences, separate
impostor repository or asset download is required. Original pinned sprites and map
are bundled. Existing third-party attribution remains in `docs/phase3_asset_provenance.md`
and `assets/skeld/`; inclusion does not imply ownership of third-party artwork.

## Selected models and strict loading

| Model | Repository path | SHA-256 |
|---|---|---|
| A — original rewards | `artifacts/crew-handoff/original/checkpoint-batch-23-choices-2058.pt` | `3b843996c5b9189da06888c89b42bc03c40479c629a01d060ad8ff4099162f7c` |
| B — shaped V2 rewards | `artifacts/crew-handoff/shaped_v2/checkpoint-batch-23-choices-2092.pt` | `88103ee9bab248c7be4bbc64d39173300ffc7c8cbbbc40c7e95182a2c38355aa` |
| Shared Phase 4 belief predictor | `artifacts/phase4/belief.pt` | `e931b1fa8a421d9a329a29280d676bf52f793cc8f07c02b8da61f2b3e5e537d1` |

The selected files are exactly 659,491 bytes each and are ordinary Git blobs.
They retain their original optimizer/RNG/training-seed metadata, not raw observation
datasets. Published metadata summaries are in the manifest. The existing
`artifacts/phase5/release/policy.pt` is also required by the original fail-closed
provenance checker; it is **not** used as an opponent by this runner. Do not delete
it or weaken `verify_frozen()` merely because it is not used for inference.

```python
from frozen_handoff.models import load_model
model_a, metadata_a = load_model('A')
model_b, metadata_b = load_model('B')
```

`load_model` verifies the selected file hash, calls the original unchanged
`FourCrewPolicy.load`, and disables gradients. It does not migrate checkpoints.
The network is the historical **51,778-parameter feedforward candidate scorer**,
with pooled player/task encoders and separate actor/value outputs. Inputs are
global 42, players 4×29, own tasks 4×8, actions 96×23, target/task-row indices and
legal masks. It has no Patrick Jane GRU or five-head action architecture.

One set of crew policy weights and one frozen Phase 4 belief network are shared.
Each actor has a distinct `ActorMemory`, `PublicMemory`, candidate list, option
lease, pending task, trajectory and sampling generator. Never reuse one actor view
for another seat. Four actors using a shared predictor are not four independently
trained belief models. Create a new match for every episode.

## Complete five-player smoke example

Model B, sampled inference, default lobby, four frozen crew, registered scripted
impostor stand-in, one completed match, optional recorded replay:

```powershell
.\.venv\Scripts\python.exe -m frozen_handoff.run --model B --seed 2760000 --replay local-runs/handoff/model-b.jsonl --summary local-runs/handoff/model-b.json
```

Use `--model A` for the original-reward policy and `--greedy` for argmax inference.
Sampling is the default; each actor uses its own deterministic generator keyed by
initialization 17, match seed and public actor ID. Replay output is exclusive: use
a new filename for repetitions. Runs have a 600-second wall safety ceiling; game
timeout is separately 900 simulated seconds. A wall abort leaves an incomplete
`.writing` recording that the viewer will not misrepresent as a complete game.

Programmatic execution with an external controller:

```python
from frozen_handoff.models import load_model
from frozen_handoff.match import FrozenCrewMatch

def factory(*, seed, player_id):
    # Return a fresh object with decide(Phase3Observation) -> Intent.
    # Load your own impostor model/encoder here; keep seed out of neural features.
    from my_impostor_bridge import Controller
    return Controller(seed=seed, player_id=player_id)

policy, _ = load_model('B')
match = FrozenCrewMatch(3600000, model='B', controller_factory=factory)
unused_crew_rollout, summary = match.collect(policy, initialization=17)
# Settle your impostor trajectory from match.game.result in trusted trainer code.
# Never insert engine/truth/reward/replay objects into policy observations.
```

Or use the CLI import hook:

```powershell
.\.venv\Scripts\python.exe -m frozen_handoff.run --model B --seed 2760000 --controller frozen_handoff.match:scripted_factory
# Replace the stand-in with: --controller my_impostor_bridge:factory
```

No partner code is copied, overwritten or automatically imported. Integration
interface: `frozen_handoff/controller.py`; coordinator: `frozen_handoff/match.py`.
The factory receives only an owner ID and private RNG seed, not game state.

## What the impostor receives; how to bridge the previous interface

`LegalImpostor.decide` passes the exact immutable `Phase3Observation` projected by
the common engine: own role/position/tasks/cooldown, public roster and map,
phase/meeting context, wall-visible players and bodies, legitimate personal events,
received public/claimed events, own navigation/interaction status and typed legal
actions. It receives no hidden crew positions/roles, crew memories, truth log,
spectator data or global belief predictions. Post-death inference is stopped.
Ballot targets become public only at result resolution. Claims remain claims.

Return the shared `Intent` type. Actions include navigation, legal kill/fake task,
report, emergency, typed structured claim, vote and skip. `WAIT` is an omission.
Parameterized locations must be inside public walkable geometry; claims must use
valid public identities/regions and past timestamps. The gate rejects stale or
illegal actions rather than silently replacing them with an advantageous action.
External Python is cooperative code, not a security sandbox.

The inspected partner branch is `eee7ee6` (full SHA in the manifest). Its earlier
`impostor_research.ImpostorMatch` uses one frozen focal crew plus three scripts,
optional opponent safeguards, and its own encoded arrays. Its V4/V4.1 wrappers
have separate option/reward/pursuit semantics. **Those numeric action indices and
tensor shapes are not this interface.** The partner must encode this legal envelope
using their matching feature schema, build their candidate list, sample/argmax the
impostor policy, then decode the chosen candidate into a shared `Intent`.

Do not treat the crewmate network's 96-candidate indices as impostor actions.
Do not import the old partner match wrapper merely to get an encoder: it constructs
a different opponent population. If their encoder requires additional own-state
fields, review an allowlisted projection explicitly; never pass `match.game` into
it. Their model and optimizer are not modified here. This handoff provides a
controller callback, **not a drop-in SB3 Gym wrapper**. A trainer can collect its
own actions/log-probabilities at callbacks and settle terminal rewards externally.
Keep elapsed simulated time for transition discounts when callbacks skip forced
meeting windows. Impostor rewards are the partner trainer's separate registered
objective; they are not the crewmate ledgers below.

## Exact active mechanics, rewards and opt-ins

The historical V2.2 lobby has **four crew + one impostor**, 0.2-second physics,
0.6-second scheduled decisions, crew vision 4.5 / impostor vision 6.75 with walls,
45-second initial/repeat kill cooldown, speed multiplier 1, two-character-width
kill distance, one emergency, 15-second discussion, 120-second voting, confirmed
ejects, nonanonymous ballots revealed together, always-updated task bar, visual
proofs, and one common/one long/two short individually owned tasks. Task victory
is enabled. Ghost tasks travel automatically, retain ownership and freeze during
meetings; ghost knowledge never enters a living controller. Ejection/parity precedes
task victory; timeout is a draw. Map/physics are documented approximations.

The original option leases, event interrupts and per-actor memory updates remain
unchanged. The handoff subclasses only match construction to allow new development
seeds and inject a checked impostor controller; exact baseline behavior is tested
against the original coordinator. It does not force learned crew to report/vote.

| Crew ledger coefficient | A | B |
|---|---:|---:|
| Living owned task | +1 | +1 |
| Accepted report | 0 | +1.5 |
| Correct / incorrect ballot | 0 / 0 | +2 / -1 |
| Skip / automatic ghost task | 0 / 0 | 0 / 0 |
| Death | -5 | -5 |
| Team win / loss / draw | +10 / -10 / -5 | +10 / -10 / -5 |
| Time per 0.6 simulated seconds | -0.005 | -0.005 |

Historical accounting is default, including its documented same-step ejection
eligibility defect. `--accounting eligible-v1` selects
`crewmate-phase21-accounting-v1`, using the pre-step eligibility ledger. This
changes reported credit, not physics, rewards coefficients or frozen action
probabilities; label results accordingly. Do not rewrite historical benchmarks.

Baseline scripted opponent remains default. `--opponent v1` selects the Phase 2
mixture; `--opponent v2` selects the Phase 2.1 goal-retaining mixture. They are
explicit behavior changes, not additional trained models. Social guards,
influence overlays, confidence-based speaking order, witness-priority overrides,
deduction-only scenarios, sabotages, additional players and learned impostors are
**disabled by default**. Relevant historical opt-in source modules are supplied;
they are not automatically applied to frozen crew. This five-player engine does
not have Patrick Jane's new sabotage system.

## Spectator V2 and original assets

```powershell
.\.venv\Scripts\python.exe run_spectator_v2.py --replay local-runs/handoff/model-b.jsonl --watch local-runs/handoff --preferences local-runs/handoff/viewer-preferences.json
```

The existing UI is preserved: original Skeld and characters, roster/roles,
individual tasks, impostor cooldown, private selected-actor probabilities and
beliefs, rewards/evidence, meetings, timeline/speed buttons, settings, library
and retrospective analytics. The only application packaging change is default
local directories. It reads completed `spectator-v1.0` files, not a live game.
Settings begin empty with omniscient/hidden-ballot reveals off; explicit spectator
debug modes never feed training. Loaded recordings may contain privileged truth,
so keep them local. No private preferences, raw replays or experimental logs are
published. Library analytics use retrospective outcomes, not independent evidence.

## Reproducibility and validation

```powershell
.\.venv\Scripts\python.exe -m frozen_handoff.verify
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_frozen_handoff.py -v
# Optional original viewer regression suite:
.\.venv\Scripts\python.exe -m pip install -r requirements-handoff-dev.txt
.\.venv\Scripts\python.exe -m pytest tests/test_spectator_v2_data.py tests/test_spectator_v2_ui.py
```

Use fresh output filenames to compare identical seeds/model/sampling mode/options.
Compare semantic event hashes and actions/rewards, not wall-clock profiling numbers.
`verify` checks selected bytes, unchanged strict source/runtime locks, bundled
artwork hashes and safe viewer defaults. Windows LF/CRLF normalization in the original
checkpoint checker is retained; the publication manifest uses canonical LF for
text and exact bytes for binary files. The legacy map JSON's raw CRLF bytes are
separately preserved by `.gitattributes` for its original strict source lock.
Do not disable a failing lock to accommodate another package version.

Allowed seeds are engineering 2760000–2760099, new handoff training
3600000–3609999, and new handoff validation 3610000–3610099. These are development
ranges, not reserved Phase 6 tests. The old registered training/validation/final
ranges are rejected by the new runner. Distinct actor decisions are not distinct
games, and repeated seeds are not independent samples.

## Limits and recommended research use

These are small exploratory policies: one initialization, approximately 2,048
decisions per arm, shared belief predictor and scripted opponents. Original
ordinary validation won 8/8 by tasks; shaped B won 4/8 ordinary games and reported
more in controlled contexts. Those eight scenario clusters do not establish
general deduction skill. Reporter shortcuts, evidence scarcity, voting reward
exploitation and the historical accounting defect remain documented. Neither
model is an accepted human-level social-deduction agent; Phase 6.1 remains pending.

First verify the partner adapter's action/feature semantics against this runner.
Then train only the impostor against sampled frozen B, retain A as a held-out
opponent/control, and evaluate both with paired fresh seeds and ordinary task wins
enabled. A later explicitly registered A/B opponent mixture can reduce overfitting.
Keep reporter role/opportunity strata, task contribution and legal evidence exposure
alongside win rate. Do not quietly add crew safeguards that make the opponent
stronger while still calling it the learned Model B.

The strict crew runtime is CPU PyTorch 2.6.0+cpu. A CUDA build or different Torch
version will fail its lock. For a separate GPU impostor stack, a future process/RPC
bridge can isolate the verified frozen CPU simulator from the GPU optimizer; this
handoff does not claim that transport is implemented. Changing versions requires
an explicit compatibility study, not metadata editing. Confirmed interface
differences above are code observations; expected benefits of opponent mixtures
or GPU separation are recommendations, not measured outcomes.
