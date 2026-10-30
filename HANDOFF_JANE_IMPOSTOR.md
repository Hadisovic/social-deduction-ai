# Lizbon V1: current execution status

30 October2026. Owner approved the90-minute pilot. Preparation complete;
pilot launch details are appended below once the frozen source commit exists.

## Branch and dependency

- Own development: codex/jane-impostor-v1, from shared foundation3766cba.
- Model name: Lizbon. New recurrent neural code and random initialization.
- Fixed native engine: Patrick Jane A2 commitd459f38674ebc537acde20647cbd07700c494533.
- Crew: frozen task-pursuit-v1 update102,10 actors. Not A2-trained.
- Impostors: two actors share new Lizbon weights; independent memory/RNG.
- Published crew inference SHA256:
  da1d43d73c140ac39bf44834746dbae605b9b36d3c6c7af8a4f2be975e535e3c.

The external source snapshot is an ignored local dependency, not copied neural
code in this branch. Its path is stored in ignored local-runs/jane-impostor-v1/source.json.
Runtime: local-runs/jane-impostor-v1/runtime/Scripts/python.exe, Python3.12.10,
Torch2.7.1+cpu. Historical runtime/weights unchanged.

## Protocol

Use tools/run_jane_impostor_pilot.py. New output directory:
local-runs/jane-impostor-v1/pilot-20261030-seed731.
Maximum90 minutes from registered start. Training at most60 minutes/four
games. Each match has a20-minute wall cap. Only natural complete matches feed
PPO. Expired optimizer chunks discard unstepped gradients. No automatic restart,
budget extension or tuning from later outcomes.

Lock the final completed optimizer boundary by checksum before development.
Initial versus trained use the same two fresh seeds9320001/2. Technical cuts are
missing outcomes, never losses. Old/new reserved final partitions remain sealed.

During healthy work, inspect pilot-status.json, train/status.json, training console
log and matches.jsonl. Preserve processes and never edit loaded v1 Python source.
After completion inspect pilot-results.json, REPORT.md, selection-lock.json,
per-run results/checkpoints/replays and graphs. Record actual timings, cuts,
failures and sufficiency in docs/jane-impostor-v1. Do not present training wins
as accuracy or claim globally best. Keep raw private outputs ignored.

## Verified implementation

29 regression cases passed in42.44s. One additional deterministic two-replay
check is run before freeze. Actual native spectator GUI capture verified using
the untrained smoke-004 replay. No full training started during engineering.

See docs/jane-impostor-v1/README.md, ENGINEERING.md and jane/impostor/STUDY_PLAN.md
for architecture, exact rewards, decisions, safeguards and deferred features.

## Viewer and collaboration

run_lizbon.py launches the native spectator only; it never starts training.
Use --watch with the train directory for live observation; --replay for playback.
Select a current impostor in the roster. Ground truth spectator values are never
fed to the actor. Keep visible GUI windows only for this requested viewer.

Owner/impostor developer: inspect active search, repeated goals, fake duration,
witnesses, self-reports and meeting choices. Friend: improve crew A2 behavior and
publish a frozen source/settings/checkpoint. Coordinate common rules before any
change; cross-play fixed releases first, adaptation/self-play later.

Finish-and-push flow: commit own code/docs/curated evidence to own branch;
no merge/deploy. Preserve unrelated existing changes in old pilot README and
NAVIGATION_V2_PLAN.md. Leave protected docs/research/ISSUES.md untouched.
