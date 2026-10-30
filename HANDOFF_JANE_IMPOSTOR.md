# Lizbon V1: current execution status

30 October2026. **PILOT STOPPED ON NATIVE GEOMETRY ASSERTION. DO NOT RESTART.**
Code, checkpoint and compact failure evidence were pushed to
codex/jane-impostor-v1 (results commit e3757a3; source commit e91d19e).
No merge or deployment. Waiting for a reviewed engine release and new training approval.

## Observed outcome

- Frozen training source: e91d19eb4bd945497893b289b18f468803800634.
- Registered start17:21:57.768 UTC; failure17:34:49.560 UTC,12m51.79s elapsed.
- Owner's90-minute cap was not spent; compute stopped on exception.
- First game: crewmate task win,263.6 simulated seconds,4 kills, no illegal actions
  or navigation failures. It used initial untrained weights.
- Saved checkpoint-001.pt:908 observations,8 optimizer passes; strict loading
  passed and Lizbon tensors changed. Model has1,036,561 parameters.
- Second game used update8, made5 kills in its preserved prefix, then failed
  at the next step after last valid tick1293/258.6s. No outcome and no PPO update
  from that match. Never infer improvement from these different scenarios.
- No selection-lock.json, final.pt, results.json or supervisor pilot-results.json
  was generated. Postmortem is failure-report.json; do not invent successful completion.
- Zero development/final evaluation games. All reserved sets remain sealed.
- Parent session7604 exited1. Supervisor82060 and train81468/62012 are gone.
- Old live viewer36524 was closed because its source job stopped. A completed
  initial-policy baseline replay is available via run_lizbon.py.
- Normal-speed baseline playback opened: launcher69980/GUI2192, initial weights,
  seed9310001, selected impostor p1. This is playback, not a new simulation.

## Confirmed cause and next action

A legal Storage closure covers dead p8's stored position after its corpse was
reported. Native assert_invariants validates all stored positions against the
dynamic door map. All living actors remain in valid geometry. A minimal saved
state fixture reproduces this with no model inference or optimizer steps.

Pinned native source remains unmodified. Proposed patch checks inactive and
underground stored positions against base floor, living surface positions against
current collision geometry. git apply --check passed; patch has not been applied.
Crew/shared developer should review it, add live/dead/vent/meeting regressions and
publish a new immutable engine release. Then obtain owner approval for another
registered study. Do not silently resume this failed one or extend its deadline.

## Evidence and release

- Report: docs/jane-impostor-v1/pilot-20261030/README.md.
- Five graphs, actual baseline GUI image, results.json, PPO/behavior counters,
  geometry-diagnostic.json and PROPOSED_NATIVE_FIX.patch live alongside it.
- Named initial.pt/checkpoint-001.pt and checksums:
  artifacts/lizbon-v1/pilot-20261030/manifest.json.
- Strict source/runtime/settings checkpoint load verified. Crew artifact hash
  unchanged; final in-memory crew hash check was not reached on exception.
- Raw preserved files remain ignored under
  local-runs/jane-impostor-v1/pilot-20261030. Original collecting status files were
  backed up before stale statuses were corrected to failed.
- Match2 raw .jsonl.writing has no fabricated result. Native viewer indexes
  completed files. Match1 playback is an INITIAL-POLICY baseline, not post-update.
- Do not edit protected docs/research/ISSUES.md. Engineering failures are recorded
  in docs/jane-impostor-v1/ENGINEERING.md and the report instead.

## Setup and collaboration

Own branch is based on shared foundation3766cba. Model name is Lizbon; Jane is the
shared game foundation. External source/runtime information remains in ignored
local-runs/jane-impostor-v1/source.json and runtime/Scripts/python.exe.
Engine pin:d459f38674ebc537acde20647cbd07700c494533; frozen task-pursuit update102
crew (not A2-trained),10 crew vs2 Lizbon actors sharing weights/private histories.
CPU Python3.12.10/Torch2.7.1+cpu, one Torch thread; i9-14900HX24C/32T.
Thirty regression checks passed before source freeze, plus strict release load,
zero-inference geometry reproduction and actual native viewer capture afterward.

Owner/impostor developer: review behavior, improve the next registered protocol
after compatibility is restored; no larger network justified by this evidence.
Crew developer: improve A2 reports, witnessed evidence, ballots and sabotage
counterplay. Publish frozen source/settings/checkpoint releases. Cross-play fixed
releases first; adaptation/self-play after competent learned base roles.

Preserve unrelated existing old pilot README and NAVIGATION_V2_PLAN.md changes.
The finish-lizbon-pilot-and-results heartbeat was removed after the interrupted
pilot, report, checkpoint and gameplay work were published. No training remains running.

## Door presentation update

Original Skeld sprite transitions replace red door bars in both viewer launch paths.
See docs/jane-impostor-v1/DOOR_VIEWER.md for provenance, real-replay GIF and tests
(30 passed,5 optional-recording skips). The0.25-second cadence is reconstructed;
original clip FPS was unavailable. The pinned engine and Lizbon neural source
remain unchanged; the native geometry assertion fix is still pending.
