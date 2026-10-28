# Compatibility audit — shared handoff versus impostor V4.1

Publication c3320583f7d562bc69ea527580a91ddc5db54f18, confirmed by developer PR12.
Original impostor source eee7ee608edb952c4b17c8d8d1b0f6303431c1a0; selected83/8192
checkpoint SHA69965afd8fab28e6ca5bdda46d9f8efdd8d6036ce57a05b86ff89fe22cf17ca9.
This document distinguishes inspected facts from checks still pending.

| Area | Verified difference / consequence |
|---|---|
| Map/base phase3 API | Git source comparison shows identical map geometry/base typed action and observation code |
| Population | Old: one full-29 learned crew +three scripts +report/witness guards. New: four frozen A/B learned crew; no social overrides by default |
| Crew architecture | A/B are51778-parameter FourCrewPolicy, not old full-29 or a new recurrent architecture |
| Runtime | Old Python3.12.3/NumPy1.26.4/Torch2.9.1+cpu; new3.12.10/2.5.3/2.6.0+cpu. Strict original load is not interchangeable |
| Task/ghost timing | New engine pauses task progress, ghost travel/work and incubation during meetings |
| Ballots | New engine validates eligible living voters/targets and resolves once all eligible ballots arrive; old waited for deadline |
| Controller boundary | New callback receives Phase3Observation and returns typed Intent, not old Gym packets/candidate indices |
| Missing old input context | Old encoder also needs public V2 task bar/events and own fake remaining time; these require an explicit actor-only projection, not engine injection |
| Navigation/fake options | Preserve neural choices; implement old bounded travel lease and persistent same-console fake behavior explicitly in the adapter |
| Crew rewards | A original versus B shaped ledgers; historical/eligible-v1 accounting explicit. These are not impostor rewards |
| Impostor rewards | Retain a separately versioned V4.1-style objective with simulated elapsed-time accounting and correct terminal handling |
| Viewer | Use published Spectator V2 recording format and actual neural/reward telemetry; no map/asset changes |
| Sabotage | Not functional in this published historical handoff; do not imply prior sabotage-first work is present |

The existing checkpoint cannot pass unchanged source/runtime validation in the new
tree. The integration uses an **explicit inference-only transfer**, not weakened
locks: load with the unchanged original loader in its original checkout/runtime,
export exact tensors without optimizer, then validate architecture, tensor hashes,
synthetic forward parity and legal observation/candidate semantics in the shared
runtime. Outcomes still face opponent/physics distribution shifts. Fresh training
or a separately approved warm-start study may be appropriate; no claim that
retraining is unnecessary.

Pending checks: strict A/B loads, exact publication verifier, synthetic transfer
parity, actor-local projection/privacy, legal actions/timing/rewards, completed
matches, deterministic replay and Spectator playback. See CHECKLIST.md.
