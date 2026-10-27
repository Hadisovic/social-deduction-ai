# Post-pilot implementation design — 2026-10-27

Owner brief: attachment fcf4edbd-ff96-499d-bdc5-f6ec9300ca41. Implementation and short engineering validation are authorized; a new full training pilot is not.

## Context map (reviewed against imports and tests)

| Files | Purpose / change |
|---|---|
| `crewmate_v2/engine.py` | V2-only eligible ballots, early resolution, meeting task/ghost freeze |
| `crewmate_v2/fourcrew/` (new) | Four private actor views, one shared policy, simultaneous intents, independent trajectories, bounded engineering runner and strict new checkpoint schema |
| `crewmate_v2/learning/config.py` / `rewards.py` | Reference historical coefficients and event attribution; keep v1 byte-identical |
| `crewmate_v2/runtime.py`, `phase6/env.py`, `phase5_features.py` | Reuse observation encoding and mechanical option projection, without modifying historical kernels |
| `crewmate_v2/collection/ppo.py` | Reuse segmented GAE/PPO update, with segments keyed by match and actor |
| Separate spectator worktree | Current-meeting display, hidden ballots, speed hitboxes, explicit private debug view, per-agent traces |
| `tests/test_v2_mechanics_repairs.py`, `tests/test_v2_fourcrew.py` | Mechanics, actor privacy, rewards, sampling, death/terminal trajectory isolation, replay and checkpoint regressions |
| `docs/PROJECT_PROGRESS.md` | Existing historical project index; extend it and link one detailed issue register |

No database or deployment migration. Existing source-locked checkpoints stay loadable from their preserved historical branch, not silently from repaired mechanics. The new four-crew schema fails closed on historical checkpoints.

## Selected architecture and alternatives

Use one match with four independent actor views and one scripted impostor. Share neural weights, never actor memories. Schedule held options independently using legitimate observations and existing mechanical intent conversion. Inference batches only actors ready to decide; random generators are seeded independently per match/actor. Freeze the policy version while collecting complete matches, then centrally update. Keep GAE segments separate by match/actor; a dead actor issues no later policy actions. Its final transition is retained through terminal settlement so delayed shared outcome credit can be discounted correctly without querying ghost observations.

A focal actor plus three scripts cannot implement the requested learning design. Four separate policy networks would lose shared learning and add avoidable complexity. One concatenated omniscient packet would violate private-observation requirements. These alternatives are rejected.

Individual task/report/vote/death credit belongs only to the responsible actor. Shared team outcome is broadcast once to each of four actor trajectories (+10/-10 per actor), explicitly reported separately from the single team outcome; summing actor returns gives four terminal credits by design. Automatic ghost work earns zero individual task credit in both new four-crew controls. No action is imposed to report, vote or flee.

Budgets count individual non-forced actor decisions, not joint ticks. Complete-match batches can exceed a target minimum; actual decisions must be logged. No equivalence to the historical focal 512-choice budget is claimed. This implementation initially prioritizes deterministic complete-match correctness; worker-throughput estimates need a new measured smoke.

Version new shaping with report +1.5 and retain historical +1.25 profiles and records. Keep all other approved coefficients. New evaluation must include innocent/impostor reporters and a zero-update reference. No full experiment launches in this task.

## Validation and stopping boundary

Test eligible ballots/timeout/early resolution, task freeze/resumption and incubations; independent memories across meetings and reset; actor-specific masks/rewards and death isolation; complete-match GAE/log probabilities; recording on/off equality; checkpoint round trips and incompatibility; meeting-scoped replay display and speed clicks after resizing. Use development engineering seeds only. Preserve artifact hashes before and after work. Record survival evidence from the existing sampled replays, with sample limits. Finish separate commits and research issue documentation, then stop for owner review.
