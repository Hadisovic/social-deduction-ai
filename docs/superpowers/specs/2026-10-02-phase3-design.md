# Phase 3 implementation design

The owner approved an implementation phase and delegated architecture/minor-rule
decisions, with a mandatory visual demo, information-boundary tests, 1,000 seeded
matches, documentation and a committed/pushed result. The later asset request
adds local original-game character/task images from the specified sources while
preserving the earlier no-unlicensed-artwork-publication condition.

Use a fixed-tick five-player rules engine with separate entities/controllers and
one trusted observation projector. Both visual and headless modes run the same
`ScriptedMatch`; the renderer is read-only. Reuse frozen map geometry and Phase 2
navigation/physics. Extend the actor API additively; preserve Phase 1 files.
Alternatives of a second movement system, retrofitting all rules into the old Gym
environment, or a general distributed event system add risk without serving this
phase. High-level learned policies can later replace `decide(observation)`.

Define private interruptible tasks, occluded local sight, legal kills and bodies,
authenticated report/meeting transitions, bounded structured claims, public votes
without role reveal, central task/ejection/parity outcomes and explicit timeout.
Keep the initial task quota; defer hidden-death reassignment until public absence
is established at a meeting. Reports precede kills in simultaneous ticks. Meetings
freeze positions and timers, cancel stale actions and retain event history.

Use independent identity/color/role/task/behavior seed domains and detached frozen
actor values. Candidate derivation depends only on the actor packet. Verify the
whole decision input and actions under hidden-world mutations, not just individual
fields. Add focused rules/physics/visual tests and retain all 197 historical tests.
Hash actor inputs/actions and privileged truth traces across independent replays.

The public repository contains original fallback artwork code, an exact pinned
asset manifest and an optional loader/installer. Requested commercial PNGs live
only in ignored `assets/phase3_local/`; normal execution is offline. Task art is
illustration for timed interactions, with no commercial task mechanics added.

Acceptance requires a no-flags `python run_phase3.py` desktop launch, inspection
of multiple complete matches, full regression, and at least 1,000 primary seeded
matches plus independent replays with no invalid state, crash, timeout, navigation
failure, illegal scripted action or deterministic mismatch. Record all results,
provenance, performance and limitations before final commit/push. Phase 4 is out
of scope. The implemented API/rules and test evidence are documented in
[`docs/social_simulator.md`](../../social_simulator.md).
