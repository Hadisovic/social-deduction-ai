# Architecture — information boundary, simulation and beliefs

## Current executable architecture (Phase 6 foundations)

Phase 5's released strategic scorer chooses one focal crewmate's goals from
legitimate actor packets, historical memory and frozen belief predictions; A*
executes navigation. Phase 6 opts into asymmetric vision through the existing
trusted engine and keeps the historical defaults. `phase6/` reuses the packet,
candidate network, PPO update and base transition engine while versioning option
durations, memory/belief conditions, route descriptors and experiment records.
Role labels/rewards remain trainer-only. A longer impostor sight range never
exposes private tasks or unseen positions. The separate preview has a frozen
learned crew and scripted opponents; no learned impostor exists yet.

See [Phase 6 plan](phase6_experiment_plan.md) and
[vision rules](phase6_vision_and_rules.md). Large batches/final tests are gated;
Phase 6 is not complete. Frozen Phase 4/5 releases and historical results remain.

## Planned Phase 6 expansion (not executable capabilities)

The [current roadmap](phase6_roadmap.md) finishes the unchanged nine-model crew
study (6.1), then trains/validates a learned impostor against learned/scripted
crew (6.2). Phase 6.3 plans fully AI-controlled matches with exactly two impostors:
6 crew + 2, **8 crew + 2 (main target)**, then 10 crew + 2. Player-count/role-count
interfaces, beliefs, legal masks, spawn safety, outcome logic and compatible
replays require review before implementation; current matches remain four crew
and one impostor. Existing task and voting semantics are not changed now.

Phase 6.4 then plans alternating role updates, historical opponent pools,
competitive adaptation and eventual self-play. Each actor must retain private
observations, its own memory and individual task ownership, even if future
policies share weights. Vision stays crew 4.5 / impostor 6.75 with native LOS.

A future local dashboard reads training telemetry and resource/ETA measurements;
a separate compatible checkpoint/replay viewer renders Skeld with labeled
spectator-only truth. Neither may inject debug state or consume training RNG.
Separate one/two-process benchmarks, optionally three, will measure reproducible
throughput before choosing future concurrency. These tools are planned and are
not added by the documentation update.

## Historical Phase 4 observer architecture

`Phase3Game` -> trusted `project_game` -> immutable `Phase3Observation` ->
`ActorMemory` -> identity-free candidate features -> shared `CandidateNet` ->
four public-ID probabilities and entropy. `ScriptedMatch` retains action control;
no belief enters a strategic policy yet. The Phase 1 fixture API and Phase 2 map/
navigation remain unchanged, as do all Phase 3 engine/controller/renderer files.

Memory/features/model modules under `belief/` cannot import world truth, labels,
controllers or replay logs. `phase4_data.py` is the trusted boundary joining actor
features with separate `role_labels` loss targets. Whole-match train/validation/
test splits and a reserved patient impostor family support evaluation. Model
selection and scalar temperature fitting use validation only. The release model
is a 2,849-parameter equivariant set scorer, with explicit memory instead of
recurrence; prediction never chooses an action. See [belief model](belief_model.md)
and [measured Phase 4 results](benchmarks/phase4/README.md).

The remaining sections preserve the rationale of the earlier foundation phases.

## Research definition

This project tests whether a compact learned crewmate using legitimate partial
observations, persistent evidence memory, and explicit role uncertainty improves
team win rate over scripted and memory-ablated baselines against held-out teammate
and opponent strategies. Known-map navigation and task mechanics are controlled
infrastructure. Improvement must generalize beyond training seeds and bot families;
a null result is informative. Commercial-client control is outside this target.

## Accepted roadmap and scope

The six phases in README separate boundary, map/navigation, simulation, belief,
strategy and multi-agent expansion. Phase 1 deliberately began with an executable
information boundary before a transition engine. Phase 3 subsequently implemented
the playable engine; Phase 4 consumes its validated actor interface.

Phase 1 provides immutable snapshots, a crew observation projector, safe action
candidates, typed event provenance, a minimal observation-history container,
independent identity assignment, and a separate role-label interface. Its truth
objects are fixtures/adapters for a future simulator, not a playable game.

## Module map and data flow

- `social_deduction/actor.py`: actor-safe immutable value types and minimal history.
- `social_deduction/truth.py`: simulator-only snapshots, private state, identity setup.
- `social_deduction/observation.py`: trusted projection and pure action candidates.
- `social_deduction/training.py`: privileged role targets; never an actor dependency.
- `test_information_contract.py`: paired-world, temporal, provenance and boundary tests.

Original Phase 1 fixture orchestration: truth -> `observe(world, player_id)` -> actor observation ->
`remember(history, observation)` -> strategic policy. Only the trusted runner calls
the projector; policies must not receive its closure, world, training labels,
simulator debug dictionaries, RNG state, or helper references. Action candidates
are computed from the safe observation only. The returned candidate tuple is the
legal-action list; no separate truth-dependent mask is allowed.

The actor module does not import simulator or training modules. The projector
constructs explicit field allowlists and validates the immutable actor graph;
there is no `asdict(world)` filtering or arbitrary event metadata passthrough.
Python modules are an architectural boundary, not a security sandbox against
malicious policy code. Process isolation can be added for untrusted policies.

Static map values are shared immutable public data. Phase 1 uses small rectangular
fixtures with range and wall-occlusion checks. It does not adapt Skeld geometry or
change navigation. Phase 3 now tests actual renderer/visibility/event integration.

## Later learning and evaluation

Start with structured-memory policy inputs, one learned crewmate and frozen mixed
bots. Role labels can supervise beliefs; a later centralized critic can receive
truth separately. Neither labels nor privileged reward feedback may enter actor
history. Do not assume SB3 masking and recurrent PPO compose automatically.

Split whole matches and bot families into train/validation/final test sets. Choose
checkpoints on validation only. Evaluate team win-rate change against a strong
scripted focal player, Brier/log loss before role disclosure, and memory/evidence
ablations across at least three training seeds. Randomize identity independently
of roles. Use paired seeds and uncertainty intervals; no raw frame-level splits.

## Phase 2 navigation infrastructure

`navigation_service.py` is separate from `social_deduction/`. A shared
`NavigationPlanner` uses public native map geometry and bounded reverse-distance
field caches. Each player owns a `NavigationService` for task/room/location targets,
physical motion commands, arrival, cancellation, replanning and explicit outcomes.
`AmongUsMapEnv.advance_motion` executes the same swept collision function used by
Gym and manual control. The service cannot assign player positions.

`benchmark_navigation.py` executes all 3,906 directed pairs of 63 destinations plus
1,008 continuous random spawns stratified over 21 room/corridor regions. See
[navigation design and API](navigation_service.md) and the
[acceptance evidence](benchmarks/phase2/README.md).

The trusted runner owns simulator references and applies speed-bounded commands.
Actors issue macro-intentions and receive only approved outcomes. Geometry changes
must be public/observed: a planner must not be refreshed from hidden door or player
state. Service diagnostics are not automatically actor observation fields. Phase 1
interfaces and legacy navigation experiments remain unchanged.

Phase 3 separately supplies the five-player match engine, tick ordering, task
rules, authenticated event publication and Skeld visibility with paired-world
leakage tests. The navigation service remains independent of those rules.
