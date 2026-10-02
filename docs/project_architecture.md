# Architecture — information boundary and navigation

## Research definition

This project tests whether a compact learned crewmate using legitimate partial
observations, persistent evidence memory, and explicit role uncertainty improves
team win rate over scripted and memory-ablated baselines against held-out teammate
and opponent strategies. Known-map navigation and task mechanics are controlled
infrastructure. Improvement must generalize beyond training seeds and bot families;
a null result is informative. Commercial-client control is outside this target.

## Accepted roadmap and scope

Keep the six phases in README. The refinement is to implement a small executable
information boundary now, without a transition engine. A documentation-only
contract would not catch leaks; building the full simulator now would entangle
rules, physics, and observations before the boundary is tested.

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

Future orchestration: truth -> `observe(world, player_id)` -> actor observation ->
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
change navigation. Phase 3 must test actual renderer/visibility/event integration.

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

Phase 3 still needs the five-player scripted match engine, tick ordering, task
rules, event publication and Skeld visibility adaptation with wrapper leakage
tests. It is not implemented by the navigation service.
