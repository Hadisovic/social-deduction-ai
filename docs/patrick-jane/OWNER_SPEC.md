# PROJECT PATRICK JANE
## Repository-Aware Implementation Specification for Codex

You are the lead implementation engineer for **Patrick Jane**, a new reinforcement learning crewmate agent for the Social Deduction AI project.

Repository: https://github.com/Hadisovic/social-deduction-ai

Patrick Jane is named after the character from *The Mentalist*. Its purpose is to independently learn how to win Among Us as a crewmate, including navigation, task completion, survival, evidence gathering, social deduction, communication, and voting.

The owner and I have already designed and approved its neural network architecture, action system, memory design, rewards, and initial training configuration.

**Your responsibility is to implement that approved design faithfully in the existing repository—not redesign it.**

---

# 1. NON-NEGOTIABLE INSTRUCTIONS

**The neural network architecture and gameplay design described in this specification are locked.**

Do not independently:

- Change the algorithm from recurrent PPO.
- Change the three encoder branches or their agreed hidden-layer widths.
- Replace the shared 256-unit GRU.
- Remove or redesign the attention mechanisms.
- Change the five action-output heads.
- Remove the timestamped event-memory system.
- Change the agreed reward values.
- Replace our navigation decisions with a different policy.
- Introduce a different learning architecture because it seems simpler or more efficient.

If you believe a significant architectural or behavioral design change is necessary, **stop, explain the proposed change and its consequences, and request the owner's approval before implementing it.**

Normal engineering decisions involving data structures, module organization, batching, testing, serialization, and efficient computation are permitted if they preserve the approved design.

The owner has explicitly approved extending the existing environment to support:

1. Configurable games with up to 15 players.
2. Two-impostor games with 10 learning crewmates.
3. Actual sabotage mechanics, including lights, O2, and reactor.
4. Shared-policy multi-agent crewmate training.
5. Parallelized simulation and integration with the existing spectator.

These are approved environmental and infrastructure changes, not permission to modify Patrick Jane's architecture.

# 2. REPOSITORY CONTEXT AND SOURCE SELECTION

You already have access to the repository and its local working directories.

Inspect the current checkout, all relevant local branches, worktrees, uncommitted files, and running processes before making changes.

Do not assume GitHub's main branch contains the latest work.

The following branches were inspected during planning:

### Primary crewmate environment

`codex/crewmate-environment-v2`

Reviewed commit:
`e62ba0b6c33ffd0317f78e18b4289d778cd5af2b`

Important files:

- `crewmate_v2/config.py`
- `crewmate_v2/api.py`
- `crewmate_v2/engine.py`
- `crewmate_v2/runtime.py`
- `crewmate_v2/tasks.py`
- `crewmate_v2/ghosts.py`
- `crewmate_v2/provenance.py`
- `docs/crewmate-v2/DESIGN.md`
- `docs/crewmate-v2/README.md`

This branch should be the preferred starting point for the new game mechanics, subject to checking whether newer local work exists.

### Training and RL infrastructure

Review:

- `phase5_env.py`
- `phase5_policy.py`
- `phase5_features.py`
- `phase5_training.py`
- `phase5_options_training.py`
- `phase6/env.py`
- `phase6/train.py`
- `phase6/config.py`

Also inspect the research work associated with:

`codex/phase6-1-seeds29-43`

Reviewed commit:
`2baa5dbc1b36b047ed97934d69526573b510c280`

The existing PPO implementation includes useful rollout collection, time-aware discounting, advantage estimation, masking, and checkpoint infrastructure.

However, `StrategicPolicy` is an older feedforward model.

**Do not reuse it as Patrick Jane's neural network.**

Reuse compatible training infrastructure while implementing Patrick Jane's approved recurrent architecture separately.

### Navigation and map

Review:

- `among_us_map.py`
- `navigation_service.py`
- `skeld_environment.py`
- `phase3_engine.py`
- `social_deduction/phase3_observation.py`

The repository already contains Skeld geometry, collision-aware movement, navigable nodes, A-star pathfinding, and a navigation service with route planning and recovery.

Reuse these components.

### Spectator and replay system

Branch:

`codex/spectator-live-integration`

Reviewed commit:
`6dd8a1bbf41177fb565e863b9157e717b4918796`

Important modules:

- `run_training_spectator.py`
- `spectator/recorder.py`
- `spectator/replay_format.py`
- `spectator/playback.py`
- `spectator/ui_renderer.py`
- `spectator/watcher.py`
- `spectator/app_state.py`

The existing spectator already supports completed-match replays, game events, player movement, and neural-action information.

Extend its integration rather than building a replacement.

### Protected work

Preserve all historical model checkpoints, existing benchmarks, saved research results, and reproducibility information.

The previous crewmate model is paused and must remain available for future comparison.

Do not overwrite its checkpoints or training outputs.

Do not modify the separate impostor development branch or the other collaborator's impostor policy.

Use the existing scripted-opponent interfaces instead.

Create an isolated Patrick Jane implementation and development branch/worktree as appropriate. Do not blindly merge complete development branches or rewrite historical source files.

Do not push, merge, deploy, or run expensive training without the appropriate authorization.

# 3. ENVIRONMENT COMPATIBILITY WORK

The inspected engine currently initializes five players: four crewmates and one impostor.

`phase3_engine.py` also contains fixed population assertions.

The existing strategic training environment uses one learned focal agent with other players controlled by scripts.

Patrick Jane requires a genuine multi-agent environment.

## Population

Initial training configuration:

| Role | Count |
|---|---:|
| Learning crewmates | 10 |
| Scripted impostors | 2 |
| Total | 12 |

Maximum supported configuration:

| Role | Count |
|---|---:|
| Crewmates | 12 |
| Impostors | 3 |
| Total | 15 |

Support smaller valid configurations through explicit lobby settings.

All 15 player slots include the focal agent. Therefore, the neural network has a maximum of 14 other players to represent.

Audit hardcoded assumptions throughout the repository, including:

- Player and role assignment.
- Identity and color generation.
- Spawn locations.
- Player-indexed event dictionaries.
- Observation projection.
- Task ownership.
- Meeting participants.
- Voting.
- Action enumeration.
- Player feature matrices.
- Reward bookkeeping.
- Replay serialization.
- Population invariants.

Particular examples include the existing four-player candidate matrix, five-player assertions, fixed task matrices, and `MAX_ACTIONS=96` candidate limit.

Replace fixed game-size assumptions with configurable population handling where required.

Use padding and legal-player masks for the neural network.

Do not rewrite older checkpoint metadata or silently reinterpret legacy observations as Patrick Jane observations.

## Simultaneous decisions

Preserve the existing principle that all participating actors receive observations from the same pre-action simulation state.

No actor may observe another actor's decision or consequences from the same tick before making its own decision.

This remains essential when 10 learning crewmates act simultaneously.

# 4. GAME RULES AND ENVIRONMENT

Reuse the newer Crewmate V2 rules as the foundation.

The existing V2 implementation already supports:

- Individually owned tasks.
- Task categories and multi-stage interactions.
- Automatic ghost task completion.
- Public task-bar progress.
- Confirmed ejections.
- Non-anonymous votes revealed at results.
- Structured meeting communication.
- Actor-local evidence.
- Wall-blocked visibility.

Preserve those features.

The approved goal is behavior close to default Among Us settings. The repository documents some approximate physics and visibility values rather than an exact commercial-game implementation.

Keep those approximations explicit and configurable. Do not claim exact retail fidelity without validation.

Start from the V2 lobby configuration unless the owner has already supplied newer settings.

Do not silently introduce deduction-only games or disable task victory.

## Sabotage implementation

The inspected crewmate engine does not contain a complete functioning sabotage system.

Implement this approved extension.

Support at least:

- Lights sabotage and reduced crewmate visibility.
- O2 sabotage with the appropriate repair objectives and failure countdown.
- Reactor sabotage with its repair requirements and failure countdown.
- Scripted impostors initiating sabotage.
- Crewmates navigating to and completing sabotage fixes.

Reuse the existing map utility positions wherever valid.

Add missing state transitions, repair actions, timer handling, and observation fields as needed.

A lights sabotage must dynamically affect what living crewmates can observe.

O2 and reactor failures must produce the appropriate loss outcome.

Ensure simultaneous repairs, meetings, and terminal conditions are resolved deterministically and tested.

Do not create freely repeatable repair rewards or shortcuts that permit impostors or crewmates to exploit sabotage mechanics.

Where exact gameplay details are missing, document the chosen approximation. Request approval for consequential rule changes rather than inventing them silently.

# 5. NAVIGATION SYSTEM

We previously agreed on the following design:

**The reinforcement learning agent chooses where to go, while pathfinding handles how to move there.**

Use:

1. Existing map geometry and collision constraints.
2. A navigation mesh or equivalent reachable-space representation.
3. A-star pathfinding.
4. Bounded-random alternative routes.
5. Visibility-coverage tracking.
6. Learned destination selection.

Do not make the network learn raw joystick movement.

## Navigation action head

Exactly three behaviors:

| Output | Behavior |
|---|---|
| 1 | Follow suggested route |
| 2 | Temporarily detour while preserving destination |
| 3 | Change destination |

When no destination exists, the model must select one.

For a temporary detour, retain the original goal and allow navigation to resume.

For a destination change, replace the previous goal.

The pathfinding system provides a suggested route, but the network is not forced to follow it.

Generate reasonable route alternatives with bounded randomness.

The network chooses its strategic behavior rather than micromanaging each A-star waypoint.

## Exploration

The agent must be able to explore the complete reachable map.

Do not restrict navigation to room centers.

Use reachable exploration locations and visibility coverage so that edges, corners, corridors, body locations, and relevant map areas can be discovered.

Track which areas have been seen and when.

Repeatedly visiting previously explored locations must not generate new exploration rewards.

Avoid route oscillation, endless detours, and repetitive destination switching.

Navigation decisions should occur at useful decision boundaries, with interruption permitted when significant observable events occur.

Keep route planning and execution separate from the learned neural policy.

# 6. OBSERVATION ARCHITECTURE

Use structured observations, not screenshots.

Maintain strict separation between actor-visible information and privileged simulator truth.

## Input branch A: self, map, navigation, and game state

Include:

- Own position and room.
- Own living/dead status.
- Assigned tasks and their locations.
- Task stages and completion progress.
- Game phase.
- Time and relevant timers.
- Known map layout.
- Static vent locations.
- Emergency meeting button locations.
- Sabotage repair locations.
- Currently observable sabotage state.
- Current destination.
- Suggested route and next waypoint.
- Available navigation alternatives.
- Reachable exploration candidates.
- Previously observed area coverage.

The previous architecture estimate allocated approximately 192 features for this combined branch.

Treat this as the provisional encoding budget. Do not fabricate information merely to fill unused feature slots.

## Input branch B: players

Represent up to 14 other players.

Initially use a 16-feature player template, with padding and attention masking.

Include appropriately encoded information such as:

- Identity or stable player slot.
- Alive/known status.
- Current visibility.
- Last observed position and room.
- Time since last observation.
- Observable movement or direction.
- Meeting interactions.
- Publicly confirmed information.
- Relevant evidence indicators.

Do not insert hand-authored suspicion probabilities.

Suspicion should be learned by the network.

Randomize identities and colors appropriately to avoid color-specific shortcuts.

## Input branch C: event memory

Include timestamped observations and social events.

Initially use an approximately 16-feature event representation.

The event record should capture:

- Event category.
- Subject and relevant participants.
- Location.
- Observation timestamp.
- Delivery timestamp where applicable.
- Information source.
- Whether the event was personally observed or merely claimed.
- Other legally observable event details.

Do not permit future events or privileged truth to leak into this branch.

# 7. MEMORY SYSTEM

**Memory is one of Patrick Jane's most important components.**

Use two complementary mechanisms:

1. A shared-architecture GRU with an independent hidden state for each crewmate.
2. Explicit timestamped event memory for each crewmate.

The GRU processes ongoing context.

The event memory retains useful historical evidence, and attention retrieves relevant events.

Start with a configurable capacity of 64 event records per agent.

Also maintain each player's last-known observation separately.

Do not continuously record every coordinate at every simulation tick.

Instead, preserve meaningful events such as:

- Player spotted in a room.
- Player entering or leaving visibility.
- Player moving toward a relevant location.
- Body discovered or reported.
- Witnessed elimination.
- Sabotage observed.
- Accusation or defense.
- Location claim.
- Previous vote.
- Meeting outcome.
- Confirmed role after ejection.
- Own previous statements.

Each record must preserve its provenance and timing.

Distinguish observation time from the time an agent learned about a claim.

Never turn a reported claim into confirmed truth merely because it is in memory.

## Example: contradictory alibi

Patrick Jane observes Red in Electrical at a particular time.

Later, Red claims to have been in MedBay at that time.

The model must be able to retrieve both records and compare their temporal and geographic context.

The network decides whether this evidence is meaningful enough to challenge Red.

**Do not hardcode “contradiction = impostor.”**

The network must learn the significance of contradictory evidence.

## Own statements

The agent must also remember its own previous public claims.

For example, it may have claimed to be in Electrical while its private memory records a different location.

Both facts must remain distinguishable.

This supports learning communication consistency and strategic deception.

GRU states and explicit memories are independent for each learning crewmate.

Reset them at the correct match boundaries.

Do not share private observations between crewmates.

# 8. LOCKED NEURAL NETWORK ARCHITECTURE

Implement the following recurrent actor-critic design.

Do not replace it with the existing Phase 5 feedforward policy.

## Encoder A: self/map/navigation

Input: structured combined features.

- Linear: 256 neurons.
- Activation.
- Linear: 128 neurons.
- Activation.

## Encoder B: players

Shared player encoder applied to each valid player:

- Linear: 128 neurons.
- Activation.
- Linear: 128 neurons.
- Activation.

Then apply:

- Four-head attention.
- Attention representation width: 128.
- Proper player masks.

Pool or aggregate the resulting representation without allowing padded player slots to influence valid player embeddings.

## Encoder C: events

For each event-memory record:

- Linear: 128 neurons.
- Activation.
- Linear: 128 neurons.
- Activation.

Then apply attention-based event retrieval.

Use four attention heads and a 128-dimensional representation.

Respect event masks and timestamps.

## Feature fusion

Combine the representations from the three encoders.

- Fusion hidden/output representation: 256 neurons.

## Shared recurrent layer

- One GRU.
- Hidden size: 256.

The GRU is shared by the Actor and Critic.

It is not shared as one global hidden state across players.

Every learning agent maintains its own recurrent state while all agents share the trainable network weights.

## Actor branch

- Linear: 256 neurons.
- Activation.
- Linear: 128 neurons.
- Activation.
- Action-output heads.

## Critic branch

- Linear: 256 neurons.
- Activation.
- Linear: 128 neurons.
- Activation.
- One scalar state-value output.

Apply appropriate numerical-stability measures and initialization without altering the architecture.

The previously discussed model estimate was approximately 946,000 parameters before final target-selection details.

This is **not an exact parameter-count requirement**.

Calculate the final parameter count from the implemented architecture and report it.

Our conceptual longest processing path has approximately nine stages counting input and output, although the network contains parallel branches.

Report both actual module counts and the longest forward path rather than describing the entire model as a simple nine-layer MLP.

# 9. LOCKED ACTION OUTPUTS

Patrick Jane uses five specialized actor output heads.

## Head 1: navigation

Three outputs:

1. Follow route.
2. Temporary detour.
3. Change destination.

## Head 2: interaction

Five outputs:

1. Do nothing.
2. Complete task.
3. Report body.
4. Fix sabotage.
5. Call emergency meeting.

## Head 3: communication

Six outputs:

1. Stay silent.
2. Share evidence.
3. Accuse.
4. Defend.
5. Ask a question.
6. Make a strategic claim.

The existing environment already supports some structured claim types through `ClaimKind`.

Extend the structured communication protocol as necessary to faithfully represent the agreed intentions.

Do not silently discard an output merely because the legacy game does not implement it.

Initially support the existing one-statement-per-participant meeting structure.

Leave the communication interface extensible for later conversational exchanges.

## Head 4: voting

A categorical distribution over eligible living players plus skip.

Support up to 14 possible other-player targets plus skip.

Mask ineligible targets and unavailable actions.

## Head 5: target selection

Use an attention-based selector over applicable candidates.

Candidate categories may include:

- Players.
- Map locations.
- Task stations.
- Exploration destinations.
- Bodies.
- Sabotage repair locations.
- Relevant evidence targets.

Use separate target-type validity masks so that unrelated candidates cannot be selected.

The number of actual available targets must not force fixed small population or map-location assumptions.

## Action execution

Use game-phase-aware masking and conditional target selection.

In roaming, valid navigation and interaction actions must be able to compete appropriately without introducing a sixth learned action head.

During discussion, expose communication decisions.

During voting, expose voting decisions.

When a chosen action requires a target, conditionally select its target using the target head.

The PPO log probability must include all actually sampled conditional choices.

Invalid actions must be masked rather than repeatedly penalized.

# 10. SOCIAL DEDUCTION AND LLM INTERFACE

Patrick Jane's strategic decisions must come from reinforcement learning.

Do not script accusations, suspicion thresholds, or alibi evaluations.

The network should learn when information is relevant to winning.

It must receive public statements, historical meeting information, votes, and evidence through its observation and memory systems.

The eventual LLM interface should receive structured information such as:

- Communication intention.
- Target player or location.
- Relevant personally witnessed evidence.
- Relevant public claims.
- Last-known player positions.
- Learned player assessments, where available.
- Prior statements made by Patrick Jane.

The LLM will eventually translate these strategic intentions into natural-language messages and parse incoming human statements into structured evidence.

**Do not train or integrate an LLM as a requirement of the first Patrick Jane release.**

Keep this as a clean future interface.

The RL system must remain functional with structured communications only.

# 11. GHOST SYSTEM

Reuse `crewmate_v2/ghosts.py` and its deterministic automatic task executor.

Do not create a separate ghost neural network.

When a crewmate dies:

- It becomes a ghost.
- Ghost movement passes through walls.
- Ghost travel respects its configured speed.
- Unfinished assigned tasks remain assigned to that player.
- Ghost tasks are completed automatically.
- The ghost does not participate in voting, reporting, sabotage repair, or living-player social decisions.
- No post-death privileged observations enter the living policy.

Preserve the existing protection against ghost information leaking to living players.

The game should continue to count ghost task completion toward the crew's legitimate task objective.

No teleportation.

# 12. LOCKED REWARD FUNCTION

Implement these exact initial reward values.

| Event | Reward |
|---|---:|
| Crewmates win | +10.0 |
| Crewmates lose | -10.0 |
| Complete assigned task while alive | +1.0 |
| Fix sabotage | +3.0 |
| Impostor successfully ejected | +4.0 |
| Innocent player ejected | -2.0 |
| Successfully report a body | +1.1 |
| Discover previously unseen area | +0.02 |
| Observe previously unseen player | +0.05 |
| Agent gets killed | -1.0 |
| Invalid action | 0.0 |
| Idle for 10 seconds | -0.01 |
| Cast a vote | 0.0 |
| Accuse or defend | 0.0 |
| Complete assigned task as ghost | +1.0 |

Important:

- Winning and losing are shared team outcomes.
- Give the agreed ejection rewards according to the actual ejection outcome, not simply because an agent cast a matching vote.
- The ghost task reward replaces the living-task reward for that completion. Do not award both.
- A task can be rewarded only once.
- A body can generate a reporting reward only once.
- A sabotage repair must represent a genuine completed repair. No repeatable farming.
- Exploration bonuses are capped at +0.3 per match.
- New-player observation bonuses are capped at +0.7 per match.
- Do not penalize unavoidable waiting.
- Do not add continuous survival-time rewards.
- Maintain legitimate terminal team rewards for all participating learning crewmates, including those whose living-policy decisions ended earlier because of death.

The previous V2 environment uses historical reward semantics, including a -5 death penalty and elapsed-time costs.

Patrick Jane must use the newly agreed reward configuration **without modifying the historical model's reward accounting or artifacts**.

All reward components must be independently logged and testable.

If an ambiguous event-crediting case appears, document and resolve its distribution without silently changing reward coefficients or creating duplicate rewards.

# 13. SHARED-POLICY MULTI-AGENT TRAINING

All 10 crewmates in the initial configuration must be controlled by Patrick Jane.

They share trainable neural network parameters.

They do not share private memory or observations.

Each agent has:

- Its own actor-local observation.
- Its own event memory.
- Its own 256-dimensional GRU state.
- Its own actions.
- Its own trajectory.

Collect experiences across all learning crewmates and parallel games.

Use centralized PPO updates to the single shared policy.

Do not independently optimize a different neural network for each crewmate.

Preserve agent identity and episode boundaries in every trajectory.

Avoid accidental recurrent-state mixing between players, workers, or matches.

The two impostors are scripted during initial training.

Do not require the other developer's impostor model.

# 14. SCRIPTED IMPOSTOR OPPONENTS

Reuse and extend the existing scripted opponent infrastructure.

The repository already includes `ScriptedController`, `BotStyle`, and different behavior variants.

Increase behavioral diversity while keeping scripted opponents compatible with the new environment.

Include strategies such as:

- Aggressive killing.
- Conservative killing.
- Opportunistic killing.
- Task faking.
- Sabotage-heavy play.
- Ambush and patient movement.
- Defensive statements.
- False accusations.
- Deceptive location claims.
- Self-reporting.
- Behavior that resembles innocent crewmates.

Use controlled randomness and diverse behavior combinations.

Ensure the scripts obey legitimate game rules and information boundaries.

Do not hardcode opponent tendencies into Patrick Jane's observations.

Keep separate opponent configurations for training and held-out evaluation.

Do not modify the separate impostor neural-network project.

# 15. LOCKED INITIAL PPO SETTINGS

| Parameter | Value |
|---|---:|
| Algorithm | Recurrent PPO |
| Learning rate | 0.0003 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| PPO clipping | 0.2 |
| Entropy coefficient | 0.01 |
| PPO epochs | 4 |
| Maximum gradient norm | 0.5 |
| Initial recurrent sequence length | 128 decisions |

Choose minibatch sizes and rollout buffers according to measured resource usage.

Preserve the configuration values above.

The environment has a physics timestep and a separate decision interval.

Different actions can therefore span different amounts of simulation time.

Use correct elapsed-time discounting rather than assuming all policy decisions represent equal physical durations.

Ensure:

- Correct recurrent PPO likelihood calculations.
- Correct action and target masking.
- Proper bootstrapping and GAE.
- No recurrent sequence mixing.
- Correct handling of episode termination.
- Correct handling of dead agents and delayed team outcomes.
- Stable numerical losses and gradients.

Reuse existing time-aware PPO utilities when compatible, but do not force the new recurrent policy through an incompatible feedforward training implementation.

# 16. PARALLEL COMPUTING

The owner wants efficient utilization of available hardware.

The intended primary machine includes an Intel Core i7-13700HX, an RTX 4070 Laptop GPU, and 32 GB RAM.

Verify the actual available machine resources at execution time.

Initially target:

**16 concurrent training environments.**

Benchmark:

- 8 environments.
- 16 environments.
- 32 environments.

Select the best practical configuration based on real measurements.

Use CPU workers for independent simulation instances and batch neural inference/training on the GPU when beneficial.

Prefer centralized updates with controlled policy synchronization.

Do not let separate workers independently train incompatible policy copies.

Do not count scripted-opponent decisions as learning-agent decisions.

Report both individual actor decisions and completed matches.

Measure:

- Learning-agent decisions per second.
- Completed games per second.
- PPO update time.
- Environment simulation time.
- GPU utilization.
- GPU memory.
- CPU utilization.
- Replay-recording overhead.
- Worker synchronization overhead.

Training should be headless.

The spectator must not slow every simulation to rendering speed.

# 17. SPECTATOR INTEGRATION

The project already has a spectator and replay studio.

Use it.

Do not create a second application without a compelling technical reason and approval.

Integrate Patrick Jane with the existing replay recorder and visualizer.

The spectator should support:

- Identifying Patrick Jane sessions.
- Showing different parallel training environments.
- Selecting a match to inspect.
- Viewing completed match replays.
- Displaying learned actions and target choices.
- Displaying relevant action probabilities.
- Displaying reward contributions.
- Displaying chosen destinations and suggested routes.
- Displaying player movement.
- Displaying ghost movement.
- Displaying meetings, claims, and votes.
- Showing the training checkpoint and update associated with a recording.

Support a practical overview of concurrent environment instances and focused inspection of one selected instance, without requiring every worker to render continuously.

Existing replay semantics and historical playback compatibility must be preserved.

Use the established `spectator-v1.0` recording pipeline or a backward-compatible versioned extension when necessary.

Never feed privileged spectator or replay information into the policy.

Recording must remain observational and must not alter random streams, action decisions, rewards, gradients, or simulation timing.

Record a configurable subset of matches to avoid unnecessary storage and rendering overhead.

The existing spectator primarily loads completed match recordings. Preserve that working behavior while integrating parallel training.

# 18. CHECKPOINTS AND CHAMPION SELECTION

Maintain two separate concepts:

**Training policy:** The policy that continues learning.

**Champion policy:** The best-performing checkpoint according to evaluation.

The owner previously suggested periodically selecting the best checkpoint and continuing from it.

Implement this safely.

Do not automatically roll back the training policy at every checkpoint.

Save training progress regularly and evaluate candidates against comparable scenarios.

Use a configurable evaluation interval, initially in the range of approximately 100,000 to 500,000 individual learning-agent decisions.

Use approximately 100 evaluation matches per checkpoint initially, with held-out opponent behaviors and reproducible scenario seeds.

Preserve:

- Policy weights.
- Optimizer state.
- Training counters.
- Configuration.
- Random generator states needed for reproducibility.
- Source/schema version.
- Checkpoint metadata.

Champion selection should prioritize actual gameplay success rather than shaped reward totals alone.

Use win rate as a primary metric with task completion and social-deduction metrics as supporting evidence.

Preserve the best checkpoint independently from the latest training checkpoint.

# 19. EVALUATION

Patrick Jane must demonstrate actual learning rather than merely accumulating shaped rewards.

Evaluate:

- Crewmate win rate.
- Wins by task completion.
- Wins by impostor elimination.
- Tasks completed.
- Sabotages successfully repaired.
- Sabotage-related losses.
- Bodies successfully reported.
- Correct impostor ejections.
- Incorrect innocent ejections.
- Skip and voting behavior.
- Evidence-based challenges.
- Contradiction-handling behavior.
- Map exploration coverage.
- Navigation failures.
- Illegal action rate.
- Average game duration.

Track learning curves and checkpoint performance.

Use held-out scripted opponent strategies for generalization testing.

When comparing Patrick Jane with the preserved previous model, use compatible conditions and distinguish environment differences from policy improvements.

Do not claim that winning against weak scripted opponents proves competent human-level social deduction.

# 20. TESTING AND SAFETY VALIDATION

Before any expensive training:

### Environment validation

Verify:

- Configurable population sizes.
- Correct role counts.
- Correct task assignment.
- Correct sabotage behavior.
- Visibility reduction during lights.
- Proper O2/reactor loss conditions.
- Ghost behavior.
- Meeting progression.
- Vote visibility and ejection results.
- No actor access to hidden information.
- No role- or identity-based data leakage.

### Neural network validation

Verify:

- Encoder tensor dimensions.
- Player padding and masks.
- Event-memory encoding.
- Attention outputs.
- Independent GRU states.
- Actor/Critic sharing of the same recurrent representation.
- All five action heads.
- Target-selection masking.
- Gradient propagation.
- Stable PPO updates.

### Multi-agent validation

Verify:

- All living crewmates actually use Patrick Jane.
- No policy update occurs independently inside simulation workers.
- Actor memories remain separate.
- Episode resets clear the correct state.
- Dead agents no longer take living actions.
- Team rewards are assigned correctly.

### Reward validation

Test each reward event individually.

Check that no event generates duplicate credit.

Specifically verify:

- Reporting a body more than once cannot farm reward.
- Sabotage repair cannot farm reward.
- Ghost tasks do not double reward.
- Repeated area visits cannot farm exploration reward.
- An innocent ejection produces the agreed -2 outcome.
- An impostor ejection produces the agreed +4 outcome.

### Spectator validation

Run recorded and unrecorded identical scenarios.

Verify identical simulation actions, rewards, outcomes, and resulting training behavior.

Verify historical replays still load.

### Training smoke test

Run a short, bounded training smoke test.

It must demonstrate:

- Successful environment resets.
- Complete games with the intended role counts.
- Nonzero learning-agent experience.
- Valid action masks.
- Finite PPO losses.
- Actual policy-weight updates.
- Working checkpoint save/reload.
- Successful replay recording and spectator playback.

A working smoke test is evidence of implementation correctness, **not proof of useful learning**.

# 21. IMPLEMENTATION ORDER

Work in the following sequence.

**Phase A — Repository audit**

Inspect the existing working state, branches, code, and compatibility boundaries.

Document the exact source commits and reusable modules.

**Phase B — Environment extensions**

Add population configuration, sabotage mechanics, and multi-agent game support.

Preserve existing V2 task, ghost, visibility, and meeting behavior.

**Phase C — Patrick Jane network**

Implement the locked encoder, attention, shared GRU, Actor, Critic, and output heads.

Validate the architecture and parameter count.

**Phase D — Memory and navigation integration**

Connect actor-local evidence memory, timestamped events, navigation decisions, and target selection.

**Phase E — Shared-policy recurrent PPO**

Implement multi-agent rollout collection, masked action sampling, time-aware advantages, and centralized updates.

**Phase F — Parallel execution**

Integrate concurrent headless matches and benchmark available worker configurations.

**Phase G — Spectator and checkpoints**

Connect real neural decisions and complete-match recordings to the existing viewer.

Add champion selection, checkpointing, and resume support.

**Phase H — Tests and smoke training**

Execute compatibility, privacy, reward, multi-agent, and training tests.

Fix implementation defects without changing the approved neural architecture or reward values.

# 22. REQUIRED FINAL REPORT

After implementation, provide a clear report containing:

1. Exact files created and modified.
2. Base branch and commit identifiers.
3. Environment features implemented.
4. Confirmation that the previous model remains untouched.
5. Confirmation that the separate impostor project remains untouched.
6. Exact Patrick Jane neural architecture.
7. Total trainable parameter count.
8. Input, hidden, and output tensor dimensions.
9. Multi-agent training behavior.
10. Parallel environment benchmark results.
11. Spectator integration status.
12. Tests executed and results.
13. Training smoke-test results.
14. Known limitations and unsupported features.
15. Any implementation deviations requiring owner approval.

Do not disguise partial implementation as a complete system.

Do not replace missing features with hardcoded behavior that gives the impression Patrick Jane has learned them.

Where implementation constraints prevent faithful execution of the approved design, stop and explicitly present the blocker.

## FINAL EXECUTION RULE

**Do not initiate full multi-million-decision training immediately after implementation.**

First complete the environment integration, neural network, parallel collector, tests, and bounded smoke test.

Then present the results, benchmark measurements, exact neural architecture, and readiness assessment.

Request authorization before launching the full expensive training run.

At that point, Patrick Jane should be ready to learn in the existing Among Us environment while the original model remains preserved for future comparison.

**Project: Patrick Jane**

**Algorithm: Recurrent PPO**

**Architecture: Three encoders → attention → 256-dimensional fusion → shared 256-unit GRU → Actor/Critic branches → five specialized action heads**

**Training: Shared-policy learning crewmates, scripted impostors, centralized updates, parallel headless environments**

**Primary objective: Learn to win Among Us through effective strategy and social deduction.**

**Preserve the approved design. Implement faithfully. Verify before scaling.**
