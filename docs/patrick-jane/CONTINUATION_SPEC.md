Continue developing Patrick Jane on `codex/patrick-jane-recurrent-ppo`, using the existing local worktree.

I reviewed your progress report and approve continuing the verification, optimization, and staged training preparation. I have three additional priorities.

## 1. OPEN THE ACTUAL SPECTATOR FOR ME

I want to visually observe Patrick Jane's progress and concurrent environments.

Do not merely implement the viewer, produce screenshots, or confirm that replay integration works.

**Launch the actual desktop spectator and leave its window open for me.**

Use the spectator implementation already integrated into this project.

The interface should let me:

- See the concurrent training instances and their current status.
- Select an individual instance to inspect.
- Watch the crewmates moving on Skeld.
- See actual navigation, tasks, sabotages, reports, meetings, and votes.
- Inspect real neural-network decisions and action probabilities.
- See live training progress: decisions collected, completed games, wins, losses, reward components, PPO updates, checkpoints, and throughput.
- Watch recent completed matches and follow newly completed replays.

Ideally, display a grid or list of concurrent instances with a selectable detailed map view.

Use real recorded or streamed simulation data, never fabricated activity.

A selected instance may publish throttled visual snapshots for monitoring. Keep the training workers independent from rendering so visualization does not significantly reduce training speed.

Preserve the existing replay and spectator system rather than replacing it.

If something prevents a persistent desktop window from opening, explain the blocker instead of claiming it opened successfully.

## 2. VERIFY DEFAULT AMONG US GAME SETTINGS

I want the simulation to reproduce the normal, out-of-the-box **Classic Among Us experience on The Skeld**, excluding special roles.

Research official Innersloth documentation and reliable gameplay references. Prefer current-version evidence over outdated defaults.

Compare those references against the actual implementation.

The documented classic baseline includes:

- Player speed: 1.0×
- Crewmate vision: 1.0×
- Impostor vision: 1.5×
- Kill cooldown: 45 seconds
- Kill distance: Medium
- Emergency meetings: 1
- Emergency cooldown: 15 seconds
- Discussion time: 15 seconds
- Voting time: 120 seconds
- Common tasks: 1
- Long tasks: 1
- Short tasks: 2
- Confirm ejects: On
- Anonymous votes: Off
- Visual tasks: On
- Task bar updates: Always

These are documented historical classic defaults, not a guarantee of the latest 2026 preset. Verify against the most recent reliable evidence available.

Keep our intentional 10-crewmate/2-impostor training configuration and configurable 15-player capacity.

Pay particular attention to these sabotage mechanics:

- O2 on The Skeld: 30-second deadline, repair in O2 and Admin.
- Reactor: 30-second deadline, two simultaneous repair participants.
- Lights: approximately fivefold reduction in living crewmate vision; impostors unaffected.
- Sabotage cooldown: documented as 10 seconds initially and 30 seconds after previous sabotage.
- Body reports cancel critical sabotages but do not automatically repair lights.
- Emergency button restrictions during sabotage.
- Correct game-ending and simultaneous-event ordering.

References to investigate:

https://www.innersloth.com/a-match-made-in-update-16-0-0-emergency-meeting-40/

https://among-us.fandom.com/wiki/Options

https://among-us.fandom.com/wiki/Sabotage

https://among-us.fandom.com/wiki/Oxygen_Depleted

https://among-us.fandom.com/wiki/Reactor_Meltdown

https://among-us.fandom.com/wiki/Fix_Lights

https://among-us.fandom.com/wiki/Report

https://github.com/roobscoob/among-us-protocol/blob/master/07_miscellaneous/01_the_structure_of_the_gameoptionsdata_object.md

Produce a comparison of reference setting, current implementation, discrepancies, corrections, and confidence/source evidence.

Correct confirmed implementation errors while preserving our agreed architecture and rewards.

Document unavoidable approximations and uncertain default values honestly. If a proposed change affects an intentionally chosen gameplay rule, ask me before applying it.

## 3. YOU HAVE ENGINEERING FREEDOM — THINK AND OPTIMIZE

I don't want you blindly following instructions when you discover a genuinely better implementation strategy.

You are encouraged to reason independently, investigate bottlenecks, challenge inefficient engineering decisions, propose improvements, and pursue promising performance optimizations.

In particular, explore:

- Parallel environment scheduling and worker counts.
- CPU/GPU utilization.
- Batched inference.
- PPO update throughput.
- Recurrent sequence batching.
- Redundant observation processing.
- Navigation caching and pathfinding overhead.
- Memory allocation and serialization.
- Worker synchronization.
- Headless simulation performance.
- Spectator overhead.
- Training bottlenecks and profiling.
- Numerical stability and checkpoint reliability.

Experiment with promising optimizations.

Measure before and after. Prefer improvements with demonstrated gains and unchanged simulation semantics.

**Important boundary:**

You may independently improve implementation efficiency and internal engineering.

You must not silently change our approved neural architecture, GRU size, attention design, action heads, reward coefficients, fundamental navigation decisions, or core RL algorithm.

If you discover a potentially superior architecture or gameplay design, present the idea, expected advantage, cost, and risks to me, and wait for approval.

The goal is to give you room for creative engineering, not permission to redesign Patrick Jane.

## 4. FINISH VERIFICATION, THEN CONTINUE TRAINING

Complete the remaining readiness checks:

- Comparable 8/16/32-worker benchmarks on the latest implementation.
- A full-game GPU PPO update.
- Checkpoint save and reload.
- Complete replay integration.
- Historical compatibility tests.
- Sabotage correctness and game-settings verification.

Fix any confirmed defects.

Once the end-to-end smoke tests pass, begin staged learning rather than immediately launching an unlimited experiment.

Start with a measurable pilot budget, approximately 250,000 individual learning-agent decisions. Use checkpoints, evaluation matches, and learning diagnostics.

If healthy, propose or continue a larger run within a clearly stated compute budget, initially targeting up to one million decisions total. Report expected wall time using the updated benchmarks before committing to the longer stage.

Pause on crashes, non-finite losses, privacy violations, repeated reward exploitation, or serious performance regressions.

Preserve the champion checkpoint separately from the latest training state.

Evaluate real gameplay learning using held-out scripted opponents, not shaped reward alone.

Make results and selected match replays visible through the spectator.

No changes to the previous model or the other developer's impostor project.

## FINAL EXPECTATIONS

I want to see a working spectator window showing real progress and concurrent environments.

I want a documented game-settings verification.

I want meaningful performance optimizations supported by benchmarks.

I want Patrick Jane to receive enough training to begin demonstrating actual learned behavior, with transparent progress and safe checkpoints.

And I want you to use your engineering judgment rather than repeatedly requesting permission for minor implementation choices.

**Preserve Patrick Jane's approved design. Be creative with implementation. Measure everything.**
