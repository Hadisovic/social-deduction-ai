# Stealth RL Navigation — Social-Deduction Research

> A research simulator for a crewmate that learns to use legitimate observations, memory, evidence, and uncertainty to improve team outcomes.

## Project definition

This project investigates whether a compact learned crewmate using visibility-filtered observations, persistent evidence memory, and explicit role uncertainty improves team win rate over scripted and memory-ablated baselines against held-out teammate and opponent strategies. Known-map navigation, collision, and task execution are controlled infrastructure. The primary target is an Among-Us-like environment we control, not the commercial game client.

The project started as continuous PPO navigation research. Stage 1 demonstrated point-to-point movement on its evaluated open-arena distribution. Stage 2 remains an optional learned-detour experiment, not a prerequisite for social learning. Stage 2.5 supplies known-map geometry and validation infrastructure; a reliable executable navigation service is still planned.

## Current status

| Category | Status |
|---|---|
| CURRENTLY IMPLEMENTED | Stage 1/2 navigation environments, PPO scripts, rewards, evaluation, inspectors; Skeld map, 40 destinations, 22-value navigation observation, validation A* |
| CURRENTLY VALIDATED | Historical Stage 1 benchmark below; Skeld's 34-test suite; 42 new information-boundary tests; 139 total pytest tests pass plus standalone regression suites |
| PHASE 1 | Implemented and ready for review/freeze; information contract and isolated observation-boundary foundation; see [contract](docs/information_contract.md) and [handoff](HANDOFF.md) |
| PLANNED | Reliable navigation controller, complete scripted social game, evidence memory/beliefs, learned strategic crewmate |
| FUTURE / OPTIONAL | Learned impostor, self-play, sabotage, vents, multiple maps/counts, CNN perception, language, learned minigames |

The social game does **not** exist yet. Contract fixtures and public-event schemas do not implement kills, meetings, voting, or task execution. Existing navigation observation vectors are not the new actor interface.

## Primary roadmap

| Phase | Deliverable | Gate |
|---|---|---|
| 1 — Definition and information contract | Truth/observation/memory/label separation; provenance; paired-world leakage tests | Contract tests and navigation regressions pass; review and freeze |
| 2 — Reliable navigation service | Planner edges consistent with physics, arrival/cancel/replan, executable-route benchmark | All 1,560 task pairs plus seeded random spawns evaluated; target >=99% success, systematic failures resolved |
| 3 — Scripted social simulator | Five players, timed tasks, visibility, bodies/reports, structured meetings/votes, outcomes and replay | 1,000 seeded matches without invalid state; intended win routes and information boundaries tested |
| 4 — Memory and beliefs | Provenance-preserving histories, rule baseline, compact supervised role predictor | Beat prior-only/rule baselines on held-out bot families using Brier/log loss |
| 5 — Learned strategic crewmate | One learned actor with frozen mixed bots and high-level actions | Held-out team-outcome improvement across >=3 training seeds; memory/evidence ablations |
| 6 — Optional expansion | Add complexity only to answer a new experimental question | Preserve the core result against unfamiliar partners/opponents |

These are proposed engineering/research gates, not guarantees. Phase 2 does not start automatically after Phase 1.

## Infrastructure versus learned intelligence

Use classical geometry, a planner/controller, and scripted task mechanics where they are reliable. Focus the learning budget on evidence interpretation, uncertainty, information gathering, task priorities, and meeting decisions. A* is allowed for the planned social agent using known geometry and observed dynamic state. The historical pure-navigation experiments retain their no-waypoint protocol.

Vision and recurrence are not prerequisites for a playable social simulator. An initial strategic baseline can consume structured evidence summaries; recurrence must justify itself experimentally. See [architecture](docs/project_architecture.md) for interfaces and scope.

## Historical navigation curriculum

| Stage | What it demonstrates | Status in the new project |
|---|---|---|
| 1 | Open-arena point-to-point navigation | Completed baseline; preserve checkpoint |
| 2 | Forced obstacle detours and recovery | Implemented; full training not run; optional bounded experiment |
| 2.5 | Known Skeld-like map, destinations and reachability validation | Implemented environment; untrained; foundation for Phase 2 |
| 3/4 | One/multiple patrol observers | Legacy arena modes; optional stealth experiments, not social milestones |
| 5 | Random geometry/patrol generalization | Optional future navigation research |

### Historical Stage 1 benchmark (43-input architecture)

Across the recorded 100 stochastic evaluation episodes: **100/100 successes**, average successful time **2.93 s** (1.83–4.83 s), average path efficiency **93.86%**, initial distance **507.86 px**, mean reward **120.93** (SD 6.43; median 119.59). These results apply to that open-arena evaluation distribution, not unseen maps or social behavior. The preserved checkpoint is `models/stage1/best_model/best_model.zip`; compatibility with it does not constrain future actor design.

![Original arena](docs/images/environment_overview.png)
![Navigation sensors](docs/images/vision_and_sensors.png)

The following reference sections document the existing navigation experiments, not the social actor contract.

## Environment & Physics

- **Arena**: 1100 &times; 700 px playable window with a 40 px outer wall margin (`1020 x 620` internal playfield).
- **Player**: Circular rigid body ($r = 15\text{ px}$), constant speed $v = 185.0\text{ px/s}$.
- **Collision Physics**: Continuous circle-to-axis-aligned-bounding-box (AABB) resolution with smooth wall sliding.
- **Simulation Frequency**: Fixed RL simulation timestep $\Delta t = \frac{1}{30}\text{ s}$ (30 Hz).
- **Gymnasium Interface**: Standard `step(action) -> (obs, reward, terminated, truncated, info)` interface.
- **Randomization**: Independent seeded random placement for player and goal on each reset, enforcing a minimum Euclidean separation of 350 px and safety clearances from obstacle boundaries.

---

## Observation Space (43 Dimensions)

The observation vector is a 1D `numpy.ndarray` of **43 `float32` values**:

| Index Range | Name | Dimensions | Normalized Range | Description |
|:---:|:---|:---:|:---:|:---|
| **0 – 1** | Player Position | 2 | `[0.0, 1.0]` | Absolute player coordinates: `[x / 1100, y / 700]` |
| **2 – 3** | Relative Goal Vector | 2 | `[-1.0, 1.0]` | Goal relative displacement: `[(gx - px) / 1100, (gy - py) / 700]` |
| **4 – 19** | Radial Obstacle Rays | 16 | `[0.0, 1.0]` | 16 distance sensors spaced every $22.5^\circ$ ($0^\circ, 22.5^\circ, \dots, 337.5^\circ$) detecting central obstacles and outer boundaries (max distance 300 px) |
| **20 – 26** | Observer 1 Slot | 7 | Real | `[active, rel_x, rel_y, cos(facing), sin(facing), vision_range / 1100, fov / pi]` *(Zeroed in Stages 1 & 2)* |
| **27 – 33** | Observer 2 Slot | 7 | Real | Observer 2 sensory slot *(Zeroed in Stages 1 & 2)* |
| **34 – 40** | Observer 3 Slot | 7 | Real | Observer 3 sensory slot *(Zeroed in Stages 1 & 2)* |
| **41** | Stuck Flag | 1 | `{0.0, 1.0}` | `1.0` if agent has experienced $\ge 15$ consecutive blocked steps (0.5s); `0.0` otherwise |
| **42** | Stagnation Progress | 1 | `[0.0, 1.0]` | Normalized measure of remaining in a 10 px radius: $\min(\text{stagnation\_steps} / 60, 1.0)$ |

---

## Action Space

Continuous 2D displacement vector:
$$\mathbf{a} \in \text{Box}(-1.0, 1.0, \text{shape}=(2,), \text{dtype}=\text{float32})$$
- Representing commanded $[a_x, a_y]$ movement.
- **Deadzone**: Actions with $\|\mathbf{a}\| < 0.05$ produce zero movement.
- **Normalization**: Vectors exceeding unit length are normalized: $\hat{\mathbf{a}} = \frac{\mathbf{a}}{\|\mathbf{a}\|}$.
- **Displacement**: $\Delta \mathbf{p} = \hat{\mathbf{a}} \cdot v_{\text{player}} \cdot \Delta t$.

---

## PPO Network Architecture

Implemented with Stable-Baselines3:
- **Policy Network (Actor)**: Fully connected MLP `[43 -> 64 -> 64 -> 2]` with Gaussian action distribution.
- **Value Network (Critic)**: Fully connected MLP `[43 -> 64 -> 64 -> 1]`.
- **Optimizer**: Adam ($\text{lr} = 3 \times 10^{-4}$).
- **Rollout Length**: $N = 2048$ steps.
- **Batch Size**: 64.
- **Epochs per Update**: 10.
- **Discount Factor ($\gamma$)**: 0.99.
- **GAE Parameter ($\lambda$)**: 0.95.
- **PPO Clip Range**: 0.2.
- **Normalization**: Raw continuous observation without VecNormalize or reward scaling to preserve absolute penalty relationships.

---

## Reward Design

### Stage 1: Basic Locomotion
- **Step Penalty**: `-0.01` per active simulation step.
- **Progress Shaping**: `+0.05` per pixel of new-best Euclidean progress toward goal (evaluated every 0.5s, minimum 5.0 px advancement, anti-farming protection).
- **Goal Reached**: `+100.0` (terminates episode).
- **Timeout**: 30.0s (truncates episode, no extra penalty).

### Stage 2: Obstacle Detours and Recovery
Stage 2 balances goal seeking with anti-stagnation and anti-looping penalties to discourage pressing into solid walls or pacing aimlessly.

| Event / Condition | Frequency / Scope | Reward / Penalty |
|:---|:---|:---:|
| **Simulation Step** | Every step | `-0.01` |
| **New-Best Progress** | Every 0.5s if advancement $\ge 5.0\text{ px}$ | `+0.03 * distance_px` |
| **Sideways / Detour Movement** | When distance doesn't decrease | `0.0` (no distance punishment) |
| **Wall Sliding** | Meaningful motion along wall ($\ge 20\%$ expected) | `0.0` extra penalty |
| **Short Blocked Steps** | Steps 1 through 14 while blocked | `-0.02` additional |
| **Prolonged Blocked Steps** | Steps 15+ while blocked | `-0.05` additional |
| **Stagnation Penalty** | Confined within 10 px for 2.0s (60 steps) | **`-25.0` ONE-TIME** |
| **Cell Revisit / Looping** | 3rd and later confirmed entry to 30 px cell | **`-2.0` ONCE** per confirmed re-entry |
| **Recovery Refund** | Break free: 10 unblocked steps + $\ge 40\text{ px}$ from anchor | **Refund 50%** of blocked penalties (capped at `+2.0`) |
| **Goal Success** | Contact with goal radius | **`+100.0`** (Terminates) |
| **Episode Timeout** | Reaching 20.0s (600 steps) | **`-50.0`** (Truncates) |

### Anti-Farming Mathematical Guarantee
Recovery refunds **only** 50% of the blocked-specific penalties (`-0.02` and `-0.05`) up to a maximum of `+2.0`. Because entering the stuck state requires at least 15 blocked steps (costing $-0.33$ blocked penalty plus $-0.15$ in time step penalties), the maximum possible refund on immediate escape is $+0.165$, leaving the net event at $\le -0.315$. Prolonged stuck events decay further into net negatives. Stagnation (`-25.0`), cell revisit penalties (`-2.0`), timeout (`-50.0`), and step penalties are **never** refunded. Deliberately getting stuck can never become profitable.

---

## Obstacle Sensing & Stuck Recovery System

### 16-Ray Radial Perception
The agent perceives solid geometry through 16 radial raycasts spaced at $22.5^\circ$ intervals covering the full $360^\circ$ circle. Rays detect both interior obstacle boundaries and outer arena walls, returning normalized distances $\in [0.0, 1.0]$. The network receives no artificial path hints, optimal route waypoints, or directional steering recommendations.

### Forced Obstacle Detour Layout Guarantee
To prevent the agent from sampling unconstrained open-field routes, Stage 2 enforces a strict geometric obstruction constraint on every episode reset:
- The finite line segment from the player spawn to the goal center must intersect at least one central obstacle.
- Obstacles are inflated by the player radius ($r = 15\text{ px}$) on all sides during the intersection test.
- Any candidate layout with an unobstructed direct line-of-sight is rejected during bounded rejection sampling.
- If sampling limits are reached, a verified deterministic fallback layout guarantees an obstructed path.
- Consequently, **100% of Stage 2 episodes** (in training, deterministic evaluation, stochastic evaluation, and live spectator modes) require navigating around solid interior barriers.

### Spatial Anti-Looping / Revisit System
To stop the agent from gaming the stagnation detector by pacing back and forth across small areas without making genuine progress:
1. **Spatial Grid Partitioning**: The arena is logically divided into $30\text{ px} \times 30\text{ px}$ cells (`STAGE2_REVISIT_CELL_SIZE = 30.0`).
2. **Confirmed Entry (Anti-Jitter)**: To prevent false visits from hovering along cell borders, entering a new candidate cell requires remaining in it for 3 consecutive simulation steps (`STAGE2_REVISIT_CONFIRM_STEPS = 3`) before the transition is committed.
3. **Visit Schedule**:
   - **Visit 1** (Spawn Cell): Counted on reset, cost: `0.0`.
   - **Visit 2**: Permitted without penalty (`0.0`) to allow legitimate exploration and normal backtracking.
   - **Visit 3 and Later**: Assesses a **`-2.0` penalty** once per confirmed re-entry (`STAGE2_REVISIT_PENALTY = -2.0`).
4. **Reward-Side Only**: Continuous occupancy inside a cell incurs no additional penalty. The revisit counter is strictly evaluated in reward space; the observation vector remains exactly 43 values.

### Deadlock and Stagnation Lifecycle
1. **Blocked Detection**: Triggered when commanded movement $\|\mathbf{a}\| \ge 0.05$ yields actual displacement $< 20\%$ of expected displacement ($< 1.233\text{ px}$ per step).
2. **Stuck State**: Declared after 15 consecutive blocked steps (0.5s). Sets `stuck_flag = 1.0` in observation index 41.
3. **Stagnation Event**: If the agent remains within a 10 px radius of an anchor point for 60 steps (2.0s), a one-time penalty of `-25.0` is assessed. The system cannot re-arm until the player escapes at least 40 px away from the stagnation anchor.
4. **Autonomous Recovery**: The environment **never** artificially turns, teleports, or steers the agent. When the neural network discovers its own escape path—remaining unblocked for 10 consecutive steps and traversing $\ge 40\text{ px}$ away from the stuck anchor—the recovery refund is awarded and the stuck flag clears.

### Path Efficiency Diagnostic Metric
Episode evaluation logs record path efficiency as the ratio of straight-line displacement from spawn to the actual distance traveled:
$$\text{Path Efficiency} = \min\left(1.0, \frac{\|\mathbf{p}_{\text{current}} - \mathbf{p}_{\text{start}}\|}{d_{\text{travelled}}}\right) \times 100\%$$
This formulation ensures mathematical validity ($\le 100.0\%$) and resolves boundary discrepancy between entity collision circles and center coordinates.

---

Both curriculum stages feature unified, single-command workflows that train headlessly at maximum speed while concurrently spawning an independent visual spectator window:

### Stage 1 Training + Live Spectator
```powershell
python train_stage1.py --watch
```
- Trains PPO on curriculum Stage 1 headlessly.
- Spawns `live_watch_training.py` in a separate process.
- Evaluates checkpoints deterministically every 10k steps and saves the best model.
- Automatically reloads the spectator when newer checkpoints appear.

### Stage 2 Training + Live Spectator
```powershell
python train_stage2.py --watch
```
- Validates that a compatible 43-input Stage 1 model exists at `models/stage1/best_model/best_model.zip`.
- Initializes Stage 2 policy and value networks directly from the Stage 1 brain.
- Spawns `live_watch_stage2.py` with real-time HUD diagnostics (Stuck flag, blocked streak, stagnation percentage, completion times).

---

## Model Evaluation

Independent evaluation scripts evaluate trained checkpoints using fixed layout seeds:

```powershell
# Stage 1 deterministic evaluation (100 episodes)
python evaluate_stage1.py --model models/stage1/best_model/best_model.zip --episodes 100 --deterministic

# Stage 2 evaluation (both deterministic and stochastic modes)
python evaluate_stage2.py --model models/stage2/best_model/best_model.zip --episodes 100
```

---

## Stage 2.5: The Skeld Navigation Environment

Stage 2.5 is a known-map navigation environment inspired by **The Skeld** from *Among Us*. Its geometry and inspector are preserved as infrastructure for Phase 2; learned navigation and visual representation experiments are optional.

> **Status**: **ENVIRONMENT BUILT / NOT TRAINED**  
> **Next step**: Phase 2 navigation-service hardening after Phase 1 review. No Skeld training has been launched; full training is not on the main critical path.

![The Skeld Overview](docs/images/skeld_overview.png)
*Figure 1: The Skeld Stage 2.5 environment. Uniform 1600.0 × 895.45 logical world space rendered with letterboxing into an 1100 × 700 display window. Shows all 14 authentic rooms, 40 task destinations, vents, and real-time internal A\* route validation.*

### Architecture & Mathematical Coordinate Separation

To eliminate aspect-ratio distortion present in early prototypes, Stage 2.5 enforces strict separation between reference coordinates, logical simulation physics, and display presentation:

1. **Reference Space**: $8565.0 \times 4794.0\text{ px}$ (Aspect Ratio $\approx 1.78660826$).
2. **Logical World Space (Physics & RL)**: $1600.0 \times 895.44658\text{ px}$.
   - Uniform World Scale: $S_{\text{world}} = \frac{1600.0}{8565.0} \approx 0.18680677$.
   - **Aspect Ratio Preservation**: Mathematically identical to reference space within machine precision ($< 10^{-7}$ relative error).
   - All physics calculations, AABB movement collision checks, wall sliding, raycasting, and goal distances operate purely in world units.
3. **Display Space (Rendering Only)**: $1100 \times 700\text{ px}$.
   - Display Scale: $s_{\text{display}} = \frac{1100.0}{1600.0} = 0.6875$.
   - Letterboxing: Vertical offset $Y = 42.19\text{ px}$ centers the ship vertically without stretching.
   - Exact invertible mappings: `world_to_screen(wx, wy)` and `screen_to_world(sx, sy)`.

### Physical Collision Geometry & Walkable Hull

The map rejects the naive "free-space except where wall rects exist" assumption by utilizing a **dual-layer safety model**:

![Collision & Walkability Debug](docs/images/skeld_collision_debug.png)
*Figure 2: Physical walkability audit. 50 solid wall rects (red) and 32 walkable floor segments (green). An internal 4px occupancy grid checks single-component grid connectivity; physical execution remains a separate validation requirement.*

- **50 Solid Wall Rectangles**: Outer perimeter hull and interior room boundaries/consoles.
- **32 Walkable Floor Regions**: Explicit navigable interior corridors and room floors.
- **Dual-Layer Movement Solver**: Axis-separated movement verifies both that candidate positions are clear of solid wall AABBs and remain strictly inside valid ship floor boundaries.
- **Exterior Vacuum Non-Walkability**: The space outside the hull is strictly non-walkable. The agent cannot leak or escape into the void.

### Calibrated Physics Ratios

- **Doorway / Player Clearance**: Narrowest doorway is 40.0 px wide. Player radius is 10.0 px (diameter 20.0 px). Ratio $\frac{\text{doorway}}{\text{diameter}} = 2.0$.
- **Player Speed**: 160.0 px/s (traverses ship width in ~10s at full throttle).
- **Goal Radius**: 12.0 px (fits cleanly within 35 px wall alcoves with $\ge 20\text{ px}$ clearance).
- **Task Interaction Radius**: 25.0 px ($2.5\times$ player radius).
- **Sensor Ray Range**: 16 rays at 22.5° intervals, max range 220.0 px ($11.0\times$ player diameter, $4.4\times$ average corridor width, $1.22\times$ average room dimension).

### 40 Authentic Task Destinations

![Task Destinations](docs/images/skeld_task_points.png)
*Figure 3: All 40 authentic task destinations extracted from community reference coordinates. All destination markers pass the grid reachability checks; executable-route validation is planned.*

All 40 task destinations are represented as rich inspectable `TaskDestination` dataclasses containing verified IDs, display names, room assignments, world coordinates, and confidence metadata.
- **Reachability**: **40 / 40 reachable (0 unreachable)** verified by occupancy-grid pathfinding.
- **All-Pairs Task Connectivity**: All destinations map to the connected grid component; the checked-in pair test queries 10 selected pairs and the route test checks 9 representative routes. Exhaustive physical execution of all 1,560 directed pairs remains Phase 2 work.
- **Goal Modes**: Supports `task` (random spawn $\to$ authentic task), `task_to_task` (crewmate chore sequence), and `room_to_room`.

### Internal Validation Pathfinder & Route Verification

An internal 4px occupancy grid ($400 \times 224 = 89,600\text{ cells}$) with radius-based clearance supports reachability validation. It is not exposed to the existing navigation PPO. Phase 2 will validate planner edges and physical controller execution before using it as social-agent infrastructure.

**Representative grid A* routes (historical measurements, not controller execution or continuous-space optimality proofs)**:
| Origin Room | Destination Room | Path Exists | Grid Route Length | Intended Route |
|---|---|---|---|---|
| **Cafeteria** | **Electrical** | **YES** | 1001.8 px | Clear (traverses Storage corridor) |
| **Cafeteria** | **Navigation** | **YES** | 977.6 px | Clear (traverses Weapons / East corridor) |
| **Navigation** | **Reactor** | **YES** | 1491.1 px | Clear (full trans-ship traversal via Cafeteria/Engines) |
| **MedBay** | **Shields** | **YES** | 1132.8 px | Clear (traverses Cafeteria & East Hall) |
| **Security** | **Admin** | **YES** | 985.1 px | Clear (traverses Lower Engine / Storage) |
| **Electrical** | **Weapons** | **YES** | 1151.5 px | Clear (traverses Storage & Cafeteria Hall) |
| **Lower Engine** | **O2** | **YES** | 1263.9 px | Clear (traverses Storage & Shields Hall) |
| **Storage** | **Reactor** | **YES** | 1045.1 px | Clear (traverses Lower Engine corridor) |
| **Weapons** | **Electrical** | **YES** | 1151.5 px | Clear (symmetric route verification) |

### Observation Architecture Status

![16-Ray Obstacle Sensors](docs/images/skeld_rays.png)
*Figure 4: Proposed 16-ray radial distance sensors (V1 experiment proposal) calibrated to local corridor geometry (220 px range).*

> **IMPORTANT ARCHITECTURAL NOTICE**: The 22-dimensional observation vector (`[0:2]` player position, `[2:4]` goal vector, `[4:20]` 16 obstacle rays, `[20]` stuck flag, `[21]` stagnation progress) is designated strictly as **`PROPOSED_STRUCTURED_OBSERVATION_V1`**. It is an **experimental proposal and is NOT permanently locked**. The current implementation uses this layout; changing it requires corresponding code and tests. Sensor sweeps and CNN representations are optional future experiments, not Phase 1 requirements.

### Interactive Map Inspector

Launch the visual developer inspector to verify map topology, doorways, and pathfinding:
```powershell
python inspect_skeld.py
```
- `W / A / S / D` or Arrow Keys: Move player
- `Left Click`: Set custom goal destination in world coordinates
- `Tab / N`: Cycle through authentic task destinations
- `R`: Toggle 16 radial obstacle raycasts
- `C`: Toggle collision debug overlay (solid wall rects + walkable floor polygons)
- `T`: Toggle task destination markers and interaction radii
- `L`: Toggle room and region name labels
- `V`: Toggle vent markers
- `G`: Toggle real-time internal A\* validation route overlay
- `H`: Toggle comprehensive diagnostic HUD (true spatial region, nearest task, clearance, FPS)
- `SPACE`: Reset episode with random spawn
- `ESC / Q`: Quit inspector

---

## Automated Test Suites

The project includes an extensive test suite verifying mathematical correctness, physics stability, and curriculum safety:

| Test File | Verification Scope |
|:---|:---|
| `test_information_contract.py` | Social Phase 1: paired-world leakage tests, action candidates, visibility, provenance, memory and training-label isolation |
| `test_observation_43dim.py` | 43-element observation vector, 16 ray angles (22.5°), observer slot indexing, stuck & stagnation flags |
| `test_stage2_forced_detour_and_revisit.py` | Forced obstacle detour generation (200 resets), seed reproducibility, fallback layout validity, spatial revisit schedule (visits 1–3+), 3-step jitter protection, trajectory anti-loop penalty (-4.0), Stage 1 immunity, bounded path efficiency |
| `test_stage2_stuck_stagnation_recovery.py` | Blocked detection, stuck escalation, stagnation -25 trigger/rearm, recovery refund formula, anti-farming proof |
| `test_rl_env.py` | Gymnasium API compliance, action bounds, step penalty consistency, terminal reward logic (41 tests) |
| `test_stage1_randomization.py` | Spawn margin safety, clearance guarantees, layout determinism from seeds (17 tests) |
| `test_stage2_randomization.py` | Obstacle clearance, 350 px minimum start-goal distance, Stage 1 (0.05) vs Stage 2 (0.03) progress scale (24 tests) |
| `test_stage2_blocked_and_timeout.py` | 20s timeout truncation, -50 timeout penalty, wall sliding vs blocked distinction (16 tests) |
| `test_stage2_evaluation_and_ranking.py` | 4-tier checkpoint ranking (success rate -> completion time -> reward -> tiebreaker) |
| `test_stage2_transfer.py` | Exact weight transfer from Stage 1 into Stage 2 PPO without reinitialization |
| `test_stage2_watch.py` | Spectator initialization, checkpoint polling, boundary switching, clean exit |
| `test_stealth_env.py` | Pygame base physics, AABB collisions, observer vision cones, line-of-sight raycasts (15 tests) |
| `test_ppo_infrastructure.py` | PPO model initialization, policy shapes, training smoke, evaluation and checkpointing (14 tests) |
| `test_skeld_environment.py` | Full Skeld Stage 2.5 suite: aspect ratio, single connected component, 40 task reachability, representative routes, dual-layer collision, hull safety, Gymnasium/SB3 compatibility (34 tests) |

To run the automated suite:
```powershell
python -m pip install -r requirements-dev.txt

# Run pytest-discoverable tests (standalone run_* suites below are additional)
python -m pytest -v

# Run individual standalone integration suites
python test_skeld_environment.py
python test_rl_env.py
python test_stage1_randomization.py
python test_stage2_randomization.py
python test_stage2_blocked_and_timeout.py
python test_stage2_forced_detour_and_revisit.py
python test_ppo_infrastructure.py
```

The infrastructure suite includes two short 512-step PPO smoke checks using
temporary models. It does not retrain the preserved Stage 1 checkpoint. Historical
suite counts include nested standalone checks and should not be added to pytest
counts as if they were unique tests. See HANDOFF for this phase's exact results.

---

## Repository Structure

```
.
├── social_deduction/                       # Phase 1 actor/truth/observation/training boundaries (no game engine)
├── test_information_contract.py            # Phase 1 information-leakage regression tests
├── HANDOFF.md                              # Current phase, decisions, validation and next work
├── config.py                               # Stages 1-2: Environment, physics, raycast, and RL hyperparameters
├── environment.py                          # Stages 1-2: Core Pygame 2D stealth environment and renderer
├── rl_environment.py                       # Stages 1-2: Gymnasium-compatible wrapper and reward calculation
├── geometry.py                             # Shared: Vector math, raycasting, and obstacle distance queries
├── player.py                               # Shared: Player entity with circle-AABB wall sliding
├── observer.py                             # Stages 3-4: Patrolling guard entity with polygon vision cones
├── skeld_config.py                         # Stage 2.5: Skeld map constants, 1600x895.45 world geometry, 40 tasks
├── skeld_navigation.py                     # Stage 2.5: 4px occupancy grid, player clearance inflation, A* pathfinder
├── skeld_environment.py                    # Stage 2.5: Gymnasium navigation environment for The Skeld
├── inspect_skeld.py                        # Stage 2.5: Interactive visual map inspector (WASD, A* path, HUD)
├── train_skeld.py                          # Stage 2.5: PPO training script (EXPERIMENTAL / NOT YET APPROVED)
├── test_skeld_environment.py               # Stage 2.5: 34-test comprehensive validation suite
├── main.py                                 # Manual keyboard interactive mode (Stages 1-4)
├── train_stage1.py                         # Stage 1 PPO training script (--watch support)
├── train_stage2.py                         # Stage 2 PPO continuation script (--watch support)
├── evaluate_stage1.py                      # Headless Stage 1 checkpoint evaluation
├── evaluate_stage2.py                      # Headless Stage 2 checkpoint evaluation
├── live_watch_training.py                  # Standalone live spectator for Stage 1
├── live_watch_stage2.py                    # Standalone live spectator for Stage 2 with HUD
├── training_callbacks.py                   # Periodic deterministic evaluation and best-model saving
├── docs/
│   ├── skeld_research.md                   # Skeld coordinate research and room topology
│   ├── skeld_accuracy.md                   # Accuracy assessment and known simplifications
│   ├── asset_sources.md                    # Research sources and copyright/licensing notes
│   ├── project_architecture.md             # New research architecture and module responsibilities
│   ├── information_contract.md             # V1 rules, legitimate knowledge, provenance and invariants
│   └── images/                             # Curated environment and sensor screenshots
├── models/                                 # Saved PPO checkpoints and best models (gitignored)
├── logs/                                   # Evaluation logs, monitor CSVs, TensorBoard (gitignored)
├── requirements.txt                        # Python package dependencies
├── requirements-dev.txt                    # Runtime dependencies plus pytest
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.10 through 3.14 (tested on Windows 11 with Python 3.12 and 3.14).
- Git.

For the Phase 1 regression environment, use **Python 3.12**. The Windows
development requirements pin Torch 2.6.0 and SB3 2.7.0: this pair imports cleanly
on the current host and loads the preserved Stage 1 checkpoint. Newer Torch
failed Windows DLL initialization here; the runtime requirements are unchanged.

### Windows Setup
```powershell
# 1. Clone repository
git clone https://github.com/Hadisovic/stealth-rl-navigation.git
cd stealth-rl-navigation

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Manual Interactive Mode
To test environment physics and observer vision cones manually using keyboard controls:
```powershell
python main.py
```
- `W / A / S / D` or Arrow Keys: Move player
- `F1`: Toggle sensor and raycast debug overlay
- `R`: Reset player and goal positions
- `ESC`: Exit

---

## Design principles and continuation

- Actor inputs come only from the [information contract](docs/information_contract.md); simulator truth and training labels remain separate.
- Memory preserves provenance: a claim never silently becomes a direct observation.
- Known geometry is legitimate; hidden dynamic world state is not.
- Validate outcome improvement and evidence use against held-out policies, with memory ablations and multiple seeds.
- Freeze reliable infrastructure instead of repeatedly redesigning it without experimental evidence.
- Keep historical Stage 1/2 behavior intact. Their no-waypoint/no-scripted-recovery restriction applies to those optional experiments only.

Read [HANDOFF.md](HANDOFF.md) before continuing. Phase 1 adds boundary types/tests only. No social gameplay or new model training is implemented.
