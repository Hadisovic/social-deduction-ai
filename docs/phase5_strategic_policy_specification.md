# TASK SPECIFICATION: PHASE 5 STRATEGIC CREWMATE POLICY & MAP GEOMETRY VERIFICATION

## AGENT DIRECTIVE & PROTOCOL
You are an expert Autonomous Systems & Reinforcement Learning Research Engineer tasked with implementing **Phase 5: Learned Crewmate Strategic Policy** in the `social-deduction-ai` repository.

### STRICT PRE-FLIGHT REQUIREMENT
> **DO NOT write, modify, or deploy implementation code immediately.**
> Your very first task is to thoroughly review the codebase, documentation, and game mechanics, and output a **Comprehensive Step-by-Step Plan and Verification Checklist**.
> **You MUST pause and request explicit user review and approval of the plan before creating or modifying any project files.**

---

## 1. MANDATORY RESEARCH & CONTEXT EXPLORATION
Before formulating your plan, you must inspect and cross-reference the following sources:
1. **Internal Architecture & Frozen Contracts**:
   - `README.md` & `HANDOFF.md`: Project status, frozen milestones, and test suite health (327 passing tests).
   - `docs/information_contract.md`: Phase 1 zero-leakage boundary. Understand why `WorldTruth` must NEVER leak into `ActorObservation` or `Knowledge`.
   - `docs/navigation_service.md`: Phase 2 deterministic A* route-planning and swept-circle collision infrastructure.
   - `docs/social_simulator.md` & `phase3_engine.py`: Phase 3 game rules, 5-player state machine, task distribution, occlusion raycasts, discussion/voting mechanics, and terminal conditions.
   - `docs/belief_model.md`, `belief/model.py`, `belief/features.py`, & `belief/memory.py`: Phase 4 memory schemas, 22-feature candidate encoding, and the calibrated 2,849-parameter `CandidateNet`.
   - `docs/benchmarks/phase4/README.md`: Phase 4 data splits, benchmarks, and baseline comparisons.
2. **Domain Grounding & Real Game Parity**:
   - Cross-reference standard *Among Us* (The Skeld) wiki mechanics regarding room topology, emergency button usage, vision line-of-sight occlusion, task completion quotas, and voting plurality/tie rules. Note where the simulator simplifies mechanics (e.g., 5 players, 4s interaction hold instead of minigames, structured claim packets instead of free chat, no vents/sabotage yet).

---

## 2. PRE-PHASE REQUIREMENT: MAP CORRIDOR & CLEARANCE AUDIT
The user has reported potential bottlenecks in two specific corridors:
- **Corridor next to O2** (the *BigYHallway* corridor between Weapons, O2, and Shields).
- **Corridor under Upper Engine** (the western passage leading down through `Door 11:12` into Security, Reactor, and Lower Engine).

### Tasks:
1. Inspect `assets/skeld/among_us_map.json` and `among_us_map.py`.
2. Confirm that all 13 doors are unlocked by default (`closed_doors=()`).
3. Check the clearance metrics: player radius is $0.2200$, but clearance drops to $\approx 0.2207$ near Upper Engine and $\approx 0.2209$ near O2.
4. **Preservation Constraint**: Do NOT alter raw coordinates in `among_us_map.json` without updating the build tools, as doing so breaks SHA-256 source hash assertions in `test_among_us_map.py`. If corridor relaxation is needed, it must be performed programmatically via clearance buffering or padding options in `AmongUsMap.__init__`, always preserving previous fallback schemes in code comments or legacy configurations.
5. Verify that agents navigate smoothly without clipping or walking through walls.

---

## 3. PHASE 5 ARCHITECTURE & DESIGN SPECIFICATION

### 3.1 Architectural Evolution
- **In Phase 4**: The agent was a passive **observer**. `BeliefModel` calculated role probabilities $P(\text{Impostor} = i)$ from memory, but all in-game actions were driven by fixed heuristic scripts (`phase3_bots.py`).
- **In Phase 5**: The agent becomes an **active autonomous decision-maker**. The frozen Phase 4 belief vector is connected directly into a learned Reinforcement Learning policy (Actor-Critic / PPO) to choose strategic intentions (Macro-Actions).

### 3.2 Strategic Gymnasium Environment (`phase5_env.py`)
Wrap `Phase3Game` and `Phase3Runner` into a clean Gymnasium interface (`CrewmateStrategicEnv`):
- Controls **1 focal crewmate** playing alongside 4 scripted agents (supporting Hunter, Patient, and Self-Report impostor archetypes).
- Decision tick: Every $0.6$ simulated seconds ($3$ physics ticks at $0.2$s) during roaming, and once per discussion/voting window during meetings.

### 3.3 Observation Space ($s_t \in \mathbb{R}^{43}$, Float32 Box)
Strictly legitimate features with zero privileged leakage:
1. **Kinematics & Room (6)**: Normalized $(x, y)$, room one-hot/index, moving/interacting/idle flags.
2. **Task Progress (4)**: Own completed ratio, distance to nearest assigned uncompleted console, public crew completed ratio ($k/8$), interaction progress ($t/4.0$).
3. **Local Vision (8)**: Visible player count, nearest visible player distance, player dispersion, body seen flag, distance to body, emergency button available flag, match elapsed ratio ($t/240.0$), match phase flag.
4. **Phase 4 Belief Vector (20)**:
   - 4 candidate role probabilities $P(\text{Impostor} = i)$ from frozen `artifacts/phase4/belief.pt`.
   - 4 candidate belief logits.
   - Belief Shannon entropy $H = -\sum p_i \ln p_i$.
   - 4 candidate sighting recencies ($t - t_{\text{last\_seen}}$).
   - 4 candidate contradiction counts from `ActorMemory`.
   - 3 direct witnessed kill flags.
5. **Meeting Context (5)**: Alive player count, claim polarity toward focal agent, top-suspicion candidate index, top-two probability margin ($P_{(1)} - P_{(2)}$), focal agent voted flag.

### 3.4 Discrete Macro-Action Space (12 Actions)
Use action masking for phase-illegal actions:
- `0: DO_NEAREST_TASK` (A* path to nearest unfinished assigned task; interacts upon arrival).
- `1: DO_FURTHEST_TASK` (Alternative task routing to avoid dangerous clusters).
- `2: PATROL_PUBLIC_ROOM` (Patrols Cafeteria/Admin/Storage to collect sighting evidence).
- `3: GROUP_WITH_TRUSTED` (Navigates toward player with $\min P_{\text{impostor}}$).
- `4: FLEE_MOST_SUSPICIOUS` (Maximizes distance from player with $\max P_{\text{impostor}}$).
- `5: REPORT_BODY` (Triggers body report if body within line-of-sight range $\le 1.3$).
- `6: CALL_EMERGENCY` (Runs to Cafeteria table and presses button if entropy $H < 0.6$).
- `7: WAIT_AND_OBSERVE` (Stands ground at corridor intersections to log sightings).
- `8-10: VOTE_CANDIDATE_1..3` (Casts vote for specific candidate in meeting).
- `11: VOTE_SKIP` (Skips vote when belief entropy is high or evidence is tied).

### 3.5 Reward Formulation (Mathematically Balanced & Exploit-Safe)
$$R_t = R_{\text{terminal}} + R_{\text{task}} + R_{\text{social}} + R_{\text{survival}} + R_{\text{step}}$$
- **Terminal Outcomes**: Crew Task Win $+10.0$; Crew Ejection Win $+15.0$; Impostor Parity Loss $-10.0$; Timeout $-5.0$.
- **Task Shaping**: $+2.5$ per finished personal task ($+5.0$ max).
- **Social/Deduction**: $+2.0$ for reporting a body; $+4.0$ for voting out the true hidden impostor; $-3.0$ for voting an innocent crewmate who gets ejected; $-1.5$ for skipping when an impostor had $P > 0.75$.
- **Survival**: $-5.0$ if the focal agent gets eliminated.
- **Urgency/Step Cost**: $-0.005$ per step to penalize stalling.

### 3.6 Policy Network & Training Pipeline (`train_phase5.py`)
- **Model**: PPO Actor-Critic with Shared Base (`Dense(128) -> ReLU -> Dense(128) -> ReLU`), separate Actor and Critic heads (`Dense(64) -> ReLU -> Out`).
- **Frozen Backbone**: Keep Phase 4 `CandidateNet` frozen; feed its outputs as observations.
- **Hyperparameters**: GAE $\lambda = 0.95$, $\gamma = 0.99$, PPO clip $\epsilon = 0.20$, entropy coeff $0.02 \to 0.001$, Adam lr $3\times 10^{-4}$, mini-batch 64, 8 vectorized environments.
- **Opponent Curriculum**: Train against naive Hunter $\to$ Self-Reporter $\to$ held-out Patient impostors.

### 3.7 Pre-Registered Evaluation & Baselines (`evaluate_phase5.py`)
Evaluate over **1,000 held-out seeded matches** against 3 required baselines:
1. **Random Strategic Baseline**: Uniform random valid macro-actions.
2. **Phase 3 Scripted Baseline**: Deterministic rule-based bot (`phase3_bots.py`).
3. **Memory-Ablated PPO Policy**: Identical network trained with belief inputs zeroed/uniform.
4. **Proposed Belief-Conditioned PPO Policy**: Full architecture.
- **Metrics to Log**: Crew Win Rate (%), Ejection Rate (%), Mean Survival Time (s), Mean Tasks Completed, Voting Accuracy (%). Record results in `docs/benchmarks/phase5/metrics.json`.

### 3.8 Visual Spectator UI (`run_phase5.py`)
- Retain the exact Pygame UI layout and asset rendering of `run_phase4.py`.
- Add a dedicated **Policy State Panel**:
  - Display active Macro-Action name and target destination.
  - Display policy action probabilities and value scalar $V(s)$.
  - Display real-time Phase 4 belief bar chart with candidate probabilities and entropy.
  - Highlight the active strategic target with an on-screen ring and route overlay.

---

## 4. SERIAL STEP-BY-STEP EXECUTION PIPELINE
You must structure your upcoming work into the following ordered phases:
- **Phase 5.0**: Map Corridor Audit, Clearance Testing & Documentation.
- **Phase 5.1**: Gymnasium Strategic Environment (`phase5_env.py`) & Unit Tests.
- **Phase 5.2**: Actor-Critic Policy Architecture (`phase5_policy.py`) & Action Masking.
- **Phase 5.3**: Training Pipeline (`train_phase5.py`) with Logging & Checkpointing.
- **Phase 5.4**: Multi-Baseline Benchmark & Evaluation Suite (`evaluate_phase5.py`).
- **Phase 5.5**: Interactive Spectator Desktop UI (`run_phase5.py`).
- **Phase 5.6**: Documentation & Benchmark Report (`docs/benchmarks/phase5/README.md`, `project-status.json`).

---

## YOUR IMMEDIATE FIRST DELIVERABLE
Respond with:
1. Confirmation that you have read and understood all frozen contracts and constraints.
2. Results of your codebase exploration regarding the corridors, belief interface, and simulator integration.
3. Your **Detailed Implementation Plan and Checklist** for Phases 5.0 through 5.6, highlighting exact file paths, schemas, and verification tests.
4. **Explicit stop and request for user approval before modifying or writing code.**


---

# Review amendment — 2026-10-03

The original pasted specification above is preserved. This amendment supplies
corrections and the implementation checklist. Where they disagree, use this
amendment for the future Phase 5 implementation. This document is a reviewed
proposal, not a claim that a strategic policy has been built or trained.

The owner separately authorized **fixing rendering now and expanding this
specification**. Those presentation changes do not require the additional Phase 5
implementation approval described in the original draft. No Phase 5 training or
client integration is part of this review.

## A. Branch and actual integration points

Reviewed local branch `main`, starting HEAD
`172f648` (merge of the public project website). The tree was clean at review start.
Phase 1 information contracts, Phase 2 navigation, Phase 3 simulation and Phase 4
belief estimation exist. Phase 5 does not. The 327-test claim is the preceding
Phase 4 milestone, not a result for the proposed policy.

Verified interfaces:

- `phase3_engine.py`: `Phase3Game`, `GameConfig`, `observe`, `step`, immutable
  projected actor packets, trusted terminal/reward bookkeeping.
- `phase3_runner.py`: **`ScriptedMatch`**, not `Phase3Runner`. It observes all
  actors before applying any same-tick action and uses three 0.2-second ticks per
  normal decision. Preserve that ordering in the focal-policy runner.
- `social_deduction/phase3_api.py`: `Phase3Observation`, actor-derived `actions`,
  `Intent`, `legal_actions`, `intent_allowed`. Existing enums encode game rules
  and wire formats; they are not learned strategic decisions. Preserve them.
- `navigation_service.py`: `NavigationPlanner`, `NavigationService`, `NavTarget`,
  explicit cancellation/arrival/failure, swept physics and bounded replanning.
  Use this service, not a new A* or waypoint teleport implementation.
- `belief/memory.py`: `ActorMemory.update(observation)`; four fixed non-self
  identities, causal event deduplication, last sightings and typed contradictions.
- `belief/model.py`: `BeliefModel.load().predict(memory)` returns `Belief` with
  `player_ids`, `probabilities`, raw `logits`, `entropy`. Preserve ID alignment.
  The network uses 4 x 22 features, shared candidate scoring and a calibrated
  temperature. `predict` uses inference mode; the Phase 5 trainer must additionally
  exclude its parameters from the optimizer and set `requires_grad_(False)`.
- `phase4_scripts.py` and the Phase 4 data configuration: inspect these for the
  controlled hunter/self-report/patient family setup. Reusing only the generic
  Phase 3 runner does not automatically reproduce that experimental distribution.
- `run_phase4.py`, `phase3_renderer.py`, `phase4_renderer.py`: spectator rendering
  remains separate from policy inputs. The mixed privileged replay/event log is
  never an observation source.

## B. Required corrections before strategic implementation

1. **Remove the live team task ratio.** The frozen contract explicitly has no
   live team task bar. A public initial quota does not reveal current completions.
   Never read `game.players` or the spectator sidebar to compute a policy feature.
   Use own task progress; introducing a team bar requires a separately versioned
   contract and hidden-task paired-world tests. Similarly, replace live alive-count
   with public meeting participants/known absences, with an unknown/stale indicator.

2. **Do not freeze the 43-value observation.** Four candidates require four direct
   witness features, not three. A room one-hot cannot occupy the single slot implied
   by the six-feature kinematic group. Missing sightings need presence masks, not
   fabricated distance/recency. Top-candidate numeric indices encode arbitrary ID
   ordering. Logits are unbounded and need documented normalization/Box bounds.

3. **Fix candidate and action coverage.** Four non-self candidates require four
   vote choices. The engine also permits self-voting. Either preserve it explicitly
   or preregister the policy restriction. There is no discussion/claim action in
   the twelve-action draft, so the proposed policy could not learn communication.
   Define whether communication is learned or transparently held fixed for all
   comparison agents. Missing actions must never silently invoke a heuristic bot.

4. **Remove strategic thresholds from legality.** Entropy < 0.6 is not an emergency
   rule, and high entropy is not a skip requirement. Legality comes from the actor
   packet and public rules. The model must choose whether to report, call a meeting,
   trust, flee or skip. Do not mask a target because the simulator secretly knows
   that player is dead or an impostor. A vote candidate can become unavailable only
   from the current public meeting roster. Keep the four historical belief slots.

5. **Use observed targets only.** Group/flee can target a currently visible player,
   or an explicitly stale last-seen location with age metadata. Neither may track
   unseen current coordinates. Fleeing should choose a reachable public waypoint;
   maximizing straight-line distance must not send movement through a wall.

6. **Handle action duration and phase transitions.** Observe every physics tick
   needed for legitimate evidence; decide at 0.6-second intervals during roaming.
   Reports/death/meetings interrupt macro execution immediately. Discussion must
   wait for the focal speaker's assigned window, not just the meeting's start.
   Voting occurs once when legal. Preserve ongoing interactions when reselecting a
   task; do not restart navigation or reset task progress every decision.

7. **Correct reward assumptions.** Task reassignment means a focal crew can finish
   more than its initial two tasks, so +5 is not necessarily the maximum. Reward
   each unique task completion once. Reporting can create a repeatable incentive
   independent of winning; test report/emergency/task cancellation loops. Penalizing
   a skip based on a belief threshold hard-codes the strategy and can punish honest
   uncertainty. Remove that term from the primary experiment.

8. **Keep reward truth in the trainer.** A reward for a correct ejection can use
   privileged roles during training, but those roles and reward components cannot
   enter actor features, memory, masks, previous-reward inputs or deployment state.
   Define the voter eligibility/attribution of a correct-vote reward. Do not claim
   the proposed hand-picked weights are mathematically exploit-safe without tests.
   Prefer equal terminal value for both crew win mechanisms, unless a deliberate
   preference for ejection over tasks is preregistered.

9. **Define elimination and timeout semantics.** Do not auto-reset at focal death
   and lose the eventual team outcome. Recommended wrapper: freeze actor decisions
   and evidence, run the remaining scripted game to terminal, assign team return
   plus any one-time elimination cost, and return a terminal transition. No ghost
   vision. A timeout that is an actual game draw is terminal; an external rollout
   cutoff is truncation with appropriate bootstrap. Never conflate the two.

10. **Use real action masking.** Plain Stable-Baselines3 PPO does not apply custom
    discrete masks automatically. Choose a version-pinned masked implementation
    (such as sb3-contrib MaskablePPO) or implement masked categorical sampling,
    log-probabilities, entropy and PPO updates consistently. At forced idle windows,
    use an explicit internal no-submission token or advance to the next choice;
    never softmax an all-invalid mask or invent an illegal WAIT intent.

11. **Preserve permutation behavior.** A flat MLP concatenating four arbitrary
    player slots loses the Phase 4 equivariance guarantee. Prefer shared candidate
    encoders and a pooled context; score candidate-target actions with shared weights.
    Relabel IDs/colors and require corresponding probabilities/actions to permute.
    Do not expose colors, seeds, labels or internal slots as numeric policy features.

12. **Fix held-out evaluation.** Patient cannot be both the final training curriculum
    stage and a held-out family. Train/validate on hunter/self-report; reserve new
    patient seeds for final out-of-family evaluation. Previously inspected Phase 4
    final seeds are not fresh Phase 5 selection data. Use at least three training
    seeds and equal budgets for the full and independently retrained ablation models.

## C. Suggested policy contract (draft, not frozen implementation)

Prefer a versioned `gym.spaces.Dict` and a candidate-action scorer over an ambiguous
43-vector and a fixed twelve-action interface. Generate dimensions from named
feature lists and record them with checkpoint hashes.

- `global`: own normalized position; explicit room vocabulary/unknown flag; own
  navigation/interaction state; own unfinished count and progress; nearest reachable
  own-task route distance with a missing flag; visible body and distance with a
  missing flag; own meeting allowance; remaining public match time; phase; own
  speaker/voted/active flags. Add only fields derivable from `Phase3Observation`.
- `players`: exactly four rows in a documented candidate-ID mapping. Include
  calibrated probability, centered/temperature-scaled logit, current visibility,
  observed relative position, seen-before flag, last-seen age/location, known body /
  public absence / ejection evidence, direct versus claim contradiction counts,
  direct witnessed-kill evidence and public vote eligibility. Missing current
  position must be masked. Do not collapse claims into direct facts.
- `actions`: actor-safe candidate records with action kind, target slot or public
  destination, own-task state, public route distance and availability. Score these
  with a shared network and padding mask. Optional mechanical macros can persist
  movement/interaction, but must not choose suspicion thresholds or vote targets.
- `action_mask`: derived from projected legal actions and public geometry only;
  version the construction and test the complete mapping back to `Intent`.

For a smaller first experiment, a corrected fixed macro set needs at least 13
choices for the original eight roaming macros plus four other-player votes and
skip. Add self-vote and learned claim choices, or explicitly state their exclusion
and use identical fixed communication across all agents. Do not claim unrestricted
strategic/communication learning for a restricted macro interface.

Treat the exact network widths and original PPO hyperparameters as starting
hypotheses. Define rollout length, update epochs, total environment-step budget,
gradient clipping, value/entropy coefficients and the schedule unit. Discounting
must account for variable simulated duration across meetings and terminal rollouts
(e.g. gamma^(elapsed/0.6), with consistent GAE). Otherwise a long meeting and a
0.6-second walk get the same discount. Benchmark eight-worker memory/planner cost
before choosing process count, and use safe main guards on Windows.

## D. Map audit results and geometry prerequisite

Executed `tools/audit_phase5_corridors.py`; full machine-readable results are in
`docs/phase5_corridor_audit.json`. No source coordinates, compiled blueprint,
player radius or geometry code were changed by this review.

- All 13 doors are open by default (`closed_doors=()`). Radius remains 0.22.
- Eight runs: both directions in each requested corridor at dt=1/30 and dt=0.2.
  All reached their destination with zero collision reports and zero replans.
- The selected straight Upper Engine south route has minimum center-to-barrier
  distance 0.47963. The O2/BigY routes approach 0.21981169 in one direction and
  0.22036424 in the other. Values depend on the chosen route and are not widths.
- **Strict circular clearance fails by about 0.00018831 units on one O2 route.**
  The map uses polygonal buffers (`quad_segs=12`); their corner chords can lie
  slightly inside an ideal circular offset. Existing `center_domain` checks accept
  the route. The audit intentionally exits nonzero for this stricter condition,
  even though all simulated movement runs succeed. Do not report an all-clear audit.

Phase 5.0 should investigate a conservative buffered radius or exact distance
predicate shared by planner and motion, preserve the old configuration as an
explicit baseline, then rerun all-pairs routes at both timesteps. Do not shrink
the player or erode walls simply to make this audit pass. Art/feet overlap is not
proof that the physical circle clips. Record sprite and collider overlays separately.

The original hash claim also needs precision: `source_hashes` asserts the bytes
of imported source JSON. It does not by itself detect every manual edit to compiled
vertices. Add deterministic builder-output comparison and a compiled-map digest
to the future geometry change gate. Changing raw sources, build rules, compiled
geometry or buffering policy requires traceable regeneration and validation.

## E. Avatar, body and bottom-UI fix delivered in this review

The local artwork pack contained **0/119 files** on inspection. Its verified
installer downloaded all 119 pinned PNGs into ignored `assets/phase3_local/`.
This is a reproducible missing-asset/fallback condition, not evidence of a GPU
failure or a need to reinstall graphics drivers.

The old offline corpse was an oval with a visor occupying a small part of its
canvas, covered by an unconditional circle. `phase3_renderer.py` now draws a
recognizable lower half-suit, boots and white bone when artwork is unavailable;
the local pack uses the existing `Dead<color>.png` sprites. Both use a readable
visible-body height. Body circles are confined to collision/debug mode. Living
map sprites have a 32-pixel logical minimum; positions and colliders are unchanged.

`run_phase4.py` previously fit its window to the smallest of every monitor, even
when displaying on monitor 0. `spectator_display.py` fits only the selected monitor;
the window is resizable, aspect-preserving, and accepts `--display` and
`--window-scale`. Enlarging graphics is presentation-only, not a physics change.

`phase4_avatar_hud.py` adds a bottom focal-avatar portrait and Report / Use /
Meeting availability cards, derived exclusively from the focal actor's legal
packet. They are clearly labeled **spectator indicators**, not manual buttons.
F changes the focal avatar as before. No world truth is passed to the HUD.
Phase 5 can add its selected action and probabilities beside these indicators.
Do not introduce a manual click-to-act path into evaluation without a separate mode.

Run:

```powershell
python run_phase4.py
python run_phase4.py --display 0 --window-scale 1
# On a fresh device only, install the optional verified artwork:
python phase3_assets.py --install
```

The artwork remains ignored and was not added to public commits. No existing
baseline screenshots/checkpoints are overwritten. Small-screen scaling cannot
make a full-map view look like the real game's close camera; a future zoomed focal
view should be an explicit presentation mode with the same world-to-screen transform.

## F. Ordered implementation plan and acceptance checklist

### Phase 5.0 — audit and freeze inputs

- [x] Review current branch, frozen actor API, navigation, engine, memory and beliefs.
- [x] Reproduce both corridors, record route-dependent clearance and open doors.
- [x] Repair avatar/body fallback, install verified local sprites, add focal HUD.
- [ ] Resolve the exact-circle buffer issue and rerun full geometry/navigation gates.
- [ ] Freeze checkpoint SHA-256, map/build/config digests, observation/action schema,
  reward version, seed split manifest and permitted communication scope.

### Phase 5.1 — strategic environment and safe runner

Proposed files: `phase5_env.py`, `phase5_runner.py`, `phase5_features.py`,
`tests/test_phase5_env.py`, `tests/test_phase5_boundary.py`.

- [ ] Control one crew, sample its public identity independently of features, and
  keep four scripted opponents/teammates with recorded family mixtures.
- [ ] Preserve simultaneous tick-start observations and 0.2-second physics; test
  macro continuation, interruption, speaker windows, vote-once and terminal rollouts.
- [ ] Make reward bookkeeping separate from actor feature construction. Deduplicate
  task IDs, reports and elimination rewards; cover reassigned tasks and stale intents.
- [ ] Paired-world tests: hidden role/position/death/task/cooldown/metadata changes
  leave observations, action records/masks and policy distributions identical.
- [ ] Gym checker, deterministic reset/replay, vector wrapper, timeout versus
  truncation, no actor action after elimination, no reward or replay-label leakage.

### Phase 5.2 — learned policy and masks

Proposed files: `phase5_policy.py`, `tests/test_phase5_policy.py`.

- [ ] Freeze the belief model including optimizer membership; verify weight hash
  before/after a training update. Keep raw versus calibrated logits explicit.
- [ ] Shared candidate/action encoding, pooled context, actor/value heads;
  ensure candidate-ID permutations permute target actions consistently.
- [ ] Identical masks for sampling and loss; zero probability on padding/illegal
  choices; finite entropy/log-probs; valid fallback for forced no-choice windows.
- [ ] No handwritten suspicion cutoff, automatic correct vote, true-role target,
  hidden position lookup or gameplay decision delegated silently to a heuristic.

### Phase 5.3 — reproducible training

Proposed files: `train_phase5.py`, `phase5_training.py`,
`artifacts/phase5/config.json`, `tests/test_phase5_training.py`.

- [ ] Small separate smoke run proves gradient flow, checkpoint round-trip and
  deterministic seeded rollout before a long run. Never overwrite Phase 4 weights.
- [ ] Preregister total steps, simulation-time discount, curriculum, three seeds,
  validation selection, process/thread limits and independently seeded workers.
- [ ] Record masks/actions, returns, outcome components, task/report/emergency
  frequencies, invalid intents, planner failures and frame-independent timing.
- [ ] Train matched full and ablated policies from scratch; evaluate held-out
  patients only after model/config selection is locked. No automatic full training
  merely from creating or opening the UI.

### Phase 5.4 — evaluation and ablations

Proposed files: `evaluate_phase5.py`, `phase5_evaluation.py`,
`docs/benchmarks/phase5/{README.md,metrics.json,split_manifest.json}`.

- [ ] At least 1,000 fresh seeded matches **per evaluated policy and training seed**,
  with a preregistered ID/patient split; same focal roles and match settings across
  learned, random-valid and scripted comparison conditions.
- [ ] Report paired differences, match-level confidence intervals, and variation
  across training seeds. Never select checkpoints using final test outcomes.
- [ ] Retrain memory-ablated PPO at equal budget. If recency/conflicts/witness flags
  remain, call it a belief-output ablation, not memory-ablation. Test a true
  current-observation-only variant retaining current public meeting context.
- [ ] Define denominators: team win rate (timeouts shown separately), correct
  ejections versus innocent ejections, correct votes conditional on voting,
  skip/abstention rate, survival time, unique own completions and report frequency.
- [ ] Include failure cases, no-claim and no-belief variants when feasible, frozen
  belief accuracy/calibration under learned behavior, and all exploit probes.

### Phase 5.5 — spectator UI and client boundary

Proposed files: `run_phase5.py`, `phase5_renderer.py`, `tests/test_phase5_ui.py`.

- [ ] Reuse corrected avatar/body assets, selected-monitor sizing and actor-only
  focal HUD. Retain Phase 4 panel and explicit privileged-spectator toggle.
- [ ] Display selected strategic action, target, probabilities, value, belief
  entropy, route and interaction progress; label truth-only displays separately.
- [ ] Check installed-pack and offline fallback paths, different colors/directions,
  live/dead/ejected states, bodies at feet anchors, focal switching, meetings,
  resizes and 100%/125%/150% desktop scaling. No distortion or body-marker occlusion.
- [ ] Rendering/UI event traces must not change seeded match outcomes or policy
  inference. Do not reuse a spectator frame as an actor input.
- [ ] A real-game client adapter is a separate milestone: define observation/input
  transport, calibration, tick synchronization and failure handling. This project
  currently consumes structured state, not real-client pixels; visual similarity
  alone does not provide sim-to-real transfer.

### Phase 5.6 — documentation and handoff

Proposed updates: `README.md`, `HANDOFF.md`, `docs/PROJECT_PROGRESS.md`,
`project-status.json`, and the Phase 5 benchmark report.

- [ ] Record exact evaluated revisions, checkpoint/map hashes, command lines,
  dependencies, training budgets, seed lists, limitations and measured results.
- [ ] Keep historical benchmark records intact. Publish only measured numbers;
  distinguish a runnable prototype, passing tests and demonstrated strategic gain.
- [ ] Review this amended contract before implementing Phase 5.1 onward. The
  approved rendering repair above is already complete; this remaining gate concerns
  the proposed strategic learning experiment, not permission to finish that repair.

## G. Domain references and deliberate simplifications

The [official Among Us description](https://www.innersloth.com/games/among-us/)
grounds the task/ejection win goals, reporting and emergency meetings. The current
simulator deliberately uses five players, timed interactions, fixed sight/ranges,
structured claims, paused meetings and no vents/sabotage/ghost gameplay. These are
research rules, not exact parity with every commercial lobby configuration.

Community cross-checks:
[task bar settings](https://among-us.fandom.com/wiki/Task_bar),
[voting and ties](https://among-us.fandom.com/wiki/Voting),
[body reporting](https://among-us.fandom.com/wiki/Report).
In particular, a task bar may be configured differently in the real game; that
does not override this repository's explicit no-live-team-bar observation contract.
Preserve unique plurality versus skip/ties and the simulator's no-ejection-role
reveal setting. Do not add newer special roles or Hide n Seek rules.

The pinned character/body sources are already documented in
`docs/phase3_asset_provenance.md` and `assets/phase3_assets.json`. No new anonymous
asset pack, generated image or guessed dimensions are needed for this rendering fix.

## Execution record — Phase 5 started, 2026-10-03

The owner's subsequent instruction explicitly authorized starting Phase 5. The
original draft and review above remain preserved as history. The owner then
requested a duration estimate and approval before longer training; that gate is
still pending. No extended experiment has run.

Implemented: conservative circular clearance, red palette decoding, missing walk
cycles, meeting-origin banners, actor-only strategic features, legal candidate
scoring, a frozen belief backbone, time-aware PPO, paired evaluation and a viewer.
The operational packet is 40 context values, four 26-value player rows, up to 96
23-value action rows, target references and a legal mask. There are 17 action
kinds. Fourteen rooms have public destination descriptors; seven hallways remain
traversable. Strategic distance features are Euclidean, not path lengths.

The true player radius remains 0.22. The conservative buffer is approximately
0.22047205. Map JSON is unchanged. Eight targeted passage tests and 4,914 routes
at each of 30 Hz and 5 Hz pass without collisions or navigation failures.
Historical Phase 4 trajectory tests explicitly retain legacy polygon clearance.

Two 512-transition smoke trainings changed policy weights and preserved belief
weights. These smoke checkpoints predate the final room-descriptor correction;
they demonstrate the pipeline and UI, not strategic quality. Fresh training
records the feature-source hash. No strategic-gain claim is accepted yet.

The proposed longer run uses three full-memory and three current-only models at
32,768 transitions each. Validation-only ranking precedes final ID/patient tests.
The implementation currently uses one serial worker, not the draft's proposed
eight. Estimated runtime is 20–35 hours including validation and final evaluation;
owner approval is required by their latest explicit instruction.

The separate teammate explanation was created outside the repository, as requested.

## Execution update — expanded learning authorized, 2026-10-03

The latest owner instruction supersedes earlier budget-approval restrictions.
Additional evidence-driven training and evaluation are authorized. The original
32,768-decision seed-17 run is preserved; it finished in 36.6 minutes with zero
own tasks in 299 training matches. Final test seeds remain unopened.

A separate `strategic-options-v2` execution protocol permits selected task/travel
actions to persist for up to 24 simulated seconds (follow/flee: 3 seconds).
Completion, phase changes, new visible bodies and witnessed eliminations interrupt
mechanical execution. No report, vote target or suspicion threshold is scripted.
A physics/reward/discount equivalence test passes. The packet and policy schema
are unchanged, and checkpoint metadata chooses the correct runtime protocol.

Current study: three full-memory initialization seeds (17, 29, 43) and a separately
trained current-only seed-17 comparison, each at 8,192 option decisions. The single
ablation seed is exploratory; it cannot establish replicated memory benefit.
Checkpoint screening uses 20 validation seeds; selected checkpoints then receive
100 validation matches. Larger final ID/patient evaluation follows a locked
selection. More learning is considered from validation trends, not final results.
The original six-run/32,768-step proposal remains historical, not a mandatory count.

Generated figures distinguish policy decisions, simulated time, match seeds,
training seeds, PPO objectives, voting accuracy and whole-match contribution.
The teammate document stays outside the repository and includes a suggested
Phase 6 sequence: prove one useful learned crewmate before adapting opponents
and self-play. Phase 6 has not been implemented.
