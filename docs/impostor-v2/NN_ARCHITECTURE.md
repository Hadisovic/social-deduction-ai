# Current learned impostor — exact neural architecture and inputs

**Latest research release:** V4.1,83/8192, SHA256
`69965afd8fab28e6ca5bdda46d9f8efdd8d6036ce57a05b86ff89fe22cf17ca9`.
Run `python run_impostor_v41.py`. Same architecture; the three overrides below
apply before reading the preserved slot tables. [Fresh results](CURRENT_RESULTS.md).

The original full slot tables describe preserved `impostor-strategic-v2`,71/8192, SHA256
`8b861c9cb7477a2145d41af3961e03aa5d9bd6c1df1d1aaea8629c2c9597698c`.
This older release remains reproducible with its original feature semantics.

## New V4.1 experiment: same architecture, three versioned feature slots

The full slot tables below describe the preserved V2 release. The corrected V4.1
experiment keeps **75,075 parameters, all tensor dimensions and12 action families**,
but uses the following explicit overrides. It is not silently compatible feature
semantics; old weights under new rules are labelled transfer demos.

| Tensor slot, one-based within its tensor | Preserved V2 table below | V4 / corrected V4.1 |
|---|---|---|
| Global4 | Boolean own interaction | Own remaining fake-task time /4;0 if not faking |
| Extra3 | Own recorded kill count /3 | Observed proximity high-water frontier |
| Extra4 | Age since own kill | Credited pursuit potential / configured0.05 lifetime scale |

V4's initial crowded spawn saturated that frontier; its study was stopped before
validation and preserved. **V4.1 clears geometry history while cooldown is positive,
then arms a non-paying visible baseline when ready.** Credit is never replenished
by cooldown/target switching and is repaid at terminal. Real-spawn tests and ordinary
training demonstrate nonzero credit in90/150 games. Fresh V4.1 confirmation:
0/20wins,34kills,0fake selections; shaping's isolated learning benefit is unproven.

The new rewards are+3kill,extra-6ejection,+10win,-10loss,-12timeout,no unconditional
time cost,-0.02 per active/roaming cooldown-ready simulated second, and bounded
pursuit potential differences. Fakes now finish; repeated active same-console
selections continue rather than restart. Strategy still comes from the NN.
See [the corrected protocol](options-v41/TRAINING_PROTOCOL.md) and
[master checklist](IMPOSTOR_MASTER_CHECKLIST.md) for current training/evaluation.

## Architecture

| Component | Exact input | Hidden layers | Output | Parameters |
|---|---|---|---|---:|
| Actor player encoder, shared across players | 21 per player; four rows | Linear 21→32, Tanh; Linear 32→32, Tanh | Four 32-value embeddings | 1,760 |
| Actor context | 35 globals + 8 extras + mean of four player embeddings (32): 75 | Linear 75→128, Tanh; Linear 128→128, Tanh | 128-value context | 26,240 |
| Candidate encoder, shared across candidates | Context 128 + targeted player embedding 32 + candidate features 51: 211 | Linear 211→64, Tanh | 64 values per candidate | 13,568 |
| Candidate score | 64 candidate values | None | One logit per candidate | 65 |
| Family embedding | Family ID 0–11 | Learned lookup | 16 values per family | 192 |
| Family score | Context 128 + mean candidate embedding within family 64 + family embedding 16: 208 | Linear 208→64, Tanh | Linear 64→1: one logit per family | 13,441 |
| Critic player encoder, independent of actor | Same 21 player features | Linear 21→32, Tanh; Linear 32→32, Tanh | Four embeddings; mean gives 32 values | 1,760 |
| Critic value head | 35 + 8 + 32: 75 | Linear 75→128, Tanh; Linear 128→64, Tanh | Linear 64→1: expected return | 18,049 |
| **Total** | Variable legal candidates, fixed tensor capacities | Feedforward; no LSTM/GRU or transformer | Action distribution + scalar value | **75,075** |

The actor has 55,266 parameters; the separate critic has 19,809. Historical memory
is computed outside the NN from legitimate evidence. It is not a recurrent layer.

## Packet shapes

| Input | Shape for one decision | Meaning |
|---|---|---|
| `global_` | 35 | Own state, public progress, phase, own region |
| `extra` | 8 | Bodies, own kill history, own goal, kill availability |
| `players` | 4 × 21 | Four other players, in actor memory's roster order |
| `actions` | 256 × 51 | Legal candidate descriptions; unused rows padded |
| `targets` | 256 | Other-player index 0–3, or -1 when none |
| `mask` | 256 | 1 = legal candidate; 0 = unavailable/padded |

The policy does not flatten these into one huge input layer. Shared encoders and
pooling score candidates. If 30 candidates exist, the other 226 are masked out.

## Global inputs — all 35 slots

Indices below are one-based within their tensor. Names describe the code's slots;
the packet itself stores arrays, not named dictionaries for individual features.

| # | Input | Meaning / normalization |
|---:|---|---|
| 1 | own_x | Own horizontal position /30 |
| 2 | own_y | Own vertical position /30 |
| 3 | navigating | 1 if own navigation status is moving |
| 4 | interacting | 1 if own interaction task is set |
| 5 | kill_cooldown | min(own remaining cooldown /45, 1) |
| 6 | elapsed_time | min(simulated match seconds /900, 1) |
| 7 | public_task_progress | Public completed / total task-bar units |
| 8 | meetings_remaining | Own emergency meetings remaining |
| 9 | accusations_against_self | Count of public suspicion claims about self /8, capped at1 |
| 10 | kill_claims_against_self | Count of public murder claims about self /8, capped at1 |
| 11 | votes_against_self | Count of observed votes against self /8, capped at1 |
| 12 | phase_roaming | One-hot phase |
| 13 | phase_discussion | One-hot phase |
| 14 | phase_voting | One-hot phase |
| 15 | phase_finished | One-hot slot; terminal packet is actually absorbing zeros |
| 16 | region_Cafeteria | Own-region one-hot |
| 17 | region_Weapons | Own-region one-hot |
| 18 | region_O2 | Own-region one-hot |
| 19 | region_Navigation | Own-region one-hot |
| 20 | region_Shields | Own-region one-hot |
| 21 | region_Communications | Own-region one-hot |
| 22 | region_Storage | Own-region one-hot |
| 23 | region_Admin | Own-region one-hot |
| 24 | region_Electrical | Own-region one-hot |
| 25 | region_Lower_Engine | Own-region one-hot |
| 26 | region_Security | Own-region one-hot |
| 27 | region_Reactor | Own-region one-hot |
| 28 | region_Upper_Engine | Own-region one-hot |
| 29 | region_MedBay | Own-region one-hot |
| 30 | region_NorthHallway | Own-region one-hot |
| 31 | region_CrossHallway | Own-region one-hot |
| 32 | region_AdminHallway | Own-region one-hot |
| 33 | region_BigYHallway | Own-region one-hot |
| 34 | region_SouthHallway | Own-region one-hot |
| 35 | region_Unknown | Own-region one-hot |

## Extra own-state inputs — all 8 slots

| # | Input | Meaning |
|---:|---|---|
| 1 | visible_body | At least one body is currently visible |
| 2 | nearest_visible_body_distance | Euclidean distance /60, capped at1; 0 if none |
| 3 | own_recorded_kills | Count of own directly recorded elimination events /3 |
| 4 | time_since_own_kill | Simulated seconds since last own recorded kill /900, capped at1; 1 if none |
| 5 | goal_dx | Current own travel goal x minus own x, /60; 0 without goal |
| 6 | goal_dy | Current own travel goal y minus own y, /60; 0 without goal |
| 7 | moving_to_goal | Own goal exists and navigation status is moving |
| 8 | legal_kill_available | At least one legal KILL candidate exists now |

## Player inputs — all 21 slots for each of four other players

| # | Input | Meaning |
|---:|---|---|
| 1 | ever_seen | Has a legitimate remembered sighting |
| 2 | last_seen_age | Simulated age /900, capped at1; 1 if never seen |
| 3 | visible_in_memory | Is in the latest memory-visible set |
| 4 | sighting_count | Retained sighting events /8, capped at1 |
| 5 | interacting_sightings | Retained sightings while interacting /8, capped at1 |
| 6 | body_evidence | Directly observed body evidence for this player /8, capped at1 |
| 7 | known_absent_or_ejected | Public meeting absence/ejection evidence, capped at1 |
| 8 | reports_made | Observed reports by this player /8, capped at1 |
| 9 | votes_cast | Observed votes by this player /8, capped at1 |
| 10 | votes_received | Observed votes targeting this player /8, capped at1 |
| 11 | accusations_received | Suspicion/murder claims targeting this player /8, capped at1 |
| 12 | defence_received | Defence claims targeting this player /8, capped at1 |
| 13 | visible_now | Appears in current actor-visible players |
| 14 | visible_dx | Visible player x minus own x, /60; 0 when unseen |
| 15 | visible_dy | Visible player y minus own y, /60; 0 when unseen |
| 16 | remembered_dx | Last legitimately seen x minus current own x, /60 |
| 17 | remembered_dy | Last legitimately seen y minus current own y, /60 |
| 18 | sighting_age | Simulated age /900, capped at1; duplicates slot2 |
| 19 | public_confirmed_crew | Public confirmed-ejection evidence says crewmate |
| 20 | public_confirmed_impostor | Public confirmed-ejection evidence says impostor |
| 21 | public_visual_task_proof | Personally received legitimate visual-task proof |

Sighting keyframes are reduced within 2 seconds when room/activity are unchanged.
Slot3 and13 are redundant in normal observations. Redundancy is documented here;
removing slots would change the current checkpoint architecture. Public confirmed
roles are public evidence, not a hidden-role oracle. In this one-impostor lobby,
slot20 is normally unused for other players.

## Action-type inputs — 22 one-hot slots

These are input encodings of candidate types, not 22 independent output buttons.
Only 17 types are currently generated for this impostor. Schema placeholders stay
for compatibility and do not create legal actions.

| # | Action input | What the impostor can do | Offered now? |
|---:|---|---|---|
| 1 | continue | Continue current navigation/activity without a new intent | Yes |
| 2 | wait | Legacy explicit wait encoding | No: represented by continue |
| 3 | go_to_task | Travel to a public task console; does not complete crew tasks | Yes |
| 4 | go_to_room | Travel to a room | Yes |
| 5 | go_to_location | Travel toward a currently visible player's observed position | Yes; not continuous automatic tracking |
| 6 | go_to_body | Approach a visible body | Yes |
| 7 | cancel_navigation | Stop navigation/activity | Yes |
| 8 | interact | Perform an assigned real crew task | No: impostor has no assigned real tasks |
| 9 | fake_task | Start fake activity at a nearby console | Yes |
| 10 | kill | Kill a legal visible nearby player when cooldown is ready | Yes |
| 11 | report | Report a legal nearby visible body, including own victim | Yes |
| 12 | call_meeting | Call an emergency meeting at the console when available | Yes |
| 13 | vote | Vote for a legal meeting participant, including self if legal | Yes |
| 14 | skip | Skip voting | Yes |
| 15 | claim_saw_player | Share a remembered player sighting | Yes, if a sighting exists |
| 16 | claim_was_in_region | State own location in a chosen room; may lie | Yes |
| 17 | claim_found_body | Body-discovery claim schema slot | No current candidate generator |
| 18 | claim_saw_player_near_body | Near-body allegation schema slot | No current candidate generator |
| 19 | claim_suspect_player | Accuse another player | Yes |
| 20 | claim_defend_player | Defend another player | Yes; no self-defence candidate |
| 21 | claim_was_at_task | Task-alibi claim schema slot | No current candidate generator |
| 22 | claim_saw_elimination | Claim another player committed a murder; may lie | Yes |

## Other action inputs — 29 slots, making 51 per candidate

| # | Input | Meaning |
|---:|---|---|
| 23 | target_dx | Destination x minus own x, /60; 0 if no position |
| 24 | target_dy | Destination y minus own y, /60; 0 if no position |
| 25 | target_distance | Euclidean distance /60, capped at1; no route-distance input |
| 26 | has_position | Candidate has a destination position |
| 27 | targets_self | Intent's target ID equals own player ID |
| 28 | claim_age | Seconds since asserted claim tick /900, capped at1 |
| 29–48 | candidate_region | Same 20 regions and order as global slots16–35; one-hot |
| 49 | target_visible | Candidate targets another player currently visible |
| 50 | nearest_visible_neighbor | Closest other visible player's distance from target /20, capped at1; default1 |
| 51 | visible_neighbors_in_4_5 | Count of other visible players within4.5 of target /3 |

Slots49–51 use only visible players. They do not reveal all actual witnesses or
unseen nearby players. A candidate may have no region, so all20 region slots can
be zero. There is no crew-task progress per action in this impostor packet.

## Outputs — learned family then learned candidate

| # | Family | Candidate types grouped here |
|---:|---|---|
| 1 | continue | continue / legacy wait |
| 2 | travel | go_to_task, go_to_room, go_to_location |
| 3 | cancel | cancel_navigation |
| 4 | fake_task | fake_task |
| 5 | kill | kill |
| 6 | approach_body | go_to_body |
| 7 | report | report |
| 8 | emergency | call_meeting |
| 9 | claim | All encoded claim types; only five generated |
| 10 | vote | vote |
| 11 | skip | skip |
| 12 | interact | interact; dormant for this impostor |

Probability is P(family | observation) × P(candidate | family, observation).
Illegal/empty families and padded candidates receive no probability. Greedy viewing
chooses the highest-probability family, then its highest-probability candidate.
Sampled play draws from the joint distribution with a private RNG. The policy
outputs one candidate index0–255, not continuous keyboard movement.

## Gameplay logic: learned decisions versus fixed mechanics

| Behavior | Who controls it | What is established |
|---|---|---|
| Where to walk and which visible target to approach | NN selects candidate; A* executes route | Available; useful sustained pursuit not demonstrated |
| Whether/whom to kill | NN; engine enforces visibility, range and45s cooldown | Actual learned kills and wins occurred |
| Whether to fake, continue or leave | NN; engine implements activity | Historical V2 restart loop is a measured failure |
| Whether to report/call emergency | NN selects legal candidate | No rule forbids self-reporting |
| Accusation, defence of others, location alibi, sighting/murder claim | NN selects structured template and target | No free text; availability does not prove effective deception |
| Vote/skip | NN selects legal participant/skip | Sensible voting is not yet demonstrated |
| Avoid killing in front of witnesses | Must be learned from outcomes | No hardcoded kill-alone rule; stealth not established |
| Living crew body reports/direct-witness votes | Explicit opponent safeguards | Rules, not learned impostor skills |
| Walls/obstacles/physics/tasks/ghost tasks | Canonical game engine | No strategy override or hidden actor information |
| Vents, sabotage, two impostors | Unsupported in this interface | Not trained or silently simulated |

Current match: **one learned impostor + one frozen full-29 learned crewmate + three
scripted crewmates**. This is five players, not ten crew plus two impostors.
The friend's newer crew is not loaded. Public roles/sight overlays are spectator
features and never enter neural input. No unseen live positions, private crew
beliefs, hidden witness truth or ghost secrets are provided.

## Learning and evidence

| Item | Current trained V2 |
|---|---|
| Algorithm | PPO, independent actor/critic gradients |
| Learning rate / optimizer | 0.0003 / Adam, epsilon0.00001 |
| Rollout / minibatch / epochs | 256 /64 /4 |
| Ratio clip / value coefficient | 0.2 /0.5 |
| Actor / critic gradient clip | 0.5 /1.0 |
| Exploration | Family entropy0.02→0.005; normalized conditional entropy0.001 |
| Discount / GAE | gamma1; physical-time lambda0.995 per simulated second |
| Historical rewards | +10win, -10loss, -12timeout, +2kill, extra-2ejection, -0.002/s |
| Timing | Physics0.2s; base choices0.6s; interruptible travel up to3s |
| Current selected evidence | 2/20 selection wins, then0/20 fresh confirmation,4kills |
| First learned win | A separate seed83 training game:2kills and an innocent crew ejection |

New reward/action work is tracked in [the live checklist](REWARD_ACTION_V4_CHECKLIST.md).
Sources: `impostor_research/strategic_v2.py`, `features.py`, `memory.py`,
`learning_v2.py`, `crewmate_v2/runtime.py`, `social_deduction/phase3_api.py`.
