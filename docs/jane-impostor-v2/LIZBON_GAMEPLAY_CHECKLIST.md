# Lizbon: complete gameplay and development checklist

Updated 31 October 2026. Branch: `codex/lizbon-classic-v2`.

**Current verdict: functional research agent, undertrained impostor.** It can
choose movement, kills, sabotage and social actions. Reliable searching, safe
kills, deception and teamwork have not been demonstrated. The current migrated
development checkpoint has **no completed validation or final-test win rate**.
No training was started for this checklist or the new lobby UI.

Status labels below distinguish **implemented rules/interfaces**, **observed but
weak behavior**, **unproven learned skills**, and **not implemented**. An unchecked
skill is a development target, not a missing hardcoded strategy. The NN must learn
when to act; legality, cooldowns, collisions and privacy belong to the engine.

## 1. One, two or three Lizbons

All impostors use **one shared Lizbon network**. Each has its own private memory,
GRU state and sampling generator. Patrick Jane crew use a separate frozen network.
Two sampling streams of training are different experiments, not two impostors
with different jobs. There is no assigned hunter, camper or leader.

| Lobby | Supported engine population | What good play should learn | Current evidence |
|---|---|---|---|
| 1 impostor | 4–15 total, 3–14 crew | Find isolated victims; create escape opportunities; balance kill speed against exposure; survive meetings without an ally. | Configuration supported. No robust one-impostor evaluation. |
| 2 impostors | 7–15 total, 5–13 crew | Avoid duplicate pursuit/wasted sabotage; avoid exposing an ally; combine independent kills or distractions; recognize when parity makes an immediate kill decisive. | Latest training used 10 crew + 2 impostors. Useful cooperation unproven; one known-ally vote observed. |
| 3 impostors | 9–15 total, 6–12 crew | Coordinate limited team resources and target coverage; avoid clustering; adapt to larger ally fraction; preserve individual evidence boundaries. | Configuration supported. No trained three-impostor competence established. |

The 7/9 minima are this shared adapter's conservative validator, not a claim
that every retail build uses those exact limits. Known ally identities are legal
inputs. Another impostor's private observations, memory or unseen live position
are not automatically shared. Simultaneous door/sabotage contention is resolved
by engine legality/order; that is not learned team coordination.

- [x] One architecture accommodates up to 15 total players / 14 other-player rows.
- [x] Separate private state per living impostor; known allies encoded.
- [x] Illegal kill targets exclude allies.
- [x] Duplicate shared-resource requests are audited without inventing a new strategy.
- [ ] Learned ally-aware voting: currently a vote against a known ally remains possible.
- [ ] Learned target coverage, planned double/triple kills, rescue/distraction and coordinated alibis.
- [ ] Replicated evaluations for each population and impostor count.

## 2. Rules and host settings

Open `python run_lizbon.py`, then **Lobby**. Player count includes impostors.
Choose Total players, Impostors and Common / Long / Short tasks, save a profile or
start an inference-only game. These controls create a new game; a replay cannot
be reconfigured. New settings are not evidence of model generalization.

| Rule or feature | Implemented now | Limit / remaining work |
|---|---|---|
| Crew victory | Complete owned tasks, including ghost task progression; or eliminate all impostors. | Strong reporting/voting competence is a separate crew-learning problem. |
| Impostor victory | Living impostor/crew parity under engine rules. | O2/reactor countdown victory is absent because those sabotages are disabled. |
| Task counts | Common 0–2, Long 0–3, Short 0–5 per crewmate. | All zero uses one Short. Categories come from the pinned task pool, not newly calibrated retail metadata. |
| Task uniqueness | No duplicate canonical task family for an owner; common plans match across crew. | Repeated stages of one task are legitimate. Exact minigames are not implemented. |
| Kill legality | Living eligible nonally, range/visibility, cooldown and valid phase. | Medium range only; short/long calibration deferred. Initial cooldown is 10 simulated seconds, subsequent configured default is 45. |
| Kill clock | Simulated elapsed time, with meeting/underground semantics in the engine. | No wall-clock or decision-count shortcut. |
| Walls, furniture, doors | Collisions and A-star navigation; timed closed doors obstruct movement. | Exact retail geometry and animation timings remain approximations. |
| Reporting | Visible/reachable corpse can start a meeting; corpses and witnesses are recorded. | Impostor self-reporting is legal and selectable, not automatically intelligent. |
| Emergencies | Configured count, cooldown and console interaction; blocked during active sabotage. | Strategic timing is unproven. |
| Discussion | Structured turn-taking, configurable duration; zero skips to voting. | No free-form text/voice generation. |
| Ballots | Eligible living participants; target vote, skip, ties/no ejection, resolution and confirm-eject setting. | Current inherited mask excludes self-votes; infinite voting is deferred. |
| Public information | Public votes/claims/results with configured anonymous-vote and task-bar visibility. | Private submitted votes and other players' minds are not actor inputs. |
| Ghosts | Crew task progression stays separated from living observations. | Impostor ghost sabotage/assistance disabled. |
| Lights | Reduced crew visibility and repairs; impostor can choose sabotage/repair. | Exact switch-panel minigame absent; crew is not A2-trained. |
| Doors | Room closures, cooldown/expiry, obstruction/recovery and local knowledge. Viewer animates actual state. | Door animation is presentation; a requested closure is not always immediately physically accepted. |
| Vents | Enter linked network, move, exit; transition timing and direct witness events. | Useful vent strategy unproven; no Clean Vent eviction. |
| Settings beyond count controls | JSON profiles support current implemented speed, vision, meetings and related switches. | Dashboard exposes population/task counts and viewing options, not every possible retail setting. Unsupported mechanics remain rejected. |
| Engine stability | Inactive-position and map/planner cache-coherence fixes in versioned adapter. | Long sampled door-heavy stability/performance still needs broader confirmation. |

Default training lobby was **12 total, 2 impostors, 1 Common / 1 Long / 2 Short**,
15s discussion, 120s voting and 45s regular kill cooldown. Read the exact profile
embedded in each recording rather than assuming these defaults for a custom game.

## 3. What the neural policy should learn, action by action

These are conditional objectives, not an if/else gameplay script. A risky kill
may be rational if it immediately wins. Waiting, faking, reporting or defending
an ally can be useful in some contexts and harmful in others.

| Gameplay decision | Desired learned behavior | Current implementation and evidence | Next verification |
|---|---|---|---|
| Search for victims | Leave unproductive positions; choose plausible rooms/routes from public map and observed history; cover gaps without omniscient tracking. | Room/console, visible-player and remembered-player goals available. Latest logs: 970 console, 314 room, 119 player-location choices. Reliable active search not established. | Time to first encounter, distance during kill-ready time, completed search goals and kills per ready roaming minute. |
| Pursue visible crew | Continue a useful route; update when a visible victim moves; abandon when risk or opportunity changes. | Visible-player goals supported; A-star executes destinations. This is not an automatically tracking hunter. | Contact conversion and time-to-range; exclude allies/dead targets; test loss of sight. |
| Use remembered positions | Treat old sightings as uncertain; search rather than track an unseen live coordinate. | Last-seen positions/time are encoded; player-goal candidates expire after 120 simulated seconds. | Stale-goal travel, target switching and lawful reacquisition. |
| Finish navigation | Keep useful commitments; cancel for a reason; revisit when circumstances change. | Arrived goals and unchanged active static goals are filtered. Latest logs still contain 663 cancellations. | Cancel-before-arrival rate, route progress and achieved goals; do not punish necessary door evasions or emergency changes. |
| Wait or fake | Use downtime and cover deliberately; finish or abandon a fake when conditions change. | Persistent finishable fake option; continuing does not restart it. Latest study: 63 fake starts, about 48.2 actor seconds. Old fake loop is not the main current failure. | Completion/interruption, opportunity cost and whether faking helps survival or team outcome. |
| Kill | Convert lawful opportunities into timely kills, with an escape or immediate parity benefit. | Kill action exists; 19 selected/accepted kills in all readable Classic study prefixes; 63 kill-eligible actor decisions. These decisions are correlated, not accuracy trials. | Contiguous opportunity windows, accepted kills, parity outcomes and missed windows by reason. |
| Avoid witnesses | Assess *observable* nearby crew/line-of-sight and likely exposure; trade it against decisive victory. | Visible counts/history are inputs. No guaranteed witness-risk skill. Offline truth: 8 of 19 kills had another living crew witness. | Paired cases with/without visible observers; record witness risk offline only. |
| React after being seen | Remember own action and public/direct evidence; avoid further contradictions; choose escape, meeting response or decisive play. | GRU/event memory and social/movement actions exist. No perfect “who saw my kill” sensor or access to another crew's memory. | Separate directly known witnesses from possible witnesses; evaluate response rather than supplying hidden witness lists. |
| Escape | Choose routes/doors/vents from legitimate knowledge; avoid returning to an exposed body without a purpose. | Available primitives; intelligent post-kill escape unproven. | Time/route from kill to safety, re-encounter risk, report/ejection outcome. |
| Self-report | Report when manipulating meeting timing or a plausible story outweighs ending kill time; otherwise leave/search. | Latest 10 report choices were 10 actual self-reports. No evidence these were strategically justified. | Controlled self-report versus leave-body opportunities; meeting outcomes and lost kill-ready seconds. Do not impose a blanket ban. |
| Approach/report another body | Exploit legitimate discovered evidence and timing where useful. | Body-travel/report candidates exist. | Distinguish own kill, ally kill and unrelated discovered body in offline diagnostics. |
| Call emergency | Use limited meetings to prevent a loss or affect voting, with lawful travel/cooldown. | Supported; no demonstrated useful emergency strategy. | Cost in paused opportunities, remaining buttons and resulting vote/task outcome. |
| Lights sabotage | Reduce crew sight when it helps an approach, kill or escape; consider repair traffic and team cooldown. | Six choices / four accepted effects in latest prefixes. The ability exists; useful timing is unproven. | Compare matched scenes without changing crew hidden state; kill opportunity and repair/evidence effects. |
| Door sabotage | Separate targets/witnesses, delay tasks or shape escape, while avoiding trapping self/allies and pointless closures. | 319 choices / 294 accepted closures; frequent closures expose runtime cost. | Useful delay/isolation, route obstruction, self/ally delay and downstream outcome. Closure count alone is not success. |
| Combine tools | Sequence search → isolate → kill → escape when appropriate; coordinate with ally resource availability. | Primitive actions available, no proven multistep plan. | Full-sequence episodes and ablations, not isolated action count rewards. |
| Use vents | Enter without exposing identity where possible, select linked exit based on known context, exit safely. | Interface/witness rules exist; zero entries selected despite two eligible decisions in latest prefixes. | Entry/move/exit legality, cooldown pause, observed vent exposure and useful escape/arrival. |
| Accuse | Select a plausible target/claim based on public story and remembered evidence; avoid provable contradictions. | Structured suspicion and witnessed-kill claims are selectable, including false assertions. Three suspicions/one witness claim observed. | Consistency, who heard it, later ballot shift and whether a crew was actually ejected. |
| Lie about location | Use plausible alibis when valuable; account for prior statements, sightings and travel time. | Location-claim choices can name rooms. Eleven location claims observed; convincing lies not demonstrated. | Offline contradiction checks using evidence available to listeners; no hidden witness truth as actor input. |
| Defend self/ally | Address the meeting context, preserve credibility and team outcome; defending an exposed ally may be harmful. | Structured defence including self is available. Two defence choices observed; no persuasive skill established. | Accusation→response→ballot chains; change in vote outcomes against stronger frozen crew. |
| Share evidence | Refer to remembered events with source/time; distinguish claims from direct facts. | Event-sharing candidates and provenance implemented; 16 shares observed. | No laundering a repeated rumour into independent direct evidence. |
| Vote or skip | Use meeting state/evidence/team context; avoid arbitrary ballots and recognize decisive voting. | 17 target votes / 21 skips; one known-ally vote. No actual ejections in latest recorded prefixes. | Ballot usefulness, ties, ally votes, innocent ejections, own ejection and parity. |
| Adapt after ally elimination | Re-evaluate risk and parity with fewer impostors; maintain private-state isolation. | Known/public roster/ejection history available; no specialized policy demonstrated. | Multi-stage matches with ally loss; no post-death privileged feed. |
| Manage task pressure | Act fast enough to reach parity before living/ghost crew complete work. | Public task bar is encoded when enabled. All three complete latest training games lost to tasks. | Remaining work/time-to-parity, sabotage delay and kill opportunities; kills alone do not stop ghost tasks. |

## 4. Information and reasoning limits

- [x] Own position, phase, settings, cooldown/fake status, navigation and recent actions.
- [x] Up to 14 other-player rows: legal current sight or timestamped last-seen information, known allies and public evidence.
- [x] Up to 64 remembered events with source/provenance; 256-unit recurrent state.
- [x] Legal action candidates scored by family and target; engine handles execution.
- [x] Separate actor/critic trunks; unchanged 1,036,561-parameter network.
- [ ] Explicit calibrated belief about **other players' suspicion of the owner**. There is no reliable suspicion scoreboard to optimize directly.
- [ ] Long-term deception/planning skill. Choosing a claim does not demonstrate understanding or persuasion.
- [ ] Free-form dialogue, voice, natural-language debate, private team chat or language-model explanations of intent.
- [ ] Guaranteed knowledge of every observer of a kill/vent, hidden votes or unseen live positions. Those must never be added as privileged actor features.

The existing critic uses the same legal packet. Offline omniscient replays help
us diagnose mistakes; they do not feed the actor. “Why the model chose it” can
describe probabilities and context, not invent an internal verbal explanation.

## 5. Not added yet

| Feature | Status / prerequisite |
|---|---|
| Shapeshifter / shapeshifting | **Not implemented.** No ability, disguise timer, appearance evidence, counterplay or trained response. Base-role Lizbon cannot shapeshift. |
| Phantom/invisibility and other special impostor roles | Not implemented; require versioned rules/observations/actions and both-role tests. |
| Scientist, Engineer, Guardian Angel and other crew roles | Disabled. Vented crew inference must change before introducing Engineer; venting is not universally incriminating once such roles exist. |
| O2 / reactor / communications | Disabled; add sabotage rules, repairs, victory/deadline semantics and crew competence together. |
| Dead-impostor sabotage | Disabled; avoid privileged ghost-to-living information flow. |
| Security cameras / Admin device observations | Not implemented as strategic observation devices. Map art is not device functionality. |
| Clean Vent eviction | Deferred until occupant/eviction semantics and tests exist. |
| Every retail task/minigame and role probability | Deferred; count/stage simulation is not an exact minigame replica. |
| Online adapting crew / self-play | Deferred until a frozen one-policy baseline works and reproducible cross-play succeeds. |
| “Best killer” / robust retail transfer | Not established. No architecture or reward value guarantees it. |

## 6. What the tests and win rates actually say

| Study/check | Completed outcome | Meaning / missing evidence |
|---|---|---|
| Older shared impostor pilot, 29 Oct | Training: 101 = 7/8, 202 = 7/8; combined **14/16 (87.5%)** against frozen B. | Different earlier system/opponents. Does not measure current Lizbon strength. Camping/self-report strategies could win there. |
| Same older pilot validation | 101/ep4 = 3/6 completed of 8 planned; 101/ep8 = 8/8; 202/ep4 = 3/8; 202/ep8 = 5/8. | 30/32 completed, execution-time ceiling stopped completion. No selection or paired A/B confirmation. The 8/8 result is not final-test accuracy. |
| Lizbon V1, 30 Oct | One completed initial-policy training game: **0/1 win**, four kills, crew task win; eight PPO passes saved. | Second game failed on inactive-position geometry. No completed evaluation of updated weights. |
| Lizbon Classic V2, 30 Oct | Three complete training games: **0/3 wins**, kills 3/5/4, all task losses. Two wall cuts and two interrupted prefixes excluded. | 24 new PPO passes across two same-start streams. No validation, selection, confirmation or final testing. These games used changing weights. |
| Classic greedy engineering game | **0/1**, no kills; each impostor waited 284/289 choices, five light choices. | Same default lobby, different scenario and greedy inference mode; cannot infer paired improvement or a trained-model win rate. |
| Current migrated checkpoint003 | Same tensors, 24 inherited total passes; explicit new geometry receipt. | **Current evaluated win rate: unavailable.** No competence improvement from migration; it performed zero new PPO updates. |
| Cache repair | 87 isolated Jane checks; three offline comparator checks. Paired engineering runs agreed on the common 110-decision prefix. | Functionality/determinism evidence; runs stopped at wall caps, not full-match performance proof. |
| Dashboard checks | See `DASHBOARD_LOBBY.md` for current UI/inference evidence and the retained failed reader-lock run. | Interactive smoke games are engineering checks, not validation or final-test samples. |

For perspective, even 0/3 has a roughly 0–56% Wilson 95% interval under an
independent fixed-policy binomial model. These are changing-policy training
games, so that interval is illustrative, not a valid final estimate. Three
losses are sufficient to reject “proven strong”, not to quantify true skill.

Latest Classic diagnostics across all readable complete/incomplete prefixes:
**4,938 decisions, 19 kills, 8 witnessed kills, 10 self-reports, zero actual
ejections.** Eight of nineteen is about **42% witnessed**, an offline descriptive
fraction with dependent events. Nineteen null-player ejection envelopes are not
nineteen eliminations. No ejections is weak evidence of stealth when crew fail
to convert witnessed evidence into ballots. Frozen crew update102 is learned
task-pursuit V1, **not A2-trained** for this full mechanic set.

## 7. Current reward, and what to tune next

Current Classic code, not earlier environment proposals:

| Reward component | Current value |
|---|---:|
| Each accepted team kill | +1 per impostor trajectory; not a divided pool |
| Natural impostor team win | +10 per impostor trajectory |
| Natural team loss | −10 per impostor trajectory |
| Living, surface, eligible kill-ready roaming time | −0.005 per simulated second |
| Extra ejection / witnessed kill / self-report / travel reward | None |
| Technical cut / wall abort | Excluded from PPO and scored win/loss denominator |

The earlier +3 kill, −6 ejection and +0.05 pursuit proposal belonged to earlier
environment work. It is **not** the current Jane Classic reward configuration.
Do not silently relabel a checkpoint or retrofit its historical reward receipt.

Increasing kill reward alone can encourage suicidal witnessed kills. Penalizing
ejections alone can reward camping while tasks finish. Penalizing every report,
wait or cancellation can destroy legitimate choices. Optimize winning with low
unnecessary exposure, not a replay that merely looks busy.

Proposed versioned experiments, after engineering gates and explicit budget approval:

1. Keep current architecture/reference fixed. Collect complete sampled and greedy
   baseline matches with full diagnostics before changing rewards.
2. Measure cancellations and candidate-family dilution. Try **one** versioned
   option-continuation/commitment design that preserves cancellation and emergency
   control. Verify route completion improves without forcing hunt/kill choices.
3. If pursuit is still absent, compare bounded observable potential shaping
   `gamma * Phi(next) - Phi(current)` at a declared scale. Define visibility loss,
   target changes, terminal states and action-duration discounting explicitly.
   Test oscillation, switching, stale memories and reset farming before training.
4. If witnessed kills remain high, first add *lawfully derived* visible-context
   risk features or an observation-based auxiliary objective in a new interface.
   Never give the actor the offline hidden-witness list. Compare with a carefully
   bounded exposure/ejection-cost ablation; retain team victory as the main goal.
5. Test report/claim/vote strategy against stronger frozen crew who actually use
   witnessed evidence. Otherwise the learner is rewarded for exploiting weak
   listeners rather than developing a credible alibi.
6. Test sabotage contribution with paired opportunity scenes and matched full
   games. Add one additional sabotage only after the crew can respond to it.

## 8. Concrete next-step checklist and responsibilities

### Owner / Lizbon developer

- [x] Repair and version geometry cache coherence; preserve all old failures/models.
- [x] Expose population/task counts through a new dashboard lobby.
- [ ] Complete longer sampled, door-heavy engineering matches after the fix; profile
  collection/recording and verify no orphan runtime children.
- [ ] Establish a complete fixed-policy reference on the 12-player default lobby.
- [ ] Diagnose idle time excluding forced meeting waits, door blockage, cooldown
  and intentional fake duration. Track redundant goals separately from cancellations.
- [ ] Register a single learning change/ablation and approve measured compute time.
- [ ] Train with enough complete matches to cover encounters, meetings, ally loss
  and tasks; tiny optimizer passes alone are not a competence gate.
- [ ] Compare independent initialization seeds as well as sampling seeds.
- [ ] Save all cuts/failures and checkpoints; never restart/extend a registered study silently.

### Friend / Patrick Jane crew developer

- [ ] Use the same versioned Classic `make_match` factory, settings/profile and geometry revision.
- [ ] Verify living body discovery→approach/report, and remembered direct kill/vent
  evidence→meeting claim/ballot. Rules safeguard privacy; do not train on ghost truth.
- [ ] Train/evaluate A2-capable repair, investigation and witness response; update102
  currently supplies task pursuit rather than proven all-mechanics competence.
- [ ] Publish a frozen inference checkpoint with source/settings/architecture hashes.
- [ ] Keep task throughput and innocent-ejection rate visible while improving reporting.

### Together: evaluation gate before stronger claims

- [ ] Freeze reference and candidate; use identical scenarios/settings/opponents
  for paired development evaluation. Report sampled and greedy modes separately.
- [ ] Record wins/natural completed matches, completion coverage, cuts/errors and
  cause-specific losses. A missing or wall-cut game is not a loss, win or game timeout.
- [ ] Record kills per kill-ready roaming minute, time to first encounter/kill,
  missed contiguous kill windows, witnessed-kill rate and task pressure.
- [ ] Record meaningful route completion, pre-arrival cancellations, voluntary idle,
  repeated goals, fake completion/interruption, self-reports and meeting timing.
- [ ] Record true ejections per eligible meeting, innocent/ally ballots, skip/tie
  rates, claim contradictions and listener vote changes. Do not call an internal
  suspicion score or action selection “accuracy”.
- [ ] Record doors/lights/vents accepted effects, useful outcomes, self/ally delay,
  privacy checks and actor-local versus offline diagnostic boundaries.
- [ ] Start with a small complete paired development screen; estimate time from
  actual match throughput before approving a larger study.
- [ ] For a later stable estimate, plan on the order of 200 natural matches per
  evaluated configuration (about ±7 percentage points worst-case 95% sampling
  uncertainty), with multiple seeds/opponent snapshots. This is a proposal, not
  permission to run now or a guarantee of sufficient research evidence.
- [ ] Select and lock on validation only. Keep every reserved final partition
  sealed until immutable selection; do not tune after inspecting final outcomes.
- [ ] Require paired win improvement without exploding cuts/witness exposure/task
  losses. Low ejection alone and high kill count alone are not success criteria.
- [ ] Frozen cross-play first, then 1/2/3-impostor and 4–15-player/task-workload
  generalization. Only then consider adapting opponents and self-play.
- [ ] Add optional roles/sabotages one at a time with both-role tests and new receipts.

## Evidence and implementation links

- [Latest stopped study and graphs](study-20261030/README.md), [decision analysis](study-20261030/decision-analysis.json).
- [Classic rules, sources and explicit approximations](RULES_AND_SOURCES.md).
- [Engineering game and known greedy failure](ENGINEERING.md).
- [Geometry repair/migration](CACHE_PAIR_FIX.md), [session checklist](CHECKLIST.md).
- [Lizbon V1 failed pilot](../jane-impostor-v1/pilot-20261030/README.md).
- [Earlier shared pilot](../collaboration/impostor-shared-integration/pilot-20261029/README.md).
- Current code: [legal features/actions](../../jane/impostor/v1/features.py),
  [network](../../jane/impostor/v1/model.py), [Classic rewards/session](../../jane/impostor/v2/session.py),
  [shared engine](../../jane/classic_v2/engine.py), [default settings](../../jane/configs/classic-v2/default_12p.json).
