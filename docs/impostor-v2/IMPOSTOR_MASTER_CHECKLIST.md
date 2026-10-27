# Learned impostor — full development checklist

Updated27 October2026, Asia/Amman. **Owner approved the bounded V4 training study;
prelaunch checks are in progress.** Checkboxes describe verified work, not a percentage of model quality.
“Best” means the strongest validated checkpoint among our tested candidates under
declared opponents/settings. It does not mean globally optimal play.

## Current position

| Item | Actual status |
|---|---|
| Architecture | Unchanged75,075 parameters; separate actor/critic;12-family then target choice |
| Current trained model | Historical V2, initialization71/checkpoint8192 |
| Previous evidence | Selected2/20wins; fresh confirmation0/20wins,4kills |
| First learned win | Yes: seed83 training game;2kills and innocent crew ejection |
| New V4 code | Reward/action/observation changes implemented and tested |
| New V4 trained model | Full study approved; prelaunch development gradient checks are separate from research training |
| Visible current game | Old V2 weights in a labelled V4 transfer demo; inference only |
| Current roster | 1learned impostor,1frozen neural crew,3scripted crew |
| Future roster | Configurable up to15 total players and1–3 impostors; validate legal combinations;10crew/2impostors is one preset |

## 1. Baseline, architecture and preservation

- [x] Work on `codex/impostor-environment-v2`, based on exact Crewmate V2 commit
  `e62ba0b6c33ffd0317f78e18b4289d778cd5af2b`, separate from main.
- [x] Read previous handoff, session summary, result reports and new reward prompt.
- [x] Keep current network architecture; address learning/interface failures first.
- [x] Document every input, hidden layer, output and strategic choice in
  [NN_ARCHITECTURE.md](NN_ARCHITECTURE.md).
- [x] Preserve all16 historical trained policy snapshots and old result files.
- [x] Strict-load8 V1 and8 V2 snapshots; frozen crew/belief checks pass.
- [x] Leave historical V1/V2 and prepared V3 runtime code unchanged.
- [x] Keep living-crew body reporting and direct witnessed-kill voting safeguards.
- [x] Preserve45s cooldown, map/physics, normal task victory and ghost isolation.
- [x] Keep existing V2 viewer available for faithful historical reproduction.
- [ ] Pin and compatibility-check the friend's newer crew before using it.

## 2. V4 reward correction — implemented

- [x] Successful kill +3, replacing+2.
- [x] Extra own-ejection penalty-6, replacing-2; ordinary loss still-10.
- [x] Preserve win+10, loss-10 and timeout-12.
- [x] Remove unconditional-0.002/s cost.
- [x] Charge-0.02/s only while alive, roaming and own cooldown ready, even in an
  empty room; calculate exact simulated-time overlap at cooldown boundaries.
- [x] Exclude meetings, eliminated actors and time after termination.
- [x] Preserve actual cooldown reset at a kill; do not punish its interval as idle.
- [x] Log terminal, kill, ejection, ready-time and pursuit components separately.
- [x] Document that shaped return can be negative for a slow win; assess true wins.
- [ ] In trained V4, check for risky witnessed kills and excessive meeting usage
  caused by the larger ready penalty. These are risks, not established outcomes.

## 3. Fake-task exploit — mechanism fixed, learned preference unresolved

- [x] Confirm V2 repeatedly restarted4s fake activity at0.6s decisions.
- [x] Make repeated same-console FAKE_TASK continue the existing activity.
- [x] Keep neural control over start, continue/repeat, cancel, travel, kill and report.
- [x] Allow completion; meetings/movement/cancel interrupt; no automatic resume.
- [x] Allow another deliberately selected fake after completion; do not force hunting.
- [x] Expose own remaining fake time in a versioned observation slot.
- [x] Test repetition, completion, interruption, continuing and deterministic replay.
- [x] Confirm complete development matches produce31/30 fake completions.
- [x] Record owner's live observation: old weights still repeatedly fake in an
  empty room and do this across their watched games. This is an unresolved policy
  failure, not the original restart bug; the owner's watched-game count is unstated.
- [ ] Train under the corrected interface after approval; watching alone does not
  change weights, even when new reward costs are displayed.
- [ ] Measure fake streak length, ready-idle seconds, visible-player count during
  faking, starts/completions/interruptions and opportunities forgone.
- [ ] Demonstrate less unproductive repeated faking on fresh scenarios through
  learned decisions, without a scripted fake limit or automatic chase.

## 4. Safe pursuit feedback — implemented, benefit untested

- [x] Use only actor-visible living players and own observation history.
- [x] Require meaningful own movement and actual distance progress across sightings.
- [x] Do not pay for a target approaching a stationary impostor.
- [x] Do not automatically choose a target, follow, kill or alter the legal action.
- [x] Bound positive credit to0.05 per match; no refill on target switch/meeting/kill.
- [x] Use potential differences and terminal correction; full-episode shaping sums0.
- [x] Test oscillation, switching, visibility loss/reacquisition, small motion and
  hidden-position isolation; document conservative high-water limitations.
- [ ] Check useful positive shaping actually occurs during exploration; current
  old-weight smoke matches earned none.
- [ ] Test whether shaping helps learning. Do not infer benefit from passing tests.

## 5. Implementation verification — completed

- [x]32 new V4 regression tests passed in56.59s.
- [x]27 original impostor-core regression tests passed in14.99s;59 checks across
  these two distinct suites. Do not add overlapping older suite totals.
- [x] Verify same network parameter count/state-dictionary shapes.
- [x] Verify masked legal neural choices and independent frozen opponent weights.
- [x] Verify checkpoint save/load and exact action-journal replay, including option
  state, actor/opponent traces and reward summary, without PPO updates.
- [x] Reject reserved final-test seeds in the new viewer.
- [x] Complete two bounded development matches in123.60s with zero illegal actions
  or navigation failures. Both lost by tasks with zero kills; not a quality estimate.
- [x] Run native/headless viewer checks; clearly label old weights as transfer.
- [x] Provide commands; owner will launch/control games. **No more computer-use work.**

## 6. Approved bounded training study

- [x] Owner approved with “go on” after the3–4h proposal; hard ceiling4h.
- [ ] Freeze a clean source/config manifest, opponent hashes, initializations and
  new scenario partitions before any new rollout optimization.
- [x] Declare **from scratch**, initializations71/83; do not pool with warm-start studies.
- [x] Keep the75,075-parameter architecture and original PPO settings initially.
  Change one cause at a time where budget permits; do not enlarge the network first.
- [x] Register initial budget: two initializations ×8192 decisions, four candidate
  checkpoints ×20 fresh paired selection cases, lock, then20 fresh cases each for
  locked V4 and unchanged V2 control. This is an initial study, not final proof.
- [x] Approve approximately3–4h CPU budget with a4h hard ceiling. This is
  a noisy estimate from old study timings and the two recent smoke games.
- [x] Register5 sampled diagnostic cases from selection after lock; no tuning
  from confirmation and no silently retried runs/extensions.
- [ ] Add action-only/reward-only ablations if approved compute permits; combined
  changes alone cannot establish which correction caused an improvement.
- [x] Verify finite gradients, PPO KL/clip fraction, entropy, value fitting and
  private RNG/checkpoint resume before scaling the approved rollout.
- [x] Leave frozen crew and belief weights outside the impostor optimizer; gradient/resume tests pass.

Prelaunch study/telemetry and V4 regression suites:36 tests passed55.79s. Includes
tiny development PPO updates and exact continuation; these are not research runs.

Exact protocol: [TRAINING_PROTOCOL.md](options-v4/TRAINING_PROTOCOL.md).80 greedy
selection +5 sampled diagnostic +40 fresh confirmation executions. No architecture
change, automatic extension, lobby expansion, merge/deployment or reserved finals.

## 7. Gameplay skills to learn and verify

These are evaluation questions, not hardcoded strategic rules or proven abilities.

- [ ] Movement: seek useful contact, retarget legitimately visible players and
  avoid long idle-ready/fake loops; A* continues to handle collision avoidance.
- [ ] Kill timing: exploit legal opportunities while learning the cost of witnesses,
  ejection and lost task time. Do not replace policy choice with a kill-alone rule.
- [ ] Cooldown management: choose useful movement/faking/claims between kills;
  do not delay indefinitely once ready or assume every available kill is wise.
- [ ] After a kill: choose movement, continued presence or self-report from evidence;
  no blanket prohibition/forced reporting rule.
- [ ] Meetings: use public claims, own sightings and votes; choose accusation,
  defence of others, own location alibi, vote or skip rather than uniform guessing.
- [ ] Deception: demonstrate beneficial false claims in matched controlled cases;
  possessing claim templates alone is not evidence of learned lying.
- [ ] Voting: improve legal/effective vote choices and reduce self-defeating votes;
  log self-votes, skips, innocent ejections and actual survival/win consequences.
- [ ] Finish games through real parity/wins, rather than reward-only activity.
- [ ] Verify no strategy depends on unseen positions, private crew beliefs, hidden
  witness labels, ghost knowledge or spectator overlays.
- [ ] Report unsupported features honestly: no functional sabotage/vents/free-text
  persuasion or second impostor in the current action contract.

## 8. Validation, graphs and model selection

- [ ] Use paired cases with fixed opponent mixture and honest train/selection/
  confirmation labels. Log wall time and simulated experience separately.
- [ ] Compare frozen untrained NN, historical selected V2 and scripted attacker;
  add compatible frozen best crew and roster conditions as separate comparisons.
- [ ] Plot training return and true wins separately, across seeds/checkpoints.
- [ ] Plot kills, legal kill opportunities, witnessed kills, task-loss/ejection/timeout
  reasons, ready-idle time, fake streaks and structured-claim/vote outcomes.
- [ ] Plot reward-component totals, pursuit credit/correction, entropy/KL/clip
  fraction/value diagnostics, decisions, simulated hours and compute time.
- [ ] Report denominators and uncertainty;20 cases alone give weak win-rate evidence.
- [ ] Choose with a declared validation rule prioritizing true wins; lock identity
  before opening fresh confirmation. Preserve all losing candidates.
- [ ] If confirmation fails, report failure. A later tuned model needs a fresh
  protocol/dataset; never reuse confirmation as an untouched test.
- [ ] Replicate useful learning across independent initializations before making
  robust claims. Expand fresh confirmation only with an approved budget.
- [ ] Open reserved final partitions only under a separate explicit approval.

## 9. Acceptance and release gates

- [ ] No illegal actions, navigation failures, privacy leaks or corrupted checkpoints
  in the declared verification/evaluation set.
- [ ] Fake activity finishes; no persistent unproductive learned loop dominates
  fresh greedy play. Quantify remaining loops rather than rely on a chosen demo.
- [ ] Reliable real kills and wins on fresh ordinary task-enabled scenarios across
  seeds, with uncertainty and opponent mix clearly reported.
- [ ] Evidence of improvement against historical/untrained baselines; compare with
  the scripted attacker without claiming superiority unsupported by the sample.
- [ ] Select the strongest tested validated candidate; do not call it globally best.
- [ ] Export immutable chosen checkpoint/config/hashes and reproduce its viewer.
- [ ] Update full report/checklist/research paper, including failed runs and limitations.
- [ ] Push/review the implementation/results when requested; merge/deploy require
  their own explicit authorization. No silent replacement of old release weights.

## 10. Crew cross-play and configurable future lobbies

- [x] Record future custom settings: up to15 total players and1–3 impostors, with
  legal combinations validated against the targeted real-game version.10crew/2
  impostors is one preset. Both models need adaptation; current5-player work remains.
- [x] Record model count: normally **one shared crew policy +one shared impostor
  policy +the existing crew belief NN**. Independent per-player observation/memory
  and RNG; not15 separately trained networks. Opponent variants are optional.
- [ ] First demonstrate useful single learned agents, then pin compatible frozen
  crew/impostor checkpoints and assess paired cross-play.
- [ ] Agree with the crew developer on a versioned variable-roster contract, up to
 14 other-player rows, encoders/masks/target indices, scaling and action capacity.
- [ ] Add client-selectable lobby presets/settings and reject unsupported/illegal
  player/impostor combinations. Version/store all settings in training/evaluation logs.
- [ ] Implement/test1–3 impostor seats, legitimate teammate identity, friendly-kill
  masks, independent cooldowns, simultaneous kills, voting and correct victory rules.
- [ ] Preserve actor-private memory and ghost isolation; prevent accidental sharing
  of an unseen teammate's observations. Test both neural pipelines.
- [ ] Preregister explicit mixed/all-neural opponent rosters; avoid silently treating
  team wins as one agent's personal contribution.
- [ ] Only then explore learned coordination, adapting opponents and self-play with
  new approval, held-out opponent pools and forgetting/exploitability checks.

## 11. Research paper evidence — maintained from now on

- [x] Create [RESEARCH_LOG.md](RESEARCH_LOG.md) with all failed trials, the first
  learned win, measured exploit/fix, controls and causal limitations.
- [x] Preserve old graphs/results and current implementation/smoke evidence.
- [ ] After every approved experiment, append protocol, source/parent hashes,
  hyperparameters, seeds, compute, failures, results and the next decision.
- [ ] Distinguish observed exploits from hypothetical design risks and unit-test
  success from learned performance; record negative/unsuccessful findings.
- [ ] Write methods, ablations, uncertainty, opponent limitations, sim-to-game
  limitations and reproducibility instructions when the evidence is ready.

Implementation detail/reward table: [V4 report](options-v4/README.md).

Run corrected transfer visualization: `python run_impostor_options.py`.
Run historical evaluated model: `python run_impostor_ai.py`.
Neither command trains. Space/P pause; R replay; N new game; Esc quit.
