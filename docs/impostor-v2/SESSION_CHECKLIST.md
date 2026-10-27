# Impostor development — live session checklist

**Complete: training/evaluation, graphs, summary and default viewer verification; code/results pushed in118715a and draft PR10 updated. No more training today.**

Primary report: [session summary](SESSION_SUMMARY.md). No further training today.

[Live training progress](../../artifacts/impostor_research/strategic-v2-20261027/LIVE_PROGRESS.md).

Draft PR: [#10 — impostor V2 environment](https://github.com/Hadisovic/social-deduction-ai/pull/10).
All 10 benchmark matches and all 239 regression tests passed. The shared chat is now
read and reconciled; the existing Edge tab worked after the earlier access failures.

Updated: 27 October 2026 (Amman). Based on your `message (5).txt` and subsequent
instructions. This file will be updated in place throughout this session.

**Latest approval: improve rewards/policy, train and evaluate the impostor, add a viewer and push the branch.**
Current registered stage: two 8,192-decision runs, 80 selection and 80 confirmation
matches, with a 3-hour ceiling. Merges, deployment and reserved final tests remain closed.

## 1. Fetch and establish an independent baseline

- [x] Fetch current GitHub branches and preserve existing work.
- [x] Verify research baseline `2baa5dbc1b36b047ed97934d69526573b510c280`.
- [x] Verify V2 baseline `e62ba0b6c33ffd0317f78e18b4289d778cd5af2b` and ancestry.
- [x] Create `codex/impostor-environment-v2` directly from V2, not main.
- [x] Read the complete attached task, V2 design and original collaborator handoff.
- [x] Record the shared planning-chat link in the resources.
- [x] Read shared-chat contents through Edge; the Among Us section confirms the existing work split and approval gates, with no extra training settings.

## 2. Integrate the frozen crewmate safely

- [x] Verify the pinned full-29 checkpoint SHA-256.
- [x] Load with original strict checkpoint/source checks intact.
- [x] Freeze crew policy and Phase 4 belief weights; verify tensor/file hashes.
- [x] Add a separate V2-to-legacy feature adapter with unchanged network dimensions.
- [x] Preserve independent actor memories and RNG streams.
- [x] Document omitted V2 inputs and distribution-shift limitations.
- [x] Probe real model behavior before adding safeguards: it selected task travel despite a legal report opportunity.

## 3. Add opponent-only safeguards

- [x] Visible body in range → prioritize legal REPORT.
- [x] Visible body outside range → approach through legal navigation, then report.
- [x] Direct witnessed kill → retain killer identity privately.
- [x] Eligible voting phase → vote for the directly witnessed killer.
- [x] Another player reports first → retain witness evidence.
- [x] Lost/resolved body or ineligible killer → stop stale/illegal actions.
- [x] No witnessed kill → do not invent killer identity from body discovery.
- [x] Dead/ghost opponents → no social actions or living-controller updates.
- [x] Support neither / body-only / eyewitness-only / both safeguards.
- [x] Log proposed and executed actions separately, including unchanged proposals.
- [x] Test learned and scripted opponents end-to-end.

## 4. Build the independent impostor environment

- [x] Keep five players: one impostor, one frozen learned crew, three scripted crew.
- [x] Reuse canonical V2 lobby, physics, permanent tasks and isolated ghosts.
- [x] Add actor-only impostor memory and observation encoding.
- [x] Add masked movement, kill, fake-task, report, claim, meeting and voting choices.
- [x] Exclude unsupported sabotage, vents and extra impostors.
- [x] Add a new, untrained neural candidate scorer and deterministic inference checks.
- [x] Keep private truth/reward/metrics outside network packets.
- [x] Add exact-tick forced-meeting-window coalescing without timer compression.
- [x] Commit implementation: `cb9b5f4fc3a606383f3b04815151602cb54f45ec`.

## 5. Verification and development measurements

- [x] New impostor tests: **27 passed**.
- [x] V2 mechanics/privacy/integrity tests: **45 passed**.
- [x] Historical regression tests: **167 passed**.
- [x] Total final regression: **239 passed**, zero failures; one legacy synthetic optimizer test deselected.
- [x] Preserve all nine selected checkpoints and both historical selection locks.
- [x] Original strict loaders still accept all six continuation checkpoints.
- [x] Controlled body/witness sequence, wall occlusion, death, false claims and privacy checks.
- [x] Ordinary-mode scripted development matches: **2/2 completed**.
- [x] Deduction-only comparison matches: **2/2 completed**.
- [x] Idle and untrained neural controls: **2/2 completed**, legal matches and no kills.
- [x] Non-reporting hunter controls for natural crew discovery: **2/2 completed**, one crew report in each match; one changed proposal.
- [x] Reference execution and optimized repeat: **2/2 completed**.
- [x] Confirm impostor input, action and event trace equality; frozen weights unchanged after the final benchmark.
- [x] Record aggregate opportunities, outcomes and measured throughput in the engineering report.

Early observations: ordinary scripted matches reached legal kill opportunities around
59 simulated seconds, killed one/two crew, and lost by tasks. The paired deduction-only
matches reached parity. These are engineering scenarios, **not learned-policy results**.

## 6. Training proposal — prepare only

- [x] Propose network, sparse terminal reward and finite-horizon objective.
- [x] Propose two 4,096-decision pilot runs; no automatic extension.
- [x] Register separate train/development/validation/reserved-final ranges.
- [x] Define validation, controls, metrics, stopping conditions and future cross-play.
- [x] Finalize budget: 1–2.5 hours estimated for two 4,096-decision pilots plus 40 validation matches; 3-hour ceiling, approval required.
- [x] State readiness and unresolved limitations clearly; no learned impostor performance claimed.

## 7. Deliver and stop at approval

- [x] Update collaborator handoff and record source references.
- [x] Finish engineering report with test results and benchmark details.
- [x] Final diff/integrity review; historical runtime and checkpoints unchanged; benchmark source hashes match.
- [x] Commit final documentation and evidence.
- [x] Push only the independent development branch.
- [x] Create and attach **draft PR #10**, based on the V2 branch.
- [x] Report completion and proposed first training budget in the linked report and session update.

## Approved pilot — current session

- [x] Owner approved the proposed pilot with “go on”.
- [x] Implement separate PPO trainer/evaluator and fixed pilot configuration.
- [x] Implement seconds-aware GAE, optimizer isolation, immutable checkpoints and replay-based resume.
- [x] Initial seven trainer preflight tests passed, including exact resumed action/update equality.
- [x] Complete combined regression: 79 passed; final trainer/report suite 9 passed (7 repeated; 81 distinct tests).
- [x] Commit and lock source/configuration before learning: `2c31f060212bd42a28d92b98e132f5f97d0d2832`; clean study manifest.
- [x] Train initialization 71: 4,096 decisions, 18 completed games, no wins/kills; 463.98 seconds wall time; frozen models unchanged.
- [x] Train initialization 83: 4,096 decisions, 18 completed games, no wins and one kill; frozen models unchanged.
- [x] Evaluate exactly 40 paired validation matches: both models 0/20 wins, zero kills, all games lost by tasks; no illegal/navigation failures.
- [x] Generate six pilot graphs and assess learning sufficiency: pilot did not learn a useful winning strategy.
- [x] Update report, checklist and existing draft PR with pilot outcomes.

## Improved policy stage — newly approved

- [x] Owner approved policy/reward/NN improvements and a runnable impostor viewer.
- [x] Implement learned family/target selection, balanced entropy, separate critic, kill shaping and interrupted travel options.
- [x] Verify 78 tests, including exact resume, privacy, rewards and headless `run_impostor.py`.
- [x] Two further controlled end-screen tests passed (win and loss); terminal absorbing packet remains valid.
- [x] Offline report fixture passed paired-aggregate, zero/all-win uncertainty and eight-graph checks; no gameplay evaluation involved.
- [x] Commit and lock upgraded runtime/configuration: `f96abbe`; clean study manifest.
- [x] Train two fresh 8,192-decision initializations (session1855 continues into evaluation).
  - Initialization 71: 82 completed games, 68 kills, 0 wins; 35.94 minutes wall.
  - Initialization 83: 78 completed games, 57 kills, 1 win; measured wall time in training.json.
- [x] Run 80 candidate-selection matches, lock selection, then 80 fresh confirmation/control matches.
  - All 80 selection executions complete. Locked seed71/8192: 2/20 wins,12 kills; checkpoint SHA begins `8b861c9c`.
  - Complete: selected0/20, older AI0/20, scripted attacker1/20, untrained NN0/20. Lock unchanged.
- [x] Generate eight upgraded graphs, assess failure and preserve all eight new policy snapshots.
- [x] Create selected checkpoint bundles; strict-load all copies; default `run_impostor_ai.py` passed eight headless frames.
- [x] Update/push existing draft PR10; run command: `python run_impostor_ai.py`.

Protocol: [strategic-v2-protocol.md](strategic-v2-protocol.md). This follow-up has a
3-hour learning/evaluation ceiling. Existing reserved final-test partitions stay closed.

The original [TRAINING_NEXT_STEPS.md](TRAINING_NEXT_STEPS.md) describes the now-completed
first pilot. The current protocol above supersedes its old training-approval gate.

## Conditional pursuit correction — same original deadline

- [x] Register the conditional trigger and fresh scenario IDs before V2 confirmation.
- [x] Implement a separate 13th learned pursuit family; keep all V2 source files unchanged.
- [x] Three tests passed: family balance/warm start, exact gradient/resume/frozen checks and both viewer schemas.
- [x] Record skipped: latest owner requested summary/push and end of today's work instead of a new study.
- [x] Preserve prepared code and three development checks; no registered pursuit training or performance study.
- **Latest owner instruction supersedes this gate: summarize completed training and push; no pursuit study will run.** The three checks above were tiny development tests, not performance evidence.
- [x] Assess and curate the explicit V2 selection; verify `python run_impostor_ai.py`; write complete session summary.
- [x] Push final code/checkpoints/graphs/summary; draft PR10 updated; no merge/deploy/new training.

See [conditional protocol](pursuit-conditional-protocol.md). This is a versioned
correction under the owner's evidence-driven instruction, not an extension of the
original three-hour wall-clock deadline. No new stage has started yet.

## Remaining approval gates

- [x] Owner approval for first pilot and subsequent improved-policy stage.
- [ ] Owner approval for reserved final-test evaluation.
- [ ] Owner approval to merge PRs.
- [ ] Owner approval to deploy.

Details: [collaborator handoff](../phase6_parallel_research_handoff.md) ·
[engineering report](README.md) · [test record](test-results.json).
