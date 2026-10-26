# Impostor development — live session checklist

**Current step: shared chat read; remaining trainer work and pilot budget documented.**

Draft PR: [#10 — impostor V2 environment](https://github.com/Hadisovic/social-deduction-ai/pull/10).
All 10 benchmark matches and all 239 regression tests passed. The shared chat is now
read and reconciled; the existing Edge tab worked after the earlier access failures.

Updated: 27 October 2026 (Amman). Based on your `message (5).txt` and subsequent
instructions. This file will be updated in place throughout this session.

**No RL training run, merge, deployment or reserved final evaluation is authorized.**

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

## Approval gates — deliberately NOT started

Remaining work is listed in [TRAINING_NEXT_STEPS.md](TRAINING_NEXT_STEPS.md): trainer,
checkpoint/resume, optimizer isolation and seconds-aware GAE verification, followed
by two pilot runs and paired validation. Allow **about 2–4 hours to first results and
graphs including preparation**. Longer training will depend on those results.

- [ ] Owner approval for any RL training run.
- [ ] Owner approval for reserved final-test evaluation.
- [ ] Owner approval to merge PRs.
- [ ] Owner approval to deploy.

Details: [collaborator handoff](../phase6_parallel_research_handoff.md) ·
[engineering report](README.md) · [test record](test-results.json).
