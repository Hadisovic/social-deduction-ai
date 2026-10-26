# Learned impostor AI — collaborator handoff (26 October 2026)

> **Audience: the collaborator's Codex agent implementing the learned IMPOSTOR.** This is **not** a request to train or modify the owner's crewmate AI, and not the main crewmate development plan.
>
> **Handoff status:** repository checkout instructions, frozen opponent contract, approved conceptual boundaries and preflight requirements. **No new impostor training, experiments, source edits or merging are authorized merely by this document.** Wait for the owner's detailed impostor implementation/training prompt and budget approval.

## CRITICAL — fetch the NEW branches, not your old clone or main

Repo: https://github.com/Hadisovic/social-deduction-ai

**There are TWO essential remote branches:**

1. **Latest complete Phase 6 experimental SOURCE / all nine crewmate models:** \`codex/phase6-1-seeds29-43\` (research PR #7), **verified immutable scientific base SHA** \`2baa5dbc1b36b047ed97934d69526573b510c280\`.
2. **NEW collaborator HANDOFF branch (this document + identical scientific source and checkpoints):** \`docs/phase6-two-track-handoff-20261026\`. This branch descends from the exact research SHA above and adds only documentation; it is the **recommended base for your new independent impostor feature branch**.

**Do not use \`main\` or old checkouts.** Main contains published roadmap and Phase 5 but lacks the complete unmerged Phase 6 research code and checkpoints. Do not treat branches #3/#4 as the latest; #7 includes the completed seed-29/43 study as well as earlier research history.

Open a terminal in the repository's existing clone, or clone fresh if absent:

\`\`\`bash
git clone https://github.com/Hadisovic/social-deduction-ai.git
cd social-deduction-ai
git fetch origin --prune
git fetch origin codex/phase6-1-seeds29-43 docs/phase6-two-track-handoff-20261026

# Confirm that the actual remote research HEAD matches the pinned study state:
git rev-parse origin/codex/phase6-1-seeds29-43
# Expected: 2baa5dbc1b36b047ed97934d69526573b510c280

# Inspect this updated handoff directly, even in an older local checkout:
git show origin/docs/phase6-two-track-handoff-20261026:docs/phase6_parallel_research_handoff.md

# Start YOUR NEW impostor work branch from the NEW handoff branch, NOT main:
git switch -c feat/learned-impostor-research origin/docs/phase6-two-track-handoff-20261026

# Verify it still descends from the pinned original scientific experiment:
git merge-base --is-ancestor 2baa5dbc1b36b047ed97934d69526573b510c280 HEAD
git status --short
\`\`\`

If you ALREADY have an existing clone, **skip \`git clone\` and \`cd\`**, but still run both fetch commands; do not assume your local tracked branches or prior files are current. If the example feature branch name already exists locally, create a new uniquely named feature branch from \`origin/docs/phase6-two-track-handoff-20261026\` rather than force-resetting or overwriting work. If the remote scientific SHA unexpectedly changes, **STOP**, inspect the changes and report them—do not silently change the fixed starting baseline. If you lack permission to fetch either branch, stop and request access rather than substituting main.

**Optional only for reading:** \`codex/phase6-1-diagnostic-publication\` (PR #8) contains a sanitized belief diagnostic; it is a *separate* branch, not a replacement for the full research source. Fetch it separately if useful, but do not merge it into the experiment automatically.

**No merges:** Do NOT merge PR #3, #4, #5, #7 or #8 into main, and do not rebase/rewrite their preserved scientific histories. Share new impostor code through its own PR later.

## Exactly what YOU are building

Train a **new reinforcement-learning IMPOSTOR** in the existing five-player simulation (four crewmates, one impostor).

- **Only the impostor network weights learn during these experiments.** Opponent crewmate neural model parameters and the Phase 4 belief predictor are frozen, loaded from GitHub checkpoints, never optimized, fine-tuned, backpropagated through, or overwritten.
- Start with the current Phase 6 crew policy as a fixed opponent; later use a **registered mix** of frozen learned and scripted crewmates to reduce overfitting to one brittle policy.
- Preserve per-actor local observations, actor-private memory and role-specific visibility (crew 4.5, impostor 6.75 simulator units), with native walls/obstacles. Never give the impostor policy world truth, unseen positions or post-game labels.
- Existing \`phase5_runner.StrategicMatch\` provides one trained focal crewmate with other scripted actors, not a ready-made learning impostor. Implement new impostor-controlled orchestration in your own namespaced code, keeping legal-action masks, deterministic replays and RNG isolation. Do not assume a trained impostor environment is already present.
- Impostor's existing legitimate actions include navigation, eliminations, fake tasks, body reports, structured meeting claims, votes and skips. Sabotage, vents, free-form chat, second impostors and human multiplayer are **not** built yet: do not invent them.
- Train against specific frozen checkpoints; report impostor win/elimination/survival/detection/ejection metrics and behavior against varied opponents. Winning only against one fixed crew does not establish general competence.
- No training or reserved final tests until separate design, seed-partition, evaluation and compute review.

## Which frozen crewmate checkpoint to pull/use

**Default recommended source-compatible baseline, from the NEW Phase 6 branch:**

\`artifacts/phase6/remaining-six/full-29/checkpoint-8192.pt\`

- Selected by validation at 8,192 decisions; full-history + frozen belief inputs; initialization seed 29.
- Recorded SHA-256: \`238904b7fe303b77e885c4cd6fdf4d9d447a62eb5b463436a391c0b4d6e1cb43\`
- Original 200-scenario Phase 6 validation: 90.0% crewmate wins, 149 correct votes / 167 cast, **zero own body reports**. This is a research baseline, **not an accepted strong crew model**.
- Second, behaviorally different comparison: \`artifacts/phase6/remaining-six/full-43/checkpoint-4096.pt\`, SHA256 \`d3abbdff88743f8b105f630e11c0e50963f38f7b1e992634dd55999087e9c0e3\`. This crew often skips; do not train solely to exploit it.
- Other selected checkpoints are in \`docs/benchmarks/phase6/remaining-six/selection.json\` and \`docs/benchmarks/phase6/seed17-stage/selection.json\`. **Use the selected checkpoint**, not an arbitrary last \`policy.pt\` or smoke model.

**Compatibility:** The seed-17 models were generated against an earlier pilot-compatible scientific source than seeds 29/43. Loading them with current continuation source may fail the strict hash guard. Keep this guard. For a seed-17 opponent, use matching original source/runtime or separately review a compatibility migration. Check dependency versions before loading. Existing source fingerprints include every \`phase6/*.py\`, \`belief/*.py\`, \`social_deduction/*.py\` and selected core source files. Even a new file inside \`phase6/\` can change source hashes. Initially put your new modules under a **new** \`impostor_research/\` package, without editing the scientific baseline merely to make checkpoint loading succeed. Load the frozen model in its compatible context.

**Before building or running training:** inspect \`phase6/train.py\` (\`load_policy\`), \`phase6/config.py\`, \`phase6/env.py\`, \`phase5_runner.py\`, \`phase5_policy.py\`, \`phase5_features.py\`, \`phase3_engine.py\`, \`social_deduction/phase3_observation.py\`, \`social_deduction/phase3_api.py\`, \`phase3_bots.py\`, \`phase4_scripts.py\`. Verify the chosen checkpoint digest against its selection lock, load successfully under strict source checks, and demonstrate legal actor-only inputs and reproducible development match traces. This handoff has **not** run a fresh new preflight.

## IMPORTANT OWNER CLARIFICATION: eyewitness rule belongs ONLY to YOUR impostor-training setup

The owner wants an explicit **conditional safeguard on frozen crewmate opponents WHILE TRAINING/EVALUATING THE IMPOSTOR**:

**If a living frozen crewmate actually witnesses an impostor kill inside its legitimate vision**, it should remember the directly witnessed killer's ID, pursue/report the associated body if it can legally do so, and vote for the witnessed killer in a subsequent meeting when the killer is still a legal vote target.

This guard is part of the **impostor training environment's opponent behavior**, **NOT** a feature to be added to the owner's separately trained/improved crewmate policy. Do **not** edit the crewmate neural parameters. Do **not** place this guard into the main crewmate training track, shared default rules, or released models.

Proposed implementation contract, subject to reviewed design before training:

1. Add a **versioned, explicitly enabled opponent-only wrapper** in \`impostor_research/\`. Its inputs are *only* that crewmate's legitimate observation and private memory, including a direct \`WitnessedElimination\` evidence event at the time of the kill. Do not peek at the game engine's hidden roles/killer truth, even to implement the guard.
2. Preserve the frozen crew model's standard behavior except when the genuine eyewitness condition is active. At an eligible legal body-report opportunity, prioritize \`REPORT\`; if not in range, use legal movement toward a legitimately seen body, not an impossible instant report. If a meeting starts first, preserve the witness ID and prioritize the legal \`VOTE\` against that killer; otherwise abstain/fall back to normal behavior.
3. **Never** override from a mere suspicion score, presence near a corpse, hearsay, a past unknown-position event, or information the actor could not see. Respect death/inactivity, reporting and meeting phases, legal-action masks, body resolution and target eligibility. No ghost voting/reporting.
4. Keep this guard optional/toggleable for matched **assisted-opponent vs ordinary frozen-opponent** comparisons. Log every triggered report/vote separately, including whether it was rule-driven; those actions are **scripted assists**, not evidence of learned crewmate reporting/reasoning.
5. This changes the **behavior** of a frozen-weight opponent in the impostor experiment; frozen weights alone do not mean identical opponent behavior. Identify the assisted opponent in the experiment manifest and evaluate both guarded and unguarded opponents so we know what was learned.

This direct eyewitness condition can prevent the impostor from safely killing while visibly watched. It is NOT a blanket mandatory report for mere discovery of an unobserved corpse; ordinary body-report decisions remain the actor's regular policy unless the owner separately approves other scripted reporting variations.

## Separate owner's crewmate track — OUT OF SCOPE for this collaborator

The owner independently plans crewmate improvements (and has not implemented them in this handoff):

- Player-owned tasks never transfer to anyone else. Note: current \`phase3_engine._reassign_publicly_absent_tasks\` **does** reassign unfinished dead/ejected-player tasks after public absence, so this requires a new opt-in rule rather than claiming it already works.
- Dead/ejected crewmates should retain tasks and operate as flying ghosts through obstacles, completing only their own tasks while not reporting/voting or transmitting observations to living players.
- Some future experiments may disable task-based victory in a controlled fraction of matches to encourage deduction while retaining actual tasks.
- The owner is separately diagnosing reporting incentives, vote decisions and reporter-biased script behavior.

**Do NOT implement those ghost/task/deduction-only changes in your initial learned-impostor branch**, and do not turn them into new default engine mechanics. Your initial crew opponents use the **unchanged historical rules except for the explicitly tagged, optional eyewitness override within your own impostor experiment**. Future cross-play with the owner's new crew versions must be a separately versioned experiment with compatible shared rule contracts.

## Scientific boundaries, datasets and future integration

- Original Phase 6.1: nine crewmate models trained and validated; Phase 6.1 **not accepted**. All 173 observed reports in the separate 180-game development diagnostic were from impostors; a reporter-based shortcut nearly reproduced high meeting identification. Do not claim demonstrated robust human-like deduction; use diverse reporters and opponents.
- Preserve original Phase 4/5 frozen artifacts, every Phase 6.1 checkpoint and recorded selection/source hashes. Historical evaluations must remain reproducible. **Never mutate research PR #7 source or checkpoints in place.**
- Fresh impostor training/validation/development partitions must be explicitly registered and checked against existing Phase 6.1 train/validation/development/final IDs. **Do not touch reserved Phase 6.1 final familiar seeds 1230000–1230499 or final patient seeds 1240000–1240499.**
- Use immutable checkpoint copies/IDs, independent policy and opponent RNG, separate directories, and bounded run budgets. Trainer/evaluator may use hidden roles only **offline for scoring/reward**, never in the impostor's input packet or opponent guard.
- When both independent tracks mature, jointly evaluate current and historical frozen crewmate policies versus the learned impostor under a deliberately shared, reviewed set of rules. Later alternate updating one side while holding the other fixed; only then consider larger agent populations or broader self-play.
- Publish raw actor trajectories and post-game role labels only with explicit owner permission. Keep sensitive local paths out of public handoff reports.
- **No PR merges, main changes, deployment, production updates, new training, final-set access or checkpoint changes are authorized by this handoff.**

## Deliverable when the owner sends the detailed impostor prompt

First return a short audited design proposal naming the exact source commit, pulled branches and verified frozen crew checkpoint; proposed impostor observations/action interface; opponent guard isolation and on/off design; reproducibility/seed isolation; training/evaluation protocol; anticipated runtime/compute. Seek approval before expensive training. Work on your independently branched checkout, not on the existing crewmate research PRs.
