# Shared crew / impostor integration — live checklist

**29 October update:** the owner approved the bounded pilot with "start training".
See [training progress](TRAINING_STATUS.md). The integration was pushed as
commit `357f889` and draft PR #13. Engineering history below is dated 28 October.

28 October 2026, Asia/Amman. **Engineering complete; ready for owner review.**
**Owner approved publication. Full PPO, merge and reserved final tests remain unapproved.**

## Starting point

- [x] Read the new owner prompt and developer handoff documentation.
- [x] Fetch all remote updates; preserve the old impostor branch/checkpoints.
- [x] Confirm the exact publication from developer PR12 and the fetched branch:
  `c3320583f7d562bc69ea527580a91ddc5db54f18`.
- [x] Create isolated branch `codex/impostor-shared-integration` from that SHA.
- [x] Identify runtime mismatch: installed Python3.12.3 / NumPy1.26.4 /
  Torch2.9.1+cpu; shared crew requires3.12.10 /2.5.3 /2.6.0+cpu.
- [x] Install isolated exact runtime; unchanged publication verifier passes all282
  published file checks, both strict model loads and119 unique artwork checks.

## Compatibility and integration

- [x] Compare map/physics, five-player rules, tasks/ghosts, vision/evidence,
  cooldown/report/meeting/voting/termination and legal action/observation interfaces.
- [x] Record crew accounting separately from the impostor objective.
- [x] Preserve the75,075-parameter impostor architecture and historical models.
- [x] Audit feature/candidate/target/mask/timing semantics. Report incompatible
  fields explicitly; never weaken original source/runtime locks.
- [x] Explicit transfer passed unchanged old strict loading and new-runtime parity
  on3 synthetic packets: identical candidate probabilities, greedy choices and
  neural tensor hash. The original checkpoint remains unchanged. This is NOT
  proof of equal gameplay under the changed environment/population.
- [x] Implement versioned actor-only projection, typed actions and option handling;
  engineering regressions and complete matches pass.
- [x] Use four independently sampled frozen B actors, private memory/RNG, no old
  social overrides. Retain A and assess mixed/opponent-pool compositions.
- [x] Keep crew/belief weights frozen and private information outside impostor inputs.
- [x] Verify elapsed-time rewards, zero-time terminal credit and complete-episode
  checkpoint/RNG/optimizer restore. Incomplete episodes fail closed; arbitrary
  mid-game resume is not claimed.

## Engineering evidence

- [x] Strict-load A and B, hash files, disable gradients.
- [x] Four-B versus scripted-impostor match complete:seed2760000,PARITY at168s,
 33.69wall seconds,0illegal actions, unchanged weights. Semantic hash matches the
  developer's publication exactly; replay is saved locally. This is a stand-in,
  not a learned-impostor win or a win-rate estimate.
- [x] Complete learned B match and recording-off repeat:one kill/report, ejected
  at84.2sim s; event/action/credit hashes match. Paired A case has same world trace.
- [x] Test legality, privacy, fixed-seed reproducibility and frozen parameter hashes.
- [x] Record failures/behavioral issues for the paper; separate engineered guards
  from learned competence and software correctness from winning skill.

## Spectator V2 demonstration

- [x] Render actual completed playback with unchanged Spectator/map/character assets.
- [x] Show actor/model identity, tasks, cooldown, reports, meetings/ballots, outcome
  and actual recorded probabilities/rewards; never invent missing telemetry.
- [x] Supply exact Antigravity-terminal commands in README.md. Owner requested no
  desktop computer-use automation; prefer commands/headless verification.

## Review and stop

- [x] Document exact source/model hashes, compatibility fixes and prior differences.
- [x] Report tests, matches/replay, timings and visualization command:84 passed,
  5 expected private-fixture skips, plus2 separate synthetic checkpoint tests.
- [x] Independently assess B-first versus A/mixed/pool conditions and likely exploits.
- [x] Propose a small matched study, zero-update reference, fixed selection rule,
  separate ordinary/evidence scenarios, metrics and measured compute budget.
- [x] Record remaining gates/limitations and stop before full training approval.
- [x] Record owner's explicit "finish and push" instruction; publish the integration
  branch while keeping raw replays/preferences/runtimes private. Training approval
  is separate and remains pending.
- [x] Owner approval of the bounded proposed study/budget on 29 October.
- [ ] Complete frozen PPO runner/registration preflight, then execute the pilot.
  No automatic extension beyond the approved pilot.

The handoff contains five players and **no functional sabotage by default**.
The latest prompt requests these published historical A/B opponents. Do not imply
sabotage or a new crewmate architecture is included. Preserve the no-sabotage
baseline; larger lobbies and self-play remain separate future work.
