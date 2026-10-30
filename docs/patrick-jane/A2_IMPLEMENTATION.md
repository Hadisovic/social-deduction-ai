# Patrick Jane A2 implementation and independent assessment

## Existing evidence and preservation

The completed12-hour export and the independent extraction report were found at the requested locations. The original archive checksum `4518f5c82fa84b26ab287f2fb3e75af78f9148f510fe08b0de16bc7bfe62b263` and all582 archived members were verified before extracting to `local-runs/imported/task-pursuit-v1-update102`. The imported raw files/checkpoints remain unchanged. Current pre-A2 source state was independently archived at `local-runs/a2-preserved-before-implementation/sources-before-a2.zip`; the delivered Task Pursuit V1 folder/ZIP remain unchanged. No commit, PR merge, partner-project change or historical deletion occurred.

The original results confirm482 task wins /504 games (95.63%), zero ejection wins,15,448 living/4,621 ghost tasks,654 correct/1,948 incorrect ballots,9,215 skips,15 impostor/48 innocent ejections. These are **on-policy training results for the combined policy and task scaffolding**, not held-out generalization. The owner's split-vote/tie observation is retained separately from the aggregate fact that1,564 meetings had no ejection. Approximately45.3% of decisions were forced; this is not proof that the NN independently discovered persistence.

The recorded active time44,843.33 seconds and the update101→102 collection interval19,301.19 seconds are preserved. Suspension is plausible based on the owner report, not established from these logs. A new A2-only Windows unbiased clock excludes sleep/hibernation going forward; no historical duration is corrected or described as a confirmed bug. See [Microsoft's API contract](https://learn.microsoft.com/en-us/windows/win32/api/realtimeapiset/nf-realtimeapiset-queryunbiasedinterrupttimeprecise).

## What changed

`a2_config.py` registers exact selected rewards; `a2_rewards.py` gives each accepted crew ballot one private credit, including a voter ejected in the same resolution. Eligibility is snapshotted at acceptance and agrees with the legal neural target set excluding self. No accepted non-skip vote is inferred from an action request. Explicit/automatic skips get zero. Full task reward is+0.5 only at final assigned-task completion; a task victory gives one+5 family, an ejection victory one+10 family, loss-10. Existing+4/-2 per-ejection event credits remain intentional separate incentives.

`a2_environment.py` adds classic-room doors and graph-constrained vents. Physics/sight share map revisions; private navigation learns closures through local sight or contact and clears stale obstruction knowledge using the publicly known maximum closure duration, not a hidden timer. Commitments survive waiting/reopening. Anti-embedding closures, animation timing, opaque-door visibility and omitted vent-lid-only animations are explicit abstractions in A2_RULES.md.

`a2_opponents.py` implements varied private tactical preferences, visible/remembered pursuit, witness-sensitive killing, fake tasks, lights, nearby door tactics, connected vent repositioning and occasional self-reporting. It consumes only immutable actor observations and known public map geometry; independent scripts have independent RNG/history. The original `opponents.py` is byte-for-byte unchanged. A50/50 per-match mixture is registered and never adapted during training. No promise that the upgraded scripts are empirically stronger is made before frozen evaluations.

Crew receive only identity-bearing vent entry/exit they actually witness, with location/time/provenance. Hidden travel/exit destination, global doors/timers and ballot correctness never enter memory or tensors. Old feature meanings and tensor widths remain; three appended event codes are versioned and initially unfamiliar. No automatic suspicion, vote, accusation, speaking-order change or social influence is implemented.

All52 update102 tensors and Adam states pass equality checks through `a2_transition.py`, with the62-source parent snapshot verified. Strict historical loading is not relaxed. Current source changes require an explicit bridge; historical reproduction uses its preserved sources. Shared GRU architecture, existing PPO and collector, task ownership/stages/ghosts, role-specific vision and legitimate terminal precedence remain.

Spectator replay/live records gain observer-only door/vent snapshots, local witnessed-evidence display, opaque-door overlays/timers in debug mode, underground occupancy only in omniscient mode, policy/regime metadata and private retrospective ballot records. Rendering does not drive physics or learning. The engineering fixture replay has actual update102 neural decisions, one witnessed vent entrance, one connected transition/exit and one door closure; its ending is explicitly TRUNCATED, not a natural-game victory or evaluation result.

## Verification and repairs

- A2 reward, privacy, mechanics, opponent, exact checkpoint/Adam and finite-PPO checks:15/15 passed in43.72 seconds before final edge-case additions.
- Expanded A2 plus historical commitment/portable regressions:38 passed,2 excluded in95.28 seconds. Exclusions were the superseded update18 bridge test (historical source contract intentionally differs) and a redundant older real recurrent PPO test. A new real A2 recurrent-PPO check did run.
- Windows spawn/CUDA check: two independent3-second engineering segments,100 actor decisions,20 finite optimizer steps on a **fresh disposable policy**, zero legitimate complete games and two explicit technical cuts. Strict saved-checkpoint reload passed. Update102 weights were never trained or overwritten. Recording on/off yielded identical action traces and rewards in the bounded test.
- Repairs caught during implementation: geometry/interaction-map identity mismatch; a test crossing segment outside real floor; an indentation defect during editing; NumPy coordinate scalars rejected by the strict typed intent API; temporary door-covered goal validation; possible stale remembered-door deadlock; private debug event filtering. The final selected source is checked after these repairs. No claimed behavioral gain is based on engineering corrections.
- The portable package installs entirely offline and its setup imported update102 with the correct2,813,879 lifetime decisions and all52 tensors verified. A zero-experience origin registration was prepared. The implementation machine is an RTX4070 laptop, so physical GTX1650 execution/performance is not independently verified. The matching CUDA12.8 wheel contains Turing support; driver and disk space remain prerequisites. Existing six-worker experience informed a hardware-bounded auto default; no worker benchmark was run.

No long A2 run, final-test access or full development comparison was executed. Read the separate verification receipt for final targeted checks and package hashes.

## Launch and comparisons

Portable folder: `dist/PatrickJane-Investigation-A2-V1`; matching ZIP beside it. From the extracted folder:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Start-PatrickJaneA2.ps1 -Hours 12
```

Visualization only:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Show-PatrickJaneA2.ps1
```

Outputs/resume: `results/investigation-a2-v1`; exports: `exports/`. Rerun the same command to resume. `-Hours 24` appends an extension. `-AdditionalDecisions` can select a bounded additional-decision budget. First launch selects a practical worker count up to6 and freezes it. The spectator watches real concurrent status and newly finished replays independently. Best checkpoint is not evaluated automatically.

A2_REGISTRATION.md and comparison-registration.json define paired development conditions and the separate2×2 reward/opponent training controls. Default mixed training is not a causal reward-only test. Use original update102 as frozen reference; new event/mechanics semantics are explicit, and old source/engine controls remain available. No comparison or normal-match performance has been fabricated.

## Independent technical opinion

1. **Vote shaping is useful but insufficient.** It supplies a local signal even for ties, with zero expectation under uniform eligible-target guessing. It can still reward a script-signature shortcut; excess accuracy over the actual base rate and direct-evidence-conditioned behavior matter more than reward accumulation.
2. **+0.5 tasks is a reasonable registered trial, not an optimum.** Four tasks now yield only+2 personal reward, less than one+3 repair. Repeated repair rewards could still dominate socially useful actions. I would not silently retune them, but would monitor reward share and task regression.
3. **+5 task win/+10 ejection win favors the intended route.** With both+4 impostor-ejection events, an ejection victory can yield+18 per owner before personal rewards. That can encourage chasing ejections; incorrect ejections are penalized, but successful guessing or predictable scripts remain possible. It must not be called evidence-grounded deduction without opportunity-conditioned tests.
4. **Vent evidence is strong and legitimate in a no-Engineer lobby.** Door closures mostly change opportunity and uncertainty, rather than identify the actor who closed them. Their value depends on actual witnesses and memory retention; invisible vent travel contributes no evidence.
5. **Opponent predictability remains a risk.** Independent stochastic preferences reduce fixed tactical cycles, but shared historical speech formats, self-report rates, door usage and vent patterns can still become signatures. The updated scripts have not yet been shown to outperform historical opponents.
6. **This is more environmental complexity, not a larger NN.** The old network must learn unfamiliar event codes and new timing. Keeping the architecture avoids a second simultaneous model change, but old successful task behavior may initially dominate attention and action selection.
7. **Commitment can conflict with investigation.** It permits body reports/lights pauses but still blocks strategic detours and emergency calls during work. The task expertise is preserved at the cost of some evidence-seeking freedom. This is an explicit remaining confound.
8. **I would not add another suspicion network yet.** If the first paired A2 comparison shows no useful gain, first separate absent evidence from ignored evidence and poor ballots. An oracle-free, separately registered evidence-conditioned learning objective could then be justified; if evidence is present but ignored during commitment, test a narrowly relaxed interruption policy. Do not respond with several unregistered reward/controller changes at once.

The positive12-hour task result is real within its training regime. A2 social improvement, opponent strength and secondary-machine cost remain unverified. Research acceptance stays pending.
