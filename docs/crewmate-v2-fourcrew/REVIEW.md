# Post-pilot repairs and four-crewmate engineering review

2026-10-27. **Implementation and engineering validation only. No full new training experiment. Phase 6.1 acceptance remains pending.**

The completed focal-agent study at `5b663f6` remains unchanged in `_v2_pretraining_diagnostic`. New mechanics and parameter-sharing PPO live in `_v2_fourcrew` on `codex/crewmate-v2-fourcrew`. The isolated viewer is `_spectator_fourcrew`, branch `codex/spectator-fourcrew`, commit `fa76497`. All new work is local; PR #11 and both original worktrees are preserved.

## Findings and repairs

| Owner observation | Confirmed cause / measured evidence | Result |
|---|---|---|
| Ghosts appear to vote | Old viewer mixes historical submissions across meetings. Zero dead submissions among 70 submissions in each eleven-replay sample. | Scope current meeting, remove ineligible rows, hide targets until public publication; engine also rejects direct dead/ejected votes and sanitizes tally inputs. |
| Voting waits after all responses | V2.1 resolves only on timer expiry. | V2.2 resolves within the last submitting step; timeout/default skips remain. |
| Ghost work during voting | Unconditional `_advance_ghosts`; 35 ghost task completions during meetings in each sampled arm. | Freeze ghost movement, work and task incubation through discussion/voting; living work is gated too. Resume own unfinished work during roaming. |
| Suspicion resets | No actual memory reset reproduced; old numeric suspicion traces unavailable. | Two-meeting persistence/reset tests, independent memory ownership, selected-actor suspicion serialization and explicit private debug. Do not force constant probabilities. |
| Speed buttons cannot be clicked | Drawn pills lack event hitboxes. | Shared drawn/click rectangles, immediate speed selection, resize tests and preserved keyboard/time scaling. |
| Focal actor dies early | 4/11 first kills, 9/11 total kills in available A replays; owner's viewing set of 15 is unavailable. | Survival concern documented; no new survival reward or hardcoded threat response. |

The old focal pilot's 5/20 versus 4/20 wins, tasks 80/80, correct votes 24/24, incorrect votes 3/5 and zero reports remain historical measurements. The repaired mechanics cannot be used to retroactively recompute those outcomes and keep the same experiment label.

Mechanics fixes are limited to V2.2: historical `phase3_engine.py`, Phase 4, Phase 5 and Phase 6 research kernels remain unchanged. Ballot targets are hidden while voting; identity/target pairs become public together at legitimate resolution. Only living meeting participants vote; duplicate/direct invalid votes are rejected, and invalid stored entries cannot affect a tally. A missing responder still receives a timeout skip. The simulator has no disconnect state; no disconnect support is claimed. Ejection/parity takes precedence over task victory. Meetings interrupt living interactions, preserving partial task work; the actor must legitimately reselect unfinished work afterward. Ghost work resumes automatically only in roaming.

## Actual four-crewmate implementation

`crewmate_v2/fourcrew/` provides four policy-controlled crewmates against one scripted impostor. No crew script issues strategic actions. One centrally updated network uses the existing 42 global, 4x29 other-player, 4x8 task and 96x23 action shapes. Actor views reuse existing observation/feature/mechanical-intent code. Candidate overflow raises an error rather than truncating; the measured 60-second engineering match used at most 29 candidates.

Each actor owns its own canonical/public memory, options, action sampler, rewards and trajectory. The shared frozen Phase 4 network is stateless at prediction time; its inputs are each actor's separate evidence. Shared weights do not transmit private information. Actor packets contain no spectator truth or reward labels. Tests inject a private proof for one actor and verify that a teammate's memory and packet remain unchanged.

Intents enter the same simultaneous engine step. Actors hold options independently, with legitimate phase, new body, witness and newly legal report interruptions. This interruption presents a choice; it never automatically reports, votes or flees. Dead/ejected actors stop policy inference, observation-memory updates, claims and votes. Automatic ghost tasks receive no individual task credit and cannot leak evidence. The global task bar remains legitimate public information.

The collector freezes a policy version for each complete-match batch. Worker policy copies have identical weights; samplers are independent by initialization/match/actor. Parent PPO updates occur only after collection. Rollout segments are keyed by match and actor. Original segmented time-aware GAE/PPO kernels are reused without changing loss coefficients, clipping or architecture. Nonterminal transitions bootstrap from that actor's next decision. A dead actor has no later actions; its final transition remains buffered through team settlement, with discounted rewards over actual elapsed time and zero terminal bootstrap. No advantage passes to another actor or match.

Budgets count **individual non-forced actor decisions**, not joint ticks. Forced no-choice waits do not become neural decisions. A complete-match collector may exceed a requested minimum budget; actual choices and match count must be registered/reported. Four actors sharing 512 choices is not four actors receiving 512 each, nor equivalent to the old focal 512-choice study. Exact decision-capped mid-match continuation is not implemented in this complete-match collector; the next protocol must register its batch/overshoot rule.

## Reward attribution and versioning

`crewmate-fourcrew-rewards-v2` uses own living task **+1**, successful legal report **+1.5**, correct vote **+2**, wrong vote **-1**, death **-5**, team win/loss **+10/-10**, draw **-5**, skip **0**, ghost task **0**, and the retained time penalty **-0.005 per 0.6 simulated seconds**. Historical `RewardConfig` retains report +1.25 and its original profiles unchanged.

Task/report/vote credit belongs only to its responsible actor, deduplicated by owner/event; illegal attempts and dead actors receive no report/vote bonus. A report must actually be accepted and trigger a meeting. Race-losing submissions earn no accepted-report credit. One team result intentionally broadcasts +10/-10 **once per actor**, including dead actors' buffered trajectories: four actor credits are not four team wins. Aggregate metrics must distinguish the single outcome from summed actor returns. Ghost completions can help the shared task bar but earn zero individual task reward in both new four-crew controls. This also means the new original control is not an exact copy of the historical focal control, which credited focal ghost work.

Reporting telemetry separates each living actor's body visibility, within-range/legal candidates, legal opportunities at choices versus during held options, selected report actions, submitted reports, accepted reports and triggered meetings. Counters are per-match unique body pairs where stated, not exposure duration. Successful both-role reports in tests are deliberately scripted engineering submissions, not learned reporting.

## Survival, suspicion and incentive limitations

An offline audit of eleven completed preselected training streams per arm found identical paired counts: focal first killed 4, killed overall 9; all nine preceding snapshots lacked another visible living crew within 4.5 sight. Last choices before death were five room options, two emergency-travel and two task options. None was a last choice to follow the eventual killer. Last decision to death was median 5.4s, maximum 18s. This supports investigation of isolation and option persistence, not a causal unsafe-route diagnosis. Scripted pursuit/kill selection uses nearest observed players and public-ID ties; it has no focal/learned-controller flag. Behavioral effects of scripted teammates remain unmeasured. Replacing them is not assumed to solve survival.

The old replay does not contain numeric belief histories. Memory code and new tests show persistence across meetings and fresh memory on new matches; switching actors and legitimate evidence changes can alter displayed probabilities. New traces explicitly record whose suspicion and at what last decision time. The predictor remains shared and frozen; four actor trajectories are not four trained belief models.

Historical task-driven skipping remains an owner-reported concern without a causal experiment. The focal pilot had six training skips per arm and zero validation skips, so that claim must not be generalized to those evaluations. +2/-1 voting can give positive immediate expected credit above one-third correctness, encouraging guessing; reporter-only accuracy already matched the focal aggregate results. No forced vote, general skip penalty or survival coefficient change was introduced. The next study must score evidence-conditioned decisions, calibration, innocent-reporter errors and win outcomes, retaining abstention as legitimate when evidence is weak.

## Validation and bounded execution

- Main relevant V2 regression suite: **134 passed**, 132.13s. A strengthened mechanics/four-actor follow-up: **17 passed**, 32.75s (overlaps the main suite; do not add counts as independent tests). This includes two new private-proof/report-interruption tests. Spectator: **34 passed**, including the original 31 and three UI/privacy tests.
- Final four-worker smoke: four development matches of 3 simulated seconds, **23 total actor choices / 16 actor trajectories**, one central PPO update, **12.85s wall time**. Loss/gradient/KL were finite; final checkpoint round-trip matched. This verifies engineering, not learning. Earlier engineering artifacts are retained, not overwritten.
- Real four-actor 60-second development match, zero updates: **32 choices**, zero illegal actions; exact recording-off/on packets, actions, log probabilities, values, rewards, discounts and outcomes. Off/on wall times **19.09/16.21s**. A single pair with warming caches and concurrent engineering activity is not a speedup or reliable overhead estimate.
- Actual private neural probabilities, suspicion and per-actor rewards were recorded. A headless frame rendered with the original map/characters and selected-actor private debug. No new visible desktop launch is claimed. [Viewer instructions](../../../_spectator_fourcrew/docs/SPECTATOR_POSTPILOT.md) explain D, speed, seek and replay controls.
- Development seeds only: registered engineering range 2760000–2760099; actually used 2760000–2760003. No full pilot, seed-29 replication or reserved final partition was opened.

## Commits and compatibility

| Commit | Scope |
|---|---|
| `77f478c` | V2.2 eligible ballots, early completion and task/ghost freeze |
| `555e3d0` | Four private shared-policy actors, collector and strict policy contract |
| `961aab9` | Per-actor replay rewards, sampling and complete-match verification |
| `732b555` | Versioned +1.5 reporting and attribution tests |
| `f8f9ff5` | Stronger privacy, held-report and ghost regression tests |
| `fa76497` (separate viewer branch) | Meeting privacy/scope, speed clicks and selected-actor debug |

Changed implementation files are `crewmate_v2/{__init__,engine}.py`, the new `fourcrew/` package, V2 regression tests and the separate spectator files. Existing learning/reward/PPO kernels remain unchanged; research documents and offline auditing tools are separate. [Design/context map](DESIGN.md) gives dependencies. [Research index](../research/README.md) and R001–R018 preserve the history; root `AGENTS.md` makes future log updates explicit.

V2.1 historical checkpoints remain byte-identical and source-compatible in `_v2_pretraining_diagnostic` at `5b663f6`. Repaired V2.2/four-crew loaders reject them; no silent weight migration is performed. New checkpoints use a separate schema and source hashes. Architecture equality alone does not establish behavioral comparability after changed mechanics, teammates and rewards. Checkpoints are taken at completed collection boundaries; no mid-match multi-agent resume is claimed. The final engineering checkpoint is the current-source verification artifact.

Local outputs: `_v2_fourcrew/local-runs/fourcrew-final-smoke/`, `fourcrew-final-capture/`, regression XMLs and `postpilot-replay-audit.json`; private image `_spectator_fourcrew/local-runs/fourcrew-engineering.png`. Historical artifacts remain in the original worktree. See [validation.json](validation.json) for integrity checks and machine-readable results.

Stop here. [The next study](NEXT_EXPERIMENT.md) is a proposal requiring owner approval, not a queued experiment. Learned impostor integration, larger lobbies and self-play remain future work.
