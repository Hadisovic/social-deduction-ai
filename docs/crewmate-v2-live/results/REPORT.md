# Seed-17 Crewmate V2 live pilot — development only

Both models completed 512 actionable choices and 20 validation games. These are 40 model-game evaluations of four distinct scenarios, with repeated modes/report rotations clustered by scenario. One initialization is exploratory; Phase 6.1 acceptance remains pending. See [the complete interpretation and limitations](REVIEW.md).

| Measure | A Original | B Shaped |
|---|---:|---:|
| Training minutes | 10.210 | 10.892 |
| Completed training episodes | 20 | 20 |
| Simulated training hours | 3.231 | 3.231 |
| Validation wins /20 | 5 | 4 |
| Living own tasks | 80 | 80 |
| Automatic ghost tasks | 0 | 0 |
| Body visibility pairs | 2 | 0 |
| Legal report opportunities at choices | 0 | 0 |
| Physical legal report pairs | 2 | 0 |
| Accepted reports | 0 | 0 |
| Reports / legal opportunity | unavailable | unavailable |
| Correct votes | 24 | 24 |
| Incorrect votes | 3 | 5 |
| Cast vote accuracy | 0.889 | 0.828 |
| Explicit skips | 0 | 0 |
| Automatic skips | 0 | 0 |
| Focal survived /20 | 8 | 9 |
| Mean common legacy return | -7.358 | -8.127 |

Total measured pilot wall time: **48.41 minutes**, including training and validation; three-hour cap retained.

![Actual PPO update curves](learning-curves.png)
![Measured validation behavior](validation-behavior.png)

## Match endings

| Seed | Mode / rotation / opponents | A | B |
|---|---|---|---|
| 2710000 | deduction / 0 / varied | PARITY | PARITY |
| 2710000 | deduction / 1 / varied | PARITY | PARITY |
| 2710000 | deduction / 2 / varied | PARITY | PARITY |
| 2710000 | deduction / 0 / original | PARITY | PARITY |
| 2710000 | ordinary / 0 / varied | TASKS_COMPLETED | TASKS_COMPLETED |
| 2710001 | deduction / 0 / varied | PARITY | PARITY |
| 2710001 | deduction / 1 / varied | IMPOSTOR_EJECTED | PARITY |
| 2710001 | deduction / 2 / varied | PARITY | PARITY |
| 2710001 | deduction / 0 / original | PARITY | PARITY |
| 2710001 | ordinary / 0 / varied | TASKS_COMPLETED | TASKS_COMPLETED |
| 2710002 | deduction / 0 / varied | PARITY | PARITY |
| 2710002 | deduction / 1 / varied | PARITY | PARITY |
| 2710002 | deduction / 2 / varied | PARITY | PARITY |
| 2710002 | deduction / 0 / original | PARITY | PARITY |
| 2710002 | ordinary / 0 / varied | TASKS_COMPLETED | TASKS_COMPLETED |
| 2710003 | deduction / 0 / varied | PARITY | PARITY |
| 2710003 | deduction / 1 / varied | PARITY | PARITY |
| 2710003 | deduction / 2 / varied | PARITY | PARITY |
| 2710003 | deduction / 0 / original | PARITY | PARITY |
| 2710003 | ordinary / 0 / varied | TASKS_COMPLETED | TASKS_COMPLETED |

## Interpretation boundaries

The shared frozen Phase4 belief predictor is unchanged. Differences in its measured scores arise from observation trajectories, not independently trained belief networks. Reporter-identity-only controls, calibration, voting participation, paired four-scenario bootstrap and direct-witness vote strata are retained in metrics.json.
An absent direct kill witness does not imply no legitimate evidence: bodies, reports, claims and public task proof may still be available. Missing reporting opportunities are separate from failures to report. Physical opportunities inside held options are separate from opportunities at policy-choice boundaries.
Sampled training accusations use only completed replays of preselected streams0–3. Hidden roles are used for offline scoring only. These counts are not a census of all training or validation claims, and their legitimate evidence support was not exported. Raw observations and role-label datasets remain local.
Correct-vote bonuses can reward guessing (+2/−1 gives positive expectation above1/3 correctness). Repeated correct ballots across meetings, reporter shortcuts, ghost/task-only contributions and reward-return scale differences must not be mistaken for learned deduction. Reward configuration is a bundle, so this pair cannot attribute effects to a single coefficient.
The viewer plays completed training matches recorded under behavior-policy updates 0–3; validation tests the fixed 512-choice checkpoint. No true in-progress frame streaming or validation replay export is claimed. No canonical simulator, existing training-kernel or reward-configuration changes, final-test access, seed-29 replication, impostor development, merge or deployment occurred.
