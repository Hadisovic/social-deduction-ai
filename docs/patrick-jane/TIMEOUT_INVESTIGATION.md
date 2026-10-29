# Match timeout investigation — 2026-10-29

Status: manual lights-run preparation; **no million-decision run started**.
Phase 6.1 acceptance remains pending. This is an engineering correction, not a
claim of improved learning.

## What the preserved records establish

| Preserved run | Training executions | Validation executions | TIMEOUT endings | Maximum simulated duration |
|---|---:|---:|---:|---:|
| `lights-launch-check-v1` | 61 | 4 | 65 | 12.0 s |
| `lights-launch-finalcheck-v1` | 32 | 4 | 36 | 12.0 s |
| `pilot-250k-seed17-v3` | 192 | 400 | 0 | 472.2 s |

The 101 TIMEOUT results in those two launcher checks were deliberately short
engineering matches. The original pilot's 592 executions all ended through
parity, impostor ejection, oxygen failure or reactor failure. The observation
alone does not establish which particular replay the owner viewed.

Inspection also confirmed a real `time >= config.timeout` game-ending rule in
the previous normal profile, whose lobby default is 900 simulated seconds. It
could end future lights-only matches even though it did not end the old pilot.
Historical files, checkpoints and their frozen source archive remain unchanged.

## New normal-run semantics

`LightsExperiment` profile `patrick-jane-lights-learning-v2` locks
`normal_match_timeout=False`. Elapsed normal time is not a victory condition.
The inherited lobby's 900-second field is inert in this profile and retained
only for compatibility. Preflight reports effective normal timeout `null`;
new replay/live metadata also describes no normal timeout. Legacy `Experiment`
retains its old behavior for reproduction and existing engineering fixtures.

Technical bounds are 1,800 simulated seconds or 300 stationary seconds without
progress. Actual movement, active task work, kills, tasks, reports, meetings or
repairs reset the stationary detector. A cut waits for the next roaming boundary
after bounded discussion/voting; genuine victory is checked first. The hard
duration therefore has a bounded meeting-phase allowance; it is not a new
canonical victory rule.

A cut records `reason: TRUNCATED`, no winner, cause, duration, no-progress time,
guard values, phase, living role counts, completed tasks, reports and kills in
offline summaries and replay result metrics. Those world diagnostics are for
research/viewing only. Actors receive only their own legitimate private
observations; hidden roles and observer metrics never enter the neural input.

## PPO boundary handling

For each living crewmate, a technical cut constructs the same private roaming
observation/encoding and memory update as ordinary inference. The critic uses
that agent's post-previous-observation GRU state. There is no sampled action,
sampling RNG draw or extra budget decision. The final transition has
`truncated=True`, `done=False` and `next_value=V(final observation)`; GAE ends at
that segment and bootstraps with the existing duration-adjusted discount.
True terminal outcomes retain `done=True` and zero bootstrap.

Dead/ejected actors have no ghost neural state. They stay terminated, retain
actually observed delayed credits, and receive no invented future ghost work
or team outcomes. Such future credits are censored at a cut. Frequent cuts would
therefore limit conclusions, even with correct living-critic handling. Operator
interruption, RPC errors and run wall limits withhold an uncertified partial
batch rather than make it a usable terminal episode.

Only the PPO boundary assertion changed. The archived PPO implementation must
match byte-for-byte after that single prescribed replacement. The GAE kernel,
network, action/feature encoders, memory, rewards and PPO hyperparameters remain
verified. The ordinary strict checkpoint/source loader remains unchanged. The
explicit pilot import verifies the original checkpoint/archive and imports its
weights, Adam and RNG into the newly registered regime.

## Viewer, results and completion

Viewer overlays and result markers distinguish crew victory, impostor victory,
TRUNCATED with cause, and HISTORICAL TIMEOUT with no victory. Summaries keep cuts
separate from wins, losses and draws. Win rates use legitimate outcomes only;
all-cut batches have an undefined win rate, not zero. Paired win comparisons
exclude pairs with either outcome cut. That conditional comparison cannot
establish performance on censored scenarios.
The existing champion selection rule is used only when all registered validation
outcomes are legitimate; any cut withholds that candidate from selection. Latest
checkpoints and raw evaluations are still saved. The archived engineering run
predates this gate and its all-cut champion pointer is not a competence result
or the manual run's warm start.

Progress counts additional actual learning-agent decisions. Percentage is capped
at 100, validation/bootstrap/script/ghost calls are excluded, and resume restores
committed counters and cumulative active time. A 100% collection counter can
coexist with running/failed state until PPO, final validation and outputs succeed.
Only then is `PATRICK JANE — TRAINING COMPLETE (100%)` displayed. The 12-hour
cumulative ceiling, original pilot warm start and manual-only launch remain.

Bounded verification measurements and remaining limitations are recorded in
`LIGHTS_LAUNCH_VERIFICATION.json` and research issues R041/R042. The short test
uses a 12-second technical guard in its separate local folder. It is not the
real run or evidence of learned competence.

[One-command launcher and output locations](RUN_LIGHTS_TRAINING.md).
