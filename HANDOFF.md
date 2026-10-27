# Continuation handoff

## CURRENT — V4.1 study COMPLETE; no jobs or further training authorized

Finished27October2026 at23:48:55 Asia/Amman. Session50520 exited0. It recovered
evaluation-only logging IO, not learning. All80 selection+5 sampled diagnostics+
40 confirmation executions are saved. No reserved final partitions were opened.
Original envelope elapsed165.09min (deadline01:03:50); corrected run87.03min,
parallel training span42.45min. Runtime source83f75c4 remains byte-unchanged.

Immutable selected83/8192 SHA
69965afd8fab28e6ca5bdda46d9f8efdd8d6036ce57a05b86ff89fe22cf17ca9.
New separate release artifacts/impostor-options-v41/release/policy.pt; all8 new
snapshots strict-loaded/copied; old models not overwritten. Actual easy wrapper
`python run_impostor_v41.py` passed8-frame headless launch; capture in report/viewer.
Old `python run_impostor_ai.py` still loads historical V2, intentionally.

Fresh20 paired cases/method:V41 0wins/34kills/0fake choices/33executed reports/
15task losses/5own ejections/0witnessed kills; V2 0wins/9kills/3654fake choices/
20task losses. New kills in20/20cases (14two-kill,6one-kill). No illegal actions
or navigation failures in125 evaluation executions; frozen models unchanged.
No reliable winning policy, causal deception or isolated shaping benefit claimed.
Training71:72games60kills0wins;83:78games76kills1win;8192 decisions each.
Positive credit42/72 and48/78 games; terminal cancellation/lifetime bound passed.

Short owner/team doc docs/impostor-v2/CURRENT_RESULTS.md; full report/11graphs/
raw cases/protocol/lock/recovery docs/impostor-v2/options-v41-study-20261027/.
Master checklist/research log/architecture/future lobbies updated. Eight attempts
total:7complete1partial;30policy snapshots, not30independent NNs. Next full study
needs approval, a fresh protocol/cases and integration of the tested terminal-time
math helper into a NEW versioned learner. Current V41 does not import that helper.

Owner asked about sabotage and reporting: missing sabotage is a hypothesis, not
proven cause; current crew have no trained functional sabotage response. Most
losses are tasks75%,ejection25%. Do not ban selfreports or invent a flat penalty:
first measure immediate-own-kill reporting and controlled leave/stay/report
consequences, then train/evaluate choices. No new training is authorized by that
question. Up-to15 total players/1–3impostors remains future work for both policies.
Owner still says no computer use; send game commands. No merge/deploy.

Completed source/results/checkpoints were pushed as99fc9d5 to the existing branch;
PR10 title/body updated and confirmed draft/open/unmerged, base still exacte62.
This later checklist-status note may have its own documentation commit. Read git
log for the latest hash. Scientific results/release/verification are complete.

## Historical execution notes — superseded, do not restart these jobs

### V4.1 evaluation IO recovery (now finished)

Training COMPLETE; selection COMPLETE and IMMUTABLE:83/checkpoint8192, SHA
69965afd8fab28e6ca5bdda46d9f8efdd8d6036ce57a05b86ff89fe22cf17ca9.
All4 candidates won0/20 selection cases; selected had28kills (tie-break).5 sampled
diagnostics completed0wins/8kills. Confirmation had6/40 saved cases when root
progress.json replacement hit transient Windows sharing violation WinError5.
Session98577 exited1 from LOGGING, not optimizer/game/source failure.

Owner already authorized completing the bounded study. Logging-only recovery via
python -u tools/resume_impostor_v41_io.py --output artifacts/impostor_research/options-v41-20261027
is active as session50520. It reuses all80 selection,5diagnostic and6 completed
confirmation files, checks the same lock, performs NO training, and retains the
ORIGINAL deadline01:03:50 +03. STOP marker was archived, not deleted. Two IO tests
passed0.36s. The recovery launcher patches ONLY bounded output/logging retries at
import in parent/spawned workers; scientific sources/weights/seeds unchanged.
Recovery provenance is a separate io-recovery-*.json; do not edit study.json/lock.
Do not edit healthy runtime/IO helper files. Avoid PowerShell Get-Content live
progress files (brief Windows sharing locks); read/close with Python if needed.

Report tool now derives original parallel training span from durable seed start/
training records: a resumed runner's training_wall_seconds is near0 because it
skips already completed learning. Total wall time includes IO downtime. Preserve
the recovery record, unfinished-case replay and worker-sum caveats in the paper.

Future trainer math audit found instantaneous terminal kills legitimately elapsed0,
but inherited GAE requires >0. Current completed learning never hit it; inference/
evaluation do not use GAE. Standalone candidate impostor_training_math.py and4tests
passed5.21s; NOT wired into frozen V4.1. Integrate in a new registered learner
before further full training. Controlled fixture is NOT a learned-policy win.

Historical runtime/study notes below retained; session98577 is stopped. Current
source remains83f75c4dc7c6c6e72cd553192e63d48dc5f32237 and current evaluation50520.

Important discovery: all75 completed seed71 games and15 durable seed83 games had
zero pursuit credit. Crowded spawn sets V4's lifetime proximity high-water to1
during initial45s cooldown, disabling intended shaping. Owner explicitly answered
“Fix and restart within original deadline (Recommended)”. Session57078 was stopped
with Ctrl-C, exit1. Preserve it as interrupted, not healthy/running/complete.
artifacts/impostor_research/options-v4-20261027/interruption.json records decision,
reason and last durable progress. Seed71:8192/75games/68kills/0wins; partial83:
1792/15games/16kills/0wins. No selection or confirmation was opened. No old source
or checkpoint was edited. Original absolute deadline is STILL2026-10-28 01:03:50 +03.

Repair is separate namespace impostor_options_v41/, root run_impostor_options_v41.py
and tools/run_impostor_options_v41.py. It preserves75,075 params, fake persistence,
all reward coefficients/physics. Only frontier arming changes: while own cooldown
is positive clear geometry history, not credit/budget; once ready establish a
non-paying baseline then pay own visible progress. Meetings do not reset frontier;
kill cooldown can begin another approach period but NEVER refills lifetime0.05.
New strict schema impostor-options-v41; original V4 loaders remain reproducible.

Read docs/impostor-v2/options-v41/TRAINING_PROTOCOL.md. New fresh training starts
3111000; selection3200140–159 (80 executions), lock,5 sampled diagnostics onfirst5
selection cases, confirmation3200160–179 (40). Two scratch71/83 ×8192, one CPU
worker, same PPO. New run must borrow original deadline from the old study SHA,
not allocate another4h. Output intended artifacts/impostor_research/options-v41-20261027.
Replacement launched as session98577 from clean source commit
83f75c4dc7c6c6e72cd553192e63d48dc5f32237. Manifest confirms dirty=False and ORIGINAL
deadline2026-10-28 01:03:50.430218 +03;161.94min remained at corrected start.
Both71/83 are active in two isolated CPU training workers. Read current persisted
artifacts/impostor_research/options-v41-20261027/progress.json and LIVE_PROGRESS.md;
root progress includes a `runs` dictionary and each seed also has progress.json.
Preserve healthy workers. DO NOT edit either impostor_options_v4/ or
impostor_options_v41/, root viewers, registered runners or either protocol.
Only unguarded offline analysis tools/status docs may change while healthy.
LATEST STAGE: both corrected learning runs COMPLETE; selection is running (80
fixed matches over4 candidates).71:8192decisions/72games/60kills/0wins;83:
8192/78games/76kills/1win. Total150 training games/136kills/1win. No selected lock
as of this update; confirmation remains unopened. Preserve session98577.
Ordinary corrected training has nonzero pursuit credit in BOTH initializations:
at1024 decisions,71 paid3/8 games (max.03955),83 paid4/8 (max.00871). The real-spawn
fix is active; terminal shaping still cancels and benefit is not yet established.

Corrected offline tools are ready OUTSIDE guarded paths:
tools/report_impostor_options_v41.py and release_impostor_options_v41.py.
Four synthetic report/curation tests passed24.93s (not performance evidence).
They use fresh3200140–159 selection/3200160–179 confirmation, selected_v41 method,
11 graphs, original-budget accounting and parallel-worker timing caveats.
After COMPLETE results.json, run:
python tools/report_impostor_options_v41.py artifacts/impostor_research/options-v41-20261027 docs/impostor-v2/options-v41-study-20261027
then inspect curves/paired results/fake streaks/ready time/true wins before curating:
python tools/release_impostor_options_v41.py artifacts/impostor_research/options-v41-20261027 docs/impostor-v2/options-v41-study-20261027 artifacts/impostor-options-v41/study-20261027
All outputs must be new directories. Never overwrite old release or tune/select
using confirmation. Viewer command for corrected chosen release:
python run_impostor_options_v41.py --checkpoint artifacts/impostor-options-v41/release/policy.pt
Use terminal headless verification only; NO computer-use/GUI automation per owner.
Report original interrupted run and repair separately; respect fixed01:03:50.

Final preflight session62944 exited0:5 tests
passed130.82s, including actual45s spawn/visible approach, bounded arming, real PPO/
exact resume, old V4 compatibility and2 isolated8-decision development workers.
The first real-world test initially failed because a room goal parked at a doorway
with no visible crew; corrected fixture uses a public-console route then legitimate
visible candidates. Only the cooldown-frontier production repair was needed.
Two PARALLEL training CPU workers/one Torch thread each, two isolated evaluation
method workers, sequential sampled diagnostics. Synthetic80-case scheduling test
also passed14.23s (six total corrected preflight checks, not new performance data);
this preserves2×8192 decisions and helps meet the original deadline. Sources and
protocol recorded this before launch. Do NOT rerun original interrupted study.

Six original V4 policy snapshots were preserved/strict-loaded in
artifacts/impostor-options-v4-interrupted/study-20261027; compact logs and honest
interruption summary in docs/impostor-v2/options-v4-interrupted-20261027/.

No computer use; user runs own game commands. Future custom lobbies documented in
FUTURE_LOBBIES.md, up to15 total/1–3impostors, normally2shared role policies +belief.
Offline V4 report/curator tools exist but require V4.1 fresh-seed/schema adaptation
before corrected results: tools/report_impostor_options_v4.py and release_impostor_options_v4.py.
Keep both interrupted V4 and corrected V4.1 evidence; account for repair time in
original budget. Historical running notes below are superseded.

## CURRENT — V4 study RUNNING, session57078 (2026-10-27)

LATEST: owner said “go on” after the3–4h training proposal. This authorizes the
registered V4 study, hard ceiling14400s, two scratch initializations71/83 ×8192
decisions. Read docs/impostor-v2/options-v4/TRAINING_PROTOCOL.md.80 greedy selection
matches3200100–119, immutable lock,5 sampled diagnostic reusing first5 selection
cases, then40 fresh greedy confirmation executions3200120–139: locked V4 and
unchanged V2. No architecture changes, reserved finals, merge/deployment or extension.

New runner: tools/run_impostor_options_v4.py; runtime impostor_options_v4/study.py
and diagnostics.py. One CPU worker/one Torch thread. Keep all source modules and
the protocol document unchanged once healthy workers start. Telemetry is proven
to leave V2/V4 physics, actor packets and traces unchanged.36 preflight tests passed
55.79s, including genuine8-step PPO gradients, frozen checks and exact continuation.
These tiny development updates are not extra research training runs. Prior59-test
implementation record and zero-optimizer smoke evidence remain historical.

Output reserved for this new study: artifacts/impostor_research/options-v4-20261027.
Start only after a clean local commit. Runner command:
python -u tools/run_impostor_options_v4.py --output artifacts/impostor_research/options-v4-20261027
Study launched as session57078 from clean source commit
b59ab8ac482fcbdd80f064d3805a27beed6cba52. Manifest confirms dirty=False.
Started2026-10-27 21:03:50 +03; deadline2026-10-28 01:03:50 +03 (Asia/Amman).
Seed71 training has completed8192 decisions; seed83 is now running. Read persisted
seed-71/training.json and episodes.json for exact outcomes. Validation is unopened.
progress.json and LIVE_PROGRESS.md persist progress; retain healthy jobs and resume
snapshots. Never restart without failure. Do NOT edit impostor_options_v4/, the
registered runner, root viewer or TRAINING_PROTOCOL.md while healthy workers run.
New offline analysis tools outside those guarded paths and status documents are
allowed. No automatic restart/extension or test-driven tuning.

Offline completion tools are prepared outside guarded namespaces:
tools/report_impostor_options_v4.py (11 plots, paired uncertainty, fake/ready/reward
metrics;3 synthetic report tests passed12.30s) and
tools/release_impostor_options_v4.py (strict8 snapshots +new separate release;
synthetic no-overwrite/identity test passed8.24s). Their fixtures are not real model
results. After results.json says complete, generate into NEW directory
docs/impostor-v2/options-v4-study-20261027, inspect figures/results, then curate
artifacts/impostor-options-v4/study-20261027 with its NEW sibling release/.
Commands: python tools/report_impostor_options_v4.py SOURCE REPORT
and python tools/release_impostor_options_v4.py SOURCE REPORT BUNDLE.
Never overwrite artifacts/impostor-ai/release or any historical models. Selected
V4 viewer command must be explicit: python run_impostor_options.py --checkpoint
artifacts/impostor-options-v4/release/policy.pt. Do NOT edit root viewer defaults,
which are source-guarded. Complete report/checklist/research log with real outcomes,
including failure, and distinguish training/selection/fresh confirmation.
If the4h ceiling stops it, preserve partial evidence and report incompleteness;
do not launch another run or claim a complete evaluation/release.
Protocol wording erratum (do NOT edit the hash-guarded protocol): “every256 updates”
in its snapshot sentence means every256 policy decisions, i.e. after each rollout/
PPO update. Runtime/checkpoint code has used256 decisions since the clean baseline;
policy-only snapshots remain0/2048/4096/8192. Report actual scheduling accurately.

Latest future-work request: client-selectable lobbies up to15 total players and
1–3 impostors, with legal combinations verified for the targeted game version.
10crew/2impostors becomes one preset. Both policies need variable-roster/team
adaptation; normally share2 role action policies +existing crew belief NN across
actors with private state, not15 independently trained networks. Documented in
master checklist/research log; NOT part of this5-player training study.

Owner expressly says no more computer-use; supply commands for games. Do not
close/control their watched game. Previous V4 viewer session75280 may still be
open; it uses OLD weights and is not learning. New study must earn improvement.
Historical notes below are superseded where they say training lacks approval.

## CURRENT — V4 correction implemented; full training awaits approval (2026-10-27)

Owner reopened work to implement the pasted reward/action prompt, a detailed NN
table, a live checklist and runnable visualization. This supersedes the previous
end-of-day stop ONLY for this implementation. **Do not start a new full PPO study.**

Branch remains codex/impostor-environment-v2, base e62ba0b6c33ffd0317f78e18b4289d778cd5af2b.
Added impostor_options_v4/ and run_impostor_options.py; old guarded V1/V2/V3 engine,
policy/trainer/viewer files and checkpoint bytes are unchanged. Same75075-parameter
12-family architecture; V4 versions the reward/action/observation semantics.
Rewards: kill+3, extra ejection-6, win/loss+10/-10, timeout-12, no unconditional
time cost, -.02 per actual simulated ready/active/roaming second. Fake repeats
continue the current4s activity; NN still chooses continue/cancel/move/kill/report.
Pursuit uses actor-only bounded potential, <=.05 lifetime positive credit, terminal
correction; no automatic chase or hidden positions. Record new feature meanings:
global_[3]=own fake remaining fraction, extra[2]=proximity high-water,
extra[3]=remaining credited potential normalized; all dimensions unchanged.

Read docs/impostor-v2/NN_ARCHITECTURE.md (all exact V2 inputs/layers/actions),
REWARD_ACTION_V4_CHECKLIST.md and options-v4/README.md. New suite32 passed56.59s,
zero optimizer steps.27 original core tests also passed14.99s (59 across these
two distinct files); all8 V1 +8 V2 historical snapshots strict-loaded unchanged.
Two full development smoke games3500000/1 completed
in123.60s: both task losses,0kills,31/30 fake completions,0illegal/nav failures.
Report docs/impostor-v2/options-v4/smoke-results.json; these are integration smoke
outcomes, not validation or evidence of better learned play. No V4 model trained.

Antigravity IDE opened the repo and exact table/checklist/new viewer source. Native
game window was visibly inspected running at89.4s. Active viewer command/session:
python run_impostor_options.py --seed3600000 --speed1, session75280. Leave it for
the owner to watch; R replays, N changes game. It explicitly labels old V2 weights
under new rules as TRANSFER DEMO / NOT TRAINED IN V4. Original run_impostor_ai.py
still reproduces the evaluated V2 release. An initial detached launch ended; the
active session above is the verified viewer, not a training job.

Current lobby remains1 learned impostor +1 frozen full29 NN crew +3 scripted crew.
Owner confirmed eventual10crew/2impostors as a future goal for both models; it is
documented, not implemented/authorized in this task. Needs separate design. Friend's newer
crew is not loaded. Proposed future bounded study estimate3–4h, not approved.
No merge/push/deploy or reserved final tests in this implementation session.
LATEST owner: no more computer-use calls; send commands for owner to run games.
Full progress plan: docs/impostor-v2/IMPOSTOR_MASTER_CHECKLIST.md (opened via Codex
file tool; queued). Research paper ledger: RESEARCH_LOG.md includes every failed
run, exploit/fix and12-player roadmap. Owner observes repeated faking in an empty
room: V4 completes activities but OLD frozen V2 weights choose to start another.
The NN does not learn while watched; fixing the interface/reward does not change
weights. Learned improvement remains an unchecked acceptance gate. No training.
Historical completed-study notes below are preserved and superseded for execution.

## CURRENT — PUSHED; owner finished for today (2026-10-27)

Code/results/checkpoints/graphs/summary pushed in118715a907de6bdc345902682f2011ee3f89108a.
Draft PR10 updated and attached: https://github.com/Hadisovic/social-deduction-ai/pull/10
Connector confirmed OPEN,DRAFT,UNMERGED, base codex/crewmate-environment-v2 at exact
e62ba0b6c33ffd0317f78e18b4289d778cd5af2b; head codex/impostor-environment-v2.
This last documentation commit marks delivery complete. No more work/training today.

Owner's latest instruction: summarize all results, PUSH ALL, then call it a day.
DO NOT start further training, including the previously prepared pursuit study.
All actual jobs finished: V1 session66647 and V2 session1855 exited0; default viewer
session5698 exited0 after8 headless frames with the actual selected model.

Four registered runs: V1 seeds71/83 each4096 decisions (18games each,0wins,0/1kills);
V2 seeds71/83 each8192 (82/78games,0/1wins,68/57kills). V1 validation0/20 each.
V2 selection locked71/8192 at2/20 wins; fresh confirmation0/20 wins,4kills,all task
losses,0ejections. Older AI0/20; scripted attacker1/20; untrained NN0/20. V2 study
117.24min; V1 study25.37min. All160 V2 evaluations legal/navigation-failure-free.

Measured failure:3560 fake-task selections in fresh20games; only4games offered legal
kill windows and all4were killed. FAKE_TASK restarts4s activity; wrapper re-decides
every.6s, so repeated greedy choices restart it. No witness/deception competence
established. Pursuit version only passed tiny development gradient/resume/viewer
tests; no registered training/performance study. It remains untrained research code.

Primary summary: docs/impostor-v2/SESSION_SUMMARY.md (plain controller names, all
failed runs, actual crew NN/script logic, failure analysis and next work).
Detailed guide: docs/impostor-v2/final-20261027/README.md, with viewer.png.
Eight V2 graphs/full data: docs/impostor-v2/strategic-v2-20261027/.
Six V1 graphs/data: docs/impostor-v2/pilot-20261027/.
16 policy snapshots preserved across artifacts/impostor-pilot-20261027 and
artifacts/impostor-v2/study-20261027; raw resume journals remain local/ignored.
Universal release: artifacts/impostor-ai/release/policy.pt, SHA
8b861c9cb7477a2145d41af3961e03aa5d9bd6c1df1d1aaea8629c2c9597698c.
Watch command: python run_impostor_ai.py. V2-only original run_impostor.py retained.
All strict loaders/frozen weights/historical source guards retained.

All intended artifacts/source pushed and draft PR10 updated; checklist marked after
verification. No merge, deploy, new training or reserved final tests. Stop here.
The historical execution notes below are superseded; do not act on old launch advice.

## Historical execution notes — improved study (superseded)

LATEST OWNER STEERING: training is finished; owner requests a complete summary
(failed runs too, impostor/crew logic and mixed roster, results/causes/next fixes,
gameplay command) then PUSH ALL. Finish healthy existing evaluation, but DO NOT
launch the conditional pursuit study. Prepared pursuit code stays documented as
tested with tiny development-only gradient checks, not a registered training run.
Use tools/report_impostor_final.py with --pursuit-skip-reason
"Owner requested summary and push after completed training; no further training."
and NO --v3-source. Final universal release must remain the locked evaluated V2
checkpoint; default watch command python run_impostor_ai.py.
The final guide generator now includes all4 actual runs, exact mixed opponent
logic, fake-task-collapse evidence and concrete next fixes. V2's selected model
confirmed0/20:20task losses,4kills,0ejections;3560fake_task,209go_to_location,
17go_to_task,96continue,264go_to_body,3claim_suspect_player,3vote. Only4matches
had legal kill opportunities; it killed in all4. Do not claim learned witness
avoidance/deception. The scripted hunter control has at least1win; finish all80.

Healthy worker: session1855, source f96abbe (clean committed manifest).
Output: artifacts/impostor_research/strategic-v2-20261027.
Runner: python -u tools/run_impostor_v2.py --output artifacts/impostor_research/strategic-v2-20261027
Live progress: same output directory / LIVE_PROGRESS.md and progress.json.
Do not restart healthy jobs or modify loaded source modules, including run_impostor.py.
Only offline analysis tools outside impostor_research/ and documentation may be edited.
The study deadline is recorded in study.json; no extension or reserved-final evaluation.
After completion curate locked checkpoint, graphs, assessment, verify viewer and push PR10.

Progress: V2 training complete. Seed71:8192 decisions/82 games/68 kills/0wins,
2156.67s wall. Seed83:8192/78games/57kills/1win,1581.21s wall. Selection COMPLETE.
Immutable lock: seed71/8192,2/20 selection wins,12 kills, checkpoint SHA
8b861c9cb7477a2145d41af3961e03aa5d9bd6c1df1d1aaea8629c2c9597698c.
Confirmation/control80 now running; do not change this selected identity.
Prepared/committed in18e401d a separate conditional pursuit correction in impostor_pursuit/ (NOT the
hash-guarded impostor_research/ namespace); 3 tests passed in26.35s, including exact
warm-start/resume and universal viewer schemas. It has NOT trained. Read
docs/impostor-v2/pursuit-conditional-protocol.md: only if V2 completed confirmation
<5/20 wins AND >=3600seconds remain on the ORIGINAL study.json deadline; no extension.
One continuation8192 steps,60 selection+40 confirmation matches with fresh IDs. The
unchanged V2 parent remains eligible. Source must be committed/clean before launch.
Use tools/run_impostor_pursuit.py with output artifacts/impostor_research/pursuit-v3-20261027,
--parent-study artifacts/impostor_research/strategic-v2-20261027,
--parent-checkpoint artifacts/impostor-v2/release/policy.pt (curate V2 first).
Universal final viewer is run_impostor_ai.py; it expects the final explicitly selected
artifact at artifacts/impostor-ai/release/policy.pt. Preserve original run_impostor.py.
Offline report/curation tools tools/report_impostor_v2.py and release_impostor_v2.py
are ready and synthetic report checks passed. New tools/report_impostor_final.py
(currently uncommitted) also passed isolated byte-copy/strict-load/zero-win guide
checks; it can handle V3 completed, skipped or stopped and writes universal release.
After V2 finishes generate report to
docs/impostor-v2/strategic-v2-20261027 and curate8 checkpoints to
artifacts/impostor-v2/study-20261027 (curator also writes V2 release and README).
If conditional gate passes, commit these results before launching V3. Do not change
loaded modules while either worker is healthy. New analysis tools outside both
runtime namespaces are allowed. If V3 skipped/stopped, explicitly release fully
evaluated V2 to universal viewer; never silently choose a partial model.

Latest owner explicitly requests fixes to policies, reasonable rewards/NN settings,
training a strong impostor, pushing results, and a file to run to watch it. The first
pilot COMPLETED (session66647 exited0): both 4096-step policies lost all 20 validation
matches with no validation kills. Study took1522.17sec; 36 training games had zero wins,
one sampled kill total. Six graphs and eight strictly loaded checkpoints are preserved
under docs/impostor-v2/pilot-20261027 and artifacts/impostor-pilot-20261027. No raw or
historical evidence was deleted/rewritten. See its README for the failed-learning assessment.

New additive modules strategic_v2.py, learning_v2.py, study_v2.py and root
run_impostor.py implement a family/target neural policy, independent critic/gradient
clipping, +2 successful-kill shaping, -2 ejection, -12 timeout, -.002/sec, original
terminal +/-10 and gamma1; interrupted3sec travel options; legitimate visible geometry.
Canonical ordinary lobby/ghosts and frozen full29/belief unchanged. 78 tests passed,
including six new policy/reward/privacy/resume/viewer checks. Improved study launched
from f96abbe in session1855. Do not edit old pilot
runtime files: old checkpoints must retain strict compatibility.

Read docs/impostor-v2/strategic-v2-protocol.md. Approved bounded follow-up: seeds71/83
x8192 decisions, then80 selection matches on3200020..39 (four checkpoints), immutable
lock, then80 confirmation/control matches on3200040..59. Three-hour learning/evaluation
ceiling. No merge, deployment or reserved-final tests. Use fresh ignored output directory
artifacts/impostor_research/strategic-v2-20261027 and update this top status with its
actual source commit/process session after launch. Finish graphs/report, curate selected
model to artifacts/impostor-v2/release/policy.pt, verify viewer and push PR#10 updates.

## CURRENT — approved impostor pilot RUNNING, session 66647 (2026-10-27)

The owner said “go on” after the 2–4-hour preparation/pilot estimate. This authorizes
two ordinary-V2 impostor initializations (71/83), 4096 decisions each, followed by
20 paired greedy validation scenarios per model (3200000–3200019), 40 total. Both
crew safeguards, pinned frozen full-29 and original belief model, canonical lobby,
gamma=1 sparse reward. Hard learning/validation ceiling: 3 hours. No additional
training, reserved final tests, merges or deployment. Friend's crew work is separate.

Trainer preflight passed: 79 combined tests and 9 final trainer/report tests (7 overlap;
81 distinct checks). Committed runtime `2c31f060212bd42a28d92b98e132f5f97d0d2832`.
`python -u tools/run_impostor_pilot.py --output artifacts/impostor_research/pilot-20261027`
is healthy as **session 66647**, seed 71 started. Study manifest confirms a clean
checkout and exact source/config/frozen hashes. Preserve this job and do not edit
runtime modules while it runs. Budget uses the original `study.json.deadline_epoch`.

Live state: `artifacts/impostor_research/pilot-20261027/progress.json` and
`LIVE_PROGRESS.md`; immutable rollout resume checkpoints and sampled training journals
are in seed-71/seed-83 directories. It will run both 4096-decision seeds before opening
exactly 40 greedy paired validation matches. Do not select earlier checkpoints or tune
from validation during this pilot. No scientific retry/extension; stop/report on errors.

After `results.json` completes, run the frozen offline report generator into a NEW
curated report directory, inspect six graphs, assess sufficiency honestly and copy
final policy-only checkpoints plus compact config/hashes/results for review. Raw actor
journals/resume binaries remain ignored locally. Update checklist/handoff/PR #10 with
actual timings, processor/runtime, outcomes and limitations. No merge/deploy/final tests.

## CURRENT — independent impostor V2 engineering complete; training unapproved (2026-10-27)

Owner authorized implementation on `codex/impostor-environment-v2` from exact V2
baseline `e62ba0b6c33ffd0317f78e18b4289d778cd5af2b`, not main. Implementation revision
`cb9b5f4fc3a606383f3b04815151602cb54f45ec` adds only independent impostor modules,
tests, benchmark and documentation; historical/V2 runtime and all checkpoints remain
unchanged. Owner's friend independently improves crew; do not train that track here.

Read `docs/impostor-v2/SESSION_CHECKLIST.md`, `docs/impostor-v2/README.md` and the
updated `docs/phase6_parallel_research_handoff.md`. Full-29 loads through its original
strict loader and a lossy V2-to-legacy adapter; crew/belief weights remain frozen.
Opponent-only body-discovery and direct-eyewitness safeguards are implemented and
toggleable. The impostor has its own actor memory, masked interface and untrained
48,738-parameter network. No optimizer/training driver is invoked by the new benchmark.

239 tests passed; 10 canonical bounded development matches completed in 374.30 seconds.
Impostor packet/action/event replay hashes match reference and repeat; zero illegal
actions/navigation failures. Ordinary scripted controls lost by tasks, paired deduction
controls reached parity. Natural crew reports occurred with non-reporting impostors.
No natural witnessed kills in these scenarios; controlled tests establish the witness
sequence. All nine checkpoint hashes/selection locks preserved; six strict loads pass.

Proposed first pilot: two 4096-decision runs plus 40 paired validation matches,
1–2.5 hours estimated, 3-hour hard ceiling. **Not approved. Do not run any RL training,
reserved final evaluation, merge or deployment.** A trainer/optimizer harness needs
reviewed implementation after design/budget approval. Only a draft PR is requested.
Shared ChatGPT planning link was read through its existing Edge tab on 27 October,
after earlier access failures. Its Among Us section confirms the same work split
and approval gates, with no additional training settings. See
`docs/impostor-v2/TRAINING_NEXT_STEPS.md`: 45–90 minutes of trainer preparation,
then the proposed pilots/validation; about 2–4 hours to first results including setup.
The completed benchmark has no running worker to resume. Earlier handoffs below are
preserved history and do not authorize training or production changes.

## CURRENT — nine-model validation complete; final evaluation unapproved (2026-10-26)

The owner approved the six seed-29/43 models. They completed on isolated research
branch `codex/phase6-1-seeds29-43`, based on PR #4 head `8187d51`, with runtime
source commit `874e0390a7a34cd96e62ecf0c795f51e20ccd52c`. Only initialization
wiring/metadata and bounded continuation/report drivers were extended. The
environment, PPO, reward, architecture, evaluation seeds and checkpoint ranking
remain frozen. Strict source loading was not relaxed; old checkpoints retain
their original source snapshots and metadata. Seed-17 research training was not repeated.

Read `docs/benchmarks/phase6/remaining-six/README.md`, results/CSV, locked selection,
integrity audit and `docs/phase6_seed29_43_continuation.md`. All nine runs trained
8192 decisions; selected steps full/history/current are 6144/4096/2048 for seed 17,
8192/6144/4096 for seed 29, and 4096/4096/8192 for seed 43. Validation only, 200
paired familiar scenarios per model; 3320 total executions including screens and
reused controls. The first 20 scenarios are reused, not independent extra evidence.

Wins by seed 17/29/43: full 90.0/90.0/88.5%, history_no_belief 87.5/91.0/89.5%,
current_only 90.0/86.0/83.0%. Mean tasks 1.922/1.952/1.797. All exceed idle/random;
superiority over scripted (91.0%) is not established. Full versus history shows
no clear win/task benefit. Voting is seed-sensitive: full seed 43 never votes;
current-only seed 29 makes 105 incorrect votes of 171. All learned own reports
are zero. Report incorrect crew-targeting votes separately from innocent team
ejections and from structured-claim accuracy, which was not separately scored.

Additional stage: 3.644 hours against 4.28 estimate / 6.42 ceiling; training 2.307h,
screening 25.76min, extended validation 52.21min. Six new runs add 49152 decisions
and 32.740 simulated hours; all nine total 73728 decisions / 49.122 simulated hours.
50 targeted tests passed; both new seeds reproduce smoke/PPO/inference, all 120
new selected screening rows replay exactly, all 30 new checkpoint copies load
and hash correctly, and 419 original research files remain byte-identical.
No recorded illegal/nav failures or validation timeouts; no scientific retry or
budget fail-safe fired. All raw runs and checkpoints remain preserved.

**Stop here. Phase 6.1 is not accepted; final familiar/patient partitions remain
unopened and need separate approval.** The report includes an unapproved proposal
for all nine finalists plus four controls: 13000 final matches, about 9.43h at
observed throughput (provisional 7.54–14.14h). It is not a final lock, a runnable
driver or permission to execute. Review voting weaknesses before spending this
budget. No learned impostor, expanded population, dashboard, parallel-training
benchmark or self-play is authorized. Do not merge PRs #3/#4/#5 or update production.

Published main remains the documentation-only PR #6 merge `a463a42`; its roadmap
is authoritative and was not copied/overwritten. Integration conflicts with main
remain separate work after experimental review. Original pilot entries below
are history where superseded by this completed, approved continuation.

## CURRENT — Phase 6.1 seed-17 pilot complete; stopped at approval gate

The approved three 8,192-decision seed-17 models and all 1,640 registered validation
executions completed successfully at frozen source revision
`cd48b65a8de9de0cab33894ddb2558057487847d`. New branch:
`codex/phase6-1-seed17-stage`, based on PR #3's `1610113`. PR #3 stays unmerged.
See `docs/benchmarks/phase6/seed17-stage/README.md` and the registered protocol.
Full / history_no_belief / current_only: 90.0% / 87.5% / 90.0% crew wins,
1.905 / 1.950 / 1.940 own tasks per 200 paired ID validation matches. Full cast
167 votes (159 correct, zero skips); the ablations cast one/zero and skipped
178/179. No own reports. No illegal actions, navigation failures or timeouts.
All beat idle/random; superiority over scripted (91.0%) is not established.

All trained to 8192; the immutable 20-seed screen selected 6144 / 4096 / 2048.
The first 20 of 200 extended seeds repeat screening exactly (60 rows). All initial
tensor hashes match; source/checkpoint/frozen release hashes verified. 431 tests
passed before execution. Fifteen experimental checkpoint copies and full curated
records are committed; all prior checkpoints and raw local outputs remain.
Actual training: 91.58 minutes and 16.382 simulated hours; whole stage: 2.746 hours.
Remaining six estimate: 4.28 additional hours, planning range 3.42–6.42, with controls
reused. No seeds 29/43, final familiar/patient matches, merge or production deploy.

**Wait for new budget approval before remaining six. Phase 6.1 is not accepted.**
Task behavior is useful; voting still needs scrutiny: full never skips and made
8 wrong votes, while ablations mostly abstain. Five innocent team ejections in
full matches do not by themselves attribute each ejection to the focal policy.
Do not claim replicated memory benefits from one initialization. Future seed
support requires a reviewed runner extension; the current CLI deliberately fixes
seed 17. Keep recorded sources intact or preserve a compatible source snapshot.
The website draft labels 6.1 in progress; 6.2–6.4 remain planned. Historical scope
limits below are superseded only by the explicit three-model staged approval.

## CURRENT — Phase 6.0 foundations, 2026-10-10

Baseline main SHA: `6b78b08e41d0791e49ff486b996d7490666db970`.
Verified research commit: `9c4e8cdc7167910ba9a774dbb18c380cdab5e348`.
Branch: `feat/phase6-vision-crewmate-foundation`. Development uses the existing
clean managed worktree; original untracked materials remain untouched.

Phase 5 is complete. Phase 6 is in progress, not complete. Opt-in
`GameConfig(rules_version=6, crewmate_sight_range=4.5, impostor_sight_range=6.75)`
changes actual LOS-projected player/body evidence and event-time elimination
witnesses. Walls/props block both roles. Legacy defaults, packet shape/config
serialization and seed-7 replay hashes are preserved; model/map releases are
byte-identical. Phase 6 spectator outlines are wall-clipped and privileged.

`phase6/` reuses the candidate scorer, PPO update and physics. It provides three
memory/belief conditions, 12/24-second options, Euclidean/route descriptors,
lr/rollout configuration, independent fresh partitions, runtime/source hashes,
and immutable output directories. Route distances are computed only at neural
decision boundaries after a costly intermediate-packet diagnostic was stopped.
Short smoke runs changed policy weights and preserved frozen belief weights.
Paired development records repeat exactly; these are infrastructure checks,
not a new strategic-performance claim. Earlier smoke artifacts are preserved
with their pre-optimization source hashes; they are not deployable under changed
runtime hashes and must not be used for selection. Do not erase or automatically
retrain them. The current route-boundary smoke uses the final runtime sources.

`python run_phase6.py` shows the frozen Phase 5 policy under new rules with four
scripted opponents. Existing Phase 4/5 viewers keep their defaults. Phase 6
impostor training, alternating learning and self-play remain unimplemented.

Read `docs/phase6_experiment_plan.md`, `docs/phase6_vision_and_rules.md` and
`docs/benchmarks/phase6/foundation/README.md` for exact checks, source records,
limitations and the nine-model Phase 6.1 compute proposal. The trainer deliberately
rejects >512 decisions; evaluation exposes only development seeds. Do not launch
a large batch or open validation/final partitions without new budget approval.
Build a reviewed batch/selection runner after approval, lock source/config/model
selection before final tests, and compare all baselines under identical new rules.

Verification: **416 passed**, zero skips (44 new tests); one existing Gymnasium
render-spec warning. Three 256-decision smoke runs and 28 development match
executions completed; repeated records are identical, with zero illegal/navigation
failures. No reserved validation/final matches ran. See the curated results JSON.

Changes belong in a PR against main. Do not merge this PR or publish experimental
claims directly. The website draft adds internal checkpoints in the existing
Phase 6 mission, preserving Phase 5 figures and collaborator navigation/colors.

## Historical handoffs — everything below predates Phase 6

The older “CURRENT” and running-session headings below are preserved history;
they do not describe active jobs or authorize restarting experiments.

## CURRENT documentation update (2026-10-04)

The owner now explicitly wants the teammate document in the repository,
superseding earlier external-only instructions. The canonical concise guide is
docs/phase5_teammate_guide.md, linked from README, with repo-relative pictures and
graph links. It explains the Phase5 fixed focal belief panel (no switch key;
F is Phase4 only), results/limits, experiment settings/timings and a proposed
Phase6 tuning and multi-agent sequence. No policy, training or viewer behavior
is changed by this documentation update. The external download mirrors the guide.

## CURRENT: Phase 5 completed and published (2026-10-04)

All approved Phase5 work is complete. PR #2 merged at commit
4212d05e4b282cc07d0546f8508ba1066eb1cb7e. GitHub Pages build/deploy run
37165652825 succeeded. Live site: https://hadisovic.github.io/social-deduction-ai/.
The published mission shows88.4%/90.4% crew wins with twelve evidence figures;
Phase6 remains future work. Red/black collaborator corrections are preserved.
Local main contains the merged release.372 tests passed; desktop/mobile browser,
palette, source/build validators and release viewer launch were checked.

The external Phase_5_Teammate_Handoff.md and Phase_5_Teammate_Package.zip remain
outside this repository. They include all experiment stages, counts, settings,
timings with caveats, Intel i9-14900HX hardware and worker details, result tables,
twelve graphs, learning assessment and the gated Phase6 sequence. Evaluation took
about3h05m elapsed. No further training or final-test selection changes occurred.
The completion heartbeat is paused after live verification. Historical entries
below describe earlier states and do not authorize restarting finished workers.

## CURRENT: final results complete; publication being verified (2026-10-04)

Session65107 exited successfully: 9,000 final matches, 90/90 chunks, nine methods.
Locked seed17/8192 wins 88.4% ID and 90.4% patient; own tasks1.67/1.68; vote
accuracy94.2%/95.7%, cast486/484. Clear gains versus idle/random, no established
superiority over scripted. No final-test selection changes. Phase5 simulator
study is accepted; real-game transfer and replicated memory benefit are unproven.
Release copied byte-for-byte to artifacts/phase5/release/policy.pt with SHA256
metadata. run_phase5.py defaults to it; dummy-video launch passed.

372 Python tests passed (203.73s, three warnings). Twelve report/public graphs
generated; Node source/build validators, palette tests and syntax passed.
External teammate document now has full ledger/timings/hardware and all result
tables. Phase_5_Teammate_Package.zip beside it includes twelve graph PNGs and
execution-metadata.json. Training sum129.4min (overlapping runs); final evaluation
about185.0min elapsed,721.8 summed worker-job minutes. Overall substantive study
span about4.67hours including reviews/gaps. Stage spans are filesystem estimates.

Remaining: finish browser verification at desktop/mobile, review/stage/commit
authorized changes, PR checks/merge and verify live GitHub Pages deployment.
No live publication claimed yet. Preserve red/black palette/animation fixes.
Automation must remain active until publication and deliverables are verified.

## CURRENT: locked final evaluation running (2026-10-04)

**Latest owner documentation requirement:** after completion, expand the external
teammate document with ALL training, validation, testing and evaluation details.
Include seeds, steps, checkpoint selection, data partitions, methods/controls,
metrics and uncertainty, measured duration per run/stage and total elapsed time,
plus processor/hardware and process/worker setup. Distinguish summed worker/run
time from elapsed wall time when jobs overlap. Recover timings from persisted
records where possible; label estimates and unavailable timings explicitly.
Explain the experimental process and the final learning-sufficiency decision.

Validation session22518 completed successfully. `phase5_analysis.py` was run
without --lock, the curves/paired comparisons/tasks/votes were inspected, and
`selection-decision.md` records the decision before final testing. No additional
training is justified in this batch. Selected full-memory seed17 checkpoint8192:
96/100 validation wins, 1.69 own tasks, 93/93 non-skip votes correct. Scripted92%,
idle50%, random42%. Paired versus scripted +4 points [0,+9]: no clear superiority.
Three full-memory seeds demonstrate useful task behavior; one ablation is exploratory.

`python phase5_analysis.py --lock` created the immutable selection lock. DO NOT
rerun selection, train/tune using final outcomes, or change the selected checkpoint.
The checkpoint curve plot now uses the SAME20 diagnostic seeds for every point;
the larger-sample comparison remains separate. No training/evaluation code changed.

**RUNNING session65107:** `python -u tools/evaluate_phase5_locked.py`, escalated
after Windows blocked process-pool pipes. Four local CPU workers evaluate NINE
methods x1,000 matches =9,000 matches, half ID/half patient. Preserve this healthy
job and inspect persisted `expanded/final-*/matches.jsonl` progress. Do not modify
source modules loaded by its workers or start duplicate evaluation jobs. Selection
is locked and the final test partition is now open. Partial outcomes are not final.

After completion inspect `final-results.json`, refresh report/figures and the
external teammate file, and assess sufficiency honestly. The report's pre-final
status detection only notices completed chunk metrics; if refreshing it mid-run,
correct any stale "unopened" wording in the documents (do not alter worker code).
After workers exit, improve that detection if needed. Update the teammate status
header and any stale pending-validation prose as well as its measured block.

Then export final graphs/story with `tools/export_phase5_site.py`, verify the
website and finish the existing GitHub Pages publication flow. Also integrate
the locked selected checkpoint as the viewer's default release after final review;
it currently defaults to the earlier smoke policy. Preserve all other artifacts.
The teammate document stays OUTSIDE the repository at the path below and includes
the gated Phase6 list. No final results or live publication have been claimed.
Keep the heartbeat active until all requested work is actually done.

## Expanded training authorization (2026-10-03, CURRENT)

**Current execution status:** all five substantive runs are complete: preserved
v1 seed 17 at 32,768 decisions, and V2 seeds 17/29/43 plus current-only seed 17 at
8,192 decisions each. V2 runs took 22.7–23.4 minutes apiece (replicates overlapped).
All four V2 checkpoint series have 20-match diagnostic validation. Seed 17 reached
100% wins at 6,144 and 8,192 decisions; other full seeds reached 95%. These are
reused small validation sets, not final evidence. Untrained initialization controls
for seeds 17/29/43 scored 50%/35%/50% wins and zero own tasks on those 20 seeds.

**RUNNING:** `tools/validate_phase5_replicates.py` session 22518 (escalated because
Windows denied process-pool pipes in the sandbox). It now evaluates 100 validation
matches for each screened full/current checkpoint, v1, and random/scripted/idle.
Its earlier checkpoint jobs are finished. Do not start another process against
the same evaluation directories. Training sessions and diagnostic/initial-control
workers have completed successfully. Preview server remains session 88477, port4175.

**Next sequence:** after session22518 completes, run `python phase5_analysis.py`
and inspect `validation-review.json`, curves and behavior. Decide from validation
whether a further training experiment is justified; do not blindly add compute
after a plateau. If selection is ready, run `python phase5_analysis.py --lock`.
Then `python -u tools/evaluate_phase5_locked.py` runs 1,000 final matches per method
(500 ID/500 patient), using four workers and disjoint 100-match chunks. This new
script may also need escalation for Windows process-pool pipes. It refuses missing
selection locks or changed checkpoint digests and checks the complete seed manifest.
It does NOT automatically start further learning. Final outcomes cannot tune selection.

After final results, regenerate `python tools/update_phase5_teammate.py "<external
teammate document path shown below>"`, inspect
the figures and write an honest learning-sufficiency assessment. The current-only
ablation has ONE seed; do not claim replicated memory benefit. Run
`python tools/export_phase5_site.py` to copy sanitized final graphs and populate
`project-status.json.phase5Study`. This exporter refuses unfinished final testing.
It does not publish. Website runtime/validator already support Phase 5 in-progress
and an expandable evidence gallery; the draft contains no old UI repair history.
Build/validate, check the browser, then finish the existing GitHub publication flow.

Current report: `docs/benchmarks/phase5/expanded/README.md` (nine figures before
final tests, twelve afterward). Full Python suite passed365 tests; seven additional
chunk/whole-match-statistics checks passed afterward. Node status/assets/palette
checks passed. The external teammate document is refreshed by the helper above;
it contains the requested gated Phase 6 list and current measured results.

The teammate helper requires the existing external document as a positional
argument; no machine-specific output path is embedded in the helper source.

The owner's latest request supersedes the earlier one-run limit: train further
as evidence warrants, create meaningful NN evaluation graphs, update the external
teammate document, and add a suggested Phase 6 list. Multi-agent learning/self-play
must wait until one learned crewmate works. No multi-agent implementation is authorized.

Preserve baseline `artifacts/phase5/runs/full-17-approved` (32,768 decisions).
The automatic `tools/complete_phase5_single.py` evaluator was STOPPED before any
validation/final matches; do not restart it while model selection is ongoing.
It would prematurely reveal the final holdout.

Baseline training showed no own tasks through 269 completed matches. A separate
`phase5_options.py` experiment preserves v1 and runs chosen travel/task actions
up to 24s (follow/flee up to 3s), interrupting on actor-visible events or completion.
It composes original physics/rewards/discounts and never chooses a social action.
An equivalence test passed. Seed-17 options training is running with 8,192 decisions
in `artifacts/phase5/runs/options-17-v2`, checkpoints every 2,048 decisions.
`phase5_options_training.py` records policy/value losses, entropy, KL, clipping,
explained variance, decisions and simulated seconds. Different option lengths
mean equal decision counts are not matched simulation budgets.

Next: validate before extending, replicate a useful approach on seeds 29/43,
compare a separately trained current-only ablation, lock selection on validation,
then open final ID/patient tests. Do not tune using final outcomes. Baselines
include idle to expose wins carried by scripted teammates. The updated evaluator
and viewer dispatch by checkpoint execution protocol and check its source hash.

The downloadable handoff remains outside the repository at
`C:\Users\masaa\.codex\visualizations\2026\10\02\01a0fc9e-88fa-7ce2-aa36-8902b0b5fd96\Phase_5_Teammate_Handoff.md`.
Its Phase 6 suggestions are explicitly gated. Update measured results after runs.
Heartbeat `finish-phase-5-training-results-and-website` has the expanded prompt.
Keep the public Phase 5 story about learned strategy/results only, not earlier
UI/map/player repairs. Final website implementation/publication is still pending.

## Authorized single-run experiment (2026-10-03, running)

The owner approved **one full-memory run, seed 17, 32,768 transitions**, followed
by graphs, outcome documentation, a judgment about further learning, and a website
Phase 5 update. Do not start the six-run protocol or another training run.
The requested website Phase 5 story must discuss strategy/learning/results only;
keep earlier map/UI/avatar repairs out of that phase's content. The website's two
collaborator characters should use the corrected palettes and walking animation.

Training is `train_phase5.py --steps 32768 --seed 17 --output
artifacts/phase5/runs/full-17-approved`. Progress is persisted in `updates.jsonl`
and `episodes.jsonl`; intermediate weights are saved every 8,192 transitions.
`tools/complete_phase5_single.py` waits for completion, evaluates 100 validation
matches per method, checks three intermediate checkpoints on the same 20 validation
seeds, records the learning assessment, then evaluates 200 final matches per method
(100 ID/100 patient). Methods are learned, original scripted and random-legal.
It generates six scientific figures and `docs/benchmarks/phase5/single-seed-17/README.md`.
Only a single initialization seed is evaluated; do not claim a multi-seed result.

The website character code now decodes the existing RGB frames at runtime and
animates both collaborators. Original PNGs are untouched. Node palette tests and
a local browser travel check passed. The public Phase 5 story/status/validator still
need updating from actual measured outcomes after evaluation completes.

Never rerun training automatically if a later plotting, evaluation or website step
fails. Preserve the checkpoint and repair the later stage. Final website publication
must be verified; do not claim the local preview is the live GitHub Pages deployment.

## Phase 5 implementation update (2026-10-03)

The owner subsequently authorized starting Phase 5 and then requested approval
before any longer training run, with an estimated duration. Two 512-transition
smoke models already exist under `artifacts/phase5/quick/`; no extended training
has run. `python run_phase5.py` loads the smoke focal-crewmate policy.

New Phase 5 modules provide actor-only feature/candidate construction, a masked
shared-candidate actor-critic, frozen Phase 4 inference, time-aware PPO, original
scripted opponents, paired evaluation and a spectator policy panel. The full
runner trains three full-memory and three current-only models, ranks full models
on reserved validation seeds, and only then evaluates final ID/patient seeds.
This is a serial reference implementation; strategic improvement is unproven.

Conservative collision buffers now cover the true radius-0.22 circle. The source
blueprint is byte-identical. Both 4,914-route benchmarks pass at dt=1/30 and .2,
with zero collisions, blocked steps or replans; eight targeted corridor tests pass.
`clearance_policy='legacy_polygon'` preserves historical replay geometry.

The red sprite source was an RGB material mask. Five other colors aliased every
walk frame to a still pose; they now use recolored complete mask animations.
Meeting banners show the public reporter/victim/room or emergency caller.
Phase 5 display interpolation uses recorded physical steps without changing rules.

The original smoke checkpoint predates the final room-action descriptor correction.
It remains a UI/pipeline demo only; fresh training uses public room coordinates and
records the feature-source hash. Fourteen rooms are strategic room destinations;
seven hallway regions remain traversable. Preserve historical smoke reports.
The owner's teammate handoff is outside the repository, as requested.

Final verification: **362 tests passed**, three dependency/Gym warnings;
`docs/benchmarks/phase5/pytest.xml`. The reviewed 16-match smoke evaluation in
`docs/benchmarks/phase5/smoke-reviewed/` recorded zero illegal actions and zero
navigation failures. Headless runs use SDL video/audio dummy drivers.

## Historical rendering repair and Phase 5 specification review (2026-10-03)

Starting review HEAD: `172f648` on `main`. The owner authorized rendering repairs
and an expanded specification, not Phase 5 training. Current UI changes improve
offline half-body sprites, restrict body circles to debug mode, increase small
avatar visibility, fix selected-monitor sizing and add an actor-only bottom HUD.
The missing pinned local artwork pack was installed (119 verified PNGs, still
ignored). Run `python run_phase4.py`; optional `--display 0 --window-scale 1`.

Validation: **334 tests passed**, including seven new UI checks; three existing
dependency/Gym warnings. Source geometry and Phase 4 weights are unchanged.
`docs/phase5_strategic_policy_specification.md` preserves the pasted original and
appends the corrected implementation checklist. Do not implement the original
43-feature/12-action draft without reading the amendment: live team task totals,
hidden live-player counts, candidate coverage and belief-threshold masks conflict
with the existing contract or learned-decision objective.

`tools/audit_phase5_corridors.py` / `docs/phase5_corridor_audit.json` show eight
successful collision-free runs, but exact circular clearance is slightly violated
by polygon-buffer corner approximation on one O2 route (~0.00018831 units).
The stricter audit exits 1 deliberately. This remains a Phase 5.0 prerequisite;
no corridor widening, player-radius reduction or map edits were made.

## Current state (2026-10-03)

- Repository: `Hadisovic/social-deduction-ai`, branch `main`.
- Phase 1/2/3 remain complete and preserved. Phase 3 baseline revision:
  `e8c5d664a3f17684c3fd844a4ef27ad089b65b12`; Phase 4 milestone commits:
  `68780df` (memory/data) and `e16d8ea` (training/evaluation/observer).
- **Phase 4 is complete** at implementation/results revision
  `fdc9f4401b781f64e3e79a2cd0bb3a80bcb515e8`. This handoff is a documentation-only
  follow-up; resolve its exact revision with `git log -1 --format=%H`.
- Final checks: full pytest suite (327 tests), five independent dataset/feature
  replays, selected-model exact retraining, final evaluation, and live rendered
  matches. The final push was to `origin/main`; see the repository history.

## Phase 4 outcome

The phase adds bounded typed event memory, permutation-safe 22-feature candidate
vectors, a calibrated set-based impostor-belief model, seeded scripted study data,
leakage/identity audits, retrained ablations, evaluation, and an interactive
belief observer. It does not add a strategic policy: task selection, reporting,
accusations and voting remain scripted.

The experiment contains 1,400 complete matches and 64,097 samples, with match-level
train/validation/ID-test/patient-family splits. Final calibrated model accuracy is
70.54% (NLL .5955) on ID and 67.41% (NLL .6375) on the held-out patient family.
These results establish performance only on the controlled five-player scripted
distribution. The current-only ablation was corrected to retain the public live
meeting roster; its NLL is 1.2455 / 1.2546. See
`docs/benchmarks/phase4/README.md` and its linked machine-readable records.

The release checkpoint is `artifacts/phase4/belief.pt`. Run the UI with
`python run_phase4.py`, regenerate data/train with `python train_phase4.py`, and
evaluate with `python evaluate_phase4.py`. Generated shards and run intermediates
are ignored; a fresh training run regenerates them. The small selected checkpoints,
selection record, benchmark report and demo captures are committed.

## Architecture and preserved contracts

`ActorMemory` accepts only typed actor observations and maintains bounded causal
events. `belief/features.py` derives candidate-relative features. `BeliefModel`
returns a distribution over legal crew candidates; it receives no role labels,
truth snapshots, match seed or hidden state. Identity/color permutation and temporal
boundary tests protect the interface. The observer reads terminal own-alive status
only for the UI and does not update model memory from it.

Phase 1/2/3 gameplay, scripts, navigation and map assets were not modified by the
Phase 4 implementation. Phase 3's 291-test contract remains in force. See
`docs/PROJECT_PROGRESS.md`, `docs/project_architecture.md`, `docs/belief_model.md`,
and `docs/phase4_design.md` for project state and technical details.

## Limits and next phase

No Phase 5 policy or win-rate claim is implemented. The next research step should
connect the belief interface to a policy, define a policy-level baseline and
pre-register match-level evaluation before tuning. Current limits include five
players, one impostor, structured claims, a static map, no sabotage/vents/free-form
language, and narrow scripted opponents. Near-perfect late meeting accuracy is a
property of this controlled script distribution, not evidence of robust human
deception reasoning.
