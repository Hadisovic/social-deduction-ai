# Reporting-diversity diagnostic registration

2026-10-27. Start from `cad6c34` on `codex/crewmate-v2-pretraining-diagnostic`.
Preserve that study, its 22 raw files, PR #9 and all canonical sources/checkpoints.
Development-only; no neural training, reward changes, final-test use, merge or deployment.

## Question and design

Separate (a) seeing a body, (b) being legally in reporting range, (c) submitting a
report, and (d) having it accepted. Impostors naturally have a proximity advantage
after their own kills. A high accepted-report fraction alone does not establish a
role-dependent reporting preference.

Reuse the eight registered development scenarios **2,630,000–2,630,007**, ordinary
A and deduction-only B, paired with the exact archived baseline. The previous
staggered two-scenario probe is background only, not the primary comparator.
Fresh fixtures use 2,630,100–2,630,109. No new research/final partition is sampled.

Execute one new, preselected **diverse reporting population** in both A/B conditions
on all eight scenarios (16 games). Each actor independently chooses one of three
styles with equal construction probability, using a private domain-separated RNG
seed before any role observation: prompt; inspect for six seconds after first
personal body sight; bypass (never approach/report bodies). The styles apply to
both roles. Inspectors may approach a currently visible body immediately, cancel
their own navigation once in reporting range, wait legally, then report. All other
task, navigation, killing, claim and vote behavior remains the original scripts.
The base impostor's kill priority remains. No new sightings, direct evidence,
claims, votes or role information are manufactured. Original impostor deception
remains structured hearsay under existing rules, not authenticated direct evidence.

This fixed population is a diagnostic perturbation, **not a selected training
mixture**. Styles are independent of role by construction; finite-sample role
correlations and behavioral shortcuts must still be measured. Do not adjust style
probabilities or choose different seeds after seeing results.

One original baseline B rerun of the first scenario must reproduce the archived
actor/public-extra/truth hashes and metrics. One diverse A rerun checks deterministic
replay. Maximum **18 fresh executions**; the 16 archived baseline games are reused,
not counted as new executions or new independent scenarios. Alternate A/B order.

Why this design: rerunning a full 32-game factorial would likely exceed 15 minutes.
Reusing hash-verified baseline evidence permits a matched eight-scenario comparison
under both victory conditions. The old baseline lacks some fine-grained latency
records; mark those unavailable instead of inventing precision. Its recorded legal
windows equal report submissions by each role, permitting a zero scheduled
opportunity-to-submission latency inference. New data records full timing/censoring.

## Measurements and shortcut tests

Record actor-body first/last visibility, visible windows, first/last legal REPORT,
legal body-action windows and actor windows, first submission, and accepted report.
Keep reporting opportunities separate from approach actions. Distinguish delays
from visibility versus from legal range; record observations censored by other
reports, death or match termination instead of substituting zero. Report rates
per distinct visible/legal actor-body pair and per legal actor-window, plus
acceptance conditional on submission and simultaneous-arbitration losses.

Keep private per-actor records locally. Attribute roles only in the offline scorer;
controllers never receive scoring labels, evaluator objects or other agents' memory.
Inspect direct crew witnesses, body discovery, claims, unique legal crew ballots,
correct/wrong/explicit/automatic skips and evidence-bearing ballot opportunities.
Split first crew exposure before/after the full task bar; count whether evidence
is available before the paired ordinary game ends. Task/ghost contribution, outcomes,
timing and forced decision windows use the previous measurement definitions.

Evaluate two fixed **reporter-identity-only** top-1 heuristics at each crew's first
legal ballot per meeting, using only current public participants and authenticated
current-meeting reporter identity, excluding the focal actor:

1. Accuse the reporter if a candidate, otherwise uniformly choose a candidate.
2. Avoid the reporter and uniformly choose among remaining candidates; use uniform
   fallback when necessary.

Score tie-averaged expected accuracy, not lucky RNG draws; compare uniform guessing
on the identical candidate sets. Report event-level P(impostor | reporter) separately.
Also use leave-one-scenario-out selection between the two directions: A/B records
of a scenario remain in the same fold. This is an offline shortcut assay, never
an agent policy or neural training. Do not report the better in-sample direction
as generalization. Report event-weighted and scenario-balanced estimates, small
sample limitations and opposite-direction shortcuts if reporting flips to crew.

## Budget and fail-safes

**900 wall seconds maximum in the new diagnostic process**, including initialization,
hashing, saved evidence, baseline check and replay. No concurrent training or benchmarks.
Save each completed match immutably. Check deadline every physics tick; save any
incomplete match privately and exclude it from completed-match denominators. Stop
early on source drift, illegal actions/navigation failures, hidden-mode leakage,
fabricated-action/evidence failures or replay divergence. No unbounded extension.
Incomplete A/B pairs must be marked; do not silently change the registered sample.

## Context map and reviewed implementation

| New files | Responsibility |
|---|---|
| `tools/reporting_diversity_v2.py` | observation-only styles, bounded runner, opportunity and shortcut scorer |
| `tests/test_reporting_diversity_v2.py` | legality, hidden-secret noninterference, style independence, denominators, heuristics/replay |
| `docs/crewmate-v2-reporting-diversity/*` | registration, concise report, aggregate/validation evidence |

Read-only dependencies: old diagnostic runner/aggregate/private replays; V2 runtime,
engine/config/provenance; original scripts/actor API/runner. No existing file is
modified. New outputs use a fresh ignored `temp_v2_reporting_diversity_run/` directory.
Record old raw hashes and all nine checkpoint hashes before/after. Publish nothing.

Design review: comparing only deduction mode would hide task-termination effects;
hand-assigning reporting styles by hidden role would invite a new shortcut. The chosen
factorial with role-independent styles addresses both within a bounded budget.
The user explicitly authorizes isolated diagnostic scripts and design judgment.
Self-review: deterministic seeds/styles, no adaptive tuning, private scoring boundaries,
valid opportunity denominators and separate approval before any next-stage training.
