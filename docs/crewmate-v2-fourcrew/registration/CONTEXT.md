# Final registration context map — 2026-10-27

Owner approved the overall design, registration, tests and one representative benchmark, **not full study execution**. This isolated worktree preserves `7b66989` and the original pilot branches/artifacts.

| Files | Responsibility |
|---|---|
| `fourcrew/registration.py` | Exact prospective seed schedule, budgets, quotas, stopping/approval contract |
| `fourcrew/scenarios.py` | Legal observed scripted prefixes; no teleportation/evidence injection; no prefix learning rewards |
| `fourcrew/execution.py` | Four independent headless jobs, actual replay metadata, bounded owned-process cleanup |
| `fourcrew/evaluation.py`, `analysis.py` | Actor-only forecasts followed by offline scoring; reporter-role controls, separate contexts and scenario-cluster uncertainty |
| `fourcrew/experiment.py` | Fresh paired initialization, immutable zero reference/checkpoints, common scenario batches, deadline and final-approval gate |
| `fourcrew/readiness.py` | One ten-minute development-only representative benchmark |
| `fourcrew/{config,match}.py` | Explicit development partitions and a read-only actor decision-audit callback; reward coefficients unchanged |
| `tests/test_v2_registration.py` | Prefix legality/coverage/privacy, matching/frozen initialization, budget rules and refusal without authorization |

Dependencies remain the existing V2.2 engine, five-player feature encoder, frozen Phase 4 predictor, segmented GAE/PPO kernels and separate viewer at `fa76497`. Existing kernels, rewards, simulator mechanics and spectator sources are not changed. New Python files enter the strict source lock; historical checkpoints stay usable from their preserved branch.

Map/design reviewed against the approved request before the representative benchmark. Eight engineering-prefix families passed; full-study partitions remain unopened. Final registration and feasibility assessment follow measured benchmark costs, without reward tuning or outcome-based checkpoint selection.
