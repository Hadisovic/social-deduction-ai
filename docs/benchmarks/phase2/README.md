# Phase 2 physical navigation benchmark

4914/4914 successful (100.00%).
Gate: **PASS**. Seed: `20261002`.

| Suite | Successes | Cases | Rate |
|---|---:|---:|---:|
| pairs | 3906 | 3906 | 100.00% |
| random | 1008 | 1008 | 100.00% |

Collisions: 0; blocked steps: 0; replans: 0.
Failures: `{}`.
Measured run time (map/planner setup, generation, execution): 826.13 s.
Planning mean / p95: 63.84 / 206.63 ms.
Controller / physics: 179.71 / 90.60 microseconds per executed step.

Every step is speed-bounded, swept-clear and applied by `AmongUsMapEnv.advance_motion`.
Path efficiency is planned executable length / actual travel; it is not a global optimality claim.
See `summary.json` for region groups, longest routes, configuration, source hashes and timing distributions.
See `cases.csv` for all individual outcomes. Wall times vary across machines.
