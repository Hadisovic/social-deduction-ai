# Social Deduction AI

A research simulator for an Among Us crewmate that completes tasks, remembers
legitimate observations and learns when to report or vote. Phase 6 is **in progress**;
Phase 6.1 acceptance remains pending.

## Current crewmate research

Nine historical Phase 6 policies have validation-selected checkpoints. Their simulator,
results and checkpoints are preserved. They are not input-compatible with Crewmate V2.
The shared frozen Phase 4 belief predictor is separate from these learned action policies.

Crewmate V2 adds owned, staged tasks, automatic task-only ghosts and explicit lobby
settings. The [V2 guide](docs/crewmate-v2/README.md) explains the implemented mechanics.
The [pre-training diagnostic](docs/crewmate-v2-diagnostic/REPORT.md) and
[reporting-diversity study](docs/crewmate-v2-reporting-diversity/REPORT.md) found scarce
crew evidence opportunities and a strong reporter-identity shortcut. These exploratory
development results do not establish reliable deduction.

The opt-in [reward/PPO pilot](docs/crewmate-v2-learning/PILOT.md) has infrastructure
verification only. Full seed-17 training and seed-29 replication require separate owner
approval. Cleanup and [performance preparation](docs/crewmate-v2-performance/CONTEXT.md)
are isolated from the impostor and spectator branches. Future self-play and larger learned
populations remain planned. Historical tests and benchmarks describe their recorded source
versions; they are not current V2 training results.

## Project structure

| Location | Purpose |
|---|---|
| `crewmate_v2/` | Opt-in V2 mechanics, observations and strategic environment |
| `crewmate_v2/learning/` | Experimental rewards, new masked policy, approval-gated PPO |
| `configs/` | Versioned lobby, reward and pilot proposals |
| `belief/`, `social_deduction/`, `phase6/` | Preserved actor boundaries and historical research |
| `among_us_map.py`, `navigation_service.py` | Shared map, collision clearance and physical navigation |
| `*_renderer.py`, `spectator_display.py`, `assets/` | Shared rendering and source-backed assets; keep interfaces stable |
| `tests/`, root `test_*.py` | Current and historical regression checks |
| `tools/` | Reproducible engineering and research utilities |
| `artifacts/` | Frozen releases and historical checkpoints; keep existing locations |
| `docs/` | Research protocols, curated measurements, architecture and provenance |
| `archive/` | Historical documentation and index of retained legacy experiments |
| `local-runs/` | Ignored engineering outputs; use a fresh directory per run |
| `site/` | Public project explanation; no research data updates implied by local work |

Standalone navigation experiments remain at their existing import/CLI paths for
compatibility. The [archive index](archive/README.md) explains why they were retained.
No large model or raw trace archive was duplicated or moved.

## Setup and safe entry points

Use Python 3.12 and the validated Windows dependency stack:

```powershell
python -m venv .venv
.venv/Scripts/activate
python -m pip install -r requirements-dev.txt
python -m crewmate_v2.learning.train --help
python run_phase6.py --help
```

The existing Phase 5/6 viewers load historical models. They are not V2 policy viewers.
Keep the separate spectator development checkout independent; no renderer is required by
headless training. `python -m crewmate_v2.learning.train pilot` requires an explicit
hash-bound approval record and compatible engineering evidence. Do not start it merely
to check installation. Smoke and benchmark commands update disposable engineering policies
only; their protocols must remain bounded.

Relevant regression checks (no training or final partitions):

```powershell
python -m pytest tests/test_crewmate_v2.py tests/test_crewmate_v2_privacy.py tests/test_crewmate_v2_integrity.py tests/test_v2_learning_rewards.py tests/test_v2_learning_pipeline.py test_among_us_map.py test_navigation_service.py test_phase3_render.py -q --basetemp=local-runs/pytest-fresh
```

Read [actor information boundaries](docs/information_contract.md),
[navigation service](docs/navigation_service.md), [architecture](docs/project_architecture.md)
and [asset notices](assets/skeld/NOTICE.md) before changing shared infrastructure.
The [earlier README](archive/research/README-before-v2-organization.md) preserves the
navigation history, historical measurements and older reproduction commands. The latest
published roadmap on `main` is authoritative; this research branch does not replace it.
