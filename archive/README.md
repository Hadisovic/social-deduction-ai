# Retained research and legacy compatibility

The [previous README](research/README-before-v2-organization.md) is archived from
`8e99383` with relative links adjusted. Historical measurements remain attached to their
original studies in `docs/benchmarks/`; they have not been erased or reinterpreted.

The following old-looking code remains at its original paths. Static imports alone cannot
prove that a CLI is unused by another branch or a developer. Relocation would also risk
breaking reproduction commands and source-locked checkpoint loaders.

| Retained group | Reason |
|---|---|
| `train_stage1.py`, `train_stage2.py`, `evaluate_stage*.py`, `watch_agent.py` | Standalone Point-A/Point-B research entry points; preserve historical commands |
| `rl_environment.py`, `environment.py`, `geometry.py`, `player.py`, `observer.py` | Legacy environment/tests still import these; retain compatibility |
| `live_watch_training.py`, `live_watch_stage2.py`, `run_live_watch_quick_check.py` | Existing watch tests and manual workflows depend on them |
| `skeld_environment.py`, `skeld_navigation.py`, `skeld_config*.py`, `inspect_skeld.py` | Preserved earlier map inspector, training and geometry checks |
| `benchmark_navigation.py` | Imported by the current navigation regression suite |
| `among_us_map.py`, `among_us_map_simulation.py`, `navigation_service.py` | Active V2 map, collision, routes and physics infrastructure |
| `*_renderer.py`, `spectator_display.py`, `assets/`, `site/assets/` | Shared rendering, existing screenshots and spectator compatibility |
| `tools/phase5_*.py`, earlier train/evaluate runners | Historical source/reproduction dependencies; review individually before any future move |

No Python entry point was relocated or removed. No binary artifact was copied. Sixteen
small, local JUnit reports from our crewmate worktree were moved to ignored
`local-runs/engineering-tests/`; the [manifest](../docs/crewmate-v2-performance/cleanup-manifest.json)
records each original/new relative path and byte hash. Raw diagnostic traces and source-
specific smoke checkpoint directories stayed in place. Files with uncertain ownership
or spectator/impostor use were retained for later coordinated review.
