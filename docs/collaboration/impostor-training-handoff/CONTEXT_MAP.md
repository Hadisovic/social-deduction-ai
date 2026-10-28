# Packaging context and source review

Publication starts from `origin/main` at `a463a427db6cd3bda47707c4c398486ed04cfa0c`.
Selected, dependency-complete sources are copied from crewmate Phase 2.1
`4bac4cdb099e81a5c6c7de982f423136f24213c5`; no research branch is merged.
The trained checkpoint source is four-crew registration
`fab5c70e63e54d4d071d3343b1e0463fa4fbb6b6`. Strict source locks pass without edits
to policy, environment, preprocessing, reward or provenance modules.

| Component | Authoritative source / boundary |
|---|---|
| `crewmate_v2/`, `phase3_engine.py`, geometry/navigation | Latest compatible Phase 2.1 tree, preserving baseline and opt-in modules |
| `artifacts/crew-handoff/` | Exact selected original/shaped checkpoint bytes |
| `belief/`, Phase 4 belief, Phase 5 release | Required frozen/source-lock dependencies; existing main bytes retained |
| `spectator/`, `run_spectator_v2.py` | Original `_spectator_v2` sources, not Patrick Jane's copy; per-file hashes record uncommitted V2 source |
| `assets/phase3_local/` | Original 120 pinned image files from the approved viewer |
| `frozen_handoff/` | New controller boundary, frozen runner and packaging verifier |
| `docs/research/ISSUES.md` | Preserve R001–R029; append handoff issue R031 |

Reviewed risks: checkpoints bind exact Python/NumPy/Torch versions; external
impostor feature/action encoders are not interchangeable with this API; optional
social guards change crew behavior; the old ballot ledger has a known same-step
ejection accounting defect. Default inference preserves historical behavior and
accounting. A separately selected pre-step ledger changes accounting only.

New tests cover strict loading, isolation, legal-controller validation, baseline
equivalence, independent memories, optional accounting and viewer packaging.
Verification also uses an actual complete Model B match, recording/repeat checks,
clean-clone execution and real spectator rendering. Review complete; no new neural
architecture, training or reward coefficients are introduced.
