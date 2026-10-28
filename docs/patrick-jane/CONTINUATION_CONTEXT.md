# Approved continuation — 2026-10-28

The owner approved the design in `CONTINUATION_SPEC.md`: preserve the current
network, navigation decisions, reward coefficients and PPO algorithm; correct
confirmed Classic mechanics in the isolated Patrick Jane implementation; extend
the existing spectator; measure 8/16/32 workers, then run an approximately 250,000
individual-decision pilot after readiness passes. No historical model or partner
branch changes. This approval supersedes additional design-approval ceremonies.

## Context map

| Files | Purpose and intended changes |
|---|---|
| `patrick_jane/config.py`, `environment.py` | Versioned Classic lobby, emergency timers, sabotage and meeting corrections |
| `patrick_jane/api.py`, `features.py` | Legitimate own timer and repair-hold observations within existing tensor widths |
| `patrick_jane/worker.py`, `collector.py`, `training.py`, `telemetry.py` | Real instance monitoring, measured scheduling, bounded training and evaluation |
| `patrick_jane/ppo.py`, `network.py` | Profile existing recurrent updates; optimize only with equivalence evidence |
| `spectator/v2/*`, `run_spectator_v2.py` | Extend existing independent renderer, real streams and training monitor |
| `tests/patrick_jane/*` | Mechanics, privacy, source locks, likelihoods, save/reload, renderer compatibility |
| `docs/research/ISSUES.md`, `docs/patrick-jane/*` | Evidence, uncertainty, benchmarks and staged pilot registration |

Dependencies `crewmate_v2`, `phase3_engine`, `social_deduction`, real Skeld geometry
and navigation service remain unchanged. Their implementation patterns are reused
through PJ subclasses; old source/checkpoint compatibility remains a boundary.

## Design and sequence

1. Preserve the prototype in local commit `2e0991757f364d763741113eba6ce9286ce2055b`.
2. Keep the persistent desktop spectator open on genuine engineering recordings.
3. Separate confirmed setting corrections from unsupported retail approximations;
   test role-local observations and event ordering before new performance runs.
4. Add observer-only throttled snapshots and a real instance list to the existing
   viewer. Workers never consume spectator truth or depend on rendering.
5. Benchmark matched complete-game workloads at 8/16/32 workers; verify a full
   GPU update, save/reload and historical replay compatibility. Preserve failed
   benchmarks, negative results and their source locks.
6. Register seeds, compute ceiling, complete-match overshoot and held-out
   evaluations from measured costs. Launch only the approved pilot, preserving
   latest and champion separately. Stop on non-finite values, crashes, privacy
   violations, exploitation or serious regressions.

The narrow extension of existing components is preferred over a replacement
viewer or architecture redesign: it preserves compatibility and keeps performance
comparisons interpretable. More complex asynchronous scheduling is conditional
on a measured bottleneck and semantic-equivalence tests.

## Review

No architecture/reward change, production deployment, publication, final-test
access, or partner-branch modification is included. New mechanics need a new
source lock. Current 2026 numerical preset defaults remain uncertain; the owner
supplied historical baseline is explicit rather than represented as verified
retail defaults. Full pilot readiness is pending, not established by short tests.
