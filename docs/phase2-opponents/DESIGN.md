# Phase 2 design and context map — 2026-10-28

The owner authorizes implementation and short engineering diagnostics, not training. Base: research `fab5c70`, isolated branch `codex/phase2-opponent-diversity`. Spectator V2 is approved and stays unchanged in its existing worktree. Historical source locks, checkpoints, results, baseline scripts, canonical mechanics and partner work are preserved.

## Decision and alternatives

Add an opt-in actor-observation-only opponent package, a read-only historical social audit, and an observational engineering harness. Do not replace the registered controller or relax checkpoint loaders. A core-script rewrite would invalidate historical locks; a set of fixed always/never-report profiles would create new signatures. Instead, use overlapping probabilistic preferences with separate RNG streams, body-level report decisions, and time-bounded within-match mode changes. A mode is an internal script parameter, never a neural input or guaranteed behavioral label. Scripted claims may lie under existing legal mechanics; false claims remain hearsay, never fabricated direct evidence.

## Context map

| New file | Responsibility |
|---|---|
| `crewmate_v2/phase2/opponents.py` | Configurable strategy mixture; local observation input only; no engine, role table or other memory access |
| `crewmate_v2/phase2/social.py` | Meeting/owner-scoped historical defense/vote and belief/action analysis; hidden roles used only for offline scoring |
| `crewmate_v2/phase2/audit.py` | Observational report/kill opportunity meters, actor decision packets, accepted-event attribution and post-match scoring |
| `crewmate_v2/phase2/scenarios.py` | Legal scripted prefixes for reproducible diagnostic evidence cases; no teleport, role injection or learned-action forcing |
| `crewmate_v2/phase2/run.py` | Bounded comparison, strict checkpoint loads, immutable output and replay creation; no optimizer |
| `tests/test_phase2_*.py` | Local-input isolation, legal actions, streams, evidence cases, historical reconstruction, noninterference |
| `docs/phase2-opponents/` | Specification, measured aggregate results, limitations and handoff |
| `docs/research/ISSUES.md` | Preserve R001–R020 and append confirmed Phase 2 findings |

Dependencies remain unchanged: `FourCrewMatch`, `ScenarioMatch`, `ActorView`, `FourCrewPolicy.load`, `ReporterController`, typed actor API and canonical engine. The harness substitutes only the impostor controller on a new match. It observes legal per-actor packets before actions and accepted events afterward. No auditor reference returns to a policy/controller. The established replay recorder and approved Spectator V2 read the resulting schema unchanged.

## Diagnostic plan and stop rules

1. Hash historical sources/artifacts, viewer implementation and protected branch refs. Audit all 96 existing completed recordings; separate 48 validation from 48 sampled training games and prefix events from neural play. Reconstruct each accepted defense followed by a vote, separating same/different actors and same/different meetings. Record probabilities and public changes, with private-memory gaps explicit. No model inference is needed for historical reconstruction.
2. Test twelve legal evidence situations using engineering seeds 2760070–2760079; reuse scenario families rather than pretending twelve fixtures are twelve independent natural games. Hidden truth configures roles/setup and scores results only. No outcome is imposed on a learned actor.
3. Compare original and mixture against frozen shaped Model B on eight matched engineering seeds 2760080–2760087. Contexts: four ordinary, two deduction-only natural starts, two legal body prefixes. Canonical lobby settings (including 900s cap) and rewards remain identical. Same policy initialization/sampling schedule; complete matches, no updates. This small opponent assay is not a new A/B learning study.
4. Keep visibility, report range, legal opportunity, decision opportunity, selected/submitted/accepted and delay denominators separate. Report kill legal windows/conversion, victim choices, navigation, claims/votes, outcomes, discovery, reporter-only macro identification, and role event shares. Profile diversity does not establish reduced exploitability or realism.
5. One process, deterministic algorithms, one Torch thread. Maximum 15 minutes for the paired execution; preserve partial outputs and report missing coverage if it cannot finish. No silent shrinking/retries/seeking better results. Historical analysis/tests are separate engineering costs. Output path must not exist; incomplete replays stay `.writing`.
6. Strict source-compatible model loads, before/after tensor hashes and historical byte manifests. Only new package/test/docs files are changed. No rewards, gameplay, actor features, PPO kernels or canonical files are modified. Final-test partitions stay sealed.

## Review

The brief resolves scope and approval: implement all three objectives and stop for review. No additional design decision needs owner input. Self-review: context separation, offline truth boundary, immutable outputs and bounded execution are explicit. Missing historical impostor opportunities cannot be recovered from map snapshots or guessed from cooldown; the new observational assay supplies those denominators prospectively. Social inconsistency is a diagnostic label, not strategic deception or a causal attribution. Future sabotage requires separate versioned legal actions, state, observations, cooldowns, physics, rewards review, replay fields and tests; none is implemented here.

## Evidence-probe clarification before the second probe

The first no-witness body probe produced 0.99983 impostor suspicion, so its absence of a direct witness does not establish weak evidence. Preserve that run as `evidence-probe-v1`. Before any further inference, define a third context: a legal emergency meeting with no kill, body, report or claim. No confidence-based scenario rejection or outcome selection. The second probe records all three histories (empty emergency, unwitnessed report, direct witness), including differing participants/legal candidates. This clarification does not change the completed sixteen-game opponent comparison, any opponent parameter or frozen actor source.
