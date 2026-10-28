# Publication verification — 2026-10-28

This is an inference/packaging handoff. No neural training, reserved final-test
evaluation, production change or merge was performed. Phase 6.1 remains pending.

## Reproducible construction

Publication is based on main `a463a427db6cd3bda47707c4c398486ed04cfa0c`, with
selective canonical/source-lock dependencies from `4bac4cdb099e81a5c6c7de982f423136f24213c5`.
The first packaging commit is `abc64a2339cdbee849bb36925082a2a114e11b3f`.
The clean-clone runtime candidate is `0835f2a73d79ef314a8035c2f25ce64003664961`;
subsequent changes add verification coverage, canonical text-hash portability and
this report, without changing frozen policy/engine behavior. The final immutable
publication SHA is supplied in the draft PR and owner-facing handoff.

A clean Git clone was created from the publication branch. Its own virtual
environment was built using CPython 3.12.10 and `requirements-handoff-lock.txt`.
Imports of `frozen_handoff`, `crewmate_v2`, `belief` and `spectator` were explicitly
checked to resolve inside that clone. No neighboring worktree's source, private
preferences or saved research data was required.

## Tests and runtime evidence

| Check | Verified evidence |
|---|---|
| Both selected checkpoints | Original unchanged `FourCrewPolicy.load` passes; exact selected SHA-256 and 51,778 parameters each; gradients disabled |
| Frozen dependencies | Phase 4 belief and transitive Phase 5 release hashes accepted by the unchanged fail-closed checker |
| Baseline coordinator equivalence | Six new handoff tests include paired original/new packets, choices, rewards and summaries |
| Isolation and external controller | Independent private/public memories; impostor-only immutable envelope; invalid/stale intent rejection; factory creates new controller per game |
| Canonical V2 mechanics/privacy | 40 original tests pass, including ghost-secret/history/timing noninterference and actually different hidden killers |
| Same-step ballot repair | Separate eligible-v1 test credits a legitimate pre-ejection ballot while historical ledger behavior stays available |
| Existing viewer and repairs | 29 tests pass; five tests that require intentionally excluded private study recordings skip |
| Real spectator playback | Standalone clean clone launches original V2 UI and renders the newly recorded completed match, including meeting state and selected-actor neural panel |
| Artwork | All 119 unique manifest-referenced images pass original Git-blob hashes; the complete selected 120-image local pack and Skeld map have publication hashes |
| Public website regression | Source validator, temporary-directory static build and built-artifact validator pass; 93 curated site assets resolve |
| Publication privacy audit | Only two new selected `.pt` files; no raw replays, private observations, local preferences, absolute personal paths, temporary environments or Patrick Jane package in the staged publication |

New handoff tests: 6 passed in 8.81 seconds initially; the clean-clone repeat passed
in 6.44 seconds. Original core/privacy tests: 40 passed in 22.58 seconds. The
clean-clone viewer/mechanics subset passed 29 tests, with five private-fixture skips,
in 13.31 seconds. These are engineering correctness checks, not trained performance.
Private-fixture skips are disclosed rather than publishing those recordings.

The final combined suite in the independent clean clone passed **75 tests**, with
the same **five private-recording skips**, in **39.87 seconds**. Its strict verifier
also passed on the committed source at `06456da6bf9fe5bb67f6615581debd10ca296820`.
The subsequent publication-evidence commit changes documentation only.

## Complete matches — frozen policies, zero updates

| Configuration | Seed | Outcome | Crew choices | Simulated seconds | Initial local wall seconds |
|---|---:|---|---:|---:|---:|
| B, sampled, original baseline, replay on | 2760000 | Impostor parity | 49 | 168.0 | 38.72 |
| B, sampled, replay off repeat | 2760000 | Same parity result | 49 | 168.0 | Measured locally; no speedup claim |
| A, greedy, original baseline | 2760001 | Crew tasks completed | 99 | 139.6 | 26.23 |

All runs completed with zero illegal actions, unchanged checkpoint weights,
independent memories and one shared frozen belief predictor. Four living crew
started under the chosen shared policy. These are only two distinct development
scenarios, with repeated controls; they are not a win-rate estimate.

The clean clone also ran B with the external-controller factory hook using the
scripted stand-in. It completed the same 168-second match and semantic event hash:

`b935b8b8d18be5b38166474b3dec86d38531b2a6351cad3c6ab9abffbb9555bd`

That hash also matches the original recorded and unrecorded runs. Replays, full
per-actor summaries and GUI frames are retained locally, excluded from Git.
The original frozen checkpoints and authoritative research worktrees were read/copied,
not modified. Existing main checkpoints are inherited public files, not newly
selected opponents; the only new checkpoints are the approved A/B pair.

## Corrections and limitations retained

The historical checkpoint locks the map JSON's raw CRLF bytes. The publication
pins those bytes with `.gitattributes`; parsed JSON is identical to main. Initial
`git diff --check` flagged those preserved CR characters as whitespace; a per-file
`cr-at-eol` attribute now makes that intentional exception explicit. No map
coordinates changed. An early clean-runtime probe ran while pip was still
installing and reported missing Torch; the completed clean installation subsequently
passed strict loading. The first local site-build destination was rejected by its
temporary-directory guard; the correctly configured external runner-temp build
and validation passed. No guard was weakened.

The partner's learned NN itself has not been integrated/tested: their actor encoder,
candidate decoder, option timing and rewards need an explicit adapter. CPU Torch
is strictly pinned; CUDA/multi-runtime RPC is not implemented. The controller API
is cooperative, not a hostile-code sandbox, and is not an SB3 Gym drop-in.
Models retain reporter/evidence confounds and small-study limitations. Social
guards and optional mechanics are not silently applied to claim stronger learned
opponents. See README.md and the unchanged historical research register.
