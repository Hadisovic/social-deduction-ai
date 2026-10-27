# Registered experiment activation — 2026-10-28

The owner authorized: “Yes, you have my approval. Start the full experiment and open the live visualization so I can watch the training progress.”

The approved driver was launched once from `codex/crewmate-v2-fourcrew-registration` at `723c8d4774a25af6ccb6a863e141c68ae3966a38`, without changing its source, rewards, seed schedule, checkpoint rules or coverage. The spectator uses the independent `codex/spectator-fourcrew-registration` worktree at `e8a815f83bc98cab5c99c862f280bbdbffc509b2`. Existing branches and PR #11 remain untouched.

Preflight verified the exact plan, runtime, 19 private readiness artifact hashes, strict checkpoint compatibility, fresh initialization tensor hash and historical frozen-source checks. A new private `local-runs/activation-20261028.json` records the owner's authorization; the false registration template was preserved.

| Contract | SHA-256 |
|---|---|
| Plan | `bc449451df3a83bdf5a38c83f7be0cb36e616895f34d9bdca2400c5d0d732fd1` |
| Runtime | `2875af6f30f25f588ec0cd52c8a14d5ac6313053c995fe3ef1ac117b2df51bca` |
| Fresh initialization / frozen reference | `1a572645635edf2bb6a802a5572892b6321747452176abd4755fe5e5cfedc3d2` |

The private output is `local-runs/fourcrew-reporting-2048-seed17-v1`. Its `registration.json` records the actual start and authorization. Checkpoints, collected summaries and raw replays remain local. Do not duplicate, overwrite, automatically restart, extend the four-hour ceiling or open final-test partitions. At activation, the viewer displayed the completed engineering readiness replay; actual training recordings subsequently appeared in its watch library. Select a training recording to enter the auto-next queue. This is completed-match playback with live progress counts, not streaming unfinished games.

Final execution status: **completed normally**, training exit0, 2,058 original/2,092 shaped decisions across46 matched games per arm; all48 validations complete. Total driver wall2,072.45 seconds. No registered inconclusive flag; acceptance remains pending. [Results](../results/registered-2048-seed17-v1/REPORT.md) retain the ordinary-performance regression, reporter confound, opportunity denominators, frozen-reference comparison and uncertainty. The independent viewer remains open with96 completed recordings. No further run or publication was initiated.
