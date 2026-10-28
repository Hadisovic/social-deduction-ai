# Corrected Patrick Jane readiness measurements

Frozen actor implementation: local commit `451e075`. Corrected comparison uses
32 distinct engineering seeds, 2920000–2920031, repeated at each worker count.
The policy stays at fresh initialization seed 17 for collection. Real four-epoch
PPO updates are timed on independent policy copies; they do not change the
collection policy. The earlier invalid comparison and failed qualification are
preserved and explained in research issues R033/R034.

| Workers | Decisions | Collection seconds | PPO seconds | Collection + PPO decisions/s | Peak process-tree RSS |
|---:|---:|---:|---:|---:|---:|
| 8 | 38,287 | 792.099 | 54.766 | 45.210 | 3.57 GiB |
| 16 | 38,287 | 549.130 | 53.822 | 63.499 | 5.05 GiB |
| 32 | 38,287 | 413.472 | 66.866 | **79.709** | 7.95 GiB |

All **32/32 action traces and results match** across counts. Legality,
navigation-failure and reward-attribution audits pass. Thirty-two workers is
the fastest measured stable configuration: 25.5% faster than sixteen and 76.3%
faster than eight including updates. This is one matched measurement on this
machine, not a universal optimal worker count. Process-tree RSS sums shared
pages and is not identical to unique physical memory use.

On the same first-eight complete trajectories (6,725 decisions), sequential
recurrent processing took 35.359 seconds; independent-sequence batching took
10.529 seconds, a **3.358×** speedup. Mathematical objective, grouping, optimizer
settings and architecture are unchanged. GPU optimizer paths are **not bitwise
equivalent**: maximum final parameter difference was 0.004929, and mean reported
losses were 1.80388 versus 1.89477. Preserve this measured numerical limitation;
do not describe trained weights as identical. The pilot freezes the batched
implementation. CPU output/gradient equivalence and fixed-configuration
reproducibility tests pass; GPU complete-game trace and strict checkpoint/
optimizer/RNG reload were separately qualified.

The observer recorded 11.533 seconds of selected-instance snapshot projection
and writing during the 413.472-second thirty-two-worker collection. This is
instrumented snapshot work, not a controlled estimate of total viewer overhead.
All benchmark configurations included monitoring. Rendering runs in a separate
desktop process; only the selected worker publishes throttled snapshots, and
monitoring on/off tests preserve action traces and rewards.

Qualification: 46 Patrick Jane tests; 46 historical replay/UI/four-crew checks;
complete-game serial/parallel GPU trace equality, finite four-epoch update and
exact weights/optimizer/RNG reload. Settings and ghost meeting behavior are
documented separately. The first pytest attempt failed on sandbox temp-directory
creation; workspace temp paths resolved it. The historical spectator-source gate
also required explicitly pointing at this worktree, without weakening the gate.

No old checkpoint, canonical historical simulator source, partner branch or
final-test partition was changed or evaluated. These are engineering results,
not evidence that Patrick Jane has learned gameplay. The learning pilot is a
separate fresh run with its own registration, checkpoints and report.
