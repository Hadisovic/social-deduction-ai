# Archived study progress — stopped30October2026

Final status: failed at20:05:49UTC. No training running. See study-20261030/README.md. The notes below preserve earlier snapshots.

# Lizbon Classic V2 live study notes

30 October2026,19:48UTC. Fixed registered study is continuing; no source or budget changes.

| Stream | First scenario | Simulated seconds | Wall seconds | Kills | Outcome | New PPO passes |
|---|---:|---:|---:|---:|---|---:|
| train101 | 9710001 | 194.4 | 901.73 | 2 | Technical wall cut | 0 |
| train202 | 9710101 | 170.4 | 918.86 | 3 | Technical wall cut | 0 |

Both initial matches reached the registered900-second match cap. Checks happen at native decision boundaries, so actual exit took901.73s and918.86s. Neither is a model loss, win or training sample; no optimizer update used these cuts. Both streams moved on to their second pre-registered training scenarios. No retry or extension.

Unlike greedy engineering playback, sampled policies moved, killed, reported and used meetings. This is descriptive behavior from the saved starting weights, not evidence of new learning. All completed-match counters are still zero at this snapshot. Old/fresh final sets remain sealed.

Inspect the preserved per-match audits for42/37 accepted door-close events,46 cancellations per actor in stream101 and36/37 in stream202 and multiple meetings; those actions increase simulation work. The neural inference timers account for only a small fraction of the wall time. A runtime investigation must preserve behavior and wait until this study ends; do not edit loaded workers or tune from confirmation.

Latest truth is persisted status/matches files, not this snapshot.

## Progress20:03UTC

Both streams completed their second registered match naturally, each a crew task victory. Stream101:216.2simseconds/640.63collectionseconds; stream202:316.6simseconds/845.16collectionseconds. Both saved checkpoint002 and eight new full PPO passes from their complete game. Their first time-cut games remain excluded. Third registered matches are now using updated weights. These training losses and differing scenarios do not establish policy improvement. Validation/confirmation have not started.
