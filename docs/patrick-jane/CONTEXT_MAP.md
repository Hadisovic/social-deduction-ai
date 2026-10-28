# Patrick Jane context map and isolation review

Approved design: [OWNER_SPEC.md](OWNER_SPEC.md). Base is local
`codex/phase21-meetings-evidence` at `4bac4cdb099e81a5c6c7de982f423136f24213c5`,
which descends from the reviewed V2 environment `e62ba0b`. The reviewed Phase 6
training commit `2baa5db` is available locally. No complete branch is merged.

| New files | Purpose |
|---|---|
| `patrick_jane/config.py`, `api.py`, `environment.py` | Separate population, sabotage and actor contract |
| `patrick_jane/memory.py`, `features.py`, `navigation.py` | Independent timestamped memory and public-map goals |
| `patrick_jane/network.py`, `actions.py` | Locked recurrent network and five conditional heads |
| `patrick_jane/collector.py`, `ppo.py`, `checkpoint.py`, `run.py` | Central updates, process simulations, resumable provenance |
| `patrick_jane/replay.py`, spectator integration | Existing completed-replay pipeline, independent viewer |
| `tests/patrick_jane/` | Mechanics, privacy, architecture, accounting, recurrence and replay checks |

| Existing dependency, preserved | Reuse boundary |
|---|---|
| `crewmate_v2/engine.py` | Inherited individual tasks, ghosts and meeting resolution; no edits |
| `crewmate_v2/tasks.py`, `ghosts.py` | Unmodified deterministic task and ghost executors |
| `phase3_engine.py` | Player/physics, routing and meeting primitives; separate reset/invariants/step |
| `social_deduction/phase3_observation.py` | Wall visibility and immutable allowlist projection |
| `among_us_map.py`, `navigation_service.py` | Existing Skeld geometry, reachable grid, A-star and movement |
| `crewmate_v2/phase21/opponents.py` | Local-observation script interface, no partner neural policy |
| `phase5_training.py` | Time-aware GAE reference; no feedforward actor reuse |
| approved `_spectator_v2/spectator/` | Pinned local viewer sources copied additively, original untouched |

Review: old five-player reset/assertions and four-candidate/96-action encoders
must never process Patrick Jane checkpoints. The new schema rejects old weights.
Truth is limited to trusted game rules, reward settlement, offline scoring and
replay. Actors receive detached local packets. Every decision batch is projected
before any action is applied. Workers simulate; only the coordinator optimizes.

Historical hashes (1077 tracked files plus 206 checkpoint/viewer sources), refs
and pause location are captured locally in
`local-runs/patrick-jane/preservation-before.json`. The earlier branch remains
clean and its results remain unchanged. No active repository training process
was found. Independent impostor work is outside this implementation.

Implementation review completed against the owner's locked architecture. The
scope includes engineering smoke updates only; full training remains gated.
