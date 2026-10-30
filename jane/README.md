# Jane: common settings, independent role projects

**Status:** teammate source reviewed; draft settings corrected; native A2
configuration adapter added. Patrick Jane is a reference only. No model code or
weights are copied, and no final joint network structure is prescribed.

Upstream implementation: `d459f38674ebc537acde20647cbd07700c494533`.
Reviewed documentation tip: `c7c0ef0fd0b6d460860f1f7b27fe34cb4f8385b2`.
See [the review](../docs/jane/UPSTREAM_REVIEW.md) for verified differences.

## Folders

| Folder | Responsibility |
|---|---|
| `shared/` | Draft settings, capability checks, configuration mapping and source receipts |
| `configs/` | Versioned match profiles; both future role adapters use the same hash |
| `crewmate/` | Actual Patrick Jane reference registration; future local adapter |
| `impostor/` | Independent learned-impostor plan and future registration |
| `../docs/jane/` | Joint rules, interface, future additions, review and evidence |

Do not duplicate an engine into each role folder. Keep old packages, studies,
checkpoints, viewer and Navigation V2 work in their original locations.

## Profiles: choose the actual regime

| File | Population | Meaning |
|---|---|---|
| `historical_5p.json` | 4 crew + 1 impostor | Unchanged historical V2.2 configuration |
| `patrick_jane_a2_12p.json` | 10 crew + 2 impostors | Exact reviewed A2 defaults; lights, doors and vents |
| `patrick_jane_a2_15p.json` | 12 crew + 3 impostors | Native constructor accepts this capacity; learning generalization untested |
| `jane_12p_base.json` | 10 crew + 2 impostors | Proposed minimal curriculum; not the registered A2 regime |
| `jane_15p_base.json` | 12 crew + 3 impostors | Proposed future minimal curriculum |
| `future_15p_full_classic.json` | 12 crew + 3 impostors | Future full classic request; not currently runnable |

The shared draft can describe 15 players with one/two impostors, but the current
upstream Population limits crew to 12. Its native adapter rejects 14+1 and 13+2.
All optional roles remain disabled. Do not silently disable A2 mechanics or enable
unregistered features to make a requested profile appear compatible.

## Inspect, do not launch training

```powershell
python -m jane jane/configs/patrick_jane_a2_12p.json --engine patrick-jane --native-config
python -m jane jane/configs/patrick_jane_a2_15p.json --engine patrick-jane
python -m jane jane/configs/historical_5p.json --engine historical-v2.2
python -m jane jane/configs/patrick_jane_a2_12p.json --engine patrick-jane --check-pair
```

The first three inspect configurations. The last intentionally exits 2: the
shared learned-impostor/crewmate pair is not implemented. Configuration-compatible
reports explicitly say `model_execution_verified: false`. No game is started.

To verify against a separate pinned upstream checkout without mixing branches:

```powershell
python tools/check_jane_patrick_config.py jane/configs/patrick_jane_a2_12p.json --source-root C:/path/to/pinned-patrick-jane
```

This verifies source/artifact bytes and constructs settings only in an isolated
child process. It does not load a model or run a match. Do not invoke the teammate's
Start-* scripts for inspection: those can immediately start training.

## Collaboration

Both collaborators can branch from `codex/jane-shared-foundation`. Shared schema
is `jane-settings-v2-draft`; existing draft-v1 profiles require explicit migration
(add the stall cut and change the version, preserving the original file).

Keep each role's parameters, histories, recurrent states and RNG independent.
Shared settings do not mean interchangeable tensors or models. Actual paired
gameplay, the new learned impostor and full future rules remain separate gates.
