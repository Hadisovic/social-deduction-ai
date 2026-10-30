# Investigation A2 V1 — registered, not trained

Starting policy: original task-pursuit-v1 update102, 2,813,879 lifetime decisions; checkpoint SHA256 `6cf7048310e889967b01dd785317d313ddc71f310cd9faedcf346b950b8fd01b`. Parent archive SHA256 `4518f5c82fa84b26ab287f2fb3e75af78f9148f510fe08b0de16bc7bfe62b263`; frozen source snapshot `2ee74da35096493975e8bfb8ef2d536704623c0fca588be14d524b8738cb42e3`. All 52 policy tensors and Adam moments/steps are retained, with an explicit source/feature bridge. No historical registration or results are rewritten.

## Rewards

| Event | Value |
|---|---:|
| Full living assigned task | +0.5 |
| Ghost task | 0 |
| Task victory, per team member | +5 |
| All-impostors-ejected victory, per team member | +10 |
| Loss, per team member | -10 |
| Individual accepted non-skip ballot | 0.2 × (correct − p) |
| Explicit or automatic skip | 0 |
| Impostor/innocent ejection event, per team member | +4 / -2 |
| Lights repair participant | +3 |
| Accepted body report | +1.25 |
| Novel exploration / player sighting | +0.02 / +0.05, existing caps |
| Death | -2 |
| Avoidable idle | -0.01, existing timing |
| Unresolved lights | -0.005 per living crew simulated second |
| Invalid / accuse / defend | 0 |

Here p is the true impostor fraction among the actual accepted ballot's eligible candidates, excluding self. Snapshot once at acceptance; roles are used only in trusted retrospective credit. A uniform random eligible-target ballot has zero expected individual reward. A valid ballot remains credited if its owner is ejected in the same resolution. Rejected, duplicate and fictional ballots get no credit. Automatic and explicit skips are recorded separately. No private ballot-correctness event is delivered to actors. Shared ejection credits remain separate from the single terminal family: two impostor ejections plus ejection victory can yield +18 per owner, versus +5 task victory. No generic +10 victory is stacked.

## Run

- Ten shared-weight, independently observed/memoried crew; two scripted impostors. Architecture 1,031,070 parameters, GRU256, five original heads; no new suspicion network.
- Unchanged PPO: Adam 0.0003, gamma .99, GAE .95, clip .2, entropy .01, four epochs, norm .5, recurrent length128, four sequences/minibatch.
- Task pursuit/commitment ON, report/lights-repair pauses preserved. Meetings and death retain original interruption semantics. No abandonment penalty, forced vote, influence or reordered speaking.
- Default opponent mixture: 50% historical and 50% upgraded **per match**, using a fixed role-independent seed stream. Both impostors receive independent script RNG/memory. Distribution never adapts during a run.
- Training seeds 3200000–3299999, allocated monotonically, interrupted allocations skipped. No reserved final partition is imported or read.
- Duration default12 active collection/PPO hours; finish a full worker wave and PPO boundary. Auto workers min(6, physical cores, total RAM/2GiB), frozen in registration; expected6 on i7-9750H/16GB. Existing collector and PPO unchanged. Optional positive additional-decision target also stops at a complete boundary; duration remains a ceiling checked between waves. No worker benchmarks.
- New A2-only Windows unbiased clock excludes sleep/hibernation. Historical 44,843.33-second record is preserved; its final wave contains a 19,301.19-second interval versus 554.62/595.04 preceding collection waves. Owner suspects suspension; cause is unconfirmed. No retrospective correction or throughput extrapolation.
- Latest complete checkpoints retain optimizer/RNG and source/configuration locks. Technical cuts use existing living-agent bootstrap. No automatic baseline/final validation, champion selection or long run during implementation.

## Controlled development comparisons

Frozen update102 and, later, a frozen A2 checkpoint each use the same 20 development scenarios 3300000–3300019, in three conditions: legacy mechanics/historical opponents; A2 mechanics/historical opponents; A2 mechanics/upgraded opponents. A2 mechanical changes and legal self-target exclusion are explicitly versioned; legacy reproduction uses the original profile. These are paired conditions, not 60 independent scenarios. No final-test partitions.

Manual command from the extracted package:

```powershell
.\runtime\python\python.exe -m tools.evaluate_pj_a2 --origin .\origin --output .\results\frozen-update102-development --games 20 --workers 2
```

Use `--checkpoint` for a later A2 checkpoint and a fresh output directory. It performs no PPO and verifies weights remain unchanged. This evaluation has **not** been run during implementation.

For causal reward/opponent training comparisons, separate runs warm-start update102 with matched RNG, seed order and **fixed equal decision budgets**, selecting `--reward-regime task-pursuit-v1` or `investigation-a2-v1`, and `--opponent-regime historical` or `upgraded-a2-v1`, through `tools.run_pj_a2`. Do not pool controls, call on-policy outcomes generalization, or attribute a combined reward/opponent effect to reward alone.

Measure legitimate task/ejection/parity endings separately from cuts; survival actor-seconds; living/ghost tasks; commitment/forced-choice counts; accepted vote accuracy and actual candidate chance baseline; skips, ties/no-ejection, ejections/accusations; firsthand vent/kill evidence availability and evidence-conditioned ballots; communication before ballots; doors/vents/witnesses/navigation recovery; PPO/numerical/runtime records. Accuracy denominator is accepted non-skip crew ballots. A correct vote is not, by itself, proof of deduction.

Acceptance remains pending. Training is owner-launched only.
