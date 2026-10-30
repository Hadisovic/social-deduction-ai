# Review of the teammate push and corrections

The owner confirmed readiness before this review. Fetched branch:
`codex/patrick-jane-recurrent-ppo`; reviewed tip `c7c0ef0fd0b6d460860f1f7b27fe34cb4f8385b2`.
The guide pins implementation/model `d459f38674ebc537acde20647cbd07700c494533`.

## What the first draft got wrong or left unknown

| Original draft assumption/gap | Verified source fact | Correction |
|---|---|---|
| Jane source unavailable | Source + update102 artifacts are now published | Pin and hash the actual source/artifacts |
| Doors/vents/lights all future | Already implemented in opt-in A2 | Add a native A2 profile instead of disabling them |
| Fifteen-player example 13+2 | Population caps crew at12 | Native 15p profile is12+3; reject13+2/14+1 |
| Emergency cooldown0 | Native classic default15s; first kill10s | Preserve actual native defaults |
| One900s match cap | A2 disables normal timeout; technical max1800/stall300 | Separate stall cap; preserve legacy field |
| Architecture separation uncertain | Encoder/attention/fusion/GRU shared; two MLP trunks | Correct reference; no false full-separation claim |
| Fifteen undifferentiated player slots | Own state separate;14 other rows; skip slot14 | Record exact row/head/reference mapping |
| Broad editable settings imply support | Native lobby fixes1/1/2 tasks, visible ballots and always task bar | Fail closed on unsupported values |
| Shape similarity could imply compatibility | Dynamic targets, role-specific features and strict checkpoint/runtime locks | No model relabelling or silent tensor reuse |

## What was actually inspected and checked

Read config/a2_config, API/features/actions/decoder, network/memory/navigation,
commitment, environment/A2 environment/geometry/opponents, worker/PPO/checkpoint,
transition, rules, implementation and integration guide. The reviewed upstream
AGENTS.md calls for research continuity. The existing ISSUES register remains
unchanged under the owner's earlier restriction; this document and the new
verification receipt preserve the compatibility findings separately.

Verified the bytes/size/SHA256 of all five files in the published artifact
manifest, including inference/full checkpoints. No weights were loaded. The
source receipt captures normalized critical-source and constructor-import hashes.
The original source was inspected in an ignored isolated snapshot, not copied as
either role's model base and not merged into this branch.

Constructed native A2 settings only in a separate Python process using that exact
source. No network, optimizer or match was constructed. The native12p translation
matches its complete default config. Fifteen-player12+3 construction is a capacity
check, not trained performance. The local CPU2.6 environment is not claimed to
match the published Torch2.7.1+cu128 model runtime.

## Remaining real compatibility work

The crew model is1,031,070 parameters /52 tensors. Update102 is trained on
task-pursuit-v1, not A2. Its504/482 outcomes are on-policy task training results,
not held-out deduction. A2 currently has10 neural crew and2 scripted impostors.

The crew observation has legitimate impostor-only teammate/ability fields, but
the crew tensor encoder and action heads do not form a learned impostor policy.
The historical75K model is five-player and incompatible without new design.
Learned pair adapters, a new impostor model, exact runtime checks and deterministic
cross-play remain pending. Optional roles/ghost-impostor sabotage/devices are
deferred; oxygen/reactor paths exist in the base environment but are not enabled
or registered in A2. Full retail equivalence is not claimed.

Primary reference: [pinned integration guide](https://github.com/Hadisovic/social-deduction-ai/blob/c7c0ef0fd0b6d460860f1f7b27fe34cb4f8385b2/PATRICK_JANE_INTEGRATION.md).
Both role structures remain provisional. Future compatibility decisions must
reference these actual interfaces, not the earlier owner-supplied description.
