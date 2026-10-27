# Conditional pursuit correction — registered before confirmation

The owner approved policy/reward fixes and evidence-driven training toward the best
tested impostor. The first improved run selected visible-player goals only 9 times
in 8,192 decisions, versus 2,685 room/task goals. The second selected 13 versus 2,527.
Both see the same candidate interface: visible-player pursuit still competes with
many public destinations *inside* the travel family. This is a concrete exploration
weakness, not evidence from reserved final tests.

The V2 study completes unchanged, including its immutable selection and all 80
confirmation/control executions. Its results, eight checkpoints and runtime stay
preserved. This separately versioned stage may run only if:

1. V2's locked model wins **fewer than 5 of 20** fresh confirmation games; and
2. **at least 3,600 seconds remain on the ORIGINAL V2 deadline**.

There is no extension of the three-hour wall-clock learning/evaluation ceiling.
If the gate fails, stop and deliver V2. If the correction hits the deadline, preserve
its partial artifacts and deliver the fully evaluated V2 parent. No automatic retry.

## One policy change

Add a 13th learned family, `pursuit`, for the existing GO_TO_LOCATION candidates,
which this interface creates only from currently visible player positions. This is
not automatic tracking, target selection or killing. The NN chooses family and target;
A* executes the chosen observed position under the same three-second interruption rules.

Continue the **locked V2 impostor**, not crew weights or a scripted expert. Copy its
actor/critic layers and first 12 family embeddings exactly; initialize the new family
embedding from its learned travel embedding. The new network has **75,091 parameters**.
Reset the Adam optimizer and use distinct private action/shuffle RNG streams.
Same rewards, features, PPO configuration, legal masks, lobby and frozen opponents.
No additional hidden information, pursuit-distance shaping or changed task rules.

The new `impostor_pursuit/` namespace and `run_impostor_ai.py` preserve previous strict
source guards. New checkpoints verify both the unchanged V2 core and all new sources,
their frozen parent file, model tensors and metadata. Old policy files are never rewritten.

## Fixed experiment

- One continuation of the selected parent; **8,192 additional decisions**. This is
  not a new independent initialization or replicated evidence.
- Fresh training scenarios start at **3101000**. Save warm-start, 2,048, 4,096 and
  8,192 policy checkpoints and per-rollout exact-replay resume states.
- Selection: unchanged V2 parent and correction checkpoints 4,096/8,192 on
  **3200060–3200079**, 20 paired matches each, **60 executions**.
- Rank by wins, kills, fewer timeouts, prefer unchanged parent on ties, earlier step.
  Lock before confirmation. The parent remains eligible; improvement is not assumed.
- Confirmation: locked selection and unchanged parent on **3200080–3200099**,
  20 paired matches each, **40 executions**. Do not change selection afterward.
- Keep the original V2 controls in the earlier report. Do not pool distinct
  partitions or count repeated scenarios as independent cases.
- Inspect true wins, kills, ejections, pursuit frequency, legal opportunities, PPO
  diagnostics, elapsed time and frozen hashes. Attribute results to a continuation
  with changed factorization, not to architecture alone versus extra experience.

Reserved final partitions remain closed. No merge, deployment, self-play or silent
replacement of the friend's crewmate checkpoint. The eventual universal viewer is
`python run_impostor_ai.py`; it strictly loads whichever evaluated checkpoint is
released. The original `python run_impostor.py` remains the preserved V2 viewer.
