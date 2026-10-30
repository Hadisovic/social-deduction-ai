# Patrick Jane crewmate

**Review the teammate's work in progress first. Patrick Jane is a reference,
not code to copy into a common base or an already-finalized architecture.**

The owner reports 1,031,070 parameters, self/player/event encoders, attention,
GRU-256, role policy heads and independent recurrent state for shared-weight
actors. The reference is recorded in `../shared/network_reference.json`.
Actual source, feature ordering, tensor layouts and checkpoint hashes are absent
from the fetched repository. `manifest.json` says `awaiting-progress-review`.
Wait for the owner's ready notice before pulling the teammate's progress. Then
read the implementation and adapt this draft for both roles before finalizing it.

## Import checklist for the crewmate collaborator

1. After the owner announces the push, fetch and read the source/commit/progress.
2. Keep the original package/checkpoint intact; record its license and hashes.
3. Document what the task commitment/navigation layer does outside the NN.
4. Map legitimate observations to the shared semantic contract.
5. Verify padding, masks and independent actor memories up to 15 total players.
6. Read the same settings profile and fingerprint as Jane Impostor.
7. Preserve current weights/settings/results; unsupported new settings must fail.
8. Add capability tests and a deterministic adapter replay before registration.
9. Only then replace null registration fields with verified values.

Future code may belong in this role folder or a preserved upstream subpackage;
the layout is provisional and must be reviewed alongside the impostor work.
Shared engine semantics belong in `../shared`, not in a role's private fork.
The current 10+2 Patrick Jane results remain owner-reported until artifacts are
available; no social-deduction or sabotage competence is inferred from task wins.
