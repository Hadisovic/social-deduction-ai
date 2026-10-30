# Jane Impostor development plan

30 October 2026. Foundation/planning branch. No new PPO is authorized by this
delivery. Latest owner clarification: **Patrick Jane is still in progress and is
a reference only, not a copied common base.** The owner will announce the teammate
push. Pull and read it first, then revise compatibility for both role projects.
This is not the final folder, interface or network structure.

## 1. Review the actual Jane progress before deciding the design

- [ ] Wait for the owner's push-ready notice, then fetch and read the progress.
- [ ] Record current source, runtime, available checkpoint hashes and incomplete work.
- [ ] Preserve its original loader and baseline before any compatibility adaptation.
- [ ] Verify the reported self `192→256→128`, player `16→128→128`, event
      `16→128→128`, attention, fusion `384→256`, GRU-256, and `256→256→128` trunks.
- [ ] Record actual actor/critic encoder and GRU sharing; separate trunks do not
      prove fully separate pathways.
- [ ] Keep its actual parameter count as a reference. Decide the impostor's own
      dimensions after review; do not copy weights/code or force a cosmetic count.

## 2. Revise the provisional common engine/profile agreement for both roles

- [x] Shared editable settings with a 15-player capacity and derived crew count.
- [x] Common rules, task/role inventory, semantic interface and version boundaries.
- [ ] Verify the current Jane engine against the shared rules and actual build.
- [ ] Agree which current Jane navigation/commitment systems remain engineered.
- [ ] Use `jane_12p_base.json` as the initial target only after compatibility tests.
- [ ] Test 4–15 player padding/masks and legal one/two/three-impostor starts.
- [ ] Keep historical five-player tests on their unchanged engine and source.

## 3. Build role-specific adapters, not strategy scripts

The crew receives only its own real task list; the impostor receives its own
legitimate fake list and ally identities. Both receive public settings, lawful
observations and events. Preserve hidden information isolation.

Shared-weight impostors keep separate GRU states, histories, option goals and
sampling RNGs. No automatic sharing of unseen ally positions, goals or memories.
Do not add rules saying hunt, kill on sight, never report, always defend an ally,
or always fake when watched. Roles/actions are engine interface labels; strategic
choices remain neural.

- [ ] Add kill, fake, body/report, emergency, claim, vote and silence candidates.
- [ ] Keep goal selection neural and physical collision/path execution deterministic.
- [ ] Preserve Navigation V2's independent arrived-goal/continuation work.
- [ ] Audit the real Jane task commitment system before applying it to fake tasks.
- [ ] Make fakes finishable, interruptible and persistent without reward farming.
- [ ] Distinguish direct evidence from claims and public results.
- [ ] Support self-defence and lawful misleading claims; no truth mask.
- [ ] Audit richer candidate combinations; factor arguments if capacity would
      overflow instead of silently truncating legal options.

## 4. Recurrent training correctness before study

- [ ] Reset state per match, retain it across meetings, validate phase transitions.
- [ ] Train contiguous actor sequences; preserve masks/log-probabilities and timing.
- [ ] Audit current Jane TBPTT-128, burn-in, hidden-state refresh, terminal bootstrap
      and reward attribution instead of blindly copying hyperparameters.
- [ ] Freeze opponents and policy version during each complete collected match.
- [ ] Define an explicit fresh-init or recorded transfer study; never silently
      reinterpret crew weights as an already-trained impostor.
- [ ] Test determinism, strict load, invalid-action masking and no hidden truth.

## 5. First learning study: narrow and measurable

Train classic base-role behavior against an independently checked frozen crew
pool. Keep meetings and exposure consequences present from the start. Separate
training, selection and fresh locked confirmation scenarios. At least three
independent initializations are desirable for architecture claims; old 101/202
are same-initial-weight sampling streams, not three independent replications.

Start the new reward comparison around team win/loss, with separately registered
kill shaping. Do not reward fake starts, distance, claims or sabotage presses.
Review individual ejection penalties before two-impostor play: an eliminated
actor can still help through future ghost sabotage. Retain old rewards exactly
in old studies. Infrastructure timeouts are not game losses.

Report wins with confidence intervals and reasons, but also ready-idle time,
search opportunities, repeated goals, witnesses, report latency, fake completion,
meeting behavior and failure replays. A parity win alone is not intelligent play.

## 6. Expand mechanics only after paired capability gates

1. Verify devices/vent/door state machines and actor-safe observations.
2. Teach or verify crew counterplay, especially repairs.
3. Enable noncritical sabotage, then critical sabotage in separate profiles.
4. Verify two-impostor action arbitration, team credit and ghost continuation.
5. Evaluate 10+2 and 15-player generalization separately; setting a maximum does
   not prove a model can play every population or setting.
6. Add optional roles only after both policies and the engine handle them.
7. Consider adaptation/self-play after both learned base roles work reliably.

Each capability gets its own settings version, test evidence and opponent pool.
See `docs/jane/FUTURE_WORK.md` for all reserved mechanics and roles.

## Approval and completion gates

Implementation and publication of this foundation are authorized. A new full
training study still needs a concrete proposal, measured throughput and owner
approval. No reserved historical final partitions, merge or deployment.

Before training: show exact source/checkpoint/settings hashes, adapters, tests,
sampling/selection protocol, decision budget, deadline and estimated time.
Before calling it compatible: both real role loaders and the shared engine must
pass deterministic, privacy and action/timing checks on the same profile.
