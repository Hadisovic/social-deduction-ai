# Jane Impostor plan after the real Patrick Jane review

30 October 2026. No training starts from this plan. Patrick Jane remains a
reference only; do not copy its implementation/weights as a final common base.

## Verified reference

- [x] Owner confirmed the teammate push; fetched/read it before revising compatibility.
- [x] Pin implementation `d459f38674ebc537acde20647cbd07700c494533` and guide tip `c7c0ef0`.
- [x] Verify published artifact checksums and read actual architecture/API/features.
- [x] Check real A2 constructors: 10+2 and 12+3 accepted; 13+2/14+1 rejected.
- [x] Translate shared settings to native A2 defaults without changing PPO/rewards.
- [ ] Implement and verify a learned-impostor adapter; configuration mapping is not this.

## 1. Agree the shared semantic boundary with the crew developer

Use the current immutable observe_pj(actor_id) envelope, including legitimate
ally identity/own abilities. Keep native Intent/API Action semantics, target IDs,
tick timing and local door knowledge explicit. Do not pass a world or another
actor's state into the policy. Native other-player rows exclude self; voting skip
is 14, and candidate target indices are packet-local/dynamic.

Current crew features do not encode all host settings, teammate flags or own
impostor cooldown/vent state. Do not reuse the crew encoder unchanged and assume
those facts reach an impostor. Decide a new role-specific feature schema after
joint review. Existing weights/layouts retain their exact meaning.

## 2. Design the impostor's independent network

Use the verified self/player/event attention + recurrent design as reference.
Keep shared weights among impostors but independent memory/history/RNG/options.
Choose actor/critic sharing deliberately: the crew model shares its backbone,
unlike the historical fully separated 75K impostor. Do not silently claim the two
designs are the same. Dimensions/head counts remain a design decision, not a copy.

Required neural choices: goals/wait/cancel, kill/target, fake start/continue/stop,
report/emergency, claims/defence/silence, vote/skip, legal door/vent/light actions.
The current crew interaction head (travel/task/report/repair/emergency) cannot
represent these by relabelling outputs. Preserve exact factor/mask/log-probability
accounting. Avoid flat claim-target Cartesian products if capacity overflows.

## 3. Integrate with the actual A2 engine without changing its frozen study

Start with the native 10+2 profile; preserve its source, task counts, lights,
doors, vents, event schema, technical cuts and independent actor boundary.
Replace the scripted controller only through the legal actor interface in a NEW
learned-impostor regime. Never mutate world state directly or share hidden ally
positions/goals. Preserve the scripted reference and opponent-mixture studies.

Keep Navigation V2's redundant-arrival repair independent. Audit the crew's task
commitment before designing fake-task options: it is engineered persistence and
must not become a forced impostor-faking loop. Impostors currently have no owned
real tasks; a legitimate fake-list contract needs explicit design.

## 4. Verification gates before learning

- [ ] Per-actor GRU/RNG/reset isolation, meeting continuity and no hidden truth.
- [ ] Legal kill/body/report/vote and source-provenance tests.
- [ ] Persistent but abandonable fake activity; no restart or task-progress exploit.
- [ ] Door/vent/light action masks, time accounting and local-knowledge paths.
- [ ] Strict model/runtime/source identities and recorded explicit transfers only.
- [ ] Recurrent sequences, conditional likelihoods and terminal/truncation handling.
- [ ] Deterministic bounded engineering matches and honest replay labels.

## 5. Register an independent learning study

Freeze competent crew opponents; record scripted versus learned opposition.
Propose rewards and budgets for this new role rather than inheriting crew task
rewards or claiming old 101/202 are a new twelve-player model. Compare win reasons,
search opportunities, ready-idle periods, witnesses, reports, fake completion and
meeting actions. Separate training, selection and locked fresh confirmation.

No full PPO before owner approval with measured throughput and approximate time.
Do not open historical final partitions, merge or deploy.

## 6. Future additions, separately versioned

Enable communications/critical sabotage only after crew counterplay and tests;
they are not registered in A2. Devices, Clean Vent, ghost impostor assistance,
optional roles, 13/14-crew support and self-play remain staged work. The existing
native fifteen-player constructor is 12+3, not proof of trained generalization.
