# Lizbon Classic V2: session checklist

Owner request: support4–15 total players, legal1–3 impostors, editable Common /
Long / Short counts, unique assigned task families per player, complete training
and evaluation, and analyze every impostor decision. Keep both roles compatible.

Branch: codex/lizbon-classic-v2, from the preserved Lizbon V1 viewer/results branch.
No historical checkpoint, settings bundle or external pinned source is overwritten.

- [x] Review the V1 failure, settings, native allocator and current neural interface.
- [x] Confirm “task level” means Common / Long / Short counts.
- [x] Verify original-game core rules and record primary sources/remaining approximations.
- [x] Add a separate shared Classic V2 settings/engine adapter for both developers.
- [x] Support every shared-accepted4–15-player/1–3-impostor combination, including14 crew +1.
- [x] Assign common types match-wide; sample other types without replacement per owner.
- [x] Prevent duplicate canonical families, including alternate garbage variants.
- [x] Keep legitimate repeated stages and ghost task ownership/progress intact.
- [x] Fix inactive/vented stored-position invariants while preserving live collision checks.
- [x] Make custom profiles runnable; embed full settings and fingerprint in replay metadata.
- [x] Preserve the V1 architecture/weights and add explicit, audited migration/resume.
- [x] Record every neural decision, legal candidates, probabilities, own observation summary,
  selected action, accepted effect and rewards. Distinguish inference from explanation.
- [x] Analyze all4,938readable choices: searching, kill opportunities/witnesses, reports, votes/claims, faking,
  ready-time idling, navigation cancellations, team contention and win/loss causes.
- [x] Test population/task settings, unique allocation, engine bug, privacy and checkpoint compatibility:74 isolated checks passed.
- [x] Run bounded deterministic gameplay checks across lobby sizes and measure throughput.
- [x] Propose budget and obtain approval: three hours, up to90 training/90 evaluation minutes.
- [x] Execute the approved study; stopped on fatal cache-coherence error after37m12.5s. No retry/extension.
- [x] Preserve failures/cuts/checkpoints; recover readable cancelled prefix separately; final partitions remain sealed.
- [ ] Complete validation/selection/confirmation: NOT RUN because the study failed. No selected/final model.
- [x] Publish five graphs, choice analysis, insufficiency and concrete next steps in study-20261030/README.md.
- [x] Reproduce cache failure withoutNN/PPO; proposed upstream patch checks cleanly but remains unapplied.
- [x] Fix and review cache coherence in shared adapter; revision `paired-map-planner-v1`;87Jane checks and3offline comparator checks passed.
- [x] Preserve original weights; explicit SHA-verified migration changes no tensors and performs no training.
- [ ] Approve/register a new training study; old study is not resumed.
- [x] Open completed trained-policy playback; publish this own branch without merge/deployment.

Core scope is Classic Skeld with base crewmate/impostor roles. Previously deferred
special roles, exact minigame implementations, devices and unverified sabotage
extensions stay explicitly listed, not silently advertised as implemented.
No hardcoded hunt/kill/report/vote strategy and no hidden-world actor inputs.
Competence must be measured; a win cannot be guaranteed for an arbitrary NN/budget.

Preflight: maintained tests/ suite291 passed,10 skipped,one existing frozen-handoff
contract failure (.gitattributes and previously edited spectator/v2/map_view.py).
Bare pytest also encounters six legacy root-script collection errors because
stable_baselines3 is absent. Neither historical contract nor environment was
modified to hide these failures. Isolated Jane regressions:74 passed.
