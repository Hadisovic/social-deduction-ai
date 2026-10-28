# Phase 2: opponent diversity and social behavior

**Development engineering only. Phase 6.1 acceptance remains pending.** No neural training, reward change, canonical game change, checkpoint migration, final-test access, push or merge is part of this work. Spectator V2 remains unchanged and approved by its owner.

Start with the [measured report](REPORT.md), [design/context map](DESIGN.md) and permanent [research register](../research/ISSUES.md). The original registered study and its negative results remain unchanged in the research branch/worktree.

## Components

- `crewmate_v2/phase2/opponents.py`: an opt-in, actor-local mixture. Aggressive, stealthy, self-report, evasive and social modes have overlapping report/kill/movement/claim/vote preferences. Independent mode/report/kill/movement/social RNG streams and bounded mode switches provide variation within a match. Reporting accumulates a stochastic body-level threshold only during legal reporting exposure. No always/never report profile, hidden role lookup or fabricated direct evidence.
- `social.py`: reconstructs accepted defense claims and subsequent submitted ballots, matching exact actor and meeting. It separates sampled training from validation by registered seed partition, public claims from missing private observations, and neural action probabilities from the shared frozen belief predictor.
- `audit.py`: an opt-in observer of each actor's own packet, legal opportunities, submitted actions and accepted events. Role labels score the completed game offline. Nothing from the auditor returns to an actor. Detailed local observation files are private research artifacts.
- `scenarios.py`: legal engineering choreography for twelve behavioral situations. Scripted setup uses ordinary navigation, cooldown, kill, report and speaker-slot rules. It never installs reporting/voting rules in learned crewmates.
- `run.py`: original-versus-mixture matches with a frozen, strictly loaded Model B. Complete matches, identical canonical settings/scoring, 15-minute execution ceiling and exclusive outputs. All A/B/zero checkpoints are loaded and checked without changing their tensors.
- `probe.py`: matched weak/direct-evidence histories at one engineering seed; frozen A/B/zero inference at discussion and voting, with no neural action executed.
- `shortcuts.py`: exploratory leave-scenario-out identification using public reporter identity and delivered claim kind. It cannot access private navigation, hidden votes, unseen positions or future outcomes as features. Labels are offline targets only.

The registered `ReporterController` and all historical source-lock files remain byte-identical. Import the new controller explicitly on a new research match; this task does not silently install it in the training runner. Profile/mode metadata is spectator-only script metadata, never a policy tensor.

## Reproduction

From this Phase 2 worktree, use the existing project virtual environment. The following creates a **new** local diagnostic output directory; do not reuse an existing path. It performs frozen inference and simulation, not PPO training:

```powershell
& ../.venv/Scripts/python.exe -m crewmate_v2.phase2.run `
  --study ../_v2_fourcrew_registration/local-runs/fourcrew-reporting-2048-seed17-v1 `
  --spectator-source ../_spectator_v2 `
  --output local-runs/new-opponent-assay
```

The exact local protocol records actor runtime/source locks, package hashes, checkpoint hashes, eight engineering scenario specifications and the execution ceiling. Frozen checkpoint loading fails on an incompatible runtime; there is no bypass. Partial games remain unfinished `.writing` files and are never counted as completed matches. Actor/body pairs, decision windows and games are different denominators.

The evidence probe is a separate short engineering operation:

```powershell
& ../.venv/Scripts/python.exe -m crewmate_v2.phase2.probe `
  --study ../_v2_fourcrew_registration/local-runs/fourcrew-reporting-2048-seed17-v1 `
  --output local-runs/new-evidence-probe
```

Both historical data and new raw observations/replays remain under ignored local output directories. Do not publish them with this package. Public research material should contain aggregate measures and sanitized representative examples only.

## Spectator V2 behavioral inspection

Open a completed new replay with the approved viewer, using an independent local preferences file:

```powershell
& ../.venv/Scripts/python.exe ../_spectator_v2/run_spectator_v2.py `
  --replay local-runs/opponent-assay-v1/replays/mixture/ordinary-2760081.jsonl `
  --watch local-runs/opponent-assay-v1/replays `
  --preferences local-runs/phase2-viewer-preferences.json `
  --paused --omniscient --reveal-ballots
```

For the preserved historical counterexample, load `ordinary-2790005.jsonl` under the registered study's `replays/shaped_v2/` directory. Select p4, seek to 126.2s and inspect Mind/Belief, then seek to 129.6s for its ballot. Pinning the two decisions in Evidence shows recorded public changes. The defended player p3 is the impostor in this particular retrospective example. Hidden truth is for researcher scoring only.

## Future boundaries

Sabotage requires a separately versioned mechanic, legal actions, actor-local observations, cooldowns, physics, replay support and correctness tests. It is not implemented here. Mixed-team learning, distillation, shared-objective PPO, architecture revisions and additional learning studies require separate approval. This mixture's diversity must not be described as increased realism, robustness or reduced exploitability without appropriate measurements.
