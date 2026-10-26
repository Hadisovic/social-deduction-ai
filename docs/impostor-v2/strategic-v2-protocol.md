# Improved impostor study — registered before learning

Owner approval: 27 October 2026, “fix the policies make reasonable rewards and nn
settings … after everything done and you push it tell me what file to run”.
This authorizes the follow-up and viewer. Merges, deployment, reserved final tests
and changes to the friend's separate crewmate model remain outside this work.

## Changes

- Factorized neural policy: choose a legal action family, then its target. Public
  travel destination count no longer determines initial family probability.
- Greedy inference also chooses family then target; it does not take the largest
  flattened probability, which would favor singleton families.
- Family entropy regularization plus a small normalized conditional entropy term;
  maximizing entropy cannot simply reward the family with the most destinations.
- Actor and critic have separate encoders and separate gradient clipping limits.
  PPO uses rollout 256, four epochs, minibatch 64, lr 3e-4, Adam eps 1e-5, ratio clip
  .2, actor gradient .5, critic gradient 1.0, family entropy .02 to .005, normalized
  conditional entropy .001, physical-time GAE lambda .995/second and gamma=1.
- Actor-local extras: visible body distance, own directly known kill history, own
  selected goal, legal kill availability and visible-neighbor geometry for targets.
  Neighbor features are not privileged actual-witness labels.
- A selected travel goal can run up to three simulated seconds. Every .2-second
  physics tick and .6-second observation/opponent schedule remains. New visible
  players/bodies, legal kill opportunities, arrival and phase changes interrupt the
  hold. No kill, target, report or claim is automatically selected.

Rewards: +10 win; -10 task loss; -12 timeout; -0.002/s; +2 per successful kill;
additional -2 on own ejection. No survival, movement, fake-task, report, claim or
hidden-witness shaping. With at most three kills before parity, wins remain positive
and every loss remains negative. Kill shaping may initially encourage reckless
behavior; confirmation must include witnessed kills/ejections and true game wins.

This is a bundled practical improvement, not an ablation identifying a single cause.
The original sparse-reward pilot, all its checkpoints and its zero-win results stay
preserved. Frozen full-29 and the belief model never enter the optimizer.

## Fixed execution and selection

1. Train initializations 71 and 83 from scratch, **8,192 decisions each**. Ordinary
   canonical V2 only: 45-second cooldown, 15/120-second meetings, original ghosts,
   task victory and both opponent safeguards. No curriculum/mode switch.
2. Save immutable initial, 2,048, 4,096 and 8,192 policy checkpoints plus per-rollout
   resume states. Start training scenarios at 3100100, a fresh subset of the existing
   registered training range, advancing independently per episode.
3. After both learning runs finish, evaluate checkpoints 4,096 and 8,192 from both
   seeds on 3200020–3200039: four candidates ×20 = **80 selection matches**.
4. Select by wins, kills, fewer timeouts, earlier step, lower initialization. Write
   an immutable lock before confirmation. Do not swap based on later outcomes.
5. On 3200040–3200059, run the selected model, preserved pilot-83, scripted hunter,
   and untrained family policy: four methods ×20 = **80 confirmation matches**.
   These are fresh confirmation-validation scenarios, not reserved final tests.
6. Stop after this stage or at the **3-hour learning/evaluation ceiling**, whichever
   comes first. Source/config/frozen drift, illegal actions or nonfinite values stop
   the job with preserved outputs. No silent retry or extension.

Measure true wins with uncertainty, kills/witnesses/ejections, task wins/timeouts,
report/claim/vote use, illegal actions, legal kill windows, selection probabilities,
PPO diagnostics, decisions, simulated experience and measured compute. Longer travel
options change simulated experience per decision; report that confound explicitly.

Preflight: **78 tests passed** in 58.02 seconds, including the six new family-policy,
dense-reward, observation-boundary, option/trace, split-gradient/resume and viewer
checks plus 72 existing impostor/V2 mechanics/privacy/integrity checks. Initial
preflight caught an unused WAIT-family mapping and viewer color-field mismatch;
both were fixed before any study run.

## Viewer deliverable

`python run_impostor.py` will load the selected evaluated checkpoint from
`artifacts/impostor-v2/release/policy.pt`. It shows the impostor, frozen opponents,
action-family probabilities, cooldown, kills and outcome. Space/P pauses, Right
steps, +/- changes speed, R resets and N starts another game. Rendering is spectator
data only. A missing checkpoint fails explicitly rather than substituting a script.

The selected model is the best of this registered candidate set according to
validation, not a claim of global optimality. Future cross-play with the friend's
improved crewmate requires a separately pinned compatible checkpoint and shared rules.
