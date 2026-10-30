# Approved Lizbon Classic V2 bounded study

Owner approved **three hours maximum** on 30 October 2026 after a measured
12-player engineering match took 142.67 collection seconds (143.18 total).
This replaces neither the historical V1 study nor its expired deadline.

## Freeze before execution

Use the commit and UTC start/deadline recorded in the supervisor registration.
Pinned external source d459f386 and frozen task-pursuit update102 weights are
verified at activation. Do not modify loaded source during healthy jobs.

Training profile: 12 total, 10 crew / 2 impostors, 1 Common / 1 Long / 2 Short,
45-second kill cooldown, lights/doors/vents, unchanged V1 network and team rewards.
Two processes each use one Torch CPU thread. GPU is not used. Machine verified:
Intel Core i9-14900HX, 24 cores / 32 logical processors.

## Fixed protocol

1. Warm-start both streams from preserved V1 checkpoint001 (eight old optimizer
   passes). Fresh Adam; private per-impostor memory/GRU/RNG. Same starting weights
   mean these are sampling streams, **not independent initialization replications**.
2. Up to four complete training matches each, scenarios 9710001â€“9710004 and
   9710101â€“9710104. Shared 90-minute training cutoff. Each match wall cap 900s.
   Complete natural games only enter PPO; cuts/failures remain preserved.
3. Candidates: starting reference plus last complete, full-PPO checkpoint in each
   stream. A fatal child error stops the study without an automatic retry.
4. Each eligible method gets the same two sampled validation scenarios:
   9720001â€“9720002. Up to two worker processes; 45-minute validation cap inside the
   total 90-minute evaluation allowance. Per-match cap 600s. Missing/cut validation
   prevents selection rather than treating it as a loss or ignoring it.
5. Select by natural win fraction, then mean eliminations, then prefer reference
   on an exact tie; lexical candidate ID is the final deterministic tie-break.
   Write `selection-lock.json` exclusively, with selected checkpoint SHA256.
6. After lock only: paired reference-versus-selected confirmation. Sampled mode
   uses fresh 9720101â€“9720102; greedy mode uses fresh 9720103â€“9720104. Eight matches
   total, two workers, within remaining evaluation/three-hour deadline. Do not tune
   or change selection from these outcomes. Cuts mean incomplete confirmation.
7. Final sets 9730001â€“9731000 and every historical final partition remain sealed.

## Analysis and stopping

Record all choices/inputs/candidate probabilities separately from offline effects.
Analyze both impostors: voluntary idle/repeated goals/cancels, visible pursuit,
kill/report windows and actual accepted events, fake durations, witnesses,
votes/claims, sabotage/vent effects, team contention, reward components and loss
causes. Adjacent choices/windows are correlated. A few wins cannot establish
intelligent play, robust strategy or a globally best model.

Graphs: training kills/return, PPO entropy/value/KL, development win fractions
with Wilson intervals, behavior distribution/motion, and processor wall times.
Report all failures/cuts and frozen crew limitations. The team must inspect actual
gameplay before deciding on another reward/network experiment. No automatic
extension or further training after the deadline.

Expected wall time depends on meetings and survival; early completion is allowed.
The supervisor records actual UTC times and uses a monotonic hard budget. Offline
report regeneration is allowed after compute ends; it never starts another match.

## Recorded outcome

The registered pilot failed after37m12.5s on an engine geometry/planner mismatch. Three complete training task losses produced24new PPO passes; two cuts and two interrupted prefixes were excluded. No validation/selection/confirmation or automatic retry occurred. See study-20261030/README.md. This plan preserves the original protocol, not a claim that its schedule completed.
