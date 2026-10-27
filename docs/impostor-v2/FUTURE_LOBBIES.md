# Future custom lobbies and shared neural players

Requested future scope: choose up to15 total players and1–3 impostors, validating
legal combinations against the real-game version we target.10crew/2impostors is a
12-player benchmark preset. This is not implemented in the current5-player study.

| Neural component | Normally trained models | Runtime state |
|---|---:|---|
| Crew action policy | 1 shared policy | Independent observations, memory and decisions for every crew actor |
| Impostor action policy | 1 shared policy | Independent observations, memory and decisions for each impostor |
| Existing crew belief predictor | 1 shared predictor | Each crew actor builds private evidence; weights do not share its evidence |

Thus15 AI-controlled seats need15 actor states, **not15 independently trained neural
models**. Actor and critic heads belong to a role policy; do not count them as a
separate policy per player. Extra opponent checkpoint variants are useful for
diversity, but are optional. The inference cost grows with seats and candidate
counts even when weights are shared. No fixed larger-lobby runtime estimate exists
yet; benchmark it before proposing a compute budget.

## Contract and settings checklist

- [ ] Agree with the crew developer on one versioned lobby schema: total players,
  impostor count, allowed standard game settings and named presets.
- [ ] Verify the targeted game version's legal combinations and timing rules from
  authoritative references; do not guess the minimum players for each impostor count.
- [ ] Add validated client controls and persist every setting with matches,
  training runs, checkpoints and evaluation reports.
- [ ] Generalize both action interfaces from4 other players to variable rosters
  with up to14 others; masks/indices must not identify unseen roles.
- [ ] Audit the belief predictor too:4-candidate memory/normalization assumptions
  and probability calibration may need adaptation even if its weights are shared.
- [ ] Reuse shared per-player/candidate encoders where appropriate; explicitly
  represent population/phase counts instead of relying on fixed five-player scaling.
- [ ] Verify action capacity without silently truncating candidates; include
  voting, public claims, sightings, visible bodies and legal kill targets.
- [ ] Version checkpoints and train across declared lobby conditions; old weights
  cannot be presumed compatible or competent merely because tensor sizes fit.

## One to three impostors

- [ ] Give each impostor its legitimate teammate identities, without granting live
  unseen teammate positions, private sightings or hidden crew beliefs.
- [ ] Prevent friendly kills and handle simultaneous actions, same-victim attempts,
  independent45s cooldowns, body/report transitions and parity correctly.
- [ ] Keep actor-private memory and action RNG separate, even with shared weights.
- [ ] Define actor-specific kill/ejection rewards versus faction-wide wins; do not
  accidentally pay every impostor an own-kill bonus for a teammate's action.
- [ ] Preserve living-player witness/report safeguards, voting legality, task victory
  and ghost-only information isolation at every roster size.
- [ ] Test permutations of IDs/seats, deaths/ejections, disconnect-like roster
  changes only if supported, empty candidate sets and deterministic replay.
- [ ] Extra game roles, sabotage and vent mechanics require their own explicit
  supported contracts; a player-count selector does not implement them.

## Learning and evaluation sequence

1. Fix and validate useful single learned crew/impostor behavior in the current
   task-enabled setting. Completed V4.1 improved killing/faking but scored0/20
   fresh wins; it does not expand lobby size or satisfy the reliable-win gate.
2. Freeze compatible checkpoints for explicit mixed and fully neural cross-play.
   Do not attribute a whole-team victory to one learned crew actor without evidence.
3. Implement/test variable lobbies, then separately approve their training protocol
   and compute budget. A policy shared across seats should encounter declared varied
   roster sizes; hold out scenarios and opponent policies for generalization checks.
4. Once individual agents work, explore coordination, adapting opponents and
   self-play. Centralized training critics are optional future designs; execution
   actors must still receive only their legitimate information.
5. Compare general policies with lobby-specific checkpoints if evidence warrants
   them. Preserve failures and report wins, uncertainty and compute by lobby size,
   impostor count, opponent mix and seed, rather than pool incompatible games.
   Count independent matches, not every shared-policy seat as an independent trial;
   report within-game/paired-scenario dependence.

Tracked in [the master checklist](IMPOSTOR_MASTER_CHECKLIST.md) and
[research log](RESEARCH_LOG.md). No code/lobby change or extra training is authorized
by this future-work note.

## Sabotage as a separate hypothesis

The owner asked whether missing sabotage makes wins harder. It may alter task pace
and encounters, but current results cannot identify that cause. V4.1 already won
one training game without sabotage; fresh losses were15 task victories and5 own
ejections. It also self-reported33 times after34 kills. Those choices warrant a
post-kill strategy audit before declaring sabotage the necessary fix.

- [ ] Version actual sabotage/repair mechanics and timing, with tests.
- [ ] Give both roles only legitimate observations and legal actions. Current
  frozen/scripted crew have no trained functional sabotage-response contract.
- [ ] Train/evaluate crew repair responses and impostor sabotage choices separately
  before cross-play. Do not make the impostor appear strong against helpless crew.
- [ ] Keep no-sabotage baseline/checkpoints; preregister paired comparisons and
  compute. Functional vents and free-text persuasion remain separate features.
