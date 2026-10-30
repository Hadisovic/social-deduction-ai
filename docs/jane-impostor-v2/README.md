# Current fix:31October2026

The cache-coherence fix is implemented in the shared adapter as `paired-map-planner-v1`. [Fix, migration and verification](CACHE_PAIR_FIX.md). No new training study started; the earlier pilot remains failed.

# Lizbon Classic V2

Status: the approved pilot **stopped on an engine geometry-cache error** after
37m12.5s. Three completed training games all lost to tasks;24new PPO passes were
saved across two streams. No validation/selection/confirmation ran. Read
[the results and exact next steps](study-20261030/README.md). No further training
started. Historical models and the failed-study source remain preserved.

## What changed and why

| Change | Purpose |
|---|---|
| Separate `jane/classic_v2` shared adapter | Both role developers can use the same rules without invalidating old checkpoints. |
| 4–15 players; 1–3 impostors | Remove the native 12-crewmate limit, including 14 crew + 1 impostor. |
| Editable Common / Long / Short counts | Host controls task workload. No separate minigame difficulty setting. |
| Unique canonical task types per owner | Prevent duplicate assignments; merge garbage/chute alternatives into one family. |
| Common types and plans selected once per match | All crew receive the same common tasks. Individual progress remains private and separate. |
| Dead/underground stored positions use base floor for invariants | Fix the V1 door crash. Living surface actors still must obey current walls/doors. |
| Versioned Lizbon V2 runner and explicit weight transfer | Reuse the saved neural weights safely; retain exact historical V1 loading. |
| Compressed log of every impostor decision | Inspect available actions, probabilities, selection, local inputs, effects and rewards. |

## Host settings

Use the dedicated Jane Python runtime. From the repository root:

```powershell
$py = '.\local-runs\jane-impostor-v1\runtime\Scripts\python.exe'
& $py tools/configure_classic_lobby.py --players 15 --impostors 3 --common 2 --long 2 --short 4 --output local-runs/my-lobby.json
```

Files are created exclusively: choose a new output name rather than overwriting a
registered profile. Inner settings are editable JSON; outer engine/allocation
versions distinguish this implementation from the preserved draft and A2 profiles.

| Setting | Supported values / behavior |
|---|---|
| Total players | 4–15 |
| Impostors | 1–3; conservative shared start minima 4 / 7 / 9 respectively |
| Tasks per crew | Common 0–2, Long 0–3, Short 0–5 |
| All task counts zero | Effective one Short task; avoids an immediate task win |
| Tasks shared between different players | Allowed and required for common tasks |
| Duplicate task family for one player | Disallowed |
| Repeated stages of one task | Allowed: fuel trips, wiring panels, upload/download, etc. |
| Task bar | Always / Meetings / Never; hidden global progress does not enter the actor |
| Votes | Public or anonymous; anonymous voter-target pairs are not actor publications |
| Discussion | Includes zero: starts voting immediately |
| Voting | Positive configured duration; zero/infinite is deferred |
| Emergency meetings | Configured count/cooldown; emergency blocked during active sabotage |
| Kill cooldown | Host setting; initial cooldown remains explicitly 10 simulated seconds |
| Kill distance | Native medium only; short/long calibration deferred |
| Other switches | Existing speed, role vision, visual tasks, confirm ejects; see limits below |

The existing neural architecture fits 15 players: 14 other-player rows. One shared
Lizbon policy controls all impostors, with separate private recurrent state, memory
and sampling generator for each. The crew policy is a different frozen network.
Changing population alone does not require a new network per avatar. It does
require training and evaluation evidence for the new population/settings.

## Run a custom engineering match

```powershell
$source = 'C:/Users/masaa/Desktop/UNI/summer_phase/stealth-rl-navigation/temp_jane_foundation/upstream-review/d459f38674ebc537acde20647cbd07700c494533'
& $py run_lizbon_classic.py --source-root $source --profile local-runs/my-lobby.json --warm-start-v1 artifacts/lizbon-v1/pilot-20261030/checkpoint-001.pt --output local-runs/my-lobby-smoke --mode smoke --smoke-observations 120 --max-minutes 3 --match-wall-seconds 150
```

This command performs bounded inference, not PPO training. The transferred model
has not been validated in that custom lobby. `--checkpoint` strictly requires the
same V2 profile/source/runtime. `--transfer-v2` explicitly copies weights to a new
profile with a fresh optimizer and records the source profile/checksum. An explicit
same-profile optimizer resume is available; it is not an automatic restart.

Playback of a completed recording, not a new simulation:

```powershell
& $py run_lizbon_viewer.py --source-root $source --replay local-runs/my-lobby-smoke/match-001.jsonl
```

## What the neural policy decides

The unchanged 1,036,561-parameter architecture encodes own state, up to 14 other
players and 64 observed/remembered events, then uses a 256-unit GRU and separate
actor/critic trunks. It scores legal candidate actions hierarchically by family
and target. Its 16 families cover continuing, cancelling, travelling, faking,
killing, reporting, emergencies, claims, voting, skipping, lights, repair, doors
and the three vent actions. Navigation executes selected goals with A-star.

Rules/masks determine what is legal; they do not choose a strategy. Arrived or
unchanged active static goals are excluded. Repeated fake-task continuation does
not restart the activity. Waiting, abandoning, revisiting and reporting remain
policy choices. There is no scripted hunting, forced kill, anti-report rule,
automatic accusation, automatic defence or guaranteed deception strategy.

The model sees its legitimate cooldown, allies, own route, visible actors/bodies
and its private observation history. It does not see unseen victims' live
positions, other players' hidden task assignments, their private beliefs or votes,
or spectator-only truth. Each actor receives its own state. Ghost task completion
does not communicate ghost information to living agents.

## Rewards and optimizer

Keep the V1 team-reward baseline while isolating the engine/task change:

| Event | Reward per impostor |
|---|---:|
| Team victory | +10 |
| Team defeat | −10 |
| Accepted team kill | +1 |
| Eligible cooldown-ready simulated time | −0.005 / second |
| Fake, movement, sabotage, report, accusation | No direct bonus |
| Technical wall/simulation/stall cut | Excluded from PPO and win-rate denominator |

Ready-time accounting handles cooldown crossings and excludes meetings, eliminated
actors, underground/transition periods and terminal intervals. Recurrent PPO uses
Adam 3e−4, four epochs, sequence length 128, clip 0.2, entropy 0.01, gradient clip
0.5, gamma 1 and per-simulated-second lambda 0.995. Update only after complete
natural matches. Preserve all incomplete/failed prefixes.

This baseline does not prove good searching or tactical sabotage. Audit results
will guide separately versioned reward experiments. Positive movement/novelty
bonuses need exploit tests before adoption; they can reward aimless running.

## Reading the decision evidence

Each match has a replay, result JSON and `.decisions.jsonl.gz`:

- `decision`: actor-local encoded inputs, candidates and their exact commands,
  joint probabilities, chosen index, log probability, entropy and value estimate.
- `effects`: separately labelled offline world events and impostor motion/timers.
  These are never fed into the actor.
- `reward`: learner-only components and totals.
- `failure` / `closed`: preserve an interpretable prefix if a run fails.

Run `python -m jane.impostor.v2.analysis RUN_FOLDER` for counts, contiguous action
availability windows, repeated targets, idle choices, movement, faking, effects
and reward components. Use the actual replay for context. A probability explains
which action was preferred, not a human-like internal reason. Availability
windows and adjacent decisions are correlated; neither is test accuracy.

## Training and evaluation plan

1. Finish engine/privacy/checkpoint regressions and a longer engineering match.
2. Measure collection, logging and PPO time. Obtain the owner's new study budget;
   the expired V1 90-minute approval cannot be silently extended.
3. Freeze source/profile/rewards and register scenario lists, wall caps, candidate
   checkpoints, selection score and stopping rules before training.
4. First establish learning at fixed 12 players / 2 impostors. Keep a same-start
   reference. Use training scenarios 9710001–9711000 and fresh development scenarios
   9720001–9721000. Do not train on validation/confirmation outcomes.
5. Compare natural wins/loss causes, kills, witnesses/ejections, ready-time idling,
   navigation loops, fake duration, accepted sabotage/report/vote/claim effects,
   and PPO entropy/KL/value/gradient diagnostics. Inspect both impostors separately.
6. Lock selection before paired held-out confirmation; reserve fresh scenario
   offsets in advance. Never open 9730001–9731000 or historical final partitions.
7. Add registered population/task generalization checks and then balanced sampling
   of 4–15-player configurations. Shared shape support is not generalization proof.
8. Repeat with independent initialization seeds and stronger frozen crew snapshots
   before considering adaptation/self-play.

## Is Lizbon good yet?

Not established. V1 completed one initial-policy loss, saved eight optimizer
passes, and crashed in game two. This new pilot saved24additional passes but
stopped on a different cache-coherence error. Its three natural games were task
losses; all validation/confirmation remained unopened. A few kills or changed
weights cannot demonstrate reliable searching, deception or improved cross-play.
The controlled cache reproduction and proposed patch are linked in the results.

## Next steps for the two developers

**Owner / impostor:** finish the fixed-profile learning study and diagnose
decisions; compare against the saved starting weights. Investigate actual loops,
kill opportunities, witnessed kills and losses before changing network size.

**Friend / crew:** use the same `HostProfile`, allocator and engine adapter. Train
crew on its own branch; preserve ownership/ghost isolation and verify living body
reporting and remembered witnessed-kill voting. Published update102 is learned
task-pursuit V1, not an A2-trained strategic opponent.

**Together:** exchange immutable checkpoint/profile/source receipts. Run frozen
cross-play before either opponent adapts. Gradually add a population/task
curriculum. Add new mechanics only after both roles can observe/respond to them.
Special roles, full sabotage coverage and self-play remain later work.

See [RULES_AND_SOURCES.md](RULES_AND_SOURCES.md) and [CHECKLIST.md](CHECKLIST.md).
