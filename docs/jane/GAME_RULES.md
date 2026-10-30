# Shared Jane gameplay rules and settings

30 October 2026. Applies to both Patrick Jane crewmate and Jane Impostor.
This is a **versioned target specification**, not a statement that all mechanics
already run. `jane/shared/catalog.json` and `rulesets.json` are machine-readable
inventories. Ruleset IDs, settings fingerprints, shared-bundle hashes and source
commits must accompany future runs.

## Current versus target

| Layer | Status |
|---|---|
| Historical V2.2 | Existing 4 crew + 1 impostor engine, authored task abstractions, frozen A/B |
| Current Patrick Jane | External system described by the owner; actual source/checkpoint needed |
| Shared settings | Validated 4–15 total players and 1–3 impostors; editable host settings |
| Base Jane profiles | Classic base roles with extension mechanics disabled for staged learning |
| Full classic target | Draft until target build, environment and both adapters are verified |

Base profiles are curriculum variants. Disabled vents/sabotage do not mean they
are absent from real classic Among Us. Do not enable every mechanic at once just
because its name appears in the registry.

## Population and roles

Capacity is **15 total players**. Crew count is `players - impostors`. Normal
classic start constraints used here are at least 4/7/9 players for 1/2/3 impostors.
One shared policy per role can serve all actors; private observations, memories,
navigation and random streams remain independent.

The initial roles are Crewmate and Impostor only. Impostors know ally identities,
not their unseen positions, private observations or goals. Innocents do not know
hidden roles. Known/unknown death status must reflect lawful observation/public
meeting information, not a spectator roster. Ghost is a life state, not a random
third living team. Optional roles are catalogued but all disabled initially.

## Host settings represented by v1

| Group | Editable values |
|---|---|
| Population | Players 4–15; impostors 1–3, subject to start constraints |
| Movement | Speed 0.5–3 in 0.25 increments |
| Vision | Crew and impostor multipliers 0.25–5 in 0.25 increments |
| Killing | Cooldown 10–60 s in 2.5 increments; short/medium/long distance |
| Meetings | 0–9 emergencies; cooldown 0–60 s in 5 s increments |
| Discussion/voting | 0–120 / 0–300 s in 15 s increments |
| Information | Confirm ejects; anonymous votes; task bar always/meetings/never |
| Tasks | 0–2 common, 0–3 long, 0–5 short; visual tasks on/off |
| Optional roles | Per-role maximum count and chance; zero initially |

These ranges describe classic host controls, not proof that the historical
adapter supports each value. See [Options](https://among-us.fandom.com/wiki/Options).
Advanced role ability settings are future schema additions. Map selection is
currently The Skeld only. Cosmetics, server/language/chat filters are presentation
or matchmaking settings and are not neural gameplay mechanics. Hide-and-Seek,
Among Us 3D and optional role modes are outside this ruleset.

The zero-task/zero-voting edge cases need target-build verification; neither is
used in base profiles. Physical simulation timestep, observation interval, match
cap and truncation policy are research controls, not real-game lobby options.
The actor does not receive private training seeds, deadlines or opponent-model IDs.

## Movement, geometry and interactions

- The NN selects a goal or activity. Public-geometry navigation handles movement.
- Walls, closed doors and collidable furniture obstruct movement. Vent covers
  and corpses are not invented solid obstacles. Verify geometry against the build.
- Reached navigation commands must not repeatedly restart completed routes.
- WAIT/continue, cancellation and a later valid revisit stay available.
- Room arrival and console usability use executor-consistent regions, not an
  arbitrary distance to a room centre. Keep the Navigation V2 comparison separate.
- Goal features, path lengths and masks may not reveal unseen players or hidden
  door changes. Collision with an unexpected blockage may produce lawful feedback.

## Tasks: complete inventory, versioned behavior

The bundled asset has **19 task-type IDs**, or 18 canonical families when
Empty Chute is treated as an Empty Garbage variant. Preserve both IDs for the old
engine. Download and Accept Power are stages, not additional independent assigned
tasks. The catalog records original categories and the target review separately.

| Task ID / name | Historical category | Stages or special handling |
|---|---|---|
| SwipeCard / Swipe Card | Common | Admin |
| FixWiring / Fix Wiring | Common | Three selected panels; common assignment shared by crew |
| ClearAsteroids / Clear Asteroids | Long | Weapons; visual; modern category reference conflicts and needs verification |
| AlignEngineOutput | Long | Both engine panels |
| SubmitScan | Long | MedBay; visual |
| InspectSample | Long | Start, incubate, return and select result |
| FuelEngines | Long | Storage pickup and engine visits |
| StartReactor | Long | Reactor task; distinct from meltdown repair |
| EmptyChute | Long | O2 to Storage; garbage-family variant |
| EmptyGarbage | Long | Cafeteria to Storage; final exterior visual |
| UploadData | Short | Selected download location, then Admin upload |
| CalibrateDistributor | Short | Electrical |
| ChartCourse | Short | Navigation |
| CleanO2Filter | Short | O2 |
| UnlockManifolds | Short | Reactor |
| StabilizeSteering | Short | Navigation |
| PrimeShields | Short | Shields; visual |
| DivertPower | Short | Electrical, then accept at the selected destination |
| VentCleaning / Clean Vent | Short | Future: occupancy, blocking and eviction interaction |

Task counts choose assignments, not every possible task in one match. Assignments
and progress are private. Crew ghosts can finish their own tasks. Common tasks
must be assigned consistently; the impostor receives only a legitimate fake list.
Fake work never advances a real task or the task bar. Visual proof requires a real
eligible task event and enabled visuals; an impostor cannot manufacture it.

Current engine timings are authored abstractions, not exact minigame replicas.
Keep stage order, interruption, saved progress, incubation and meeting pauses
explicit. Verify source-version category differences before enabling the target
pool. [Task reference](https://among-us.fandom.com/wiki/Tasks).

## Killing, bodies and reporting

Living impostors choose whether/whom to kill among eligible visible targets.
Range, cooldown, phase and animation rules are mechanical constraints, not a
script saying kill whenever possible. No ally kills. Multiple requests cannot
kill one victim or award the event twice.

Killing creates the correct corpse and life transition. Dead players cannot vote,
report, call a meeting or execute living actions. Bodies remain reportable until
a rule removes them. Living crew and impostors may report; self-report is legal.
Report opportunities must use actual range/phase, not an override silently moving
the actor. Witness evidence must come from real visibility and timing.

Target conformance must cover the first-round cooldown, resets after kills and
meetings, kill reposition/animation, vent/device/repair pauses and exact native
range. Do not infer all pause rules from one generic "busy" flag.
[Impostor reference](https://among-us.fandom.com/wiki/Impostor).

## Meetings, claims and votes

- A lawful report or emergency starts the meeting; identify caller/reporter publicly.
- Emergency count/cooldown, active sabotage and alive status restrict the button.
- Keep discussion, voting, results/ejection and return-to-roaming transitions explicit.
- Structured statements may accuse, defend, describe sightings/location/body or
  witness claims. Questions/replies need a real grammar and receiving-policy support.
- Claims need not be true; preserve sender and source provenance. Do not expose
  hidden truth to validate a lie or script a persuasive response.
- Ballots remain private until publication. Anonymous mode hides voter identity.
- Highest eligible tally controls ejection; ties/skip follow the pinned rules.
- Confirm-eject settings control role/remaining-impostor disclosure. Never leak it
  through another field when confirmation is disabled.
- Dead actors do not talk to or vote with living actors. No ghost information relay.
- Full classic target uses Cafeteria meeting relocation and configured post-meeting
  cooldown behavior; historical in-place/preserved-cooldown behavior stays historical.

[Meeting reference](https://among-us.fandom.com/wiki/Emergency_meeting).

## Vents, doors, sabotage and devices

These extensions are OFF in base profiles; each needs engine and both-policy tests.

| Mechanic | Required shared semantics |
|---|---|
| Vents | Walk to entry, enter, hold, connected hops, exit; observable animations and legal timing |
| Doors | Room-group closure, timed reopening, collision/visibility, replanning and meeting interaction |
| Lights | Crew sight reduction and repair; no automatic impostor/ghost vision reduction |
| Communications | Correct task-bar/task-list/device restrictions and repair; no invented task freeze |
| Oxygen | Critical timer and both repair consoles; sequential completion can be possible |
| Reactor | Critical timer and two distinct simultaneous living repair participants |
| Admin | Lawful occupancy information, not identities or a full simulator dump |
| Security | Lawful camera feeds/activation cues, not global visibility |

The asset has 14 vents in six components, 13 physical doors in seven groups and
four camera entries. That data is not an executable mechanic. The current source
geometry build is unknown and its metadata has a different version history.

Do not teleport between disconnected vents. Vented impostors cannot kill or
sabotage under the target classic profile; kill cooldown pauses. Clean Vent needs
correct eviction/blocking, not a decorative timer.
[Vent rules](https://among-us.fandom.com/wiki/Vent).

Skeld door/non-door sabotage restrictions, shared cooldowns and concurrency must
be enforced. Reference timings in `rulesets.json` remain draft until measured
against the chosen build. Body reports remain possible during sabotage, resolve
critical emergencies and reopen Skeld doors; noncritical emergencies may persist.
Impostors may perform legal repairs. Do not add a reward for pressing sabotage.
[Sabotage](https://among-us.fandom.com/wiki/Sabotage),
[reporting](https://among-us.fandom.com/wiki/Report),
[The Skeld](https://among-us.fandom.com/wiki/The_Skeld).

## Ghosts, victory and terminal handling

Crew ghosts work only on their own tasks and cannot share private information with
living actors. If an impostor is ejected while an ally survives, a future enabled
ghost policy can sabotage. No ghost kill/report/emergency/vote/vent/living repair.
Retain that actor's own history; never share its hidden state with the ally.
[Ghost reference](https://among-us.fandom.com/wiki/Ghost).

Crew win by completing the eligible crew task set or eliminating all impostors.
Impostors win by living parity or an unresolved enabled critical sabotage.
Resolve simultaneous events in a documented, verified engine order. Disconnections
require explicit roster/task handling and are future work, not an assumed win rule.
If all impostors are eliminated, do not continue ghost sabotage after termination.

A real-game match does not inherently end at the research cap. New profiles mark
that cap as truncation; historical timeout losses remain unchanged. A process
timeout is never a neural gameplay loss.

## Future activation rule

For every extension: pin the rule/build, implement the state machine, define actor
observations/actions, add conformance/privacy tests, verify crew counterplay, then
enable it in a NEW profile and study. Keep completed studies immutable.
