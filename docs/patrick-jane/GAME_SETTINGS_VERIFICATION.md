# Classic Skeld settings verification — 2026-10-28

Scope: Patrick Jane only. The previous V2 model, environment and checkpoints are
historical controls. No special roles; intentional population remains 10 crew /
2 impostors, with configurable 4–15 total players. New PJ source locks are
required after the corrections below. No claim of exact retail equivalence.

## Sources and confidence

Innersloth's [16.0.0 lobby update](https://www.innersloth.com/a-match-made-in-update-16-0-0-emergency-meeting-40/)
is primary evidence for adjustable lobby/role settings, not every numerical
default. The [historical GameOptionsData specification](https://github.com/roobscoob/among-us-protocol/blob/master/07_miscellaneous/01_the_structure_of_the_gameoptionsdata_object.md)
explicitly lists the owner's baseline defaults. Its old player-capacity limit
and versioned protocol make it unsuitable as proof of a 2026 preset.
The recently indexed [Options reference](https://among-us.fandom.com/wiki/Options)
corroborates settings and 4–15 capacity, but available indexed formatting does not
reliably identify underlined defaults. Current numerical Core preset defaults
are **unverified**. We retain the owner's explicit historical profile.

| Setting | Registered historical profile | Earlier PJ implementation | Corrected PJ / confidence |
|---|---|---|---|
| Speed | 1.0× | 1.0× (2.5 map units/s) | Unchanged; historical default high, unit conversion simulator approximation |
| Crew / impostor vision | 1.0× / 1.5× | 4.5 / 6.75 map units | Unchanged; role-local visibility and walls; absolute unit calibration approximate |
| Kill cooldown | 45 s repeating | 45 s, including initial | Repeat 45 s; initial 10 s; reset to 45 after meeting |
| Kill distance | Medium | Two character widths | Retained metric approximation; historical enum 1=Medium, physical retail distance unverified |
| Emergency meetings | 1 per actor | 1 | Unchanged; high historical confidence |
| Emergency cooldown | 15 s after meeting | 0 | 15 s after meeting, actor-visible remaining time; initial button available |
| Discussion / voting | 15 / 120 s | 15 / 120 | Unchanged; voting may end early after all eligible ballots |
| Tasks | 1 common / 1 long / 2 short | Four individually owned multistage assignments | Unchanged; abstract interactions, not retail minigame execution |
| Confirm ejects | On | On | Unchanged; publicly confirmed role at resolution |
| Anonymous votes | Off | Off | Unchanged; targets revealed together at resolution, never early |
| Visual tasks | On | On | Unchanged; proof only for actors with legitimate line of sight |
| Task bar | Always | Always | Unchanged; aggregate progress public |
| Population | Intentional 10+2 | 10+2; up to 15 | Unchanged target; minimum total corrected to four |

## Confirmed mechanics and corrections

The [Sabotage reference](https://among-us.fandom.com/wiki/Sabotage) supports
initial 10-second and subsequent 30-second cooldowns. PJ previously used 30
initially; corrected. Both Skeld critical deadlines remain 30 seconds. Oxygen
uses Admin and O2; reactor requires two live actors simultaneously holding
distinct real map scanners. A held scanner now persists until the actor changes
intention, leaves, dies, or a meeting starts, instead of requiring identical-tick
button submissions. Only actual contributors receive the unchanged repair reward.

[Fix Lights](https://among-us.fandom.com/wiki/Fix_Lights) supports the existing
0.2 living-crew vision multiplier and unaffected impostor sight. Lights now also
blocks the emergency button. Reports cancel oxygen/reactor but leave lights
active, consistent with the [Report reference](https://among-us.fandom.com/wiki/Report).
One supported sabotage is active at a time; the next 30-second cooldown starts
at resolution/cancellation. The shared sabotage clock continues through meetings.

The [Kill reference](https://among-us.fandom.com/wiki/Kill) supports the initial
10-second timer and configured cooldown after meetings, plus the killer's lunge
to the victim's position. Witness eligibility is computed before that movement.
The [Emergency button reference](https://among-us.fandom.com/wiki/Emergency_button)
supports its cooldown after meetings and sabotage restrictions. Surviving actors
return to valid Cafeteria spawn positions after resolution; private memories
persist and ghosts retain their task ownership. Cafeteria positions approximate
the retail table arrangement.

High confidence applies to these documented mechanics, not every game's UI,
animation duration or retail default. Finite physics ticks use deterministic
ordering: pre-action report/emergency before kills; repair on the last legal tick
before critical failure; failure before task completion. Exact same-frame retail
engine ordering is unverified and this simulation ordering is explicitly retained.

## Remaining approximations

No vents, doors, communications sabotage, cameras/Admin console intelligence,
special roles, kill/meeting animation delays or manual task minigames. Lights
and oxygen repairs abstract their panels as legal console interactions. Reactor
holds model participation, not a GUI. Ghosts use the historical deterministic
task executor rather than learned ghost decisions. Task incubation and ghost work
pause during meetings. A 900-second simulation timeout is a research bound, not a
retail win rule. These omissions prevent claiming full out-of-box equivalence.

No unverified physical Medium distance or new sabotage subsystem was silently
introduced. Such additions require a separately documented implementation/design
decision. Role labels and spectator records remain observer-only; actors receive
only their own legitimate observations, timer, and repair-hold state.
