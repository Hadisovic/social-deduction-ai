# Future features: reserved, disabled and staged

Do not turn the learning problem into every Among Us mode at once. The current
delivery adds settings/registries/plans, not these engine implementations. Base
Jane profiles enable only base Crewmate/Impostor roles. Features become usable
only after both roles and the engine pass the same capability contract.

## Mechanics backlog

| Stage | Capability | Gate before activation |
|---|---|---|
| Foundation | Current Jane import and 15-slot variable lobby | Real source, loader, private-state/padding checks |
| Foundation | Full ordinary Skeld task semantics | Console/stage/cancellation/timing and visual-proof tests |
| Foundation | Meetings, claims, ballots and settings | Truth isolation, anonymity, cooldowns and relocation |
| Extension 1 | Security and Admin | Actor-safe device views and communications behavior |
| Extension 1 | Vents and Clean Vent | Real connectivity, occupancy, entry/exit exposure, eviction |
| Extension 1 | Doors | Timers, reopening, collision/LOS and private-knowledge replanning |
| Extension 2 | Lights and communications | Effects, repairs and meaningful crew responses |
| Extension 2 | Oxygen and reactor | Critical timers, repair concurrency, outcomes and report effects |
| Extension 3 | Two/three-impostor coordination | Shared team state arbitration; independent actor memories |
| Extension 3 | Ghost impostor assistance | Life-transition memory and restricted sabotage capabilities |
| Research | Mixed/adapting opponents and self-play | Both learned base roles first demonstrate held-out competence |
| Later | Disconnect/reconnect and late lifecycle events | Roster/task/victory semantics and private-memory rules |
| Later | Other maps/modes | Separate map/rule contracts; no forced reuse of Skeld metadata |

All task types are inventoried now; Clean Vent remains excluded from the old
runtime. Exact minigame input learning is separate from the current timed-task
abstraction. Do not claim timed completion taught the NN the real minigame.

## Optional-role catalog

All are set to zero count/chance in every supplied profile. Settings v1 reserves
their count/chance only. Their advanced ability controls need versioned schemas,
engine state, observations, masks, tests and retrained/adapted opponents.

| Role | Side/life | Future capability and configuration to define |
|---|---|---|
| Scientist | Crew | Portable vitals, battery and recharge controls |
| Engineer | Crew | Legal vent access, cooldown and maximum vent duration |
| Guardian Angel | Crew ghost | Protection, cooldown, duration and visibility |
| Shapeshifter | Impostor | Appearance identity, duration/cooldown and evidence |
| Noisemaker | Crew | Death alert timing/duration and impostor visibility option |
| Tracker | Crew | Tracking use, delay/duration/cooldown and legitimate minimap evidence |
| Phantom | Impostor | Invisibility lifecycle, cooldown/duration and action restrictions |
| Detective | Crew | Notes/case state and limited interrogations |
| Viper | Impostor | Acid/body dissolution stages, timing and reporting interactions |

Detective and Viper are confirmed in Innersloth's
[17.0.0 announcement](https://www.innersloth.com/new-roles-are-on-the-scene-of-17-0-0-emergency-meeting-41/).
Existing advanced-role controls are described in the
[options reference](https://among-us.fandom.com/wiki/Options). New future roles
must be added explicitly; "all roles" is a reviewed catalog, not an unrestricted
plugin that silently enables unknown abilities.

Ghost is a life state, not another independently assigned base role. Hide-and-Seek
and 3D roles are separate projects. Cosmetics and UI additions should not expand
the neural observation contract without a reason.

## Avoid premature complexity

No separate NN per player, vent or sabotage. Start with one shared-weight model
per role, private memory per actor, the in-progress Jane architecture as reference and
legal capability masks. Do not add language-model persuasion, privileged ally
channels, a large transformer or a new graph encoder by default. Measure a missing
capability before adding capacity. Change one study factor at a time.
