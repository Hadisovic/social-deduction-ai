# Phase 6 vision and rules

Opt in with `GameConfig(rules_version=6, crewmate_sight_range=4.5,
impostor_sight_range=6.75)` or the Phase 6 reference experiment.
`GameConfig()` still means historical shared 4.5 vision. Supplying role distances
with legacy rules fails instead of silently changing old experiments. Distances
must be finite and positive; report range must fit both sight ranges and kill
range must fit impostor sight. Independent role values are configurable; the
registered reference experiment additionally requires crew < impostor.

Innersloth lists separate [crewmate and impostor vision settings](https://innersloth.zendesk.com/hc/en-us/articles/8531332682004-What-accessibility-options-are-available-in-Among-Us).
It does not establish a conversion to this map's distance units. Our 4.5/6.75 pair
preserves the historical crew reference and tests a 1.5× impostor radius. It is a
research configuration, not an exact commercial preset. A point exactly at the
effective range is eligible; walls, props and boundary contact still block sight.
Lighting/sabotage, visual cones and through-wall information are not implemented.

`GameConfig.effective_sight_range(role)` is the sole role-range resolver.
`Phase3Game.settings_for(role)` supplies a detached `ObservationSettings` with
that effective range to `project_game`. The public map view carries the same
effective range. No extra field or reference is added to actor packets: own
impostor cooldown remains role-local; other roles, tasks/cooldowns and unseen
positions stay hidden. Sight does not change kill/report/navigation distances.

Witness delivery uses the observer's role/range **at the elimination tick**, and
requires clear LOS to both killer and victim. Moving closer later cannot acquire
an old witness event. Moving away cannot erase an event legitimately observed.
Observation caches are per actor and invalidated on world transitions. Direct
evidence comes from the actual projector; future/privileged logs are not forwarded.
Eliminated actors have no local sight/actions; meetings expose legitimate public
context rather than roaming geometry.

The Phase 6 spectator overlay resolves the same radius and ray-clips at native
barriers. Its sampled outline is approximate rendering; exact entity/event
eligibility uses `visible_between`, not the pixels. Press V in `run_phase6.py`.
The overlay is explicitly labeled spectator-only and never enters policy input.
Legacy viewers retain their original range-circle rendering.

Config serialization omits new fields for legacy version 1, preserving replay
hashes and old dictionaries. New replay metadata records version 6 and both
distances. Historical release files, map JSON, old benchmarks and Phase 5 runtime
sources are unchanged. Fresh asymmetric evaluations must compare all methods
under the same new rules; historical 88.4%/90.4% results remain Phase 5 evidence.
