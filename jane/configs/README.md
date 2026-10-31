# Editable profiles and native limits

`jane-settings-v2-draft` validates host settings for 4–15 total players and 1–3
impostors. Crew count is derived. The native engine may support fewer settings
than the draft can represent; always run its explicit compatibility check.

The reviewed Patrick Jane A2 configuration fixes 1 common/1 long/2 short tasks,
one emergency, non-anonymous ballots, always-visible task bar and medium mapped
kill range. Its default emergency cooldown is 15 seconds, first kill cooldown
10, later kill cooldown 45. It enables lights, doors and vents. Modern optional
roles, Clean Vent, ghost impostor sabotage, devices and additional sabotage are
not enabled in that registration.

Current Population permits at most 12 crew. Therefore the supplied native 15p
profile is 12+3. A proposed 13+2/14+1 profile is representable but rejected by the
native adapter. Shape capacity is not proof of engine support or trained skill.

Simulation settings are private research controls. A2 keeps the old lobby.timeout
field at 900 but disables its gameplay ending. Actual technical cuts are 1800s
maximum / 300s stall by default. Historical V2.2 keeps terminal timeout behavior
and no separate stall cut. A process wall timeout is never a game loss.

To migrate an old draft-v1 JSON: keep a copy, use schema `jane-settings-v2-draft`,
and add simulation.stall_cap_seconds (0 for historical, 300 for the future/A2
templates). Do not automatically change population, ruleset or task categories.
To target real A2, start from its native profile rather than relabelling a minimal
curriculum. Frozen studies keep their original files and fingerprints.

Settings exports are detached values. Both future role adapters use the same
settings and shared-bundle fingerprint. Unknown/duplicate fields fail. Advanced
role parameters and unsupported new mechanics require explicit schema/engine work.
