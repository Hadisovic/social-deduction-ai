# Editable profiles

Both future role adapters should consume one `jane-settings-v1-draft` profile. The loader
requires all fields and rejects unknown or duplicate keys. Values are immutable
after loading. `GameSettings.fingerprint()` binds the full profile to a run;
`public_settings()` omits trainer timing/profile bookkeeping from actor input.

Change `lobby.players` (4–15) and `lobby.impostors` (1–3); crew count is derived.
Classic start minima are 4/7/9 players for one/two/three impostors. Capacity is
15 total players, not 15 crewmates plus impostors. A 12-player profile is 10+2.

Modify task counts, meetings, vision and cooldowns in the same file for both
roles. All modern optional-role counts/chances start at zero. Advanced role
ability parameters remain future schema work, not silently ignored JSON fields.

`mechanics` flags are requested capabilities. A true flag does not implement its
state machine. The runtime gate must reject unavailable capabilities; no silent
fallback to a different game. The initial profiles are curriculum variants.

The first-round cooldown, meeting relocation and cooldown reset are ruleset
semantics, not arbitrary switches. `historical-v2.2` retains its old behavior;
`classic-skeld-v1-draft` describes the future verified target. The exact real
build is intentionally null until conformance is established.

Zero tasks and zero voting time are representable UI options. Their resulting
match/phase semantics must be verified by the new engine, and they are not used
in the initial learning profiles. No adapter may invent those semantics.
