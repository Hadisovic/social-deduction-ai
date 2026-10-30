# Shared interface and compatibility agreement

`jane-settings-v2-draft` validation and native A2 configuration mapping are
implemented. `jane-actor-contract-v2-draft` remains a future paired design. The
owner announced the push and the actual source was reviewed first. No network is
copied or finalized; actual learned role adapters remain pending.

## Same game; separate policies

Both roles use one settings JSON file, its SHA256, the shared bundle receipt and
one engine/ruleset. Role adapters expose the same semantic event/action language,
but role-specific masks and private task/ability information. Tensor schemas and
trained weights can differ. Neither an old FourCrewPolicy nor the historical 75K
impostor checkpoint is relabelled as Jane.

Current tensor widths, feature order and sharing were verified from the pinned
source. `network_reference.json` records the reference, not a copied network.
Its actor/critic share encoders, attention, fusion and GRU, with separate trunks.

## Semantic envelope

Each packet identifies match, actor, tick/time, phase and own life state. It
contains public settings, own state, lawful current/remembered observations, own
real/fake tasks, public events and available actions. Capacity is 15 public roster
slots including self. Native neural rows are14 OTHERS, with own state separate.
Voting is14 other slots plus skip at14; target indices are dynamic/packet-local.
Future task capacity is10 but native A2 fixes four. Expanding tasks needs a new
feature-layout and commitment audit, not silent truncation or repacking.
Roster order must not depend on hidden roles. Do not pad with true unseen state.

Events distinguish direct observation, public claim, public result and own action.
Include source/sender, subject, time/age and location validity. Public ballots are
available only after release and obey anonymity. Belief predictions are labelled
estimates, not true role fields. Reject world objects, back-references, future
events, private ballots and other actors' observations/goals/memory.

Impostors may know ally identities. Shared parameters are not a communication
channel. Every actor owns its recurrent state, history, activity, RNG and goals.
Ghost information cannot be transferred privately to a living teammate.

## Actions and long options

Start from the real Jane navigation/interact/communication/vote/target heads.
Add impostor capabilities only where the engine can execute them. Preserve neural
choices: WAIT, continue, cancel, fake, kill, report, sabotage, claim and vote are
not selected by a strategy script.

Use executor-consistent target regions. Separate an arrived travel command from
an available interaction/fake action. Continuing a route or fake activity does
not restart it. A destination can become selectable after leaving it. The existing
Navigation V2 work remains separate and reproducible.

For rich actions, capacity must be proven. If claim type × player × location
exceeds the candidate budget, use conditional arguments with exact summed
log-probabilities. Never silently drop legal actions. Do not change the existing
Jane head layout until the source and candidate contracts have been inspected.

## Recurrent and lifecycle agreement

New match resets private state. Meetings preserve memory. Valid ghost transitions
preserve that actor's memory and change its capability mask. No stale state across
actors/matches; no automatic reset after each action. Held navigation options still
need observation/event updates. PPO trains ordered sequences and records every
mask, argument selection, duration and lifecycle/truncation boundary.

## Compatibility gates

1. Strict settings validation, including unknown/duplicate-key rejection.
2. Engine capability and ruleset verification for the selected population/settings.
3. Real source/checkpoint/runtime/tensor schema registration for both role models.
4. Same semantic observation/action version and exact settings/bundle hashes.
5. Deterministic adapter checks, legal-action tests and hidden-state isolation.
6. Confirm no private truth through observations, masks, route costs or devices.
7. Recurrent PPO and terminal/truncation/reward checks before training.

`check_pair()` only checks declarations. It cannot authenticate a hand-written
manifest; the future trusted loader must validate actual bytes/source and tests.
`legacy_lobby()` constructs the unchanged historical settings object only. It
does not load weights or launch a game. Old model validity still depends on its
original registered study settings, not only constructor acceptance.

No auto-resize, `strict=False` load or inferred migration. New head dimensions or
feature ordering mean a new tensor schema. Preserve original weights, results and
source hashes, and record any explicit transfer separately.

## Editing safely

Shared definitions live only in `jane/shared`; do not duplicate them into both
role folders. An incompatible edit bumps the schema/ruleset and adds a migration.
The CLI hashes the settings and exact shared definition files, so a same-name
catalog/rules edit changes the bundle hash. Role manifests are excluded from that
bundle to avoid circular hashes. A run must also record source commits and actual
checkpoint hashes. Add a new profile instead of editing a frozen run's profile.
New Jane paths use LF line endings so shared bundle fingerprints survive Windows
and Linux checkouts. Existing historical raw-byte asset rules remain unchanged.

## Actual native differences

The source Observation has impostor teammate/own ability fields, but the crew
encoder does not expose all of them to an impostor policy. Its five interaction
outputs are travel/task/report/lights repair/emergency, not kill/vent/sabotage.
Not every host setting is encoded in the192-wide self vector. A role-specific
feature/head design and settings-conditioning audit remain necessary.

Preserve native event codes: A2 appends vent-enter/exit/obstruction without
renormalizing old values. The code uses direct/public/claim provenance. Shared
semantic labels must map explicitly, never reinterpret old weights by shape.
Runtime2.7.1+cu128 and strict source-locked checkpoints remain upstream; this
branch's2.6CPU environment only tested configuration construction, not the model.
