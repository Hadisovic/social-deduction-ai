# Shared interface and compatibility agreement

`jane-settings-v1-draft` validation is implemented. `jane-actor-contract-v1-draft` is a semantic
design. It is not an invented tensor adapter for absent Patrick Jane code.
Both models are in progress. Wait for the owner's teammate-push announcement,
pull/read that work, then revise compatibility for both. No network copy or final
folder/tensor structure is prescribed by this draft.

## Same game; separate policies

Both roles use one settings JSON file, its SHA256, the shared bundle receipt and
one engine/ruleset. Role adapters expose the same semantic event/action language,
but role-specific masks and private task/ability information. Tensor schemas and
trained weights can differ. Neither an old FourCrewPolicy nor the historical 75K
impostor checkpoint is relabelled as Jane.

The current Jane self/player/event widths, scaling, tensor ordering and recurrent
sharing must come from its real source. The reported reference remains in
`jane/shared/network_reference.json`; it is not an executable model definition.

## Semantic envelope

Each packet identifies match, actor, tick/time, phase and own life state. It
contains public settings, own state, lawful current/remembered observations, own
real/fake tasks, public events and available actions. Capacity is 15 public roster
slots including self; shorter lobbies are masked. Task capacity is 10 assignments.
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
