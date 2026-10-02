# Information contract v1

Status: Phase 1 contract and executable crew-only boundary. Game rules below are
settled defaults for the future simulator, not implemented gameplay.

## Rules that determine information

- Initial game: four crew and one impostor, stable public five-player roster.
- Static geometry, room topology and console locations are public known-map data.
  Exact own position/room and exact observed entity positions are accepted
  localization abstractions. Internal simulation ticks are public timestamps.
- Sight is 360 degrees, range limited, blocked by opaque closed rectangles.
  A point exactly at range is visible; a wall touch blocks sight. No hearing,
  through-wall identities, visual task verification, vents or sabotage in v1.
- Tasks are private assignments with private own progress. No live team task bar.
  A visible interaction animation is ambiguous and can be mimicked; it is never
  labeled as genuine completion. Task durations are nonzero in Phase 3.
- Deaths are not globally announced at occurrence. A visible body confirms its
  victim; a report publicly reveals reporter, victim and room, not killer or death
  time. Reports do not reveal the reporter's prior route or exact body coordinates.
- Meetings publicly announce their participant roster; this reveals absence/death
  status at that point, not continuously during roaming. Previously seen alive
  players remain last-seen-alive in memory until evidence supersedes that record.
- Discussion is turn-based; each speaker may make a bounded structured claim.
  Claims may be false, including claims by crew. The engine authenticates the
  speaker and delivery time, never the claim's asserted fact or timestamp.
- Voting is a distinct phase. Votes are public when cast, at most one per eligible
  player per meeting. Skip and self-vote are allowed. Ties/skip cause no ejection.
  Ejection reveals identity but **not role**. Full roles remain training-only even
  at match end in this actor API; a separate spectator may reveal them.
- Own elimination is known immediately. Eliminated actors receive no local vision
  and have no actions; no ghost access. Public announcements may still be recorded.
- Proposed Phase 3 outcomes: crew win if all assigned tasks complete or impostor
  ejected; impostor win at parity. Eliminated crew tasks stay in the fixed quota
  and will be reassigned privately to surviving crew (never silently removed).
  Meetings pause movement/task timers. A fixed match deadline yields a public draw.
  Rule transitions, reassignment and simultaneous-action ordering are Phase 3 work.
- Phase 1 fixture defaults: sight 220 world units, interaction/report radius 25.
  Emergency meetings require proximity to a known meeting console and one unused
  personal meeting allowance. These distances are explicit public map parameters.

## Four distinct domains

| Domain | Contents | Permitted consumers |
|---|---|---|
| `WorldTruth` | Roles, all positions, private tasks/cooldowns, bodies/death ticks, internal events | Simulator, trusted projector, training/evaluation |
| `ActorObservation` | Own state/tasks, public roster/map/context, currently visible entities, typed received events, safe action candidates | Actor and actor memory |
| `Knowledge` | Sequence of that actor's legitimate immutable observations only | Actor and later evidence summarizer |
| `RoleLabels` | Full per-player role targets keyed by match/tick | Training/evaluation only |

No actor object contains a reference to truth, labels, debug metadata or a helper
that can query them. Another player's role, cooldown, task assignment/progress,
unseen position/action/death or death timestamp must not enter actor fields.
Unknown status is not false/alive by default: the static roster makes no current
life-status claim. Visible-player and body records supply timestamped evidence.

The crew API rejects an impostor actor explicitly. Future impostor observations
need a separately reviewed contract; the existence of role labels is not support
for an impostor policy. Actor targets/masks cannot depend on hidden roles.

## API and provenance

`observe(world, player_id)` returns a complete safe snapshot. The future trusted
simulator must provide accurate public context and a public-only publication log.
Its internal event log is separate and is ignored by the projector.

`Evidence` stores an observation/delivery tick, provenance and a typed payload:

- DIRECT: visible player or body snapshot, timestamped when observed. Body records
  do not include time of death. Seeing an interaction does not reveal its truth.
- PUBLIC: typed report, meeting roster, vote, ejection or match-result announcement.
- CLAIM: authenticated speaker, subject, asserted room and claimed occurrence tick.
  Delivery tick remains separate from the claimed tick; no simulator fact lookup.

Delivered public log records must be ordered by publication time. Future-dated
records are ignored without inspecting payloads or their ordering. Same-tick
ordering is public publication order; no hidden event ID,
global sequence gap, raw count or hidden RNG is serialized. Local entity records
are sorted by public ID, so internal container order cannot encode roles.

Snapshots include the public log prefix so delivery is stateless and reproducible.
Consumers must not count repeated snapshots as repeated independent evidence.
Phase 4 will add deduplication/compression and last-seen summaries. `remember` now
only appends a validated observation, rejects cross-actor/cross-match or non-forward
time, and starts fresh per match; it does not infer roles or refresh unseen state.

Public publication is a trusted simulator responsibility: the typed allowlist
prevents arbitrary internal payload passthrough, but cannot tell whether a report
was legitimately triggered. Phase 3 must test publication triggers and timing
against rules. Phase 1 tests these with fixtures, not an implemented meeting engine.

## Actions and their information boundary

Actions are intentions valid at the current snapshot, not guarantees of execution.
World changes after observation can make an action fail; the simulator must handle
that without returning a hidden explanation. Candidates are derived exclusively
from the actor snapshot:

- Roaming: wait; navigate to public rooms/consoles; follow visible players; interact
  with incomplete own tasks in visible range; report visible nearby bodies; call a
  meeting at a visible nearby meeting console with personal allowance remaining.
- Discussion: claim only on the publicly assigned speaker's turn.
- Voting: vote for public eligible IDs or skip, only if own public vote is uncast.
- Finished or personally eliminated: no actions.

Hidden deaths never remove roster entries or unseen player targets from an
otherwise identical actor input. Navigation to a public room remains available
regardless of unseen occupants. Vote eligibility comes from the announced roster,
not live hidden life flags. There is no truth-based "safe room" or "real task" mask.

## Identity and training boundaries

Per-match public IDs are opaque `p0`... labels, stable within a match. Identity setup
shuffles both internal-slot-to-ID mapping and display names using an identity RNG;
roles are assigned with a separate role RNG. No role argument enters identity
assignment. Neither seed nor internal slot is actor-visible. The future match
runner must vary these seeds independently and not expose role-correlated match IDs.
Match IDs are opaque public bookkeeping, never derived from hidden state/seeds.

`role_labels(world)` is an explicit separate training call. Never append its result
to actor observations/memory, auxiliary inference inputs, candidate lists, replay
observations or previous reward. Label availability in training is legitimate;
execution-time access is not. A centralized critic must have a separate input
path. Phase 1 implements no critic, reward wrapper or learned policy.

## Required invariants and tests

For fixed public settings, actor state and legitimate history, hidden-only changes
must leave **the entire actor packet** equal, including candidate ordering. With a
fixed policy RNG state this also implies the same policy action distribution.

Paired fixtures cover hidden role swaps, unseen positions/deaths/interactions,
private tasks/cooldowns, body death times, internal log content/order, and internal
player order. Positive controls vary visible position, own tasks and public reports.
Temporal tests prove sightings remain historical when players leave view. Tests
also cover wall occlusion, range edges, eliminated vision, provenance, publication
cutoffs, label separation, serialization, immutable snapshots and role-independent
identity assignment. Runtime allowlist validation rejects malformed actor/public
payloads instead of silently forwarding extra fields.

These are regression protections, not a proof against every future integration
leak. Repeat them at the Phase 3 wrapper, replay, reward and policy-call boundaries.
