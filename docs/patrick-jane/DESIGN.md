# Patrick Jane implementation decisions

The owner's locked specification is authoritative. This is a new research regime,
not a migration of the paused feedforward model. Architecture: self/map 192→256→128;
players 14×16→128→128 with four-head 128-wide self-attention; events 64×16→128→128
with four-head 128-wide retrieval; concatenation 384→256; one shared GRU(256,256);
actor and critic each 256→256→128. Five actor outputs are navigation(3), interaction(5),
communication(6), voting(14 other slots + skip), and a dynamic attention selector.
The target selector has a 16→128 candidate embedding, four-head 128-wide retrieval
and a query projection. The critic outputs one scalar. Tanh activations are used.

There are **1,031,070** parameters, one GRU, three attention modules, and 21 Linear
modules counting attention output projections (packed Q/K/V matrices are additional
parameters inside attention). The longest dependency path runs through a two-layer
encoder, attention, fusion, GRU, two-layer actor, target retrieval/query and scoring.
This parallel graph is not a nine-layer MLP. Initial output heads are small-gain
orthogonal projections. No historical StrategicPolicy or belief predictor is used.

## Game approximations and deterministic order

V2 default lobby remains: crew vision 4.5, impostor vision 6.75, 45-second kill
cooldown, .2-second physics and .6-second decision cadence, four individually owned
tasks, 15-second discussion and 120-second voting, visible ballots at results,
confirmed ejections, task victory enabled, 900-second timeout. Match identities,
colors and roles are randomized with separate streams. Twelve initial actors are
10 learning crew and two scripts; padding supports 15 total, at most three impostors.

Sabotage uses native SwitchConsole, both NoOxyConsole utilities and both reactor
hand utilities. Chosen **explicit abstractions**, configurable in SabotageSettings:
30-second shared initial/post-repair cooldown, one active sabotage, 30-second O2 and
reactor countdowns, lights multiply crew sight by .2, impostor sight unchanged.
Lights require one legal panel use. O2 requires both distinct panels (persistent
completion). Reactor requires two distinct actors using different panels in the
same physics tick. Panel mini-games/continuous hand holding are abstracted as
legal panel-use actions. Both roles may legitimately repair. Contributors receive
+3 each once per genuinely completed sabotage. These timings/interaction durations
are not validated retail fidelity. No opponent identity is disclosed to crew.

All proposed actions are validated against one pre-action state. Stable public ID
breaks simultaneous report ties. Reports/emergencies precede sabotage and kills;
critical sabotage blocks emergency calls, while a legal body report starts a meeting
and cancels the critical crisis without repair credit. Lights persist. Simultaneous
kills require the killer/victim still active and cannot target an impostor teammate.
Repairs on the last legal tick precede countdown failure. Failure precedes task
victory; ejection/parity resolution precedes task victory. V2 automatic ghosts
retain tasks, travel at speed through walls and freeze during meetings. No living
actor receives ghost observations. Default task completion is never disabled.

## Observations, memory and navigation

Packets contain own state/tasks, public task bar/map/phase/claims/results,
wall-visible actors/bodies, personally witnessed events, public sabotage state and
own route feedback. Impostor teammates and sabotage ability exist only in the
scripted impostor's legitimate packet. No hidden role roster, kill truth, opponent
profile, seed, frozen suspicion score or global event counter enters crew tensors.
Last-seen positions remain separate from 64 timestamped events. Own room transitions
and own public statements have distinct provenance. Claims retain assertion and
delivery time; contradictions never receive a scripted suspicion value.

All 67,874 existing reachable-grid points remain eligible destinations through a
seeded cyclic candidate window plus bounded nearby alternatives. Tasks, utilities,
bodies and retained evidence have dynamic targets; no MAX_ACTIONS=96 limit.
Coverage uses .75-map-unit spatial cells with exact wall-ray tests at representative
reachable nodes. It is an explicit spatial abstraction, not exact continuous vision
coverage. Discovering new coverage in a decision observation pays +.02 once, capped
at +.3; revisiting only previously seen cells pays zero. First sightings pay +.05
once per other identity, capped at +.7. Goal replacements are bounded by a 2.4-second
replanning interval while moving; failed/finished/no goals can change immediately.
Detours retain the original goal, use nearby targets, cannot nest, and can resume on
completion/8-second bound through Follow. The network is never forced to report,
vote, accuse or follow a particular suggested route.

Roaming first samples interaction. Choosing Do nothing samples navigation; all
other interactions select a compatible target. Discussion samples communication
and its conditional target; voting samples the masked other-player/skip head.
Only active factors enter the joint PPO log likelihood. Actor memories stay inside
CPU simulation workers; independent GRU states and RNG streams are keyed by
worker/match/owner in the coordinator. All crew use the same frozen policy version
during a complete-match collection batch; only the coordinator updates weights.

## Rewards, trajectories and reproducibility

Exact coefficients are in `patrick_jane/rewards.py`. Ejections pay +4 impostor or -2
innocent to each participating learning crewmate, independent of their ballot or
living status. Terminal +10/-10 similarly settles all crew, including earlier deaths.
Draw/timeouts have neither win nor loss credit. A completed task pays its owner
+1 exactly once, under either living or automatic-ghost completion. Reporting pays
+1.1 once per body globally. Death is -1. Vote/accusation/defense/invalid actions
pay zero. Idle -0.01 requires ten actual avoidable seconds; movement, meetings,
interaction and pending incubation do not incur waiting penalties. No continuous
survival reward is present. Component ledgers are distinct from discounted returns.

An actor's last living transition stays open until match settlement, receiving
discounted later ghost and team outcomes without any post-death policy inference.
Discount is .99^(elapsed/.6), including within-transition reward arrival times.
GAE uses .95^(elapsed/.6), reused from the old time-aware kernel. Truncated BPTT
uses contiguous sequences of at most 128 actor decisions and their collected initial
hidden states. There is no cross-owner recurrence. Stored chunk-start states are
fixed during PPO epochs (no burn-in); this standard truncation approximation is a
known limitation. Minibatch size is an engineering parameter, four sequences initially.

PPO is fixed at lr .0003, clip .2, entropy .01, four epochs, grad clip .5. Sampling
uses independent CPU generators after batched inference transfers, avoiding per-actor
GPU synchronization. Workers never import an optimizer. Source/schema/runtime locks
are checked on reload; checkpoints are exclusive files saved at complete-match
boundaries with optimizer/counters/RNG. Resume advances the registered scenario cursor.
The latest learning policy continues independently of a held-out win-rate champion;
there is no automatic rollback. Default candidate evaluation is 100 held-out matches
every 100,000 decisions. No reserved Phase 6 partition reader is invoked.

Development partitions: training 2900000–2900999, validation 2910000–2910099,
engineering 2920000–2920099. Engineering smoke and throughput runs are not learning
or generalization evidence. Large-run mechanics and training require further review
and separate owner authorization, which the full-run API requires explicitly.
