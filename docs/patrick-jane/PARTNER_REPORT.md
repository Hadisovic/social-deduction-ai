# Patrick Jane — implementation progress for partner review

Date: 2026-10-28. Status: **local implementation draft; research readiness pending**.

## Separation and preservation

- New branch: `codex/patrick-jane-recurrent-ppo`.
- Separate worktree: `_patrick_jane`, based on the latest local crewmate research
  commit `4bac4cdb099e81a5c6c7de982f423136f24213c5`.
- Previous model/research stays on `codex/phase21-meetings-evidence`. Its worktree
  is clean. The historical models were not retrained or migrated.
- Hash verification found all 1,077 original tracked files unchanged before this
  report, plus all 206 separately protected checkpoint/viewer source files unchanged.
  The only subsequent historical-file edit is an appended R030 research entry;
  its previous content is preserved byte for byte. Existing branch/tag references
  remain unchanged. Codex's own temporary capture references are not research branches.
- The partner's impostor project, existing PRs, production and final-test partitions
  were not modified. Nothing was pushed, merged or deployed.
- New implementation files are still **uncommitted local drafts**. Branch HEAD
  remains the base commit; there is no new publication commit or PR.

## Implemented draft

There are 20 new Python modules under `patrick_jane/`, separate design/isolation
documentation, 29 correctness tests and a copied/additively extended existing
spectator. Historical training and environment modules remain unchanged.

**Locked network:** self/map/navigation 192→256→128; up to 14 other players,
16→128→128 plus four-head 128-wide attention; up to 64 timestamped events,
16→128→128 plus four-head 128-wide retrieval; 384→256 fusion; shared GRU256;
separate actor/critic branches 256→256→128. The actor has exactly five heads:
navigation 3, interaction 5, communication 6, voting 14 others plus skip, and a
dynamic attention target selector. Critic output is scalar.

**Exact trainable count: 1,031,070.** One GRU, three attention modules including
target selection, 21 Linear modules including attention output projections. It
is a parallel recurrent graph, not a simple nine-layer MLP.

**Game:** initial 10 learned crew + two scripted impostors; capacity up to
12 crew + three impostors. Individually owned/multistage V2 tasks, automatic
task-only ghosts, public task bar, confirmed ejections, result-only ballot
revelation and wall visibility are retained. Lights, O2 and reactor use existing
Skeld utility positions. Decisions are projected before simultaneous execution.

**Actor isolation/navigation:** separate actor observations, explicit timestamped
memory, own statements and retained goals. No frozen belief scores or hidden-role
roster enters the policy. Learned destination/detour/follow decisions use existing
map/pathfinding; all existing grid points remain possible exploration goals.

**Learning infrastructure:** one central shared policy, independent actor GRU/RNG
states, CPU simulation workers, batched GPU inference, masked conditional joint
likelihoods, elapsed-time GAE, recurrent sequences up to 128 decisions, exact
approved reward coefficients, delayed ghost/team settlement, strict exclusive
checkpoints and separate champion selection. The full-run API requires explicit
owner approval; it has not been executed.

**Spectator:** the existing application loads 12-player recordings and displays
real choices, separate conditional head probabilities, rewards, routes and
checkpoint/update labels. A parallel-environment monitor was added to its existing
study view. Playback was rendered and visually inspected; this is not a second app.

## Verified so far

The current code passed **29 tests in 22.8 seconds**. Coverage includes population
and task ownership; lights and critical sabotage; simultaneous repairs and ordering;
hidden ballots; private role changes; ghost freezing; event provenance; padding;
independent recurrence; conditional likelihood reconstruction; collision-clear
A-star routes; actual finite PPO updates; checkpoint reload; and the approval gate.

Recorded versus unrecorded short correctness matches produced identical actions,
rewards, trajectories and updated weights. Inline versus process-worker execution
also matched. These short tests are engineering checks, not learning evidence.

The first full untrained 12-player game took 62.6 seconds for 1,332 crew decisions
and 117 simulated seconds. It exposed six communication rejections from floating
timestamp comparisons. That defect was repaired; later complete batches had zero
illegal actions. The negative run is preserved.

The latest completed eight-worker measurement at the time of interruption was:

| Measurement | Result |
|---|---:|
| Complete games | 8 |
| Learned individual actor decisions | 10,806 |
| Wall time | 223.8 s |
| Throughput | 48.34 actor decisions/s; 2.15 complete games/min |
| Simulated experience | 1,132 match-seconds |
| Reports / meetings / ballots | 8 / 23 / 227 |
| Completed tasks / repaired sabotages | 61 / 4 |
| Correct / innocent ejections | 0 / 11 |
| Skipped ballots | 31 |
| Illegal actions / navigation failures | 0 / 0 |
| CPU utilization, whole machine | 39.5% mean; 79.6% peak |
| Research process-tree RSS | 3.33 GB mean; 3.67 GB peak |
| GPU utilization, whole GPU | 45.1% mean; 52% peak |
| Whole-GPU used memory | 3,420 MiB peak |
| PyTorch peak allocated memory | 79.3 MiB |
| Batched inference | 10.84 s |
| Environment plus synchronization | 212.57 s |
| Estimated RPC/synchronization overhead | 2.37 s |
| Instrumented recording time, selected replay | 0.43 s |

All eight games were **untrained-policy losses**: six critical-sabotage failures
and two parity losses. No PPO update occurred in this throughput batch. These
results demonstrate executable mechanics, not useful strategy or deduction.
Tasks include ghost work and ballots include both roles. Hardware utilization
includes background system activity. The recording figure is instrumentation time,
not a controlled causal overhead estimate.

Important version limitation: this batch predates the most recent A-star routing,
vectorized PPO-likelihood and provenance refinements. Those refinements passed
unit/correctness tests, but have not yet completed a new full-game benchmark or
GPU PPO smoke. Earlier benchmark revisions reused development scenarios; their
game totals must not be pooled as independent scientific evidence.

## Runtime and explicit approximations

Hardware was verified: i7-13700HX, 24 logical CPUs, approximately 32 GB RAM,
RTX 4070 Laptop GPU with 8 GB VRAM. The old PyTorch 2.6 CPU environment was kept.
A separate ignored local runtime uses PyTorch 2.7.1+cu128 for Patrick Jane.

Mechanics approximations needing review before a research training run include
30-second O2/reactor deadlines, a 30-second sabotage cooldown, lights sight factor
0.2, abstract panel-use repairs and same-tick two-actor reactor repair. They are
documented choices, not verified retail fidelity. Coverage uses 0.75-unit cells.
Chunk-start recurrent states are fixed during PPO epochs without burn-in. These
limits are explicit in DESIGN.md; no architecture or coefficient was substituted.

## Remaining work and recommendation

1. Freeze/review the current implementation and mechanics approximations.
2. Complete comparable 8/16/32-worker measurements on that frozen revision. Only
   eight-worker measurements exist; 16 workers has not been established as best.
3. Run the combined bounded smoke on full default-lobby matches: actual GPU PPO,
   finite gradients, checkpoint/reload and completed replay from one frozen source.
4. Finish broader historical replay/checkpoint compatibility and spectator-control
   regression checks, then finalize runtime setup and readiness documentation.
5. Commit the reviewed draft. Request separate approval before expensive training.

No long Patrick Jane training, held-out champion study or trained-policy comparison
has been run. The previous Phase 6.1 acceptance remains pending. At the reporting
checkpoint, no Patrick Jane training, benchmark or test process remains active.

**Recommendation:** partners should review the preserved architecture and explicit
mechanics assumptions now. Treat the project as an implemented prototype undergoing
verification, not a trained agent or a fully qualified training release.
