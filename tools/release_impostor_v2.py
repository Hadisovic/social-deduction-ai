"""Curate the immutable selected model and documentation after the study completes."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def curate(source,report,bundle):
    import torch
    from impostor_research.frozen import FrozenModels
    from impostor_research.learning_v2 import load
    torch.set_num_threads(1)
    source=Path(source).resolve();report=Path(report).resolve();bundle=Path(bundle).resolve()
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    summary=read(report/'summary.json');study=read(source/'study.json');lock=read(source/'selection-lock.json')
    results=read(source/'results.json');selected=lock['selected']
    if results['status']!='complete' or summary['selected']!=selected or results['selected']!=selected:
        raise ValueError('Only completed, consistently locked results can be released')
    if results['final_tests_opened'] or len(results['confirmation'])!=80:raise ValueError('Unexpected evaluation protocol')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    chosen=ROOT/selected['path']
    if digest(chosen)!=selected['sha256']:raise ValueError('Locked selected checkpoint changed')
    models=FrozenModels();entries=[]
    bundle.mkdir(parents=True,exist_ok=False)
    for seed in (71,83):
        folder=bundle/f'seed-{seed}';folder.mkdir()
        for step in (0,2048,4096,8192):
            path=source/f'seed-{seed}'/f'checkpoint-{step:05d}.pt'
            policy,metadata=load(path,models)
            if metadata['seed']!=seed or metadata['decisions']!=step or metadata['development'] or 'optimizer' in metadata:
                raise ValueError('Checkpoint identity mismatch')
            target=folder/path.name;shutil.copy2(path,target)
            if digest(target)!=digest(path):raise ValueError('Checkpoint copy changed')
            entries.append(dict(seed=seed,decisions=step,path=target.relative_to(ROOT).as_posix(),sha256=digest(target),strictly_loaded=True))
    release=bundle.parent/'release';release.mkdir(exist_ok=False)
    target=release/'policy.pt';shutil.copy2(chosen,target)
    policy,metadata=load(target,models)
    if digest(target)!=selected['sha256']:raise ValueError('Release differs from immutable selection')
    models.assert_unchanged()
    metrics=summary['confirmation']['selected_family']
    status=('observed_wins_comparative_uncertainty' if metrics['wins'] else 'insufficient_winning_strategy')
    if metrics['wins'] and metrics['win_rate']<=max(summary['confirmation'][m]['win_rate'] for m in ('pilot83','untrained_family')):
        status='observed_wins_without_control_advantage'
    release_manifest=dict(schema=metadata['schema'],source_commit=study['record']['source_commit'],
        selected=selected,checkpoint_sha256=digest(target),selection_lock_sha256=digest(source/'selection-lock.json'),
        confirmation=metrics,quality=status,report=report.relative_to(ROOT).as_posix(),
        frozen_opponent=study['record']['checkpoint'],frozen_opponent_sha256=study['record']['checkpoint_sha256'],
        frozen_weights_unchanged=True,final_tests_opened=False,
        report_script_sha256=summary['analysis_sha256'],summary_sha256=digest(report/'summary.json'),
        curator_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        disclaimer='Best of four preregistered candidates by selection validation. Not globally best or final-test accepted.')
    write=lambda p,v:p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write(release/'release.json',release_manifest);write(report/'checkpoints.json',entries)
    write(report/'release.json',release_manifest)
    total=summary['seconds'];train=sum(v['wall_seconds'] for v in summary['training'].values())
    selection_rows=read(report/'selection.json');selection_seconds=sum(r['seconds'] for r in selection_rows)
    confirmation_seconds=sum(v['wall_seconds'] for v in summary['confirmation'].values())
    stamp=lambda t:datetime.fromtimestamp(t,ZoneInfo('Asia/Amman')).strftime('%Y-%m-%d %H:%M:%S %Z')
    table=[]
    labels={'selected_family':'Selected learned impostor','pilot83':'Preserved old pilot 83','scripted_hunter':'Scripted hunter','untrained_family':'Untrained family network'}
    for method in labels:
        m=summary['confirmation'][method];ci=m['wilson95']
        table.append(f'| {labels[method]} | {m["wins"]}/20 | {100*m["win_rate"]:.0f}% ({100*ci[0]:.1f}–{100*ci[1]:.1f}%) | {m["mean_kills"]:.2f} | {m["own_ejections"]} |')
    assessment=(f'The selected model wins {metrics["wins"]}/20 fresh confirmation games. This shows a winning behavior in this narrow setup; '
        '20 scenarios and two training seeds are too small to establish broad robustness. Compare the controls below before crediting learning.'
        if metrics['wins'] else
        f'The selected model wins 0/20 fresh confirmation games. Its {metrics["mean_kills"]:.2f} mean kills per game are not a useful winning strategy. '
        'This release is an evaluated research checkpoint, not a strong finished opponent. More of the same training is not justified by wins alone.')
    trainrows=[]
    for seed in ('71','83'):
        t=summary['training'][seed];episodes=read(report/f'seed-{seed}'/'episodes.json')
        trainrows.append(f'| {seed} | {t["decisions"]:,} | {len(episodes)} | {sum(r["impostor_win"] for r in episodes)} | {sum(r["kills"] for r in episodes)} | {t["wall_seconds"]/60:.2f} min |')
    comparisons=[]
    for method,item in summary['paired_comparisons'].items():
        lo,hi=item['paired_bootstrap95']
        comparisons.append(f'- Versus {labels[method].lower()}: {100*item["win_rate_difference"]:+.1f} percentage points; paired bootstrap 95% interval {100*lo:+.1f} to {100*hi:+.1f}.')
    text=f'''# Learned impostor — results and teammate guide

Run from the repository root on `codex/impostor-environment-v2`:

```powershell
python run_impostor.py
```

This loads the locked selected model: initialization **{selected['seed']}**, checkpoint
**{selected['step']:,} decisions**. It plays against frozen full-29 and three scripted
crewmates under the canonical V2 rules. Space/P pauses; Right steps; +/- changes speed;
R replays the current scenario; N starts another. The side panel shows the learned
action-family probabilities. `1` toggles spectator roles, G routes, V sight, Esc exits.
Use `--stochastic` to sample rather than use the evaluated greedy policy.

**Assessment:** {assessment}

## What changed and why

- Old pilot: both seeds lost all 20 validation games with zero kills. Its eight
  checkpoints and six graphs remain in [the pilot report](../pilot-20261027/README.md).
- The new NN chooses an action family, then a target. Many travel destinations no
  longer crowd out kill exploration. No script forces the impostor to kill or chase.
- The critic has separate features and gradient clipping. Learning the value of a
  losing state cannot dominate the actor's gradient.
- Own visible geometry and legitimate memory help compare targets. Hidden roles,
  unseen positions, actual witness labels, ghosts and private crew beliefs stay out.
- Travel choices can continue up to 3 simulated seconds. New players/bodies, legal
  kills, arrival and phase changes interrupt them. Physics still ticks every .2s.
- Rewards: **+10 win, -10 loss, -12 timeout, +2 successful kill, -2 own ejection,
  -0.002/second**. No reward for moving, waiting, lying, reporting or fake tasks.
  Every loss remains negative and every win positive. Kill shaping can still change
  the policy's risk preference; it does not mathematically preserve the maximum
  win-rate policy. True wins, not shaped return, determine checkpoint selection.
- Frozen crew and belief tensors, all historical checkpoints, ordinary task victory,
  ghost-task ownership and living-crew report/eyewitness safeguards are preserved.

### Which gameplay logic is learned?

The NN can move toward visible players, choose a kill target, fake a task, report,
call a meeting, accuse, defend, claim a location/sighting/elimination, vote or skip.
There is no hard-coded "never report", "always vote for X" or "only kill alone"
impostor rule. Kill avoidance near witnesses must be learned from the game outcome;
living crew retain direct witnessed-kill evidence and vote for that killer.
The impostor sees legitimate observations and its own memory, never private crew
beliefs or actual hidden-witness labels. Claims use the game's structured vocabulary,
not natural-language chat. Its own location claim can serve as an alibi; the current
DEFEND_PLAYER candidates refer to other players. Sighting candidates use remembered
observations, while location and accusation templates permit false claims. It does
not generate arbitrary sentences, timestamps or multi-step persuasion scripts.
Availability of an accusation/defence action is not proof
of learned deception. Vents and sabotage are not supported in this environment.

## NN and training settings

| Setting | Value |
|---|---|
| New network | {study['record']['network']['parameters']:,} parameters; 12 legal action families and target scorer |
| Inputs | 35 global + 8 own extras; 4×21 player features; up to 256×51 candidate features |
| Actor | player encoder 32→32; context 128→128; candidate/family heads 64 |
| Critic | independent player encoder and 128→64→1 value head |
| PPO | rollout 256; 4 epochs; minibatch 64; learning rate 0.0003; Adam epsilon 0.00001 |
| Stability | ratio clip .2; actor gradient .5; critic gradient 1; value coefficient .5 |
| Exploration | family entropy .02→.005; normalized conditional entropy .001 |
| Credit assignment | gamma 1; physical-time GAE lambda .995/second |
| Lobby | 5 players; 1 impostor; 45s cooldown; 15s discussion + 120s voting; permanent tasks |

Training scenarios start at 3100100. Initializations use separate random streams;
only impostor parameters enter the optimizer. Checkpoints, optimizer/RNG resume
states and replay journals are preserved. The exact configuration is in
[study.json](study.json); the prior [registered protocol](../strategic-v2-protocol.md)
fixed the budget, candidate set, controls and selection rule before learning.

| Initialization | Decisions | Completed training games | Training wins | Kills | Wall time |
|---|---:|---:|---:|---:|---:|
{chr(10).join(trainrows)}

These are sampled training outcomes, not held-out accuracy. A partly collected final
episode contributes decisions but is not counted as a completed game.
A win may involve other players' mistakes. It does not, by itself, show that a
particular accusation or alibi caused the win; that requires a separate causal test.

## Selection and fresh confirmation

After both runs finished, four candidates (4,096 and 8,192 from each seed) played
20 paired selection scenarios, 3200020–3200039: **80 matches**. The fixed ranking
was wins, kills, fewer timeouts, earlier checkpoint, then lower initialization.
[The immutable selection lock](selection-lock.json) was written before confirmation.

The selected model and three controls then played the same 20 fresh scenarios,
3200040–3200059: **80 confirmation matches**. Selection was not changed afterward.
These are confirmation-validation results; reserved final tests remain unopened.
Win rate is the appropriate outcome metric here, not classification accuracy.

| Method | Wins | Win rate (Wilson 95% interval) | Mean kills | Own ejections |
|---|---:|---|---:|---:|
{chr(10).join(table)}

All 160 selection/confirmation executions had zero illegal actions/navigation
failures. Paired win-rate differences versus controls:

{chr(10).join(comparisons)}

These are descriptive small-sample comparisons without multiplicity correction.
An all-equal paired sample can produce a collapsed bootstrap interval; that is not
proof of equivalence or zero uncertainty. Use the Wilson intervals above as well.
The 80 executions per partition contain 20 independent scenarios, not 80 independent
cases. Architecture, rewards and travel options changed together; this is not an
ablation attributing improvement to one change. Old pilot/hunter use their original
decision interfaces and rewards; compare true wins/kills, not cross-method return.

## Graphs — how to read them

![Training outcomes](figures/learning_curves.png)
Higher win rate and more wins matter most. Shaped returns/kills can rise without
winning. The two colors are independent initialization seeds, not extra scenarios.

![PPO diagnostics](figures/ppo_diagnostics.png)
Actor/critic losses, family entropy, KL and clipping show optimizer behavior. Finite
losses and a good value fit alone do not establish a useful gameplay strategy.

![Kill exploration](figures/kill_exploration.png)
The left plot shows cumulative learned kill probability only when killing is legal.
The middle shows sampled kill frequency in those opportunities. The right shows
unclipped gradient norms; separate clipping is applied before every optimizer step.

![Action families](figures/action_families.png)
Travel destinations are pooled. This reveals whether the policy only travels,
cancels or waits, rather than using effective actions.

![Checkpoint selection](figures/checkpoint_selection.png)
These curves were used for selection, not final acceptance. Four candidates were
compared on the same scenarios; their selected maximum can be optimistic.

![Fresh confirmation](figures/confirmation_results.png)
Whiskers show win-rate uncertainty. Kills and end reasons distinguish useful wins,
reckless ejections and failure to finish before task victory.

![Paired scenarios](figures/paired_scenarios.png)
Each column is the same scenario for all methods. Small vertical offsets separate
overlapping points; they do not change integer wins/kills.

![Compute](figures/compute.png)
Environment plus frozen-crew inference and PPO optimization are measured separately.
Travel options change simulated experience per decision; decision counts alone do
not establish a fair speed comparison with the old pilot.

## Time, processor and verification

- Study start: **{stamp(summary['started_epoch'])}**.
- Finish: **{stamp(summary['finished_epoch'])}**.
- Total training/selection/confirmation: **{total/60:.2f} minutes** (ceiling 180 minutes).
- Training wall: {train/60:.2f} min; timed selection: {selection_seconds/60:.2f} min;
  timed confirmation: {confirmation_seconds/60:.2f} min. Total also includes saves,
  loading, integrity checks and orchestration. Preparation/graphs are outside it.
- Compute: one CPU worker, one Torch thread, {study['record']['compute']['logical_cpus']}
  logical CPUs available; no GPU. Processor string:
  `{study['record']['compute']['processor']}`. Exact marketing model was unavailable.
- Runtime: Python {study['record']['versions']['python']}, Torch {study['record']['versions']['torch']},
  NumPy {study['record']['versions']['numpy']}; {study['record']['compute']['os']}.
- Improved preflight: **78 tests passed**, including real gradients, private
  observations, exact resumed update/action equality, reward/option semantics and
  headless viewer. Two additional controlled win/loss end-screen tests passed in
  14.76 seconds without changing the running worker's source. The offline report
  also passed synthetic zero/all-win interval and eight-graph checks; those fixtures
  are not gameplay results. Earlier engineering: 239 regression tests and 10 matches passed.
- All eight policy checkpoints were copied byte-for-byte and strictly loaded.
  [Checkpoint hashes](checkpoints.json), [release manifest](release.json),
  [aggregate results](summary.json), [per-match confirmation](confirmation.json).
- Frozen full-29 SHA-256: `{study['record']['checkpoint_sha256']}`.
  New runtime source: `{study['record']['source_commit']}`. Strict source/schema/file
  and tensor guards remain enabled. No merges, deployment or reserved final tests.

## Next research step and future best-crewmate play

1. Inspect whether progress is task-loss, witnessed-kill/ejection or navigation
   limited. More training is useful only if fresh validation supports it.
2. Test one change at a time: richer actor-local temporal memory, pursuit target
   features, longer credit horizons or entropy schedule. Keep legal masks, task
   victory, cooldowns and ghost isolation unchanged. Register fresh validation IDs.
3. If kills rise without wins, reduce reckless-kill shaping or use potential-based
   shaping from legitimate observations. Never reward hidden true witness counts.
4. Freeze this impostor release. When the friend's improved crewmate is ready,
   verify its V2 schema, pin its exact checkpoint/hash, then run a paired cross-play
   matrix including both old and new crew. Do not silently replace full-29.
5. Only after one learned crewmate and one useful learned impostor work, explore
   adapting opponents and self-play with frozen opponent pools and held-out tests.

This release is the best of the registered candidate set by selection, not a claim
of global optimality. The frozen crewmate is an assisted legacy-to-V2 transfer;
results do not prove equal behavior to its original training environment.
'''
    (report/'README.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(release=str(target),selected=selected,quality=status,strict_copies=len(entries)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('report',type=Path);p.add_argument('bundle',type=Path)
    args=p.parse_args();curate(args.source,args.report,args.bundle)
