"""Offline final guide/release; follows existing locks and never chooses from confirmation."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib,json,shutil,sys
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.report_impostor_v2 import aggregate,wilson


def build(v2_source,v2_report,destination,v3_source=None,release_directory=None,pursuit_skip_reason=None):
    import torch
    from impostor_research.frozen import FrozenModels
    from impostor_pursuit.learning import load,digest
    torch.set_num_threads(1)
    v2_source=Path(v2_source).resolve();v2_report=Path(v2_report).resolve();destination=Path(destination).resolve()
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    v2=read(v2_source/'results.json');v2plan=read(v2_source/'study.json');v2summary=read(v2_report/'summary.json')
    if v2['status']!='complete' or len(v2['confirmation'])!=80 or v2['final_tests_opened']:raise ValueError('Complete V2 first')
    if v2['selected']!=read(v2_source/'selection-lock.json')['selected']:raise ValueError('V2 lock drift')
    destination.mkdir(parents=True,exist_ok=False)
    selected=v2['selected'];path=ROOT/selected['path'];selection_stage='V2';v3=None;v3plan=None;v3status='not_run'
    rows=[r for r in v2['confirmation'] if r['method']=='selected_family'];comparison=None;training=None;figures=[]
    write=lambda p,v:p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    models=FrozenModels()
    if v3_source:
        v3_source=Path(v3_source).resolve()
        v3plan=read(v3_source/'study.json');shutil.copy2(v3_source/'study.json',destination/'pursuit-study.json')
        if (v3_source/'results.json').exists():
            v3=read(v3_source/'results.json');v3status=v3['status']
            if v3status!='complete' or len(v3['confirmation'])!=40 or v3['final_tests_opened']:raise ValueError('Invalid completed continuation')
            if v3['selected']!=read(v3_source/'selection-lock.json')['selected']:raise ValueError('Pursuit lock drift')
            selected=v3['selected'];path=ROOT/selected['path'];selection_stage='conditional pursuit comparison'
            rows=[r for r in v3['confirmation'] if r['method']=='selected'];parent_rows=[r for r in v3['confirmation'] if r['method']=='parent_v2']
            if len(rows)!=20 or len(parent_rows)!=20 or {r['seed'] for r in rows}!=set(range(3200080,3200100)):raise ValueError('Incomplete fresh confirmation')
            comparison=aggregate(parent_rows)
            for name in ('results.json','selection-lock.json'):shutil.copy2(v3_source/name,destination/('pursuit-'+name))
            write(destination/'pursuit-selection-matches.json',[read(p) for p in sorted((v3_source/'selection').glob('*.json'))])
            if len(list((v3_source/'selection').glob('*.json')))!=60:raise ValueError('Incomplete pursuit selection')
            write(destination/'pursuit-confirmation.json',v3['confirmation'])
            bundle=ROOT/'artifacts/impostor-pursuit/study-20261027';bundle.mkdir(parents=True,exist_ok=False)
            snapshots=[]
            for step in (0,2048,4096,8192):
                original=v3_source/'training'/f'checkpoint-{step:05d}.pt';policy,metadata=load(original,models)
                if metadata['decisions']!=step or metadata['development'] or 'optimizer' in metadata:raise ValueError('Wrong continuation snapshot')
                target=bundle/original.name;shutil.copy2(original,target)
                if digest(target)!=digest(original):raise ValueError('Changed continuation copy')
                snapshots.append(dict(decisions=step,path=target.relative_to(ROOT).as_posix(),sha256=digest(target),strictly_loaded=True))
            write(destination/'pursuit-checkpoints.json',snapshots)
        else:
            v3status='incomplete_preserved_V2_fallback'
            if (v3_source/'progress.json').exists():shutil.copy2(v3_source/'progress.json',destination/'pursuit-stopped-progress.json')
        if (v3_source/'training/training.json').exists():
            training=read(v3_source/'training/training.json');(destination/'pursuit-training').mkdir()
            for name in ('training.json','episodes.json','updates.json','start.json'):shutil.copy2(v3_source/'training'/name,destination/'pursuit-training'/name)
    if digest(path)!=selected['sha256']:raise ValueError('Selected file differs from lock')
    policy,metadata=load(path,models);models.assert_unchanged()
    metrics=aggregate(rows)
    if len(rows)!=20 or metrics['illegal_actions'] or metrics['navigation_failures']:raise ValueError('Invalid final confirmation')
    release=Path(release_directory).resolve() if release_directory else ROOT/'artifacts/impostor-ai/release'
    release.mkdir(parents=True,exist_ok=False)
    shutil.copy2(path,release/'policy.pt')
    if digest(release/'policy.pt')!=selected['sha256']:raise ValueError('Universal release differs')
    load(release/'policy.pt',models);models.assert_unchanged()
    parent_selected=metadata['schema']!='impostor-pursuit-v3'
    summary=dict(status='evaluated_research_release_not_final_test_acceptance',selection_stage=selection_stage,selected=selected,
        schema=metadata['schema'],confirmation=metrics,same_partition_parent=comparison,pursuit_status=v3status,
        pursuit_skip_reason=pursuit_skip_reason,
        checkpoint_sha256=digest(release/'policy.pt'),frozen_opponent_sha256=v2plan['record']['checkpoint_sha256'],
        frozen_weights_unchanged=True,final_tests_opened=False,source_commit=metadata['record'].get('source_commit',metadata['record'].get('core',{}).get('source_commit')),
        v2_seconds=v2['seconds'],pursuit_seconds=v3['seconds'] if v3 else None,
        original_deadline_epoch=v2plan['deadline_epoch'],analysis_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        note='Selection follows immutable validation lock, never confirmation results. One continuation is not replication; no global-best claim.')
    write(release/'release.json',summary);write(destination/'summary.json',summary)
    if training:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figures_dir=destination/'figures';figures_dir.mkdir()
        episodes=read(v3_source/'training/episodes.json');updates=read(v3_source/'training/updates.json')
        def save(fig,name):fig.tight_layout();fig.savefig(figures_dir/name,dpi=160);plt.close(fig);figures.append(name)
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        for ax,key,title in zip(axes.flat,('return_','impostor_win','kills','impostor_ejected'),
            ('Shaped return','Cumulative training win rate','Kills / completed game','Cumulative own-ejection rate')):
            y=np.asarray([r[key] for r in episodes],float)
            if key in ('impostor_win','impostor_ejected'):y=np.cumsum(y)/np.arange(1,len(y)+1)
            ax.plot([r['decisions'] for r in episodes],y,'.-');ax.set_title(title);ax.set_xlabel('Additional decisions')
        fig.suptitle('One pursuit continuation — training, not held-out accuracy');save(fig,'pursuit_learning.png')
        fig,axes=plt.subplots(2,3,figsize=(12,7))
        for ax,key in zip(axes.flat,('policy_loss','value_loss','entropy','approximate_kl','clip_fraction','kill_probability_when_available')):
            ax.plot([r['decisions'] for r in updates],[r[key] for r in updates],'.-');ax.set_title(key.replace('_',' '));ax.set_xlabel('Additional decisions')
        save(fig,'pursuit_ppo.png')
        families=Counter()
        for kind,n in training['actions'].items():
            family='pursuit' if kind=='go_to_location' else 'travel' if kind in ('go_to_room','go_to_task') else kind
            families[family]+=n
        fig,ax=plt.subplots(figsize=(12,4));ax.bar(list(families),np.asarray(list(families.values()))/8192)
        ax.tick_params(axis='x',rotation=65);ax.set_ylabel('Training choice fraction');ax.set_title('Pursuit separated from room/task travel');save(fig,'pursuit_actions.png')
        if v3:
            selection=[read(p) for p in sorted((v3_source/'selection').glob('*.json'))]
            names=['parent_v2','pursuit4096','pursuit8192'];stats=[aggregate([r for r in selection if r['method']==n]) for n in names]
            fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].bar(names,[r['win_rate'] for r in stats]);axes[0].set_ylim(0,1)
            axes[0].set_title('Selection wins / 20 scenarios');axes[1].bar(names,[r['mean_kills'] for r in stats]);axes[1].set_title('Selection mean kills')
            for ax in axes:ax.tick_params(axis='x',rotation=20)
            save(fig,'pursuit_selection.png')
            fig,axes=plt.subplots(1,3,figsize=(12,4));data=[metrics,comparison];labels=['Locked selection','Unchanged V2']
            rates=[r['win_rate'] for r in data];ci=[r['wilson95'] for r in data]
            errors=np.maximum(0,np.asarray([[p-c[0] for p,c in zip(rates,ci)],[c[1]-p for p,c in zip(rates,ci)]]))
            axes[0].bar(labels,rates,yerr=errors,capsize=5);axes[0].set_ylim(0,1);axes[0].set_title('Fresh confirmation / Wilson 95% CI')
            axes[1].bar(labels,[r['mean_kills'] for r in data]);axes[1].set_title('Mean kills / confirmation game')
            axes[2].bar(labels,[r['own_ejections'] for r in data]);axes[2].set_title('Own ejections / 20 games')
            for ax in axes:ax.tick_params(axis='x',rotation=20)
            save(fig,'pursuit_confirmation.png')
    lo,hi=metrics['wilson95']
    quality=('The model has not demonstrated a useful winning strategy on this fresh confirmation set.' if not metrics['wins'] else
        'It demonstrates some winning behavior in this setup. The small sample does not establish robust or globally best play.')
    if comparison and metrics['win_rate']<=comparison['win_rate']:quality+=' It does not beat the unchanged V2 parent on win rate in this comparison.'
    timing=('No pursuit study: '+pursuit_skip_reason if pursuit_skip_reason else 'Not run: conditional gate or available time did not permit another stage.') if v3_source is None else (
        f'Continuation study: {v3["seconds"]/60:.2f} minutes, including training and 100 evaluation executions.' if v3 else
        'Continuation incomplete/stopped: its artifacts are retained; the already evaluated V2 model is released.')
    if training:timing+=f' Training alone: {training["wall_seconds"]/60:.2f} wall minutes, {training["decisions"]:,} additional decisions.'
    pursuit_text=('The pursuit family was prepared and passed tiny development gradient/resume checks, but its registered training study did not run. '+(pursuit_skip_reason or '')) if v3_source is None else (
        f'The conditional study completed. Its fixed validation ranking selected **{selected["method"]}**; '+('the unchanged parent remains the release.' if parent_selected else 'the pursuit continuation is the release.') if v3 else
        'The conditional study did not complete within its operational budget. No partial checkpoint replaces the evaluated V2 parent.')
    links='\n'.join(f'![{n.replace("_"," ").replace(".png","")}](figures/{n})' for n in figures)
    other=''
    if comparison:
        a=metrics;b=comparison
        other=f'\nSame fresh scenarios: unchanged V2 won {b["wins"]}/20, with {b["mean_kills"]:.2f} mean kills and {b["own_ejections"]} ejections.\n'
    run_table=[]
    pilot=read(ROOT/'docs/impostor-v2/pilot-20261027/summary.json')
    for seed in ('71','83'):
        episodes=read(ROOT/f'docs/impostor-v2/pilot-20261027/seed-{seed}/episodes.json');t=pilot['training'][seed]
        run_table.append(f'| First pilot {seed} | 4,096 | {len(episodes)} | {sum(r["impostor_win"] for r in episodes)} | {sum(r["kills"] for r in episodes)} | {t["training_wall_seconds"]/60:.2f} min |')
    for seed in ('71','83'):
        episodes=read(v2_source/f'seed-{seed}/episodes.json');t=read(v2_source/f'seed-{seed}/training.json')
        run_table.append(f'| Improved V2 {seed} | 8,192 | {len(episodes)} | {sum(r["impostor_win"] for r in episodes)} | {sum(r["kills"] for r in episodes)} | {t["wall_seconds"]/60:.2f} min |')
    control_table=[]
    for method,label in (('selected_family','Selected trained V2'),('pilot83','Old trained pilot 83'),('scripted_hunter','Handwritten hunter'),('untrained_family','Randomly initialized V2 NN')):
        m=v2summary['confirmation'][method]
        control_table.append(f'| {label} | {m["wins"]}/20 | {m["mean_kills"]:.2f} | {m["own_ejections"]} |')
    selection_table=[]
    for c in v2['selection']:selection_table.append(f'| {c["seed"]} | {c["step"]:,} | {c["wins"]}/20 | {c["kills"]} |')
    selected_v2=[r for r in v2['confirmation'] if r['method']=='selected_family']
    v2_actions=sum((Counter(r['actions']) for r in selected_v2),Counter())
    opportunities=sum(r['first_legal_kill'] is not None for r in selected_v2)
    text=f'''# Impostor AI — session summary and teammate guide

```powershell
python run_impostor_ai.py
```

Run from the repository root on `codex/impostor-environment-v2`. Install the normal
requirements if needed: `python -m pip install -r requirements.txt`. The script loads
the explicit release checkpoint documented in the
[release manifest](../../../artifacts/impostor-ai/release/release.json); it does not
train or substitute a scripted impostor. The game contains
one learned impostor, frozen full-29 and three scripted crew, with report/witness
safeguards. Your friend's newer crewmate is not silently swapped in.

**Result:** {metrics['wins']}/20 fresh confirmation wins; {100*metrics['win_rate']:.0f}%
win rate (Wilson 95% interval {100*lo:.1f}–{100*hi:.1f}%), {metrics['mean_kills']:.2f}
mean kills/game, {metrics['own_ejections']} own ejections. **{quality}**
{other}
The release follows the immutable **{selection_stage}** selection. Confirmation did
not change its identity. These are validation/confirmation results, not reserved
final-test accuracy. All old models and failed runs remain preserved.

## Controls

Space/P pause; Right single-step; +/- speed; R replay the same scenario; N new
scenario; 1 spectator roles; G routes; V sight; Esc quit. The orange outline marks
the learned impostor. The right panel shows legal neural action-family probabilities,
kill cooldown, last choice and outcome. `--stochastic` samples the learned policy;
the reported evaluation uses greedy family-then-target choices.

## What the NN decides

The released V2 impostor has **75,075 parameters and 12 action families**. It was
initialized from scratch, separately from the crewmate. Inputs are 35 globals plus
8 own extras, 4×21 player features and up to 256×51 legal candidate features. Its
actor and critic use separate encoders. PPO uses lr0.0003, rollout256, minibatch64,
four epochs, clip0.2 and physical-time GAE; full settings are in the V2 report.

- Movement, pursuit of visible player positions, cancelling/continuing and fake tasks.
- Which legal nearby player to kill and when. There is no forced-kill/kill-alone rule.
- Whether to report or call a meeting. No rule forbids self-reporting.
- Structured accusations, defence of other players, own location alibis and remembered
  sightings; voting or skipping. No free-text conversation or persuasion script.
- Vents and sabotage are outside this interface. A* executes navigation; it does
  not select the impostor's target or strategy.

Inputs include visible players/bodies, own cooldown/goals, the public task bar,
legitimate memory and public meeting evidence. There are no hidden positions,
private crew beliefs, ghost secrets or privileged actual-witness labels. Living crew
can report bodies and must vote for directly witnessed killers. Avoiding witnesses,
effective alibis and sensible voting are behaviors to learn, not established skills
merely because the action vocabulary includes them.

Rewards: +10 win, -10 loss, -12 timeout, +2 successful kill, additional -2 ejection,
-0.002 per simulated second. No movement/fake/report/claim reward. Kill shaping can
change risk preference; true wins assess strength. Frozen crew/belief never train.

## Exactly which crewmates it faces

| Slot | Controller | Learns during these runs? |
|---|---|---|
| 1 impostor | Our separate PPO NN | Yes during training; frozen during evaluation |
| 1 crewmate | Frozen `full-29/checkpoint-8192.pt` + Phase 4 belief NN | No |
| 3 crewmates | Scripted `StudyController` variants | No |

The learned crew seat is deterministically randomized per match. This is **not four
learned crewmates**. We did not train or improve the friend's crewmate in this session.
Five players, 45s kill cooldown, 15s discussion/120s voting, ordinary task victory
and permanent ghost tasks remain unchanged.

The one learned crewmate has two frozen neural components:

1. Its actor-safe memory records sightings, bodies, public claims/votes and direct
   witnessed kills. The Phase 4 belief NN estimates suspicion over other players.
2. The full-memory PPO policy uses that belief, task/progress/location features and
   memory to score legal task travel, room travel, follow/flee, reports, emergencies,
   claims, votes and skips. It chooses greedily in these tests. A* executes movement;
   the adapter keeps longer goals up to 24s (follow/flee up to 3s) with interruptions.

This is a **legacy-to-V2 projection**: 40 global, 4×26 player and 96×23 candidate
features. V2-only task-stage/readiness, confirmed-role and visual-proof details are
not new neural inputs. Legal readiness is still checked before executing an intent.
Thus its historical training win rate is not a measured V2 cross-play win rate.

All four living crew also receive explicit safeguards: approach/report visible
bodies and retain direct witnessed-kill evidence to vote for the killer when legal.
Those are overrides, not newly learned skills; body discovery alone never identifies
the killer. Dead crew have no social actions or shared private memory.

The other three crew use scripts: nearest/shuffled/social task ordering, A* movement,
body reporting, and evidence/cautious/skeptical meeting styles. Voting scores direct
witness evidence, nearby-body sightings and public accusations, then applies a
threshold and skips ambiguous ties. It is **not a neural vote model or simply a
uniform random vote**, but it can make mistakes. Some uncertain scripted claims use
seeded randomness. The mandatory direct-witness override remains above those rules.

## Every completed training run, including failures

| Run | Decisions | Completed training games | Training wins | Training kills | Training wall |
|---|---:|---:|---:|---:|---:|
{chr(10).join(run_table)}

Four registered runs, **24,576 decisions total**. These are training counts, not
test accuracy. The first pilot's two final checkpoints each scored **0/20 validation
wins and zero validation kills**. Those are different scenarios from V2; do not pool
them as one model's test sample. All failed checkpoints, logs and graphs remain.

V2 checkpoint selection used the same 20 fresh scenarios for each candidate:

| Initialization | Checkpoint decisions | Selection wins | Total kills |
|---|---:|---:|---:|
{chr(10).join(selection_table)}

Initialization71/checkpoint8192 was locked before fresh confirmation. All methods
below then played the same 20 new scenarios; 80 executions are not 80 independent
cases. None of these are the reserved final tests.

| Impostor control | Fresh wins | Mean kills | Own ejections |
|---|---:|---:|---:|
{chr(10).join(control_table)}

## Why the wins look like this

The measured failure is **stalling**, not frequent ejection. In V2's 20 selected-model
confirmation games it chose FAKE_TASK **{v2_actions.get('fake_task',0):,} times**,
GO_TO_LOCATION {v2_actions.get('go_to_location',0)} times, and had legal kill opportunities
in only {opportunities}/20 games. It made {v2_actions.get('kill',0)} kill selections.
Every selected-model confirmation game ended by crew task completion; no own ejection
occurred. This does not prove that it learned clever witness avoidance.

The engine's FAKE_TASK action restarts a four-second activity. The current wrapper
re-decides non-travel actions every .6s, so repeated fake selections restart it before
completion. This is permitted game behavior, but a poor learned use of the interface.
The sampled training policy sometimes moves away from that loop; greedy evaluation
can keep choosing its highest-scoring fake action repeatedly. Its exact optimizer
cause has not been isolated.
No positive fake-task reward exists. Sparse delayed game outcomes, too little useful
pursuit experience and imperfect credit assignment are plausible contributors,
not proven single causes. More steps are not automatically the remedy.

The one training win involved two kills and a wrongly ejected crew member. That
does not prove a learned lie caused the mistake. Likewise the whole mixed crew
team winning does not prove that full-29 alone carried the match: the three task
scripts and ghost task completion contribute. Current aggregate logs do not isolate
the learned crewmate's own task contribution.

The selected V2's 2/20 selection wins fell to 0/20 fresh confirmation wins. Small
samples and selecting the maximum among candidates can produce that optimism;
the unchanged lock and fresh scenarios expose it. The model is **not yet a strong
winning impostor**, and advanced deception/sensible voting are not established.

## What changed

1. First pilot: travel destinations overwhelmed rare useful actions. Both seeds
   won 0/20 validation games with zero validation kills; **25.37 minutes** total.
   [Preserved results/six graphs](../pilot-20261027/README.md).
2. V2: learned action-family/target hierarchy, separate critic and clipping, kill
   shaping, legitimate visible geometry and interrupted three-second travel choices.
   Two 8,192-decision runs; 80 selection + 80 fresh control/confirmation matches.
   [Full settings, eight graphs, controls, processor and timings](../strategic-v2-20261027/README.md).
3. Pursuit correction: visible-player goals get their own learned family rather
   than competing with many room/task goals. Warm-start only our own impostor;
   retain identical PPO/reward/observation/rules. {pursuit_text}
   [Registered conditional protocol](../pursuit-conditional-protocol.md).

## Graphs and interpretation

V2's linked report explains all eight training, PPO, kill-exploration, selection,
paired-confirmation and compute graphs. Training wins/returns are not test accuracy;
falling value loss alone does not prove good strategy. The original control results
use different scenarios from the pursuit comparison; do not compare their percentages
as a direct paired test. One continuation is not a second independent training seed.

{links}

The continuation curves (when present) show additional experience, not a fresh
replicate. Pursuit frequency tests the intended exploration change. Selection plots
show the candidates used for the lock. Confirmation whiskers show small-sample
uncertainty; higher kills without more wins can still mean a poor policy. Extra
experience and changed factorization are confounded, so this is not a pure ablation.

## Time, checks and remaining limitations

- V2 study: {v2['seconds']/60:.2f} minutes, including its two runs and 160 evaluations.
- {timing}
- The original three-hour deadline is unchanged. Preparation, reports and UI checks
  are outside timed learning/evaluation execution; concurrent preparation is reflected
  in wall time. Single CPU worker, one Torch thread, 32 logical CPUs available,
  no GPU; processor/runtime details are in V2's study manifest.
- 78 improved preflight tests, two controlled win/loss end-screen tests and three
  pursuit/warm-start/resume/viewer tests passed. Earlier engineering passed 239
  regression tests and 10 deterministic/control matches. Suites overlap; do not sum
  these counts as unique repository tests.
- Every released/copied checkpoint strictly loaded and retained its byte hash.
  Frozen full-29/belief/history checks remain enabled. Final confirmation has zero
  illegal actions or navigation failures. A recorded win is not proof of deception.
- No reserved final tests, merged PRs, deployment or multi-agent self-play.
- Results concern one frozen legacy-to-V2 learned crew plus three scripts. They do
  not establish strength against human opponents or four improved learned crew.

## Next stage

1. Fix the repeated FAKE_TASK interface in a new version: test a finishable,
   interruptible option, or suppress only mechanically redundant restarts while the
   same fake is active. Keep the decision to fake/leave learned; do not force hunting
   or alter the frozen canonical engine. Verify effects before another training run.
2. Test the prepared pursuit family so visible-player goals do not compete with
   many room/task destinations. It still chooses family/target neurally; it does not
   auto-chase or auto-kill. No pursuit training study ran in this delivery.
3. Register fresh scenarios. Compare sampled and greedy behavior explicitly; inspect
   actor KL/entropy, useful opportunities and task-loss rate before extending training.
   Then tune learning rate/entropy or temporal memory one factor at a time.
4. If shaping is needed, test potential-based rewards from legitimate observations
   with terminal handling, rather than hidden witness labels or rewards for fake-task
   activity. Keep real wins primary and do not make cooldown/task rules easier.

Freeze the chosen impostor. When your friend's best compatible crewmate is ready,
pin its checkpoint/hash, verify its V2 inputs and ghost isolation, and run paired
cross-play against both old/new crew. Only after useful single learned agents work,
explore adapting opponents and self-play with frozen opponent pools and held-out
tests. This release is best within its registered selection set, not globally best.
'''
    (destination/'README.md').write_text(text,encoding='utf-8')
    print(json.dumps(summary,indent=2,allow_nan=False));return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v2-source',required=True,type=Path);p.add_argument('--v2-report',required=True,type=Path)
    p.add_argument('--destination',required=True,type=Path);p.add_argument('--v3-source',type=Path)
    p.add_argument('--release-directory',type=Path)
    p.add_argument('--pursuit-skip-reason')
    a=p.parse_args();build(a.v2_source,a.v2_report,a.destination,a.v3_source,a.release_directory,a.pursuit_skip_reason)
