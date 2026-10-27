"""Offline complete-study graphs/report. Never selects, evaluates or trains a model."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo
import numpy as np


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(wins,n):
    if not n:return [0.,1.]
    z=1.959963984540054;p=wins/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,float(center-half)),min(1.,float(center+half))]


def aggregate(rows):
    n=len(rows)
    if not n:raise ValueError('Cannot infer performance from an empty partition')
    wins=sum(r['impostor_win'] for r in rows)
    total=lambda key:sum(r.get(key,0) for r in rows)
    mean=lambda key:total(key)/n
    rewards=Counter()
    for row in rows:rewards.update(row.get('reward_components',{}))
    return dict(matches=n,wins=wins,win_rate=wins/n,wilson95=wilson(wins,n),mean_kills=mean('kills'),
        kills=total('kills'),witnessed_kills=total('witnessed_kills'),own_ejections=total('impostor_ejected'),
        fake_starts=total('fake_starts'),fake_completions=total('fake_completions'),
        fake_selections=total('fake_selections'),fake_selections_no_visible=total('fake_selections_no_visible'),
        mean_max_fake_streak=mean('max_fake_selection_streak'),fake_activity_seconds=total('fake_activity_seconds'),
        ready_faking_seconds=total('ready_faking_seconds'),ready_seconds=total('measured_cooldown_ready_seconds'),
        pursuit_positive_total=total('pursuit_positive_total'),
        rewards=dict(rewards),
        crew_reports=total('crew_reports'),impostor_reports=total('impostor_reports'),
        correct_crew_votes=total('correct_crew_votes'),incorrect_crew_votes=total('incorrect_crew_votes'),
        innocent_team_ejections=total('innocent_team_ejections'),
        claims=dict(sum((Counter(r.get('impostor_claims',{})) for r in rows),Counter())),
        actions=dict(sum((Counter(r.get('actions',{})) for r in rows),Counter())),
        reasons=dict(Counter(r['reason'] for r in rows)),
        illegal_actions=total('illegal_actions'),navigation_failures=total('navigation_failures'),
        simulated_seconds=total('simulated_seconds'),wall_seconds=total('seconds'))


def paired(a,b):
    amap={r['seed']:r for r in a};bmap={r['seed']:r for r in b}
    if set(amap)!=set(bmap):raise ValueError('Unpaired cases')
    delta=np.asarray([int(amap[s]['impostor_win'])-int(bmap[s]['impostor_win']) for s in sorted(amap)])
    rng=np.random.default_rng(20261027);boot=rng.choice(delta,(10000,len(delta)),replace=True).mean(1)
    return dict(win_rate_difference=float(delta.mean()),paired_bootstrap95=list(map(float,np.quantile(boot,[.025,.975]))),
        v4_only_wins=int((delta==1).sum()),v2_only_wins=int((delta==-1).sum()),
        caveat='20 paired cases; descriptive interval. Degenerate all-equal bootstrap is not population equivalence.')


def generate(source,destination):
    source=Path(source);destination=Path(destination)
    study=read(source/'study.json');result=read(source/'results.json');lock=read(source/'selection-lock.json')
    if result['status']!='complete' or result['final_tests_opened'] or result['selected']!=lock['selected']:
        raise ValueError('Need complete results with unchanged immutable selection')
    if study['steps']!=8192 or study['initializations']!=[71,83] or study['budget_seconds']!=14400:
        raise ValueError('Unexpected study protocol')
    methods=('selected_v4','unchanged_v2')
    confirmation={m:[r for r in result['confirmation'] if r['method']==m] for m in methods}
    expected=set(range(3200120,3200140))
    if any(len(rows)!=20 or {r['seed'] for r in rows}!=expected for rows in confirmation.values()):
        raise ValueError('Incomplete fresh confirmation')
    selection=[read(p) for p in sorted((source/'selection').glob('*.json'))]
    if len(selection)!=80 or len(result['diagnostic'])!=5:raise ValueError('Incomplete selection/diagnostic')
    if {r['seed'] for r in result['diagnostic']}!=set(range(3200100,3200105)):
        raise ValueError('Unexpected sampled diagnostic cases')
    for candidate in result['selection']:
        rows=[r for r in selection if r['method']==candidate['method']]
        if len(rows)!=20 or {r['seed'] for r in rows}!=set(range(3200100,3200120)):
            raise ValueError('Incomplete candidate selection')
    if any(r['illegal_actions'] or r['navigation_failures'] for r in selection+result['confirmation']+result['diagnostic']):
        raise ValueError('Gameplay integrity failure')
    for p in (source/'confirmation').glob('*.json'):
        if p.stat().st_mtime < lock['locked_epoch']:raise ValueError('Confirmation predates selection lock')
    training={s:read(source/f'seed-{s}/training.json') for s in (71,83)}
    episodes={s:read(source/f'seed-{s}/episodes.json') for s in (71,83)}
    updates={s:read(source/f'seed-{s}/updates.json') for s in (71,83)}
    scores={m:aggregate(rows) for m,rows in confirmation.items()}
    selected=result['selected'];chosen=Path(__file__).resolve().parents[1]/selected['path']
    if digest(chosen)!=selected['sha256']:raise ValueError('Locked checkpoint changed')
    summary=dict(status='complete_validation_confirmation_not_reserved_final',selected=selected,training=training,
        confirmation=scores,paired=paired(confirmation['selected_v4'],confirmation['unchanged_v2']),
        diagnostic=aggregate(result['diagnostic']),seconds=result['seconds'],started_epoch=study['started_epoch'],
        finished_epoch=study['started_epoch']+result['seconds'],locked_epoch=lock['locked_epoch'],
        source_commit=study['record']['core']['source_commit'],frozen_weights_unchanged=result['frozen_weights_unchanged'],
        final_tests_opened=False,analysis_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        caveats=['Bundled reward/action changes; no isolated causal ablation.',
            'Two training initializations; only one selected checkpoint on20 fresh cases.',
            'Old V2 control uses original feature/reward/action semantics; shaped returns are not comparable.',
            'Five sampled diagnostics reuse selection cases; not confirmation or an unbiased greedy/sampled estimate.',
            'Mixed crew: frozen full-29 NN and3 scripts with explicit reporting/witness safeguards.',
            'No reserved final tests, new lobby size or model identity changes using confirmation.'])
    destination.mkdir(parents=True,exist_ok=False);figures=destination/'figures';figures.mkdir()
    def write(name,obj):(destination/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write('summary.json',summary);write('selection.json',selection);write('confirmation.json',result['confirmation'])
    for name in ('study.json','results.json','selection-lock.json'):shutil.copy2(source/name,destination/name)
    for s in (71,83):
        folder=destination/f'seed-{s}';folder.mkdir()
        for name in ('training.json','updates.json','episodes.json','start.json'):shutil.copy2(source/f'seed-{s}'/name,folder/name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    colors={71:'#1676b6',83:'#df7b24'}
    def save(fig,name):fig.tight_layout();fig.savefig(figures/name,dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for s,rows in episodes.items():
        for ax,key,title in zip(axes.flat,('return_','impostor_win','kills','impostor_ejected'),
            ('Shaped return (not accuracy)','Cumulative sampled-training win fraction','Kills per training game','Cumulative own-ejection fraction')):
            values=np.asarray([r[key] for r in rows],float)
            if key in ('impostor_win','impostor_ejected'):values=np.cumsum(values)/np.arange(1,len(values)+1)
            ax.plot([r['decisions'] for r in rows],values,'.-',color=colors[s],label=f'init{s}');ax.set_title(title)
            ax.set_xlabel('Training policy decisions')
    axes[0,0].legend();axes[0,1].set_ylim(0,1);axes[1,1].set_ylim(0,1);save(fig,'learning_curves.png')
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for ax,key in zip(axes.flat,('policy_loss','value_loss','entropy','approximate_kl','clip_fraction','explained_variance')):
        for s,rows in updates.items():ax.plot([r['decisions'] for r in rows],[r[key] for r in rows],'.-',color=colors[s],label=f'init{s}')
        ax.set_title(key.replace('_',' '));ax.set_xlabel('Training policy decisions')
    axes[0,0].legend();save(fig,'ppo_diagnostics.png')
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for s,rows in episodes.items():
        x=[r['decisions'] for r in rows]
        for ax,key,title in zip(axes.flat,('max_fake_selection_streak','fake_selections_no_visible','fake_completions','fake_activity_seconds'),
            ('Longest consecutive FAKE choices','FAKE choices with no player visible','Completed fake activities','Actual simulated faking seconds')):
            ax.plot(x,[r[key] for r in rows],'.-',color=colors[s],label=f'init{s}');ax.set_title(title);ax.set_xlabel('Training policy decisions')
    axes[0,0].legend();save(fig,'fake_behavior.png')
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for s,rows in episodes.items():
        x=[r['decisions'] for r in rows]
        axes[0,0].plot(x,[r['measured_cooldown_ready_seconds'] for r in rows],'.-',color=colors[s],label=f'init{s}')
        axes[0,1].plot(x,[r['ready_faking_seconds'] for r in rows],'.-',color=colors[s])
        axes[1,0].plot(x,[r['pursuit_positive_total'] for r in rows],'.-',color=colors[s])
        axes[1,1].plot(x,[r['reward_components']['cooldown_ready'] for r in rows],'.-',color=colors[s])
    for ax,title in zip(axes.flat,('Cooldown-ready roaming seconds','Ready time spent actually faking','Positive pursuit credit (terminal correction repays)','Cooldown-ready reward component')):
        ax.set_title(title);ax.set_xlabel('Training policy decisions')
    axes[1,0].set_ylim(-.001,.051);axes[0,0].legend();save(fig,'ready_and_pursuit.png')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for s,rows in updates.items():
        x=[r['decisions'] for r in rows]
        axes[0].plot(x,[r['kill_probability_when_available'] for r in rows],'.-',color=colors[s],label=f'init{s}')
        axes[1].plot(x,[r['kill_selections']/max(1,r['kill_available_decisions']) for r in rows],'.-',color=colors[s])
        axes[2].plot(x,[r['actor_gradient_norm'] for r in rows],color=colors[s],label=f'actor{s}')
        axes[2].plot(x,[r['critic_gradient_norm'] for r in rows],'--',color=colors[s],label=f'critic{s}')
    for ax,title in zip(axes,('Mean KILL probability when legal','Sampled KILL / legal opportunities','Gradient norms before clipping')):
        ax.set_title(title);ax.set_xlabel('Training policy decisions')
    axes[0].set_ylim(0,1);axes[1].set_ylim(0,1);axes[0].legend();axes[2].legend(fontsize=8);save(fig,'kill_exploration.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for s in (71,83):
        candidates=[c for c in result['selection'] if c['seed']==s]
        axes[0].plot([c['step'] for c in candidates],[c['wins']/20 for c in candidates],'.-',color=colors[s],label=f'init{s}')
        axes[1].plot([c['step'] for c in candidates],[c['kills']/20 for c in candidates],'.-',color=colors[s])
    axes[0].set_ylim(0,1);axes[0].set_title('Selection wins /20 paired scenarios');axes[0].legend()
    axes[1].set_title('Selection mean kills');save(fig,'checkpoint_selection.png')
    labels=['Locked V4 NN','Unchanged V2 NN'];palette=['#1676b6','#a17d6a']
    fig,axes=plt.subplots(1,3,figsize=(13,4));rates=[scores[m]['win_rate'] for m in methods]
    intervals=[scores[m]['wilson95'] for m in methods]
    error=np.maximum(0,np.asarray([[p-ci[0] for p,ci in zip(rates,intervals)],[ci[1]-p for p,ci in zip(rates,intervals)]]))
    axes[0].bar(labels,rates,color=palette,yerr=error,capsize=5);axes[0].set_ylim(0,1);axes[0].set_title('Fresh win rate / Wilson95% CI')
    axes[1].bar(labels,[scores[m]['mean_kills'] for m in methods],color=palette);axes[1].set_title('Mean kills / fresh match')
    bottom=np.zeros(2)
    for reason in sorted({r['reason'] for r in result['confirmation']}):
        counts=np.asarray([scores[m]['reasons'].get(reason,0) for m in methods]);axes[2].bar(labels,counts,bottom=bottom,label=reason);bottom+=counts
    axes[2].set_title('End reasons /20 per method');axes[2].legend(fontsize=8)
    save(fig,'confirmation_results.png')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,title in zip(axes,('mean_max_fake_streak','ready_faking_seconds','fake_completions'),
        ('Mean longest FAKE choice streak','Total ready-faking seconds /20 cases','Total fake completions /20 cases')):
        ax.bar(labels,[scores[m][key] for m in methods],color=palette);ax.set_title(title)
    save(fig,'confirmation_fake_loops.png')
    fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
    for i,m in enumerate(methods):
        rows=sorted(confirmation[m],key=lambda r:r['seed'])
        axes[0].plot(range(20),[int(r['impostor_win'])+i*.025 for r in rows],'.-',label=labels[i],color=palette[i])
        axes[1].plot(range(20),[r['kills']+i*.025 for r in rows],'.-',color=palette[i])
    axes[0].legend();axes[0].set_ylabel('Win0/1 (small display offset)');axes[1].set_ylabel('Kills (small display offset)')
    axes[1].set_xlabel('Paired fresh cases3200120–3200139');save(fig,'paired_cases.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    collection=[sum(r['collection_seconds'] for r in updates[s])/60 for s in (71,83)]
    optimization=[sum(r['optimization_seconds'] for r in updates[s])/60 for s in (71,83)]
    axes[0].bar(['71','83'],collection,label='Environment/inference');axes[0].bar(['71','83'],optimization,bottom=collection,label='PPO optimization')
    axes[0].set_ylabel('Wall minutes');axes[0].legend();axes[0].set_title('Training wall breakdown')
    axes[1].bar(['Train','Select','Sample diag','Confirm'],[sum(t['wall_seconds'] for t in training.values())/60,
        sum(r['seconds'] for r in selection)/60,sum(r['seconds'] for r in result['diagnostic'])/60,
        sum(r['seconds'] for r in result['confirmation'])/60]);axes[1].set_ylabel('Wall minutes')
    axes[1].set_title(f'Total study {result["seconds"]/60:.1f}min including overhead');save(fig,'compute.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,keys,title in ((axes[0],('crew_reports','impostor_reports','own_ejections'),'Reports and own ejections'),
                         (axes[1],('correct_crew_votes','incorrect_crew_votes','innocent_team_ejections'),'Crew votes/ejections (not NN accuracy)')):
        x=np.arange(len(keys))
        for i,m in enumerate(methods):ax.bar(x+(i-.5)*.36,[scores[m][k] for k in keys],.36,label=labels[i],color=palette[i])
        ax.set_xticks(x,[k.replace('_',' ') for k in keys],rotation=25,ha='right');ax.set_title(title)
    axes[0].legend(fontsize=8);save(fig,'meetings_and_votes.png')
    stamp=lambda value:datetime.fromtimestamp(value,ZoneInfo('Asia/Amman')).strftime('%Y-%m-%d %H:%M:%S %Z')
    chosen_score=scores['selected_v4'];control=scores['unchanged_v2']
    assessment=('The selected model has no fresh confirmation wins; useful winning strategy is still unproven.'
        if not chosen_score['wins'] else
        'The selected model has observed fresh wins in this narrow mixed-roster setup. The small sample does not establish broad reliability or globally best play.')
    lines=['# V4 learned impostor — study results','',assessment,'',
        f"Run `python run_impostor_options.py --checkpoint artifacts/impostor-options-v4/release/policy.pt` after curation.",
        '',f"Immutable selection: initialization{selected['seed']}, {selected['step']} decisions. Do not choose another using confirmation.",
        '', '| Fresh method | Wins | Wilson95% interval | Mean kills | Mean longest FAKE streak |',
        '|---|---:|---:|---:|---:|']
    for m,label in zip(methods,labels):
        v=scores[m];lo,hi=v['wilson95'];lines.append(f"| {label} | {v['wins']}/20 | {lo:.1%}–{hi:.1%} | {v['mean_kills']:.2f} | {v['mean_max_fake_streak']:.1f} |")
    lines+=['','Shaped returns differ between versions; true wins are the comparison.40 executions are20 paired cases.',
        f"Paired win difference: {summary['paired']['win_rate_difference']:+.1%}; descriptive bootstrap interval {summary['paired']['paired_bootstrap95']}.",
        '', '## Training and timing','', '| Initialization | Decisions | Complete games | Training wins | Kills | Wall min |', '|---|---:|---:|---:|---:|---:|']
    for s in (71,83):
        t=training[s];rows=episodes[s]
        lines.append(f"| {s} | {t['decisions']} | {len(rows)} | {sum(r['impostor_win'] for r in rows)} | {sum(r['kills'] for r in rows)} | {t['wall_seconds']/60:.2f} |")
    lines+=['',f"Started {stamp(study['started_epoch'])}; finished {stamp(summary['finished_epoch'])}; total{result['seconds']/60:.2f}min.",
        f"Frozen runtime commit `{summary['source_commit']}`. One CPU worker/one Torch thread; execution details in study.json.",
        '', '## How to read the graphs','']
    captions=[('learning_curves','Sampled training progress; returns and training wins are not held-out accuracy.'),
        ('ppo_diagnostics','Loss, entropy, KL, clipping and value estimates; stable optimization alone does not imply good play.'),
        ('fake_behavior','Whether learned fake streaks/empty-visibility faking decline, beyond merely fixing activity completion.'),
        ('ready_and_pursuit','Idle-ready cost and bounded pursuit credit; terminal correction removes net pursuit payment.'),
        ('kill_exploration','Learned kill probability/opportunity use and gradient norms.'),
        ('checkpoint_selection','Selection cases only; these picked the immutable checkpoint before confirmation.'),
        ('confirmation_results','Fresh wins, kills and actual loss reasons; whiskers show uncertainty.'),
        ('confirmation_fake_loops','Behavioral fake-loop comparison under each original version, across the same20 cases.'),
        ('paired_cases','Match-by-match outcomes under identical scenario IDs; display offsets separate overlapping lines.'),
        ('compute','Environment/frozen inference versus optimization and study phase wall times.'),
        ('meetings_and_votes','Reports, crew votes and ejections. This is not a natural-language deception score.')]
    for name,caption in captions:lines+=['',caption,'',f'![{caption}](figures/{name}.png)']
    lines+=['','## Limits and next decision','',*['- '+c for c in summary['caveats']],
        '', 'Five sampled diagnostics are in results.json, reuse selection cases, and did not change the lock.',
        'If fake loops/wins remain weak, record failure and propose a fresh study; do not silently add steps or tune from confirmation.',
        'Future up-to15-player/1–3-impostor settings and cross-play remain separate. No merge/deployment or historical release overwrite.']
    (destination/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path);parser.add_argument('destination',type=Path)
    args=parser.parse_args();summary=generate(args.source,args.destination)
    print(json.dumps(dict(selected=summary['selected'],confirmation=summary['confirmation'],seconds=summary['seconds']),indent=2))
