"""Offline graphs and auditable aggregates; never trains, evaluates or selects."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def wilson(k,n):
    if not n:return [0.,1.]
    z=1.959963984540054;p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,float(c-h)),min(1.,float(c+h))]


def aggregate(rows):
    n=len(rows);wins=sum(r['impostor_win'] for r in rows)
    return dict(matches=n,wins=wins,win_rate=wins/n,wilson95=wilson(wins,n),
        mean_kills=float(np.mean([r['kills'] for r in rows])),
        witnessed_kills=sum(r['witnessed_kills'] for r in rows),
        own_ejections=sum(r['impostor_ejected'] for r in rows),
        crew_reports=sum(r['crew_reports'] for r in rows),
        impostor_reports=sum(r['impostor_reports'] for r in rows),
        correct_crew_votes=sum(r['correct_crew_votes'] for r in rows),
        incorrect_crew_votes=sum(r['incorrect_crew_votes'] for r in rows),
        crew_skips=sum(r['crew_skips'] for r in rows),
        innocent_team_ejections=sum(r['innocent_team_ejections'] for r in rows),
        impostor_claims=dict(sum((Counter(r['impostor_claims']) for r in rows),Counter())),
        reasons=dict(Counter(r['reason'] for r in rows)),
        actions=dict(sum((Counter(r['actions']) for r in rows),Counter())),
        illegal_actions=sum(r['illegal_actions'] for r in rows),
        navigation_failures=sum(r['navigation_failures'] for r in rows),
        simulated_seconds=sum(r['simulated_seconds'] for r in rows),
        wall_seconds=sum(r['seconds'] for r in rows))


def action_family(kind):
    if kind=='wait':return 'continue'
    if kind.startswith('go_to_') and kind!='go_to_body':return 'travel'
    if kind=='cancel_navigation':return 'cancel'
    if kind=='go_to_body':return 'approach_body'
    if kind=='call_meeting':return 'emergency'
    if kind.startswith('claim_'):return 'claim'
    return kind


def generate(source,destination):
    source=Path(source);destination=Path(destination)
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    study=read(source/'study.json');result=read(source/'results.json');lock=read(source/'selection-lock.json')
    if result['status']!='complete' or result['final_tests_opened'] or len(result['confirmation'])!=80:
        raise ValueError('Only a completed registered study can produce this report')
    if result['selected']!=lock['selected']:raise ValueError('Selection identity drift')
    methods=('selected_family','pilot83','scripted_hunter','untrained_family')
    rows={m:[r for r in result['confirmation'] if r['method']==m] for m in methods}
    expected=set(study['confirmation'])
    if any(len(r)!=20 or {v['seed'] for v in r}!=expected for r in rows.values()):raise ValueError('Incomplete paired confirmation')
    if any(r['illegal_actions'] or r['navigation_failures'] for r in result['confirmation']):raise ValueError('Gameplay verification failed')
    selection_files=sorted((source/'selection').glob('*.json'))
    if len(selection_files)!=80:raise ValueError('Incomplete selection')
    selected_rows=[read(p) for p in selection_files]
    if any(r['illegal_actions'] or r['navigation_failures'] for r in selected_rows):raise ValueError('Selection gameplay verification failed')
    destination.mkdir(parents=True,exist_ok=False);figures=destination/'figures';figures.mkdir()
    training={s:read(source/f'seed-{s}'/'training.json') for s in (71,83)}
    updates={s:read(source/f'seed-{s}'/'updates.json') for s in (71,83)}
    episodes={s:read(source/f'seed-{s}'/'episodes.json') for s in (71,83)}
    confirmation={m:aggregate(r) for m,r in rows.items()}
    selection={c['method']:aggregate([r for r in selected_rows if r['method']==c['method']]) for c in result['selection']}
    paired={m:{r['seed']:r for r in records} for m,records in rows.items()}
    comparisons={};rng=np.random.default_rng(20261027)
    for m in methods[1:]:
        a=np.asarray([int(paired['selected_family'][s]['impostor_win']) for s in sorted(expected)])
        b=np.asarray([int(paired[m][s]['impostor_win']) for s in sorted(expected)])
        delta=a-b;boot=rng.choice(delta,size=(10000,len(delta)),replace=True).mean(1)
        comparisons[m]=dict(win_rate_difference=float(delta.mean()),paired_bootstrap95=list(map(float,np.quantile(boot,[.025,.975]))),
            selected_only_wins=int(((a==1)&(b==0)).sum()),control_only_wins=int(((a==0)&(b==1)).sum()),
            caveat='20 paired scenarios; descriptive intervals, no multiplicity adjustment. A degenerate all-equal bootstrap does not establish population equivalence.')
    summary=dict(status='locked_confirmation_validation_not_final_test',selected=result['selected'],
        training=training,selection=selection,confirmation=confirmation,paired_comparisons=comparisons,
        seconds=result['seconds'],started_epoch=study['started_epoch'],finished_epoch=study['started_epoch']+result['seconds'],
        selection_locked_epoch=lock['locked_epoch'],source_commit=study['record']['source_commit'],
        frozen_weights_unchanged=result['frozen_weights_unchanged'],final_tests_opened=False,
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        caveats=['Bundled architecture/reward/option change, not a controlled single-factor ablation.',
            'Only two initialization seeds and 20 independent confirmation scenarios per method.',
            'Greedy family-then-target evaluation differs from sampled learning.',
            'Travel options change simulated experience per decision; raw decision counts are not equal compute.',
            'Old-pilot and hunter controls retain their original decision interface and sparse reward.',
            'All methods use identical ordinary V2 rules, frozen full-29 plus three scripts, and crew safeguards.',
            'No model identity is changed using confirmation outcomes.'])
    def write(name,value):(destination/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write('summary.json',summary);write('confirmation.json',result['confirmation']);write('selection.json',selected_rows)
    for name in ('study.json','selection-lock.json','results.json'):shutil.copy2(source/name,destination/name)
    for s in (71,83):
        folder=destination/f'seed-{s}';folder.mkdir()
        for name in ('training.json','updates.json','episodes.json','start.json'):shutil.copy2(source/f'seed-{s}'/name,folder/name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    colors={71:'#1676b6',83:'#e47f21'}
    def save(fig,name):fig.tight_layout();fig.savefig(figures/name,dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for s,data in episodes.items():
        for ax,key,title in zip(axes.flat,('return_','impostor_win','kills','impostor_ejected'),
            ('Shaped episode return','Cumulative sampled-training win rate','Kills / completed training game','Cumulative own-ejection rate')):
            y=np.asarray([r[key] for r in data],float)
            if key in ('impostor_win','impostor_ejected'):y=np.cumsum(y)/np.arange(1,len(y)+1)
            ax.plot([r['decisions'] for r in data],y,'.-',color=colors[s],label=f'seed {s}')
            ax.set_title(title);ax.set_xlabel('Training policy decisions')
    axes[0,1].set_ylim(0,1);axes[1,1].set_ylim(0,1);axes[0,0].legend()
    fig.suptitle('Improved impostor — training, not held-out accuracy');save(fig,'learning_curves.png')
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for ax,key,title in zip(axes.flat,('policy_loss','value_loss','entropy','approximate_kl','clip_fraction','explained_variance'),
        ('Actor loss','Critic squared error','Action-family entropy','Approximate KL','PPO clipped fraction','Critic explained variance')):
        for s,data in updates.items():ax.plot([r['decisions'] for r in data],[r[key] for r in data],'.-',color=colors[s],label=f'seed {s}')
        ax.set_title(title);ax.set_xlabel('Training policy decisions')
    axes[0,0].legend();save(fig,'ppo_diagnostics.png')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for s,data in updates.items():
        x=[r['decisions'] for r in data]
        axes[0].plot(x,[r['kill_probability_when_available'] for r in data],'.-',color=colors[s],label=f'seed {s}')
        axes[1].plot(x,[r['kill_selections']/max(1,r['kill_available_decisions']) for r in data],'.-',color=colors[s])
        axes[2].plot(x,[r['actor_gradient_norm'] for r in data],color=colors[s],label=f'actor {s}')
        axes[2].plot(x,[r['critic_gradient_norm'] for r in data],'--',color=colors[s],label=f'critic {s}')
    axes[0].set_title('Mean learned KILL probability when legal');axes[0].set_ylim(0,1);axes[0].legend()
    axes[1].set_title('Cumulative sampled KILL / legal opportunities');axes[1].set_ylim(0,1)
    axes[2].set_title('Gradient norms before separate clipping');axes[2].legend(fontsize=8)
    for ax in axes:ax.set_xlabel('Training policy decisions')
    save(fig,'kill_exploration.png')
    families=sorted({action_family(k) for t in training.values() for k in t['actions']})
    fig,ax=plt.subplots(figsize=(12,4));x=np.arange(len(families))
    for i,s in enumerate((71,83)):
        counts=Counter()
        for kind,n in training[s]['actions'].items():counts[action_family(kind)]+=n
        ax.bar(x+(i-.5)*.38,[counts[k]/8192 for k in families],.38,label=f'seed {s}',color=colors[s])
    ax.set_xticks(x,families,rotation=40,ha='right');ax.set_ylabel('Fraction of sampled training actions');ax.legend()
    ax.set_title('Action-family usage — travel destinations pooled');save(fig,'action_families.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for s in (71,83):
        keys=[f'seed{s}-step{step}' for step in (4096,8192)];data=[selection[k] for k in keys]
        axes[0].plot([4096,8192],[r['win_rate'] for r in data],'.-',label=f'seed {s}',color=colors[s])
        axes[1].plot([4096,8192],[r['mean_kills'] for r in data],'.-',color=colors[s])
    axes[0].set_ylim(0,1);axes[0].set_title('Selection wins / 20 paired scenarios');axes[0].legend()
    axes[1].set_title('Selection mean kills / match')
    for ax in axes:ax.set_xlabel('Checkpoint training decisions');ax.set_xticks([4096,8192])
    fig.suptitle('Selection only — immutable lock precedes confirmation');save(fig,'checkpoint_selection.png')
    labels=['Selected NN','Old pilot 83','Hunter script','Untrained NN'];palette=['#1676b6','#86939c','#a271ac','#df9c30']
    fig,axes=plt.subplots(1,3,figsize=(13,4));rates=[confirmation[m]['win_rate'] for m in methods]
    intervals=[confirmation[m]['wilson95'] for m in methods]
    errors=np.maximum(0,np.asarray([[p-c[0] for p,c in zip(rates,intervals)],[c[1]-p for p,c in zip(rates,intervals)]]))
    axes[0].bar(labels,rates,color=palette,yerr=errors,capsize=5);axes[0].set_ylim(0,1);axes[0].set_title('Confirmation win rate / Wilson 95% CI')
    axes[1].bar(labels,[confirmation[m]['mean_kills'] for m in methods],color=palette);axes[1].set_title('Mean kills / confirmation match')
    reasons=sorted({r['reason'] for r in result['confirmation']});bottom=np.zeros(4)
    for reason in reasons:
        counts=np.asarray([confirmation[m]['reasons'].get(reason,0) for m in methods])
        axes[2].bar(labels,counts,bottom=bottom,label=reason);bottom+=counts
    axes[2].set_title('End reasons (20 games per method)');axes[2].legend(fontsize=7)
    for ax in axes:ax.tick_params(axis='x',rotation=35)
    save(fig,'confirmation_results.png')
    fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
    for i,m in enumerate(methods):
        data=[paired[m][s] for s in sorted(expected)]
        axes[0].plot(range(20),[int(r['impostor_win'])+i*.025 for r in data],'.-',label=labels[i],color=palette[i])
        axes[1].plot(range(20),[r['kills']+i*.025 for r in data],'.-',color=palette[i])
    axes[0].set_ylabel('Win (0/1; slight display offsets)');axes[0].legend(ncol=4,fontsize=8)
    axes[1].set_ylabel('Kills (slight display offsets)');axes[1].set_xlabel('Paired confirmation scenario index: 3200040–3200059')
    fig.suptitle('Same 20 scenarios per method; 80 executions are not 80 independent cases');save(fig,'paired_scenarios.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    collection=[sum(r['collection_seconds'] for r in updates[s])/60 for s in (71,83)]
    optimize=[sum(r['optimization_seconds'] for r in updates[s])/60 for s in (71,83)]
    axes[0].bar(['71','83'],collection,label='Environment + frozen inference');axes[0].bar(['71','83'],optimize,bottom=collection,label='PPO optimization')
    axes[0].set_ylabel('Minutes');axes[0].set_title('Measured training compute');axes[0].legend(fontsize=8)
    evaluation=sum(r['seconds'] for r in selected_rows+result['confirmation'])
    axes[1].bar(['Training','Selection','Confirmation'],[sum(t['wall_seconds'] for t in training.values())/60,
        sum(r['seconds'] for r in selected_rows)/60,sum(r['seconds'] for r in result['confirmation'])/60])
    axes[1].set_ylabel('Wall minutes');axes[1].set_title(f'Total study {result["seconds"]/60:.1f} minutes; includes overhead')
    save(fig,'compute.png')
    print(json.dumps(summary,indent=2,allow_nan=False))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('destination',type=Path)
    args=p.parse_args();generate(args.source,args.destination)
