"""Offline pilot summaries/graphs. Does not run games, train or select checkpoints."""
from collections import Counter
import json
import math
from pathlib import Path
import numpy as np


def wilson(wins,n):
    if not n:return [0.,1.]
    z=1.959963984540054;p=wins/n;d=1+z*z/n
    center=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [center-half,center+half]


def generate(source,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source=Path(source);destination=Path(destination)
    result=json.loads((source/'results.json').read_text())
    if result['status']!='completed_pilot_not_final_acceptance' or len(result['validation'])!=40:
        raise ValueError('Do not label partial pilot results complete')
    destination.mkdir(parents=True,exist_ok=False)
    figures=destination/'figures';figures.mkdir()
    training={s:json.loads((source/f'seed-{s}'/'training.json').read_text()) for s in (71,83)}
    updates={s:json.loads((source/f'seed-{s}'/'updates.json').read_text()) for s in (71,83)}
    episodes={s:json.loads((source/f'seed-{s}'/'episodes.json').read_text()) for s in (71,83)}
    records={s:[r for r in result['validation'] if r['initialization_seed']==s] for s in (71,83)}
    summaries={}
    for s,rows in records.items():
        wins=sum(r['impostor_win'] for r in rows)
        legal_times=[r['first_legal_kill'] for r in rows if r['first_legal_kill'] is not None]
        summaries[str(s)]=dict(matches=len(rows),wins=wins,win_rate=wins/len(rows),wilson95=wilson(wins,len(rows)),
            mean_kills=float(np.mean([r['kills'] for r in rows])),witnessed_kills=sum(r['witnessed_kills'] for r in rows),
            reasons=dict(Counter(r['reason'] for r in rows)),mean_seconds=float(np.mean([r['simulated_seconds'] for r in rows])),
            reports=sum(r['impostor_reports'] for r in rows),crew_reports=sum(r['crew_reports'] for r in rows),
            ejections=sum(r['impostor_ejected'] for r in rows),
            illegal_actions=sum(r['illegal_actions'] for r in rows),navigation_failures=sum(r['navigation_failures'] for r in rows),
            validation_wall_seconds=sum(r['seconds'] for r in rows),
            mean_first_legal_kill=float(np.mean(legal_times)) if legal_times else None)
    paired={s:{r['seed']:r for r in rows} for s,rows in records.items()}
    seeds=sorted(paired[71]);differences=np.asarray([float(paired[83][s]['impostor_win'])-float(paired[71][s]['impostor_win']) for s in seeds])
    rng=np.random.default_rng(20261027)
    boot=rng.choice(differences,size=(10000,len(differences)),replace=True).mean(1)
    summary=dict(status='pilot_validation_only',training=training,validation=summaries,
        paired_seed83_minus71=dict(win_rate_difference=float(differences.mean()),bootstrap95=list(map(float,np.quantile(boot,[.025,.975])))),
        seconds=result['seconds'],frozen_weights_unchanged=result['frozen_weights_unchanged'],final_tests_opened=False,
        checkpoint_search=False,notes=['20 paired validation scenarios per initialization; two trained seeds only.',
            'Greedy validation differs from sampled training actions.',
            'No same-partition untrained/scripted impostor evaluation control is included in this approved 40-match budget.',
            'Engineering controls use other scenarios and cannot establish a causal learning improvement.'])
    (destination/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (destination/'validation.json').write_text(json.dumps(result['validation'],indent=2,allow_nan=False)+'\n')
    colors={71:'#2389da',83:'#e67e22'}
    def save(fig,name):
        fig.tight_layout();fig.savefig(figures/name,dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for s,rows in episodes.items():
        x=[r['decisions'] for r in rows]
        metrics=[('return_','Training episode return'),('impostor_win','Training wins: cumulative rate'),
                 ('kills','Kills per completed training episode'),('simulated_seconds','Episode game seconds')]
        for ax,(key,title) in zip(axes.flat,metrics):
            y=np.asarray([r[key] for r in rows],float)
            if key=='impostor_win':y=np.cumsum(y)/np.arange(1,len(y)+1)
            ax.plot(x,y,'.-',color=colors[s],label=f'seed {s}',alpha=.85);ax.set_title(title);ax.set_xlabel('Training decisions')
    axes[0,0].legend();fig.suptitle('Impostor pilot — training episodes (not validation accuracy)')
    save(fig,'learning_curves.png')
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for ax,key,title in zip(axes.flat,('policy_loss','value_loss','entropy','approximate_kl','clip_fraction','gradient_norm'),
                            ('Policy loss','Value loss','Policy entropy','Approximate KL','Clipped ratio fraction','Gradient norm before clipping')):
        for s,rows in updates.items():ax.plot([r['decisions'] for r in rows],[r[key] for r in rows],'.-',label=f'seed {s}',color=colors[s])
        ax.set_title(title);ax.set_xlabel('Training decisions')
    axes[0,0].legend();save(fig,'ppo_diagnostics.png')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    rates=[summaries[str(s)]['win_rate'] for s in (71,83)]
    cis=[summaries[str(s)]['wilson95'] for s in (71,83)]
    axes[0].bar(['71','83'],rates,color=list(colors.values()),yerr=np.maximum(0,np.asarray([[p-c[0] for p,c in zip(rates,cis)],[c[1]-p for p,c in zip(rates,cis)]])),capsize=6)
    axes[0].set_ylim(0,1);axes[0].set_title('Validation win rate / Wilson 95% CI')
    axes[1].bar(['71','83'],[summaries[str(s)]['mean_kills'] for s in (71,83)],color=list(colors.values()));axes[1].set_title('Mean kills / validation match')
    reasons=sorted({r['reason'] for r in result['validation']});bottom=np.zeros(2)
    for reason in reasons:
        counts=np.asarray([summaries[str(s)]['reasons'].get(reason,0) for s in (71,83)])
        axes[2].bar(['71','83'],counts,bottom=bottom,label=reason);bottom+=counts
    axes[2].set_title('Validation end reasons (20 each)');axes[2].legend(fontsize=7)
    save(fig,'validation_results.png')
    fig,ax=plt.subplots(figsize=(12,4));kinds=sorted({k for rows in updates.values() for k in rows[-1]['actions']})
    x=np.arange(len(kinds))
    for i,s in enumerate((71,83)):
        counts=updates[s][-1]['actions'];ax.bar(x+(i-.5)*.4,[counts.get(k,0)/4096 for k in kinds],width=.4,label=f'seed {s}',color=colors[s])
    ax.set_xticks(x,kinds,rotation=55,ha='right');ax.set_ylabel('Fraction of training selections');ax.legend();ax.set_title('Sampled training actions')
    save(fig,'action_distribution.png')
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    for s in (71,83):
        axes[0].plot(range(20),[int(paired[s][k]['impostor_win']) for k in seeds],'.-',label=f'seed {s}',color=colors[s])
        axes[1].plot(range(20),[paired[s][k]['kills'] for k in seeds],'.-',color=colors[s])
    axes[0].set_ylabel('Validation win (0/1)');axes[0].legend();axes[1].set_ylabel('Kills');axes[1].set_xlabel('Paired validation scenario index (3200000–3200019)')
    fig.suptitle('Same scenarios, two trained initializations — no checkpoint search');save(fig,'paired_scenarios.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for s,rows in updates.items():
        axes[0].plot([r['decisions'] for r in rows],[r['seconds']/60 for r in rows],'.-',label=f'seed {s}',color=colors[s])
    axes[0].set_xlabel('Training decisions');axes[0].set_ylabel('Collection + optimization minutes');axes[0].legend()
    collect=[sum(r['collection_seconds'] for r in updates[s])/60 for s in (71,83)]
    optim=[sum(r['optimization_seconds'] for r in updates[s])/60 for s in (71,83)]
    axes[1].bar(['71','83'],collect,label='Environment + inference');axes[1].bar(['71','83'],optim,bottom=collect,label='PPO optimization');axes[1].set_ylabel('Minutes');axes[1].legend()
    save(fig,'compute.png')
    return summary
