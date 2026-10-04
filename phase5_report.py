"""Measured single-run Phase 5 figures and an explicit learning assessment."""
import json
from pathlib import Path
import numpy as np


def assess(validation, policy_method):
    learned=validation['metrics'][policy_method]['ID']
    scripted=validation['metrics']['scripted']['ID']
    comparison=validation['paired'][policy_method]['ID']
    lower,upper=comparison['paired_match_bootstrap95']
    unsafe=learned['illegal_actions'] or learned['navigation_failures']
    if unsafe:
        decision='Repair correctness before further training'
    elif upper<0:
        decision='More learning or a revised training objective is needed'
    elif lower>0:
        decision='Promising single-seed improvement; replicate before claiming robust success'
    else:
        decision='Improvement is unproven; inspect behavior and extend validation before declaring success'
    return {'decision':decision,'based_on':'Reserved validation matches only; final tests are descriptive',
            'policy_win_rate':learned['crew_win_rate'],'scripted_win_rate':scripted['crew_win_rate'],
            'paired_difference':comparison,'policy_mean_own_tasks':learned['mean_own_tasks'],
            'scripted_mean_own_tasks':scripted['mean_own_tasks'],
            'additional_training_started':False,'training_seeds':1,
            'limitation':'No independent training-seed replication or current-only ablation in this approved experiment.'}


def report(training_dir,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);training=json.loads((Path(training_dir)/'training.json').read_text())
    validation=json.loads((output/'validation/metrics.json').read_text())
    final=json.loads((output/'test/metrics.json').read_text())
    rows=[json.loads(line) for line in (output/'test/matches.jsonl').read_text().splitlines()]
    methods=list(final['metrics']);policy=next(m for m in methods if m not in ('random','scripted'))
    names={m:('PPO / seed 17' if m==policy else 'Scripted' if m=='scripted' else 'Random legal') for m in methods}
    colors=['#8796a5','#b67b2e','#178879'];figures=output/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'figure.dpi':145,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
    episodes=training['episodes'];history=training['history']
    steps=np.array([e['training_step'] for e in episodes]);window=min(30,len(episodes))
    def smooth(field):return np.convolve([float(e[field]) for e in episodes],np.ones(window)/window,mode='valid')
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,field,title in zip(axes.flat[:3],('crew_win','return','own_tasks'),('Training crew win rate','Episode return','Own tasks completed')):
        ax.scatter(steps,[float(e[field]) for e in episodes],s=7,alpha=.15,color=colors[2])
        ax.plot(steps[window-1:],smooth(field),color=colors[2],label=f'{window}-match rolling mean')
        ax.set(xlabel='Training transitions',ylabel=title);ax.legend(fontsize=8)
    axes[1,1].plot([h['steps'] for h in history],[h['loss'] for h in history],color='#4379b8')
    axes[1,1].set(xlabel='Training transitions',ylabel='PPO objective (not accuracy)')
    fig.suptitle('Phase 5 learning curves | one training run, seed 17 | training outcomes are not test results')
    fig.tight_layout();fig.savefig(figures/'learning_curves.png');plt.close(fig)
    curve=json.loads((output/'validation-curve.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for ax,field,title in zip(axes,('crew_win_rate','mean_own_tasks'),('Validation crew win rate','Validation own tasks')):
        ax.plot([r['steps'] for r in curve],[r[field] for r in curve],marker='o',color=colors[2])
        ax.set(xlabel='Training transitions at saved checkpoint',ylabel=title)
    axes[0].set_ylim(0,1)
    fig.suptitle('Same 20 reserved validation matches | checkpoints from ONE run, seed 17 | small-sample diagnostic')
    fig.tight_layout();fig.savefig(figures/'validation_curve.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for ax,family in zip(axes,('ID','patient')):
        for i,m in enumerate(methods):
            r=final['metrics'][m][family];p=r['crew_win_rate'];lo,hi=r['win_rate_wilson95']
            ax.bar(i,p,color=colors[i]);ax.errorbar(i,p,yerr=[[max(0,p-lo)],[max(0,hi-p)]],fmt='none',color='#283644',capsize=5)
            ax.text(i,p+.035,f'{p:.0%}',ha='center')
        ax.set(xticks=range(len(methods)),xticklabels=[names[m] for m in methods],ylim=(0,1.15),ylabel='Crew win rate',title=f'{family}: {r["matches"]} paired match seeds')
    fig.suptitle('Final evaluation | 95% Wilson intervals across matches, not training seeds')
    fig.tight_layout();fig.savefig(figures/'evaluation.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for ax,family in zip(axes,('ID','patient')):
        for i,m in enumerate(methods):
            r=final['metrics'][m][family];accuracy=r['voting_accuracy'];votes=r['votes_cast']
            if accuracy is None:ax.text(i,.03,'N/A\nno cast votes',ha='center',color='#666666')
            else:
                ax.bar(i,accuracy,color=colors[i]);ax.text(i,accuracy+.035,f'{accuracy:.0%}\nn={votes}',ha='center')
            ax.text(i,-.17,f'{r["skips"]} skips',ha='center',fontsize=9)
        ax.set(xticks=range(len(methods)),xticklabels=[names[m] for m in methods],ylim=(0,1.22),ylabel='Correct impostor votes / non-skip votes',title=family)
    fig.suptitle('Voting accuracy is conditional on voting; compare with vote counts and crew wins')
    fig.tight_layout();fig.savefig(figures/'voting_accuracy.png');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,5.2))
    for ax,family in zip(axes,('ID','patient')):
        seeds=sorted({r['seed'] for r in rows if r['family']==family})
        lookup={(r['method'],r['seed']):r for r in rows}
        matrix=[[int(lookup[(m,s)]['crew_win']) for s in seeds] for m in methods]
        from matplotlib.colors import ListedColormap
        ax.imshow(matrix,aspect='auto',interpolation='nearest',cmap=ListedColormap(['#cf6a65','#178879']),vmin=0,vmax=1)
        positions=np.unique(np.linspace(0,len(seeds)-1,6,dtype=int))
        ax.set(yticks=range(len(methods)),yticklabels=[names[m] for m in methods],xticks=positions,xticklabels=[str(seeds[i]) for i in positions],xlabel='Shared evaluation match seed',title=family)
    fig.suptitle('Paired outcomes by match seed | green: crew win; red: loss/timeout | one PPO training seed')
    fig.tight_layout();fig.savefig(figures/'seed_outcomes.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for ax,field,title in zip(axes,('mean_own_tasks','mean_survival'),('Own tasks completed','Survival (simulated seconds)')):
        for i,family in enumerate(('ID','patient')):
            x=np.arange(len(methods))+(i-.5)*.35
            ax.bar(x,[final['metrics'][m][family][field] for m in methods],.35,label=family)
        ax.set(xticks=range(len(methods)),xticklabels=[names[m] for m in methods],ylabel=title);ax.legend()
    fig.suptitle('Task contribution and survival | avoid interpreting team wins alone as learned skill')
    fig.tight_layout();fig.savefig(figures/'behavior.png');plt.close(fig)
    assessment=assess(validation,policy)
    summary={'training_seed':training['config']['seed'],'training_transitions':training['config']['steps'],
        'training_episodes':len(episodes),'training_seconds':training['seconds'],
        'belief_frozen_verified':training['config']['belief_frozen_verified'],
        'checkpoint_sha256':training['checkpoint_sha256'],'assessment':assessment,
        'validation':validation,'validation_curve':curve,'final':final,'figures':[p.name for p in sorted(figures.glob('*.png'))]}
    (output/'results.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    lines=['# Phase 5: one strategic-policy training run','',
        f"Training seed **17**; **{training['config']['steps']:,} transitions**; **{len(episodes)} complete training matches**.",
        f"Training duration: **{training['seconds']/60:.1f} minutes**. Phase 4 belief weights remained frozen.",'',
        '## Learning assessment','',assessment['decision']+'.',
        'This decision uses reserved validation matches. Final tests are not used to select another model.',
        'Only one training seed was authorized. There is no replicated-seed or ablation claim.','',
        '| Set | Method | Crew win rate | Own tasks | Voting accuracy | Votes | Skips | Illegal / navigation failures |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for split,result in [('Validation',validation),('Final',final)]:
        for m,families in result['metrics'].items():
            for family,r in families.items():
                accuracy='N/A' if r['voting_accuracy'] is None else f"{r['voting_accuracy']:.1%}"
                lines.append(f"| {split} {family} | {names[m]} | {r['crew_win_rate']:.1%} | {r['mean_own_tasks']:.2f} | {accuracy} | {r['votes_cast']} | {r['skips']} | {r['illegal_actions']} / {r['navigation_failures']} |")
    lines+=['','## Figures','']
    for name in summary['figures']:lines += [f'![{name.replace("_"," ")}](figures/{name})','']
    lines+=['## Reproduction','',f'Checkpoint SHA-256: `{summary["checkpoint_sha256"]}`.',
        'Training configuration/source digests and exact match seeds are retained with the results.',
        'Action-type frequencies and policy entropy are recorded per evaluation match in matches.jsonl.',
        'Crew win rate is the strategic objective. Voting accuracy is conditional on non-skip votes; PPO loss is not accuracy.',
        'Three scripted teammates can carry a weak focal policy. Task contribution, action frequencies and paired outcomes matter.',
        'No additional learning is started automatically after this assessment.','']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    return summary
