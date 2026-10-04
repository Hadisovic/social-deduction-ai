"""Regenerate honest Phase 5 figures from persisted training/evaluation records."""
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'docs/benchmarks/phase5/expanded'


def read_rows(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # A live writer may be between writes. Never manufacture a data row.
            continue
    return result


def label(method):
    if method in {'random', 'scripted', 'idle'}:
        return {'random': 'Random options', 'scripted': 'Scripted', 'idle': 'Idle'}[method]
    path = Path(method)
    group = path.parent.name.replace('full-17-approved', 'V1 seed 17').replace('options-', 'V2 seed ')
    group = group.replace('-v2', '').replace('-ablation', ' current only')
    group = group.replace('-initial-control', ' untrained')
    stage = path.stem.replace('checkpoint-', '')
    return f'{group} / {stage}'


def build(output=DEFAULT):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    output = Path(output)
    figures = output / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'figure.dpi': 145, 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.bbox': 'tight'})
    runs = []
    for path in sorted((ROOT / 'artifacts/phase5/runs').glob('*/run-config.json')):
        config = json.loads(path.read_text())
        if path.parent.name != 'full-17-approved' and not path.parent.name.startswith('options-'):
            continue
        episodes = read_rows(path.parent / 'episodes.jsonl')
        updates = read_rows(path.parent / 'updates.jsonl')
        if not episodes or not updates:
            continue
        episodes = [e for e in episodes if e['training_step'] <= updates[-1]['steps']]
        if not episodes:
            continue
        name = label(str(path.parent / 'policy.pt')).replace(' / policy', '')
        runs.append(dict(name=name, directory=str(path.parent.relative_to(ROOT)), config=config,
                         episodes=episodes, updates=updates,
                         complete=(path.parent/'training.json').exists()))
    saved = []
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures/name)
        plt.close(fig)
        saved.append(name)

    if runs:
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        for run in runs:
            episodes, updates = run['episodes'], run['updates']
            window = min(30, len(episodes))
            steps = [e['training_step'] for e in episodes][window-1:]
            for ax, field, title in zip(axes.flat[:3], ('crew_win', 'return', 'own_tasks'),
                                        ('Crew win rate', 'Undiscounted return', 'Own completed tasks')):
                values = np.convolve([float(e[field]) for e in episodes], np.ones(window)/window, mode='valid')
                ax.plot(steps, values, label=run['name'])
                ax.set(xlabel='Policy decisions', ylabel=f'{title} ({window}-match mean)')
            axes[1,1].plot([u['steps'] for u in updates], [u['seconds']/60 for u in updates], label=run['name'])
        axes[0,0].legend(fontsize=8)
        axes[1,1].set(xlabel='Policy decisions', ylabel='Training wall time (minutes)')
        fig.suptitle('Training curves | stochastic training outcomes are not evaluation results')
        save(fig, 'learning_curves.png')

        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        for run in runs:
            episodes = run['episodes']
            # V1 did not log this accumulator: reconstruct completed-match time.
            simulated = np.cumsum([e['time'] for e in episodes]) / 3600
            for ax, field, title in zip(axes.flat[:3], ('own_tasks', 'crew_win', 'return'),
                                        ('Own tasks / match', 'Crew wins / match', 'Episode return')):
                window = min(30, len(episodes))
                values = np.convolve([float(e[field]) for e in episodes], np.ones(window)/window, mode='valid')
                ax.plot(simulated[window-1:], values, label=run['name'])
                ax.set(xlabel='Completed-match simulated hours', ylabel=f'{title} (rolling mean)')
            axes[1,1].plot(simulated, [e['training_step'] for e in episodes], label=run['name'])
        axes[0,0].legend(fontsize=8)
        axes[1,1].set(xlabel='Completed-match simulated hours', ylabel='Policy decisions')
        fig.suptitle('Interaction budget | different option durations make decision counts unequal budgets')
        save(fig, 'interaction_budget.png')

        diagnostic_fields = ('policy_loss', 'value_loss', 'entropy', 'approximate_kl',
                             'clip_fraction', 'explained_variance', 'gradient_norm', 'mean_option_seconds')
        fig, axes = plt.subplots(4, 2, figsize=(12, 11))
        for run in runs:
            for ax, field in zip(axes.flat, diagnostic_fields):
                rows = [u for u in run['updates'] if u.get(field) is not None]
                if rows:
                    ax.plot([u['steps'] for u in rows], [u[field] for u in rows], label=run['name'])
                ax.set(xlabel='Policy decisions', ylabel=field.replace('_', ' ').capitalize())
        axes[0,0].legend(fontsize=8)
        fig.suptitle('PPO optimization diagnostics | V1 did not record these fields; loss is not accuracy')
        save(fig, 'ppo_diagnostics.png')

    evaluations = []
    for path in sorted(output.glob('*/metrics.json')):
        metrics = json.loads(path.read_text())
        evaluations.append((path, metrics))
    validation = []
    for path, evaluation in evaluations:
        if evaluation['status'] != 'validation':
            continue
        for method, families in evaluation['metrics'].items():
            r = families['ID']
            validation.append(dict(method=method, label=label(method), source=str(path.relative_to(output)), **r))
    if validation:
        # Keep largest available validation sample for a method, not best score.
        best_sample = {}
        for r in validation:
            if r['method'] not in best_sample or r['matches'] > best_sample[r['method']]['matches']:
                best_sample[r['method']] = r
        selected = list(best_sample.values())
        fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(selected)*.5)))
        for i, r in enumerate(selected):
            lo, hi = r['win_rate_wilson95']
            p = r['crew_win_rate']
            axes[0].errorbar(p, i, xerr=[[max(0,p-lo)], [max(0,hi-p)]], fmt='o', capsize=3)
            axes[1].barh(i, r['mean_own_tasks'])
        for ax in axes:
            ax.set(yticks=range(len(selected)), yticklabels=[f"{r['label']} (n={r['matches']})" for r in selected])
        axes[0].set(xlabel='Crew win rate / 95% Wilson interval', xlim=(0,1))
        axes[1].set(xlabel='Own completed tasks per match', xlim=(0,max(2,max(r['mean_own_tasks'] for r in selected)*1.1)))
        fig.suptitle('Reserved validation | reused for selection, not an unbiased final test')
        save(fig, 'validation_comparison.png')

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        groups = {}
        # Curves compare the identical 20 diagnostic seeds at every checkpoint.
        # The separate comparison figure uses the largest available sample.
        for r in validation:
            if r['matches'] != 20:
                continue
            path = Path(r['method'])
            if not path.stem.startswith('checkpoint-'):
                continue
            groups.setdefault(path.parent.name, []).append((int(path.stem.split('-')[-1]), r))
        for name, rows in groups.items():
            rows.sort()
            for ax, field in zip(axes, ('crew_win_rate', 'mean_own_tasks')):
                ax.plot([x for x, _ in rows], [r[field] for _, r in rows], marker='o', label=name)
                ax.set(xlabel='Saved checkpoint decisions', ylabel=field.replace('_', ' '))
        for ax in axes:
            if groups: ax.legend(fontsize=8)
        fig.suptitle('Checkpoint validation | same 20 reserved diagnostic seeds; no final-test tuning')
        save(fig, 'validation_curve.png')

    # Prefer final-test files when available; otherwise explicitly label validation.
    final = [(p,e) for p,e in evaluations if e['status']=='evaluation']
    chosen = final or [(p,e) for p,e in evaluations if e['status']=='validation']
    split = 'FINAL TEST' if final else 'VALIDATION ONLY — final tests unopened'
    rows_by_key = {}
    for path, evaluation in chosen:
        for row in read_rows(path.parent/'matches.jsonl'):
            rows_by_key[(row['method'], row['family'], row['seed'])] = row
    rows = list(rows_by_key.values())
    methods = sorted({r['method'] for r in rows})
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(methods)*.5)))
        for i, method in enumerate(methods):
            subset = [r for r in rows if r['method']==method]
            cast = sum(r['votes_cast'] for r in subset)
            correct = sum(r['votes_correct'] for r in subset)
            skip = sum(r['skips'] for r in subset)
            if cast:
                axes[0].barh(i, correct/cast)
                axes[0].text(min(.92,correct/cast+.02), i, f'{correct}/{cast}', va='center', fontsize=8)
            else:
                axes[0].text(.02,i,'N/A: no votes',va='center')
            axes[1].barh(i, correct, label='Correct' if i==0 else None, color='#168571')
            axes[1].barh(i, cast-correct, left=correct, label='Incorrect' if i==0 else None, color='#d57c60')
            axes[1].barh(i, skip, left=cast, label='Skipped' if i==0 else None, color='#99a3b1')
        for ax in axes:
            ax.set(yticks=range(len(methods)),yticklabels=[label(m) for m in methods])
        axes[0].set(xlabel='Correct impostor votes / cast non-skip votes', xlim=(0,1.15))
        axes[1].set(xlabel='Recorded vote events');axes[1].legend(fontsize=8)
        fig.suptitle(f'Voting accuracy and coverage | {split} | pooled families; votes are not independent matches')
        save(fig, 'voting_accuracy.png')

        counts = {m: Counter() for m in methods}
        for row in rows: counts[row['method']].update(row['action_counts'])
        kinds = sorted({k for c in counts.values() for k in c})
        matrix = np.array([[counts[m][k]/max(1,sum(counts[m].values())) for k in kinds] for m in methods])
        fig, ax = plt.subplots(figsize=(14,max(4,len(methods)*.45)))
        im = ax.imshow(matrix,aspect='auto',cmap='Blues',vmin=0,vmax=1)
        ax.set(yticks=range(len(methods)),yticklabels=[label(m) for m in methods],
               xticks=range(len(kinds)),xticklabels=kinds)
        plt.setp(ax.get_xticklabels(),rotation=55,ha='right')
        fig.colorbar(im,ax=ax,label='Fraction of controller decisions (not fraction of elapsed time)')
        fig.suptitle(f'Action distribution | {split} | scripts use original intent names')
        save(fig,'action_distribution.png')

        families = sorted({r['family'] for r in rows})
        fig, axes = plt.subplots(len(families),1,figsize=(14,max(4,len(families)*3)),squeeze=False)
        for ax,family in zip(axes.flat,families):
            seeds=sorted({r['seed'] for r in rows if r['family']==family})
            matrix=np.full((len(methods),len(seeds)),np.nan)
            for i,m in enumerate(methods):
                for j,s in enumerate(seeds):
                    if (m,family,s) in rows_by_key:matrix[i,j]=int(rows_by_key[m,family,s]['crew_win'])
            cmap=ListedColormap(['#ce7763','#168571']);cmap.set_bad('#ededed')
            ax.imshow(matrix,aspect='auto',interpolation='nearest',cmap=cmap,vmin=0,vmax=1)
            positions=np.unique(np.linspace(0,len(seeds)-1,min(8,len(seeds)),dtype=int))
            ax.set(yticks=range(len(methods)),yticklabels=[label(m) for m in methods],xticks=positions,
                   xticklabels=[seeds[i] for i in positions],xlabel='Evaluation match seed',title=family)
        fig.suptitle(f'Paired match outcomes | {split} | green win / red loss or timeout / grey unmeasured')
        save(fig,'seed_outcomes.png')

        fig, axes=plt.subplots(1,2,figsize=(13,max(4,len(methods)*.45)))
        for ax,field,title in zip(axes,('own_tasks','survival'),('Own completed tasks','Survival in simulated seconds')):
            values=[[r[field] for r in rows if r['method']==m] for m in methods]
            ax.boxplot(values,vert=False,tick_labels=[label(m) for m in methods],showmeans=True)
            ax.set(xlabel=title)
        fig.suptitle(f'Contribution and survival distributions | {split}')
        save(fig,'behavior.png')

    final_path=output/'final-results.json'
    if final_path.exists():
        result=json.loads(final_path.read_text())
        metrics=result['metrics'];final_methods=list(metrics)
        fig,axes=plt.subplots(1,2,figsize=(14,max(4,len(final_methods)*.5)))
        for ax,family in zip(axes,('ID','patient')):
            for i,m in enumerate(final_methods):
                r=metrics[m][family];p=r['crew_win_rate'];lo,hi=r['win_rate_wilson95']
                ax.errorbar(p,i,xerr=[[max(0,p-lo)],[max(0,hi-p)]],fmt='o',capsize=3)
            ax.set(yticks=range(len(final_methods)),yticklabels=[label(m) for m in final_methods],
                   xlabel='Crew win rate / 95% match interval',xlim=(0,1),title=f'{family}: 500 matches per method')
        fig.suptitle('Locked final evaluation | no checkpoint selection from these outcomes')
        save(fig,'evaluation.png')

        compared=[m for m in final_methods if m not in {'scripted','random','idle'}]
        fig,axes=plt.subplots(1,2,figsize=(14,max(4,len(compared)*.5)))
        for ax,family in zip(axes,('ID','patient')):
            for i,m in enumerate(compared):
                r=result['paired'][m][family]['scripted'];p=r['win_rate_difference'];lo,hi=r['bootstrap95']
                ax.errorbar(p,i,xerr=[[max(0,p-lo)],[max(0,hi-p)]],fmt='o',capsize=3)
            ax.axvline(0,color='#777',linestyle='--')
            ax.set(yticks=range(len(compared)),yticklabels=[label(m) for m in compared],
                   xlabel='Crew win-rate difference vs scripted / paired bootstrap 95%',title=family)
        fig.suptitle('Paired complete-match differences | intervals resample matches, not votes or training seeds')
        save(fig,'paired_differences.png')

        fig,axes=plt.subplots(1,2,figsize=(12,4.8))
        for family,marker in (('ID','o'),('patient','s')):
            for ablation,color in ((False,'#168571'),(True,'#bb7638')):
                values=[]
                for m in final_methods:
                    config_path=ROOT/Path(m).parent/'run-config.json'
                    if m in {'random','scripted','idle'} or not config_path.exists():continue
                    config=json.loads(config_path.read_text())
                    if config.get('initial_control') or config.get('execution_protocol')!='strategic-options-v2':continue
                    if bool(config['ablation'])!=ablation:continue
                    values.append((config['seed'],metrics[m][family]))
                values.sort()
                for ax,field in zip(axes,('crew_win_rate','mean_own_tasks')):
                    ax.plot([s for s,r in values],[r[field] for s,r in values],marker=marker,
                            color=color,linestyle='--' if family=='patient' else '-',
                            label=f"{'Current only' if ablation else 'Full memory'} / {family}")
                    ax.set(xlabel='Independent training initialization seed',ylabel=field.replace('_',' '),xticks=[17,29,43])
        axes[0].legend(fontsize=8)
        fig.suptitle('Training-seed replication | one current-only seed is an exploratory ablation')
        save(fig,'training_seed_comparison.png')

    summary = dict(status='final_evaluation_available' if final else 'research_in_progress',
                   final_test_opened=bool(final), figures=saved,
                   runs=[dict(name=r['name'],directory=r['directory'],seed=r['config']['seed'],
                              ablation=r['config']['ablation'],complete=r['complete'],
                              decisions=r['updates'][-1]['steps'],episodes=len(r['episodes']),
                              own_tasks=sum(e['own_tasks'] for e in r['episodes']),
                              training_seconds=r['updates'][-1]['seconds']) for r in runs],
                   validation=validation, evaluation_matches=len(rows),
                   caution='Repeated checkpoints from one initialization are not independent training seeds. No best-model claim from training wins.')
    (output/'progress.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Phase 5 expanded learning study','',
           '**Status:** '+summary['status'].replace('_',' ')+'.',
           'Final test partitions have '+('been opened after selection.' if final else '**not** been opened.'),'',
           '| Run | Training seed | Decisions | Complete matches | Own tasks | Training minutes | Finished |',
           '|---|---:|---:|---:|---:|---:|---|']
    for r in summary['runs']:
        lines.append(f"| {r['name']} | {r['seed']} | {r['decisions']:,} | {r['episodes']} | {r['own_tasks']} | {r['training_seconds']/60:.1f} | {r['complete']} |")
    lines+=['','Training outcomes are exploratory. Three scripted teammates may carry an idle focal player.',
            'Compare task contribution, voting coverage, idle and scripted controls, not team wins alone.',
            'V1 and V2 use different action durations. Interaction-budget plots show this difference.',
            'PPO loss is not accuracy. Voting accuracy is conditional on cast non-skip votes.',
            'The frozen belief model is reused; all strategic networks began from random weights.',
            'Initial controls reproduce the same RNG and environment-before-policy construction order without PPO updates.',
            'The separately trained current-only ablation has one seed. Its result cannot establish replicated memory benefit.','']
    if final_path.exists():
        result=json.loads(final_path.read_text());selected=result['selected_checkpoint']
        lines+=['## Locked final results','',f'Selected on validation: **{label(selected)}**.',
                'Every method uses the same 500 ID and 500 held-out patient match seeds.',
                'All final comparisons report this frozen selection; they do not choose a new checkpoint.','',
                '| Family | Method | Crew wins | 95% match interval | Own tasks | Voting accuracy | Cast votes | Skips | Illegal / navigation failures |',
                '|---|---|---:|---|---:|---:|---:|---:|---:|']
        for m,families in result['metrics'].items():
            for family,r in families.items():
                lo,hi=r['win_rate_wilson95'];accuracy='N/A' if r['voting_accuracy'] is None else f"{r['voting_accuracy']:.1%}"
                lines.append(f"| {family} | {label(m)} | {r['crew_win_rate']:.1%} | {lo:.1%}–{hi:.1%} | {r['mean_own_tasks']:.2f} | {accuracy} | {r['votes_cast']} | {r['skips']} | {r['illegal_actions']} / {r['navigation_failures']} |")
        lines+=['','## Learning assessment','']
        for family in ('ID','patient'):
            r=result['metrics'][selected][family]
            comparison=result['paired'][selected][family]
            lines.append(f"**{family}:** selected policy wins {r['crew_win_rate']:.1%} and completes {r['mean_own_tasks']:.2f} own tasks per match.")
            for baseline in ('scripted','random','idle'):
                pair=comparison[baseline];lo,hi=pair['bootstrap95']
                lines.append(f"Against {baseline}: {pair['win_rate_difference']:+.1%} crew win-rate difference; paired 95% interval [{lo:+.1%}, {hi:+.1%}].")
        deficit=any(result['paired'][selected][f]['scripted']['bootstrap95'][1]<0 for f in ('ID','patient'))
        useful=all(result['paired'][selected][f][b]['bootstrap95'][0]>0 for f in ('ID','patient') for b in ('idle','random'))
        if deficit:
            verdict='The selected policy remains demonstrably below scripted play on at least one test family. More research is needed; these opened tests must not guide selection of a replacement.'
        elif useful:
            verdict='The selected policy contributes useful learned behavior relative to idle and random controls. Superiority to scripted play depends on the paired intervals above; a globally best policy is not established.'
        else:
            verdict='Useful improvement across both families is not established. Inspect failure modes and register a fresh evaluation set before further tuning.'
        lines+=['',verdict,
                'Final observations can motivate a new research question, but they cannot be reused to pick a better checkpoint in this experiment.',
                'Three full-memory seeds measure initialization sensitivity. One current-only run is an exploratory comparison, not a definitive ablation.',
                'This fixed five-player simulator, fixed scripted opponents and research map do not establish real-game or human-opponent performance.',
                'See final-results.json for vote uncertainty from resampling whole matches, exact counts and paired comparisons.','']
    else:
        lines+=['Final-test assessment is pending. Additional training decisions must use validation only.','']
    lines+=['## Figures','']
    for name in saved:lines += [f'![{name.replace("_"," ")}](figures/{name})','']
    lines += ['## Phase 6 gate','','Only after one learned crewmate works, explore adapting opponents and self-play.',
              'The teammate handoff contains the proposed sequence. This study does not implement multi-agent learning.','']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    return summary


if __name__ == '__main__':
    summary=build()
    print(json.dumps({k:summary[k] for k in ('status','figures','runs','evaluation_matches')},indent=2))
