"""Development-pilot graphs and a concise report; never labels wins as accuracy."""
from collections import Counter
import json
import math
from pathlib import Path


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def wilson(wins, total):
    if total <= 0:
        return [0., 1.]
    p=wins/total; z=1.96; denominator=1+z*z/total
    center=(p+z*z/(2*total))/denominator
    radius=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0.,center-radius),min(1.,center+radius)]


def generate(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output); figures=output/'figures'; figures.mkdir(exist_ok=True)
    results=json.loads((output/'results.json').read_text(encoding='utf-8'))
    registration=json.loads((output/'registration.json').read_text(encoding='utf-8'))
    training={stream:read_lines(output/f'stream-{stream}'/'training.jsonl') for stream in (101,202)}
    def save(fig,name):
        fig.tight_layout();fig.savefig(figures/name,dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for stream,rows in training.items():
        xs=[r['episode'] for r in rows];label=f'Sampling stream {stream}'
        axes[0,0].plot(xs,[r['match']['credit']['reward_sum'] for r in rows],marker='o',label=label)
        axes[0,1].plot(xs,[r['match']['behavior']['kills'] for r in rows],marker='o',label=label)
        axes[1,0].plot(xs,[sum(x['match']['impostor_win'] for x in rows[:i+1]) for i in range(len(rows))],marker='o',label=label)
        axes[1,1].plot(xs,[r['match']['credit']['cooldown_ready_seconds'] for r in rows],marker='o',label=label)
    for ax,title in zip(axes.flat,['Training return (shaped objective)','Kills per training episode','Cumulative training wins','Cooldown-ready roaming seconds']):
        ax.set_title(title);ax.set_xlabel('Complete episode');ax.grid(alpha=.2);ax.legend()
    save(fig,'training.png')
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    metrics=['approximate_kl','clip_fraction','family_entropy','value_loss','actor_gradient_norm','critic_gradient_norm']
    for ax,metric in zip(axes.flat,metrics):
        for stream,rows in training.items():ax.plot([r['episode'] for r in rows],[r['ppo'][metric] for r in rows],marker='o',label=str(stream))
        ax.set_title(metric.replace('_',' '));ax.set_xlabel('Episode update');ax.grid(alpha=.2);ax.legend()
    save(fig,'ppo.png')
    candidates=json.loads((output/'candidates.json').read_text(encoding='utf-8'))
    selected=[json.loads((output/f'selection-{c["stream"]}-ep{c["episode"]}'/'summary.json').read_text(encoding='utf-8')) for c in candidates]
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar([f'{c["stream"]} / ep {c["episode"]}' for c in candidates],[s['wins'] for s in selected])
    ax.set_title('Validation wins: 8 ordinary B scenarios per candidate');ax.set_ylabel('Wins / 8');ax.set_ylim(0,8)
    save(fig,'selection.png')
    confirmed=results['confirmation']; rows=[]; paired={}
    for item in confirmed:
        wins=item['wins']; n=len(item['matches']); interval=wilson(wins,n)
        rows.append(dict(label=item['label'],wins=wins,matches=n,win_fraction=wins/n,wilson95=interval,
            kills=item['kills'],ejections=item['ejections'],
            outcomes=dict(Counter(r['reason'] for r in item['matches']))))
    for crew in ('A','B'):
        chosen=next(x for x in confirmed if x['label']==f'confirmation-selected-{crew}')
        reference=next(x for x in confirmed if x['label']==f'confirmation-reference-{crew}')
        a={r['seed']:r['impostor_win'] for r in chosen['matches']};b={r['seed']:r['impostor_win'] for r in reference['matches']}
        paired[crew]=dict(selected_only=sum(a[s] and not b[s] for s in a),reference_only=sum(b[s] and not a[s] for s in a),
            both_win=sum(a[s] and b[s] for s in a),neither_win=sum(not a[s] and not b[s] for s in a))
    fig,ax=plt.subplots(figsize=(10,5))
    for i,row in enumerate(rows):
        p=row['win_fraction'];lo,hi=row['wilson95']
        ax.bar(i,p,color='#cf4446' if 'selected' in row['label'] else '#597da7')
        ax.errorbar(i,p,yerr=[[max(0,p-lo)],[max(0,hi-p)]],color='black',capsize=4)
    ax.set_xticks(range(len(rows)),[r['label'].replace('confirmation-','') for r in rows]);ax.set_ylim(0,1)
    ax.set_title('Fresh development confirmation: win fraction, 95% Wilson intervals');ax.set_ylabel('Wins / 16')
    save(fig,'confirmation.png')
    assessment=('Insufficient: no candidate won a validation game.' if results['selection']['all_zero_selection_wins'] else
        'A candidate won validation games. This small warm-start pilot does not establish reliable or general play.')
    aggregate=dict(selection=results['selection'],confirmation=rows,paired=paired,assessment=assessment,
        study_wall_seconds=results['elapsed_wall_seconds'],training_cpu_seconds=sum(r['match']['collection_cpu_seconds']+r['ppo']['optimization_cpu_seconds'] for rs in training.values() for r in rs),
        evaluation_cpu_seconds=sum(r['collection_cpu_seconds'] for s in selected+confirmed for r in s['matches']),
        final_tests_opened=False)
    (output/'aggregate.json').write_text(json.dumps(aggregate,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# Shared impostor pilot results','',assessment,'',
        f'Elapsed study wall time: {results["elapsed_wall_seconds"]/60:.1f} minutes. Two CPU workers, one Torch thread each.',
        'The four crewmates and belief networks were frozen. Both streams started from identical historical impostor weights.',
        'Training: 8 complete episodes per stream. Validation: 4 candidates × 8 B scenarios. Fresh confirmation: selected/reference × A/B × 16 scenarios.',
        'Historical reserved final sets remained sealed. These are development results, not test accuracy.','',
        '| Method / crew | Wins | Kills | Own ejections |','|---|---:|---:|---:|']
    lines += [f'| {r["label"].replace("confirmation-","")} | {r["wins"]}/{r["matches"]} | {r["kills"]} | {r["ejections"]} |' for r in rows]
    lines += ['', '![Training returns, kills, wins, and ready time](figures/training.png)',
        'Training curves show learning experience; rewards include shaping and time costs. They are not held-out performance.',
        '', '![PPO diagnostics](figures/ppo.png)',
        'KL and clipping show policy movement. Entropy shows exploration. Value loss and gradient norms are optimizer diagnostics, not playing skill.',
        '', '![Validation selection](figures/selection.png)',
        'The fixed rule chooses wins, then kills, fewer ejections, earlier checkpoint, then lower stream. Selection is locked before confirmation.',
        '', '![Fresh confirmation](figures/confirmation.png)',
        'Error bars describe binomial uncertainty for each 16-match block. The same scenario seeds pair the selected and untouched policies.',
        '', '## Next decision',
        'Do not extend or tune using confirmation outcomes automatically. Review task-loss/ejection reasons, self-report behavior, and value diagnostics before proposing a new registered study.',
        'Consider a checked all-A/all-B/mixed-team pool after this B-only adaptation. Add sabotage as a separate versioned both-role task. Larger lobbies and self-play remain future work.',
        'Witness counts are retrospective. Claims and innocent ballots do not establish causal deception. No new escape or deception success is claimed.','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return aggregate
