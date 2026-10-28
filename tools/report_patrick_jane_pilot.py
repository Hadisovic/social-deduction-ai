"""Offline aggregate report; never loads actors, checkpoints or final partitions."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import numpy as np


def wilson(wins,n):
    if not n:return [0.,1.]
    z=1.959963984540054;p=wins/n;d=1+z*z/n
    middle=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [middle-half,middle+half]


def aggregate(matches):
    total=Counter();actions=Counter();exposure=Counter();reasons=Counter();sabotages=Counter();repairs=Counter()
    for m in matches:
        total.update(dict(games=1,wins=m['winner']=='crewmate',losses=m['winner']=='impostor',draws=m['winner'] is None,
            decisions=m['individual_decisions'],simulation_seconds=m['simulation_seconds'],
            tasks=m['metrics']['task_completions'],ghost_tasks=m.get('ghost_tasks',0),
            task_total=4*m['crew'],meetings=m['metrics']['meetings'],reports=m['metrics']['reports'],
            crew_reports=m.get('crew_reports',0),impostor_reports=m.get('impostor_reports',0),
            repairs=m['metrics']['repairs'],kills=m['metrics']['eliminations'],
            correct_ejections=m['correct_ejections'],innocent_ejections=m['innocent_ejections'],
            illegal_actions=m['metrics']['illegal_actions'],navigation_failures=m['metrics']['navigation_failures'],
            critical_cancellations=m.get('critical_cancellations',0)))
        actions.update(m.get('neural_actions',{}));exposure.update(m.get('exposure',{}));reasons[m['reason']]+=1
        sabotages.update(m.get('sabotages_by_kind',{}));repairs.update(m.get('repairs_by_kind',{}))
    return dict(totals=dict(total),win_rate=total['wins']/max(1,len(matches)),win_rate_wilson95=wilson(total['wins'],len(matches)),
        task_fraction=total['tasks']/max(1,total['task_total']),living_tasks=total['tasks']-total['ghost_tasks'],
        vote_accuracy=actions['correct_vote']/max(1,actions['correct_vote']+actions['incorrect_vote']),
        actions=dict(actions),exposure=dict(exposure),reasons=dict(reasons),sabotages=dict(sabotages),repairs=dict(repairs))


def paired(reference,current):
    old={m['seed']:m for m in reference};new={m['seed']:m for m in current};keys=sorted(old)
    if set(old)!=set(new):raise ValueError('Paired development scenarios differ')
    delta=np.array([int(new[k]['winner']=='crewmate')-int(old[k]['winner']=='crewmate') for k in keys],np.float64)
    rng=np.random.default_rng(293_017)
    samples=delta[rng.integers(0,len(keys),(10000,len(keys)))].mean(1)
    return dict(scenarios=len(keys),win_rate_difference=float(delta.mean()),
        scenario_bootstrap95=list(map(float,np.quantile(samples,[.025,.975]))),bootstrap_seed=293_017,replicates=10000)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-commit',required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    registration=json.loads((args.pilot/'registration.json').read_text())
    history=json.loads((args.pilot/'history.json').read_text()) if (args.pilot/'history.json').is_file() else []
    completion=json.loads((args.pilot/'completion.json').read_text()) if (args.pilot/'completion.json').is_file() else None
    failure=json.loads((args.pilot/'failure.json').read_text()) if (args.pilot/'failure.json').is_file() else None
    evaluations=[json.loads(path.read_text()) for path in sorted((args.pilot/'evaluations').glob('*.json'))]
    evaluations.sort(key=lambda row:row['update']);reference=next((row for row in evaluations if row['label']=='zero'),None)
    stages=[dict(label=row['label'],update=row['update'],metrics=aggregate(row['matches']),
        paired_vs_zero=paired(reference['matches'],row['matches']) if reference else None) for row in evaluations]
    training=aggregate([m for row in history for m in row['rollout']['matches']])
    collection=sum(row['rollout']['elapsed'] for row in history);ppo=sum(row['ppo']['elapsed'] for row in history)
    rewards=Counter()
    for row in history:
        for components in row['rollout']['reward_components'].values():rewards.update(components)
    benchmark=json.loads(args.benchmark.read_text())
    elapsed=(completion or failure or {}).get('elapsed',0)
    total_evaluated=sum(len(row['matches']) for row in evaluations)
    evaluation_and_overhead=max(0,elapsed-collection-ppo)
    training_rate=training['totals'].get('decisions',0)/max(1,collection+ppo)
    # Whole-run experience, not a promise: fixed validation overhead is included separately.
    million_training=1_000_000/max(.01,training_rate)
    million_evaluation=1100*evaluation_and_overhead/max(1,total_evaluated)
    result=dict(status='complete' if completion else 'stopped' if failure else 'incomplete',source_commit=args.source_commit,
        acceptance='pending',development_only=True,workers=registration['workers'],target=registration['target_decisions'],
        training=training,reward_components=dict(rewards),validation=stages,
        runtime=dict(wall_seconds=elapsed,collection_seconds=collection,ppo_seconds=ppo,
            validation_and_other_overhead_seconds=evaluation_and_overhead,training_decisions_per_second=training_rate,
            pilot_decisions_per_total_second=training['totals'].get('decisions',0)/max(1,elapsed)),
        larger_estimate=dict(one_million_training_seconds=million_training,
            estimated_validation_and_overhead_seconds=million_evaluation,total_seconds=million_training+million_evaluation,
            basis='Observed pilot training throughput plus 1100 fixed development validation executions; extrapolation, not a started run'),
        benchmark=[dict(workers=int(n),training_rate=r['decisions_per_training_second'],
            collection_rate=r['decisions_per_collection_second'],collection_seconds=r['collection_seconds'],ppo_seconds=r['ppo_seconds']) for n,r in benchmark['results'].items()],
        failure=failure['error'] if failure else None)
    (args.output/'aggregate.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    xs=[next((h['counters']['individual_decisions'] for h in history if h['counters']['update']==s['update']),0)/1000 for s in stages]
    axes[0].plot(xs,[s['metrics']['win_rate']*100 for s in stages],marker='o');axes[0].set(xlabel='Training decisions (thousands)',ylabel='Crew win rate (%)',ylim=(0,100),title='Fixed held-out development scenarios')
    axes[1].plot(xs,[s['metrics']['task_fraction']*100 for s in stages],marker='o',label='All tasks')
    axes[1].plot(xs,[s['metrics']['living_tasks']/max(1,s['metrics']['totals']['task_total'])*100 for s in stages],marker='s',label='Living crew only')
    axes[1].set(xlabel='Training decisions (thousands)',ylabel='Assigned tasks completed (%)',ylim=(0,100),title='Task contribution includes scripted ghosts');axes[1].legend()
    axes[2].bar([str(row['workers']) for row in result['benchmark']],[row['training_rate'] for row in result['benchmark']])
    axes[2].set(xlabel='Worker count',ylabel='Individual decisions / second',title='Matched collection + PPO throughput')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(args.output/'pilot-summary.png',dpi=180);plt.close(fig)
    lines=['# Patrick Jane — bounded development pilot','',f"Status: **{result['status']}**. Acceptance remains **pending**. Source commit `{args.source_commit}`.",'',
        'The approved architecture and rewards are unchanged. This is one initialization against scripted opponents, with repeated paired development validation. Reserved final partitions remain unopened.','',
        '| State | Crew wins | Wilson 95% interval | Tasks / assigned | Living / ghost tasks | Correct / wrong neural votes | Neural skips | Crew / impostor reports |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in stages:
        m=s['metrics'];t=m['totals'];a=m['actions'];lo,hi=m['win_rate_wilson95']
        lines.append(f"| {s['label']} · update {s['update']} | {t['wins']}/{t['games']} | {lo:.1%}–{hi:.1%} | {t['tasks']}/{t['task_total']} | {m['living_tasks']} / {t['ghost_tasks']} | {a.get('correct_vote',0)} / {a.get('incorrect_vote',0)} | {a.get('skip',0)} | {t['crew_reports']} / {t['impostor_reports']} |")
    lines+=['','![Measured pilot summary](pilot-summary.png)','',
        f"Training collected {training['totals'].get('decisions',0):,} individual decisions across {training['totals'].get('games',0)} complete matches using {registration['workers']} workers. Collection: {collection/60:.2f} minutes; PPO: {ppo/60:.2f} minutes; total measured run: {elapsed/3600:.3f} hours. Effective collection-plus-update rate: {training_rate:.2f} decisions/s.",'',
        'Validation uncertainty is paired across the same scenarios; repeated states are not independent datasets. Exact values, per-stage actions, report/repair opportunity denominators and critical-failure counts are in `aggregate.json`. Opportunity counts are legal choices at learner decision points, not unique bodies or independent games. Selected report/repair intentions may exceed accepted actions because simultaneous reports end the roaming tick.','',
        'Task progress is separated into living crew and scripted ghost completions. Ghosts receive their unchanged owner reward and do not act or complete tasks during meetings. Increasing shaped return alone is not evidence of better deduction or gameplay. Neural vote accuracy is scored offline against roles; those role labels never enter neural observations.','',
        f"A one-million-decision run with the same evaluation schedule extrapolates to {(million_training+million_evaluation)/3600:.2f} hours before a contingency margin. This estimate uses pilot throughput, treats validation plus miscellaneous overhead conservatively together, and is not authorization or launch of the larger stage.",'',
        'No fixes, reward tuning, final-test evaluation, production changes or partner-impostor changes are included in this report. Review the paired results, sabotage responses and evidence limitations before proposing the next experiment.']
    if failure:lines+=['',f"Automatic stop: `{failure['error']}`. Preserved complete checkpoints remain available; no completion claim."]
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],decisions=training['totals'].get('decisions',0),hours=elapsed/3600,output=str(args.output))))


if __name__=='__main__':main()
