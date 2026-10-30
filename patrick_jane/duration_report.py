"""Training-only summaries; no validation, checkpoint selection or actor execution."""
from collections import Counter
from pathlib import Path
from .learning_report import read,summarize,component_totals
from .progress import atomic_json


def generate(folder,state,*,active_seconds,error=None,plots=False):
    folder=Path(folder);reg=read(folder/'registration.json',{});history=read(folder/'history.json',[])
    latest=read(folder/'latest.json',{});c=latest.get('counters',{})
    matches=[m for row in history for m in row['rollout']['matches']]
    training=summarize(matches);components=Counter()
    for row in history:components.update(component_totals(row['rollout']['reward_components'].values()))
    budget=read(folder/'duration-budget.json',[{'target_seconds':43200}])[-1]['target_seconds']
    if state=='complete' and (active_seconds<budget or not history):raise ValueError('Cannot claim completion before duration and safe PPO boundary')
    result=dict(schema='patrick-jane-duration-results-v1',state=state,error=error,reward_regime='task-pursuit-v1',
        base_lifetime_decisions=1518786,lifetime_decisions=c.get('individual_decisions',1518786),
        additional_decisions=c.get('run_decisions',0),ppo_updates=c.get('run_updates',0),
        active_training_seconds=active_seconds,target_training_seconds=budget,
        progress_percent=min(100,100*active_seconds/budget),workers=reg.get('workers',6),
        training=training,reward_components=dict(components),latest=latest,
        checkpoint_history=[{k:row[k] for k in ('checkpoint','sha256','counters','ppo')} for row in history],
        validation=dict(performed=False,best_checkpoint=None,reason='No evaluation requested; on-policy waves are not comparable champion tests'),
        limitations=['New +2 task reward and pursuit action regime; historical rewards are separate.',
            'Training outcomes change policy and scenarios; no measured held-out improvement.',
            'Task commitment scaffolds execution; forced decisions are not voluntary persistence.',
            'Ghost task progress, scripted opponents and reporter patterns remain confounds.',
            'Technical truncations are censored, not legitimate losses or victories.',
            'No reserved final-test partition opened; acceptance pending.'])
    atomic_json(folder/'training_results.json',result)
    t=training['totals'];d=training['diagnostics'];a=training['actions']
    lines=['# Patrick Jane — Task Pursuit V1','',f'**{state.upper()}** — development training; acceptance pending.','',
        f"Active collection/PPO: {active_seconds/3600:.4f} / {budget/3600:g} hours. Setup, reporting, export and downtime excluded.",
        f"New trained decisions: {result['additional_decisions']:,}; lifetime: {result['lifetime_decisions']:,}; PPO rounds: {result['ppo_updates']}.",
        f"Six complete matches per wave; unchanged recurrent PPO. Living full-task +2; ghosts 0; all other rewards unchanged.",
        f"Legitimate games {t.get('legitimate_games',0)}; wins {t.get('wins',0)}; losses {t.get('losses',0)}; technical cuts {t.get('truncations',0)}.",
        f"Living tasks {training['living_tasks']}; ghost tasks {t.get('ghost_tasks',0)}; win rate {training['win_rate']}.",
        f"Crew deaths {t.get('kills',0)}; correct/innocent ejections {t.get('correct_ejections',0)}/{t.get('innocent_ejections',0)}.",
        f"Correct/wrong/skip neural votes {a.get('correct_vote',0)}/{a.get('incorrect_vote',0)}/{a.get('skip',0)}.",
        '', '## Task pursuit and commitment','', '| Metric | Value |','|---|---:|']
    lines.extend(f'| {k} | {v:.5f} |' for k,v in sorted(d.items()))
    lines+=['','Pursuit decisions select an owned current-stage console. Starts count actual engine task-start events; full completions count assigned tasks only. Stage/resume starts are not additional completed tasks.',
        '', '## Rewards','', '| Component | Total |','|---|---:|']
    lines.extend(f'| {k} | {v:.5f} |' for k,v in sorted(components.items()))
    lines+=['','## PPO and checkpoints','', '| Update | New decisions | Optimizer steps | KL | Loss |','|---|---:|---:|---:|---:|']
    for row in history:
        p=row['ppo'];lines.append(f"| {row['counters']['update']} | {row['counters']['run_decisions']} | {p['updates']} | {p['approx_kl']:.6f} | {p['loss']:.6f} |")
    lines+=['',f"Latest resumable checkpoint: `{latest.get('checkpoint','none')}`.",
        f"SHA256: `{latest.get('sha256','none')}`.",
        'Best checkpoint: **not evaluated**. No baseline/final evaluation games were run.',
        'Rerun the launcher to resume; increase -Hours to append an extension. Interrupted/uncheckpointed experience is never reported as trained.',
        '', '## Limitations','', *['- '+x for x in result['limitations']],
        '', '![Training-only curves](learning_curves.png)']
    if error:lines+=['',f'Exit error: `{error}`. This is not successful completion.']
    (folder/'TRAINING_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if plots and history:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        ms=[summarize(row['rollout']['matches']) for row in history];xs=[row['counters']['run_decisions'] for row in history]
        fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
        series=[('Crew win rate',[m['win_rate'] for m in ms]),('Living tasks/game',[m['living_tasks']/len(history[i]['rollout']['matches']) for i,m in enumerate(ms)]),
            ('Commitments completed',[m['diagnostics'].get('commitment_ended_completed',0) for m in ms]),
            ('Pursuit selections',[m['diagnostics'].get('task_pursuit_selections',0) for m in ms]),
            ('PPO KL',[row['ppo']['approx_kl'] for row in history]),('Training decisions/sec',[row['rollout']['individual_decisions']/max(.001,row['rollout']['elapsed']+row['ppo']['elapsed']) for row in history])]
        for ax,(title,ys) in zip(axes.flat,series):ax.plot(xs,ys,marker='.');ax.set(title=title,xlabel='New trained decisions');ax.grid(alpha=.2)
        fig.suptitle('On-policy training; no held-out improvement claim');fig.savefig(folder/'learning_curves.png',dpi=130);plt.close(fig)
    return result
