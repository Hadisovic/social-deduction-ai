"""Offline/exit reporting for the manual continuation; no model updates or role inputs."""
from collections import Counter
import json
from pathlib import Path
from .progress import atomic_json


def read(path,default=None):
    try:return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:return default


def summarize(matches):
    from tools.report_patrick_jane_pilot import aggregate
    result=aggregate(matches);d=Counter()
    for m in matches:d.update(m.get('learning_diagnostics',{}))
    result['diagnostics']=dict(d)
    cuts=[m for m in matches if m.get('truncated') or m.get('reason')=='TRUNCATED']
    result['totals']['truncations']=len(cuts);result['totals']['draws']=result['totals'].get('draws',0)-len(cuts)
    result['totals']['legitimate_games']=sum(m['winner'] in ('crewmate','impostor') for m in matches)
    result['truncation_causes']=dict(Counter(m.get('truncation',{}).get('cause','unspecified') for m in cuts))
    result['win_rate_all_collected']=result['win_rate']
    result['win_rate']=result['totals']['wins']/result['totals']['legitimate_games'] if result['totals']['legitimate_games'] else None
    a=result['actions'];result['vote_participation']=a.get('vote',0)/max(1,a.get('vote',0)+a.get('skip',0))
    return result


def component_totals(values):
    totals=Counter()
    for components in values:totals.update(components)
    return dict(totals)


def generate(folder,state,*,active_seconds=0.,error=None,plots=True):
    folder=Path(folder);history=read(folder/'history.json',[]);reg=read(folder/'registration.json',{})
    latest=read(folder/'latest.json',{});c=latest.get('counters',{})
    training=summarize([m for row in history for m in row['rollout']['matches']])
    components=Counter()
    for row in history:
        for values in row['rollout']['reward_components'].values():components.update(values)
    evaluations=[]
    for label in ('entry','final'):
        ev=read(folder/f'evaluations/{label}.json')
        if ev:evaluations.append(dict(label=label,metrics=summarize(ev['matches']),elapsed=ev['elapsed']))
    if state=='complete' and (c.get('run_decisions',0)<reg.get('additional_target',1)
            or len(evaluations)!=2 or any(e['metrics']['totals']['games']!=len(reg.get('validation_seeds',[])) for e in evaluations)):
        raise ValueError('Cannot report success before target and complete entry/final validation')
    paired=None
    if len(evaluations)==2:
        from tools.report_patrick_jane_pilot import paired
        old=read(folder/'evaluations/entry.json')['matches'];new=read(folder/'evaluations/final.json')['matches']
        valid={m['seed'] for m in old if not m.get('truncated')} & {m['seed'] for m in new if not m.get('truncated')}
        paired=paired([m for m in old if m['seed'] in valid],[m for m in new if m['seed'] in valid]) if valid else dict(scenarios=0,reason='No jointly non-truncated match outcomes')
    collection=sum(r['rollout']['elapsed'] for r in history);ppo=sum(r['ppo']['elapsed'] for r in history)
    result=dict(schema='patrick-jane-learning-results-v1',state=state,success=state=='complete',
        development_only=True,acceptance='pending',configuration=reg,additional_decisions=c.get('run_decisions',0),
        lifetime_decisions=c.get('individual_decisions',reg.get('base_decisions',255083)),
        completed_training_matches=training['totals']['legitimate_games'],
        collected_training_segments=c.get('run_matches',0),technical_truncations=training['totals']['truncations'],ppo_updates=c.get('run_updates',0),
        active_seconds=active_seconds,collection_seconds=collection,ppo_seconds=ppo,
        training_decisions_per_second=c.get('run_decisions',0)/max(1,collection+ppo),
        training=training,reward_components=dict(components),validation=evaluations,paired_win_comparison=paired,
        checkpoint_history=[dict(checkpoint=r['checkpoint'],sha256=r['sha256'],counters=r['counters'],
            ppo=r['ppo'],matches=len(r['rollout']['matches']),batch_metrics=summarize(r['rollout']['matches']),
            batch_reward_components=component_totals(r['rollout']['reward_components'].values())) for r in history],
        latest=latest,error=error,limitations=['Warm-start changed regime, not directly comparable to the original pilot',
            'Eight paired development scenarios: exploratory, one initialization and scripted opponents',
            'On-policy training curves change scenarios and policy together; reward alone is not improvement',
            'Automatic ghost work and reporter/script patterns remain confounds',
            'Technical cuts censor future ghost/team credits; win rates and paired comparisons condition on uncensored outcomes',
            'No reserved final-test partition used'])
    atomic_json(folder/'training_results.json',result)
    t=training['totals'];d=training['diagnostics'];a=training['actions']
    lines=['# Patrick Jane — lights-only training report','',f'**State: {state.upper()}**. Phase 6.1 acceptance pending. Development only.','',
        f"Additional trained decisions: **{result['additional_decisions']:,}** / {reg.get('additional_target',0):,}. Lifetime decisions: {result['lifetime_decisions']:,}.",
        f"Legitimate complete training matches: {result['completed_training_matches']}; technical cuts: {result['technical_truncations']}; collected segments: {result['collected_training_segments']}; PPO updates: {result['ppo_updates']}; active wall time: {active_seconds/3600:.3f} hours.",
        f"Collection + PPO: {collection:.3f} + {ppo:.3f} seconds; throughput: {result['training_decisions_per_second']:.2f} individual decisions/s.",'',
        'Ten independent-memory crewmates share the unchanged network. Two scripted impostors can kill; only lights sabotage is enabled. The new penalty is -0.005 per living crew per simulated second in darkness, including meetings; all other rewards are unchanged.','',
        '## Training outcomes (changing policies and scenarios)','',
        f"Legitimate wins/losses: {t.get('wins',0)} / {t.get('losses',0)}. Technical TRUNCATED segments: {t.get('truncations',0)} (not wins, losses or draws). Living / ghost task completions: {training['living_tasks']} / {t.get('ghost_tasks',0)}.",
        f"Kills (crew deaths): {t.get('kills',0)}; innocent / impostor ejections: {t.get('innocent_ejections',0)} / {t.get('correct_ejections',0)}.",
        f"Accepted crew / impostor reports: {t.get('crew_reports',0)} / {t.get('impostor_reports',0)}; legal crew reporting decision points: {training['exposure'].get('legal_report',0)}.",
        f"Correct / incorrect neural votes: {a.get('correct_vote',0)} / {a.get('incorrect_vote',0)}; skips: {a.get('skip',0)}; non-skip accuracy: {training['vote_accuracy']:.1%}; participation: {training['vote_participation']:.1%}.",
        f"Correct / wrong accusations: {a.get('correct_accusation',0)} / {a.get('incorrect_accusation',0)}. Action probabilities are not calibrated impostor beliefs.",'',
        '## Task and navigation diagnostics','',
        '| Measurement | Count / simulated actor-seconds |','|---|---:|']
    lines += [f'| {k} | {v:.4f} |' for k,v in sorted(d.items())]
    lines+=['','Movement time counts physics ticks with actual movement; task work counts active interaction ticks. These are actor-seconds, not wall time. Meetings pause tasks/ghost work; darkness penalties continue only for living crew while lights remain unresolved. Task stages may start repeatedly and are not the same as complete owned tasks. Navigation interruptions are measured from actual submitted navigation intents.','',
        '## Reward components (raw totals)','', '| Component | Total |','|---|---:|']
    lines += [f'| {k} | {v:.5f} |' for k,v in sorted(components.items())]
    lines+=['','Reward accumulation is not proof of learning. In particular, automatic ghost rewards and repeated script patterns can obscure weak living-task execution or deduction. Use matched entry/final behavior and opportunity denominators.','',
        '## Paired lightweight development checks','','| State | Wins / legitimate games | TRUNCATED | Living tasks | Ghost tasks | Vote accuracy |','|---|---:|---:|---:|---:|---:|']
    for e in evaluations:
        m=e['metrics'];lines.append(f"| {e['label']} | {m['totals']['wins']}/{m['totals']['legitimate_games']} | {m['totals']['truncations']} | {m['living_tasks']} | {m['totals']['ghost_tasks']} | {m['vote_accuracy']:.1%} |")
    if paired and paired.get('scenarios'):lines+=['',f"Paired win-rate difference: {paired['win_rate_difference']:.1%}; scenario-bootstrap 95% interval {paired['scenario_bootstrap95']}. Jointly non-truncated pairs: {paired['scenarios']}; censored pairs excluded. Only eight distinct paired scenarios; do not claim established generalization."]
    else:lines+=['','No uncensored paired comparison is available yet. No learned improvement claim is made.']
    lines+=['','## Technical truncation diagnostics','',f"Training causes: `{training['truncation_causes']}`.",
        'Normal lights matches have no 900-second game-ending rule. A technical cut is not an MDP victory or loss. Living policies bootstrap from their final legitimate roaming state with their own recurrent memory; no new action or random draw is made. True terminal outcomes use zero bootstrap. Dead/ejected actors remain terminated, keep actually observed delayed credits, and receive no fabricated future ghost work or team outcome. Unobserved post-cut ghost credits are censored; this is a limitation if truncations become common. Operator interruption/run wall limits still withhold incomplete batches instead of pretending to complete matches.','',
        f"Registered technical guards: {reg.get('experiment',{}).get('technical_max_seconds','unavailable')} simulated seconds or {reg.get('experiment',{}).get('technical_stall_seconds','unavailable')} seconds stationary without progress. Engineering checks use a deliberately short bound and are not learning evidence. Cuts wait for a roaming boundary after bounded meeting phases, and a genuine victory on the boundary takes precedence. Per-match cause and world diagnostics are retained in raw batches and replay result metrics. Win/loss rates exclude technical cuts; on-policy reward totals do not prove competence."]
    lines+=['','## Checkpoints and recovery','',f"Latest complete checkpoint: `{latest.get('checkpoint','none')}`.",f"SHA256: `{latest.get('sha256','unavailable')}`.",'',
        'All completed checkpoints, source archive, immutable registration, attempt journals, selected full replays and stdout/stderr are retained in this run folder. Rerun the same launcher command to resume an incomplete run; interrupted partial batches are excluded from PPO and retried from the previous complete boundary. A completed run is not extended automatically.','',
        '## Curves','','![On-policy curves; descriptive only](learning_curves.png)','',
        'Exact counters, component totals, diagnostics, checkpoint history and limitations are in `training_results.json`. Old pilot/champion and partner project remain untouched.']
    if error:lines+=['',f'Exit reason: `{error}`. This is not a successful completion.']
    (folder/'TRAINING_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if plots and history:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        metrics=[summarize(r['rollout']['matches']) for r in history];xs=[r['counters']['run_decisions'] for r in history]
        fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
        series=[('Batch crew win rate',[m['win_rate'] for m in metrics]),
            ('Living tasks per match',[m['living_tasks']/max(1,m['totals']['games']) for m in metrics]),
            ('Ghost tasks per match',[m['totals']['ghost_tasks']/max(1,m['totals']['games']) for m in metrics]),
            ('Navigation task interruptions per match',[m['diagnostics'].get('task_interruptions_navigation',0)/max(1,m['totals']['games']) for m in metrics]),
            ('Mean raw reward per crew per match',[sum(sum(v.values()) for v in r['rollout']['reward_components'].values())/max(1,10*len(r['rollout']['matches'])) for r in history]),
            ('Dark seconds per match',[m['diagnostics'].get('lights_active_seconds',0)/max(1,m['totals']['games']) for m in metrics])]
        for ax,(title,ys) in zip(axes.flat,series):
            ax.plot(xs,ys,marker='o');ax.set(title=title,xlabel='Additional trained decisions');ax.grid(alpha=.2)
            if all(v is None for v in ys):ax.text(.5,.5,'Undefined: no legitimate outcomes',ha='center',va='center',transform=ax.transAxes)
        fig.suptitle('On-policy batches: changing scenarios; reward alone is not learned competence')
        fig.savefig(folder/'learning_curves.png',dpi=140);plt.close(fig)
    return result
