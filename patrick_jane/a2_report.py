"""A2 denominators and separate regime summaries; offline truth never feeds policy."""
from pathlib import Path
from collections import Counter
import json
from .learning_report import read,summarize
from .progress import atomic_json


def metrics(matches):
    result=summarize(matches);mechanics=Counter()
    from tools.report_patrick_jane_pilot import wilson
    legitimate=result['totals']['legitimate_games']
    result['win_rate_wilson95']=wilson(result['totals']['wins'],legitimate) if legitimate else None
    for m in matches:mechanics.update(m.get('a2_metrics',{}))
    ballots=[b for m in matches for b in m.get('accepted_ballots',[]) if b['voter_role']=='crewmate']
    cast=[b for b in ballots if b['target_id'] is not None]
    evidence=[b for b in cast if b['direct_vent_evidence'] or b['direct_kill_evidence']]
    without=[b for b in cast if b not in evidence]
    def vote_group(group):
        n=len(group)
        return dict(accepted_nonskip_ballots=n,correct=sum(b['correct'] for b in group),
            accuracy=sum(b['correct'] for b in group)/n if n else None,
            mean_uniform_target_base_rate=sum(b['base_rate'] for b in group)/n if n else None,
            excess_correct_over_chance=sum(float(b['correct'])-b['base_rate'] for b in group),
            mean_chance_adjusted_reward=sum(b['reward'] for b in group)/n if n else None)
    result['investigation']=dict(all_cast=vote_group(cast),direct_evidence_cast=vote_group(evidence),without_direct_evidence_cast=vote_group(without),
        explicit_skips=sum(b['target_id'] is None and b['explicit'] for b in ballots),automatic_skips=sum(b['target_id'] is None and not b['explicit'] for b in ballots),
        cast_ballots_after_communication=sum(b.get('communication_before_ballot',0)>0 for b in cast),
        notes='Accuracy denominator: accepted non-skip crew ballots. Chance baseline uses the actual accepted-ballot eligible roster, excluding self. Evidence conditions describe private witnessed vent/kill availability, not proof of reasoning.')
    result['mechanics']=dict(mechanics)
    groups={}
    for regime in sorted({m.get('opponent_regime','historical') for m in matches}):
        cohort=[m for m in matches if m.get('opponent_regime','historical')==regime];groups[regime]=summarize(cohort)
    result['opponent_groups']=groups
    return result


def generate(folder,state,*,active_seconds,error=None,plots=False):
    folder=Path(folder);reg=read(folder/'registration.json',{});history=read(folder/'history.json',[]);latest=read(folder/'latest.json',{})
    c=latest.get('counters',{});matches=[m for row in history for m in row['rollout']['matches']]
    training=metrics(matches);components=Counter()
    for row in history:
        for v in row['rollout']['reward_components'].values():components.update(v)
    target=read(folder/'duration-budget.json',[dict(target_seconds=43200)])[-1]['target_seconds']
    decision_target=reg.get('additional_decisions')
    met=active_seconds>=target or decision_target is not None and c.get('run_decisions',0)>=decision_target
    if state=='complete' and (not met or not history):raise ValueError('No completed approved budget/update boundary')
    regime=reg.get('reward_regime','investigation-a2-v1')
    result=dict(schema='patrick-jane-a2-results-v1',state=state,error=error,reward_regime=regime,
        base_lifetime_decisions=2813879,lifetime_decisions=c.get('individual_decisions',2813879),additional_decisions=c.get('run_decisions',0),
        active_training_seconds=active_seconds,target_training_seconds=target,additional_decision_target=decision_target,
        ppo_updates=c.get('run_updates',0),training=training,reward_components=dict(components),latest=latest,
        checkpoint_history=[{k:row[k] for k in ('checkpoint','sha256','counters','ppo')} for row in history],
        source_version=reg.get('source_version',{}),initial_checkpoint=reg.get('import_provenance',{}),
        validation_performed=False,best_checkpoint='not evaluated',acceptance='pending',
        limitations=['On-policy mixed-opponent outcomes do not establish generalization or isolate reward effects.',
            'Door/vent controls require matched frozen-policy comparisons; legacy engine differs from versioned A2 semantics.',
            'Correct ballots use hidden truth for trusted post-action credit only; chance adjustment alone does not establish deduction.',
            'Task commitment scaffolds execution and restricts strategic investigation interruptions.',
            '50% historical/50% upgraded is selected per match and never adapted to performance.',
            'No final-test partitions opened. Optional modern roles and non-lights critical sabotages are disabled.'])
    atomic_json(folder/'training_results.json',result)
    t=training['totals'];lines=['# Patrick Jane — Investigation A2 V1','',f'**{state.upper()}**. Development only; acceptance pending.','',
        f"Warm-start: original update102 at 2,813,879 lifetime decisions. New decisions {result['additional_decisions']:,}; PPO waves {result['ppo_updates']}.",
        f"Source publication: {reg.get('source_version',{}).get('revision','unpublished-working-tree')}. Origin identity: {reg.get('import_provenance',{}).get('checkpoint_sha256','unavailable')}.",
        f"Active collection/PPO {active_seconds/3600:.4f}/{target/3600:g} hours; optional decision target {decision_target}.",
        f"Registered reward regime: {regime}. Exact coefficients: {reg.get('reward_coefficients',{})}. Per-ejection +4/-2 and terminal awards are different events.",
        'A2 accepted target ballot: 0.2*(correct - actual impostor fraction among eligible targets excluding self); skips zero. Original-reward controls have no individual ballot credit.',
        f"Games {t.get('legitimate_games',0)}; task/ejection/parity endings {training['reasons']}; technical cuts {t.get('truncations',0)}.",
        f"Living/ghost tasks {training['living_tasks']}/{t.get('ghost_tasks',0)}.",
        '', '## Investigation and opportunity denominators','', '```json',json.dumps(training['investigation'],indent=2),'```',
        '', '## Doors, vents and meetings','', '```json',json.dumps(training['mechanics'],indent=2),'```',
        '', '## Task pursuit and commitment','', '```json',json.dumps(training['diagnostics'],indent=2),'```',
        '', '## Opponent mixture','']
    for key,value in training['opponent_groups'].items():lines.append(f"- {key}: {value['totals']['games']} games; crew win rate {value['win_rate']}; living tasks {value['living_tasks']}.")
    lines+=['','## Reward ledger','', '```json',json.dumps(dict(components),indent=2),'```',
        '', '## PPO','', '| Update | Decisions | Optimizer steps | KL | Loss | Entropy | Grad norm |','|---|---:|---:|---:|---:|---:|---:|']
    for row in history:
        p=row['ppo'];lines.append(f"| {row['counters']['update']} | {row['counters']['run_decisions']} | {p['updates']} | {p['approx_kl']:.6f} | {p['loss']:.6f} | {p['entropy']:.6f} | {p['grad_norm']:.6f} |")
    lines+=['',f"Latest checkpoint: `{latest.get('checkpoint','none')}`. Best: **not evaluated**.",
        '', '## Limitations','',*['- '+v for v in result['limitations']], '', '![Training curves](learning_curves.png)']
    if error:lines+=['',f'Failure/interruption: `{error}`.']
    (folder/'TRAINING_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if plots and history:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        xs=[r['counters']['run_decisions'] for r in history];ms=[metrics(r['rollout']['matches']) for r in history]
        fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
        series=[('Crew win rate',[m['win_rate'] for m in ms]),('Living tasks',[m['living_tasks'] for m in ms]),('Vote excess above chance',[m['investigation']['all_cast']['excess_correct_over_chance'] for m in ms]),('Vent witnesses',[m['mechanics'].get('crew_witnessed_vent_enter',0)+m['mechanics'].get('crew_witnessed_vent_exit',0) for m in ms]),('PPO KL',[r['ppo']['approx_kl'] for r in history]),('Forced fraction',[m['diagnostics'].get('forced_decisions',0)/max(1,m['totals']['decisions']) for m in ms])]
        for ax,(label,ys) in zip(axes.flat,series):ax.plot(xs,ys);ax.set(title=label,xlabel='A2 decisions');ax.grid(alpha=.2)
        fig.suptitle('On-policy A2; separate frozen development comparisons required');fig.savefig(folder/'learning_curves.png',dpi=130);plt.close(fig)
    return result
