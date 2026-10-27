"""Summarize only completed local pilot evidence; never runs games or training."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def ratio(a,b):return a/b if b else None
def fmt(value):return 'unavailable' if value is None else str(value) if isinstance(value,int) else f'{value:.3f}'
def mean(values):return statistics.mean(values) if values else None


def summarize(run,checkpoint):
    import torch
    result=run['result'];games=run['games']
    if result['actionable_decisions']!=512 or len(games)!=20:raise ValueError('Incomplete arm; do not report finished pilot')
    counts=Counter();actions=Counter();evidence={k:Counter() for k in ('with_direct_witness','without_direct_witness')}
    for g in games:
        counts.update(g['reward_event_counts']);actions.update(g['action_counts'])
        for key in evidence:evidence[key].update(g['voting_evidence'][key])
    metadata=torch.load(checkpoint,map_location='cpu',weights_only=True)['metadata']['training_state']
    opportunity=metadata['evidence']['opportunities']
    training=Counter();training_components=Counter()
    for e in result['episodes']:
        training.update(e['reward_counts']);training_components.update(e['reward_components'])
    for row in result['partial_streams'].values():
        training.update(row['reward_counts']);training_components.update(row['reward_components'])
    legal=sum(e['policy_report_legal_unique'] for e in result['episodes'])+sum(len(x['reports']) for x in opportunity)
    reporting=sum(g['report_legal_unique'] for g in games)
    return dict(training=dict(decisions=512,completed_episodes=len(result['episodes']),partial_streams=len(result['partial_streams']),
                simulated_seconds=result['simulated_seconds'],seconds=result['seconds'],event_counts=dict(training),
                reward_components=dict(training_components),legal_report_opportunities=legal,
                reporting_rate=ratio(training['reports'],legal),actions=result['actions'],
                checkpoint_sha256=__import__('hashlib').sha256(Path(checkpoint).read_bytes()).hexdigest()),
        validation=dict(games=20,wins=sum(g['crew_win'] for g in games),losses=sum(not g['crew_win'] for g in games),
                reasons=dict(Counter(g['reason'] for g in games)),event_counts=dict(counts),action_counts=dict(actions),
                report_opportunities=reporting,reports=counts['reports'],reporting_rate=ratio(counts['reports'],reporting),
                physical_report_pairs=sum(g['physical_report_legal_unique'] for g in games),
                visible_body_pairs=sum(g['body_visible_unique'] for g in games),
                vote_accuracy=ratio(counts['correct_votes'],counts['correct_votes']+counts['incorrect_votes']),
                survived=sum(g['focal_survived'] for g in games),
                mean_survival_seconds=mean([g['episode']['survival'] for g in games if g['episode'].get('survival') is not None]),
                mean_common_legacy_return=mean([g['common_legacy_return'] for g in games]),
                mean_counterfactual_shaped_return=mean([g['counterfactual_shaped_return'] for g in games]),
                repeated_correct_votes=sum(g['repeated_correct_vote_count'] for g in games),
                direct_witness_vote_strata={k:dict(counts=dict(v),cast_accuracy=ratio(v['correct_votes'],v['correct_votes']+v['incorrect_votes'])) for k,v in evidence.items()},
                modes={str(mode):dict(games=sum(g['mode']==mode for g in games),wins=sum(g['crew_win'] for g in games if g['mode']==mode),
                          reasons=dict(Counter(g['reason'] for g in games if g['mode']==mode))) for mode in (0.,1.)}))


def sampled_claims(replays):
    result={arm:Counter() for arm in ('original','shaped')}
    files={arm:0 for arm in result}
    for arm in result:
        for p in (Path(replays)/arm).glob('*.jsonl'):
            header=None;claims=[]
            for line in p.open(encoding='utf-8'):
                row=json.loads(line)
                if row.get('record')=='header':header=row
                elif row.get('record')=='event' and row.get('kind')=='claim':claims.append(row)
            if header is None:continue
            files[arm]+=1;focal=header['config']['focal_player'];roles=header['agent_roles']
            for claim in claims:
                if claim['speaker_id']!=focal:continue
                result[arm]['all_focal_claims']+=1
                result[arm]['kind:'+claim['claim_kind']]+=1
                if claim['claim_kind'] in ('suspect_player','suspect','accuse'):
                    result[arm]['accusations']+=1
                    role=roles.get(claim.get('subject_id'))
                    if role is not None:result[arm]['correct_accusations' if role=='impostor' else 'incorrect_accusations']+=1
    return {arm:dict(completed_replays=files[arm],**dict(v)) for arm,v in result.items()}


def preupdate_reproduction(replays):
    records={arm:{} for arm in ('original','shaped')}
    for arm in records:
        for p in (Path(replays)/arm).iterdir():
            if not p.name.endswith(('.jsonl','.jsonl.writing')):continue
            header=None
            for line in p.open(encoding='utf-8'):
                try:row=json.loads(line)
                except ValueError:continue
                if row.get('record')=='header':header=row
                if row.get('record')=='neural' and row.get('training_step',1000)<=128:
                    key=(header['config']['stream'],row['training_step'])
                    records[arm][key]={k:row[k] for k in ('selected_action_index','action_probabilities','value_estimate','policy_entropy','policy_update')}
    common=set(records['original']) & set(records['shaped'])
    return dict(sampled_decisions_compared=len(common),expected_sampled_decisions=64,
                identical_before_first_update=all(records['original'][k]==records['shaped'][k] for k in common),
                equal_coverage=set(records['original'])==set(records['shaped']),
                scope='Preselected streams0–3, first128 global choices; includes valid prefixes of incomplete recordings without labeling them completed.')


def figures(runs,summaries,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    fig,axes=plt.subplots(2,2,figsize=(11,7),facecolor='#0a1520')
    colors={'original':'#6ce5de','shaped':'#ff927d'}
    for run in runs:
        arm=run['arm'];label='A Original' if arm=='original' else 'B Shaped';rows=run['result']['updates']
        for ax,key in zip(axes.flat,('loss','entropy','approximate_kl','gradient_norm')):
            ax.plot([r['decisions'] for r in rows],[r[key] for r in rows],'-o',label=label,color=colors[arm]);ax.set_title(key.replace('_',' ').title());ax.set_xlabel('Actionable choices');ax.grid(alpha=.15)
    axes[0,0].legend();fig.suptitle('Seed-17 PPO pilot — four measured updates per model');fig.tight_layout()
    fig.savefig(output/'learning-curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.6),facecolor='#0a1520')
    for ax,key,title in zip(axes,('wins','report_opportunities','reports'),('Validation wins /20','Legal report opportunities','Accepted focal reports')):
        values=[summaries[a]['validation'][key] for a in ('original','shaped')]
        bars=ax.bar(['A Original','B Shaped'],values,color=list(colors.values()))
        ax.bar_label(bars,padding=3);ax.set_ylim(0,max(values+[1])*1.25)
        ax.yaxis.set_major_locator(__import__('matplotlib').ticker.MaxNLocator(integer=True))
        ax.set_title(title);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Development validation — 20 games/model over four scenario clusters');fig.tight_layout();fig.savefig(output/'validation-behavior.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path);p.add_argument('--replays',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    pilot=json.loads((a.run/'pilot.json').read_text());comparison=json.loads((a.run/'comparison.json').read_text())
    if pilot['status']!='development_pilot_complete_acceptance_pending':raise ValueError('Only complete pilots may use this report')
    a.output.mkdir(parents=True,exist_ok=False)
    summaries={r['arm']:summarize(r,a.run/f"{r['arm']}-17"/'checkpoint-512.pt') for r in pilot['runs']}
    claims=sampled_claims(a.replays);reproduction=preupdate_reproduction(a.replays)
    record=dict(status='exploratory_development_only_acceptance_pending',wall_seconds=pilot['seconds'],arms=summaries,
                comparison=comparison,sampled_training_claims=claims,preupdate_reproduction=reproduction,
                seed29_started=False,final_partitions_opened=False)
    (a.output/'metrics.json').write_text(json.dumps(record,indent=2)+'\n')
    lines=['# Seed-17 Crewmate V2 live pilot — development only','',
           'Both models completed512 actionable choices and20 validation games. These are40 model-game evaluations of four distinct scenarios, with repeated modes/report rotations clustered by scenario. One initialization is exploratory; Phase6.1 acceptance remains pending.','',
           '| Measure | A Original | B Shaped |','|---|---:|---:|']
    rows=[('Training minutes',lambda x:x['training']['seconds']/60),('Completed training episodes',lambda x:x['training']['completed_episodes']),
          ('Simulated training hours',lambda x:x['training']['simulated_seconds']/3600),('Validation wins /20',lambda x:x['validation']['wins']),
          ('Living own tasks',lambda x:x['validation']['event_counts'].get('alive_tasks',0)),('Automatic ghost tasks',lambda x:x['validation']['event_counts'].get('ghost_tasks',0)),
          ('Body visibility pairs',lambda x:x['validation']['visible_body_pairs']),('Legal report opportunities at choices',lambda x:x['validation']['report_opportunities']),
          ('Physical legal report pairs',lambda x:x['validation']['physical_report_pairs']),('Accepted reports',lambda x:x['validation']['reports']),
          ('Reports / legal opportunity',lambda x:x['validation']['reporting_rate']),('Correct votes',lambda x:x['validation']['event_counts'].get('correct_votes',0)),
          ('Incorrect votes',lambda x:x['validation']['event_counts'].get('incorrect_votes',0)),('Cast vote accuracy',lambda x:x['validation']['vote_accuracy']),
          ('Explicit skips',lambda x:x['validation']['event_counts'].get('skips',0)),('Automatic skips',lambda x:x['validation']['event_counts'].get('automatic_skips',0)),
          ('Focal survived /20',lambda x:x['validation']['survived']),('Mean common legacy return',lambda x:x['validation']['mean_common_legacy_return'])]
    for title,fn in rows:lines.append(f'| {title} | {fmt(fn(summaries["original"]))} | {fmt(fn(summaries["shaped"]))} |')
    lines+=['',f'Total measured pilot wall time: **{pilot["seconds"]/60:.2f} minutes**, including training and validation; three-hour cap retained.','',
            '![Actual PPO update curves](learning-curves.png)','![Measured validation behavior](validation-behavior.png)','',
            '## Match endings','', '| Seed | Mode / rotation / opponents | A | B |','|---|---|---|---|']
    for x,y in zip(pilot['runs'][0]['games'],pilot['runs'][1]['games']):
        assert (x['seed'],x['mode'],x['rotation'],x['opponents'])==(y['seed'],y['mode'],y['rotation'],y['opponents'])
        lines.append(f"| {x['seed']} | {'deduction' if x['mode'] else 'ordinary'} / {x['rotation']} / {x['opponents']} | {x['reason']} | {y['reason']} |")
    lines+=['','## Interpretation boundaries','',
            'The shared frozen Phase4 belief predictor is unchanged. Differences in its measured scores arise from observation trajectories, not independently trained belief networks. Reporter-identity-only controls, calibration, voting participation, paired four-scenario bootstrap and direct-witness vote strata are retained in metrics.json.',
            'An absent direct kill witness does not imply no legitimate evidence: bodies, reports, claims and public task proof may still be available. Missing reporting opportunities are separate from failures to report. Physical opportunities inside held options are separate from opportunities at policy-choice boundaries.',
            'Sampled training accusations use only completed replays of preselected streams0–3. Hidden roles are used for offline scoring only. These counts are not a census of all training or validation claims, and their legitimate evidence support was not exported. Raw observations and role-label datasets remain local.',
            'Correct-vote bonuses can reward guessing (+2/−1 gives positive expectation above1/3 correctness). Repeated correct ballots across meetings, reporter shortcuts, ghost/task-only contributions and reward-return scale differences must not be mistaken for learned deduction. Reward configuration is a bundle, so this pair cannot attribute effects to a single coefficient.',
            'The viewer plays completed training matches recorded under behavior-policy updates0–3; validation tests the fixed512-choice checkpoint. No true in-progress frame streaming or validation replay export is claimed. No source/reward/gameplay change, final-test access, seed29 replication, impostor development, merge or deployment occurred.']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    figures(pilot['runs'],summaries,a.output)
    print(json.dumps({k:v['validation'] for k,v in summaries.items()},indent=2))


if __name__=='__main__':main()
