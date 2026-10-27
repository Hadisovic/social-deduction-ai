"""Registered aggregate endpoints; no raw role-label dataset is published."""
from collections import Counter
import numpy as np
from crewmate_v2.learning.comparison import calibration


def summarize(games):
    counts=Counter();report=Counter();opportunities=Counter();actions=Counter()
    for game in games:
        for actor in game['actors'].values():
            counts.update(actor['reward_counts']);report.update(actor['reporting'])
            opportunities.update(actor['opportunities']);actions.update(actor['actions'])
    def ratio(a,b):return a/b if b else None
    cast=counts['correct_votes']+counts['incorrect_votes']
    return dict(games=len(games),scenario_clusters=len({g['seed'] for g in games}),wins=sum(g['crew_win'] for g in games),
        endings=dict(Counter(g['reason'] for g in games)),actor_event_counts=dict(counts),report_actions=dict(report),
        reporting_opportunities=dict(opportunities),actor_report_conversion=ratio(report['accepted'],opportunities['choice_legal']),
        games_with_crew_reports=sum(any(v['reporting'].get('accepted',0) for v in g['actors'].values()) for g in games),
        vote_accuracy=ratio(counts['correct_votes'],cast),action_counts=dict(actions),
        frozen_belief=calibration([g['audit'] for g in games],'frozen_belief'),
        policy_vote_ranking=calibration([g['audit'] for g in games],'policy_vote_ranking'),
        selected_incorrect_suspect_claims=sum(g['audit']['selected_incorrect_suspect_claims'] for g in games))


def review(evaluation,training):
    groups={};flags=[];coverage={}
    for label,games in evaluation.items():
        groups[label]={name:summarize([g for g in games if (g['context']=='ordinary' if name=='ordinary' else g['context']!='ordinary')]) for name in ('ordinary','controlled')}
        groups[label]['body']=summarize([g for g in games if g['context']=='body'])
        groups[label]['meeting']=summarize([g for g in games if g['context'].startswith('meeting')])
        meetings=[g for g in games if g['context'].startswith('meeting')]
        per_role={}
        for role in ('crewmate','impostor'):
            applicable=[g for g in meetings if g['audit']['nonreporter_observer_policy'].get(role,{}).get('n',0)]
            per_role[role]=dict(games=len(applicable),policy_top1_macro=float(np.mean([g['audit']['nonreporter_observer_policy'][role]['top1'] for g in applicable])) if applicable else None,
                accuse_reporter_macro=float(np.mean([g['audit']['nonreporter_reporter_only'][role]['accuse'] for g in applicable])) if applicable else None)
        groups[label]['controlled_reporter_role_macro']=per_role
        pairs=groups[label]['body']['reporting_opportunities'].get('choice_legal',0)
        ballots=sum(g['audit']['controlled_prefix_actor_meetings'] for g in meetings)
        coverage[label]=dict(body_actor_pairs=pairs,legal_actor_meetings=ballots,
                             reporter_role_meetings={r:v['games'] for r,v in per_role.items()})
        if len(games)!=16 or pairs<12 or ballots<12 or any(v['games']<2 for v in per_role.values()):flags.append(label+':evaluation_coverage_inconclusive')
    train_coverage={}
    for label,games in training.items():
        body=[g for g in games if g['context']=='body']
        pairs=sum(sum(a['opportunities']['choice_legal'] for a in g['actors'].values()) for g in body)
        reports=sum(sum(a['reporting'].get('accepted',0) for a in g['actors'].values()) for g in games)
        train_coverage[label]=dict(body_matches=len(body),body_actor_pairs=pairs,accepted_learning_reports=reports)
        if len(body)<8 or pairs<24:flags.append(label+':training_opportunity_coverage_inconclusive')
        if reports<2:flags.append(label+':reporting_learning_endpoint_inconclusive_fewer_than_two_reports')
    intervals={};rng=np.random.default_rng(20261027)
    if 'original' in evaluation and 'shaped_v2' in evaluation:
        for name in ('ordinary','controlled'):
            arms={a:{g['seed']:g for g in evaluation[a] if (g['context']=='ordinary')==(name=='ordinary')} for a in ('original','shaped_v2')}
            seeds=sorted(arms['original'].keys()&arms['shaped_v2'].keys())
            deltas=np.asarray([float(arms['shaped_v2'][s]['crew_win'])-float(arms['original'][s]['crew_win']) for s in seeds])
            samples=rng.choice(deltas,(5000,len(deltas)),replace=True).mean(1) if len(deltas) else []
            intervals[name]=dict(scenario_clusters=len(seeds),mean_shaped_minus_original=float(deltas.mean()) if len(deltas) else None,
                descriptive_percentile_95=np.percentile(samples,[2.5,97.5]).tolist() if len(samples) else None)
    return dict(status='exploratory_development_only_acceptance_pending',groups=groups,training_coverage=train_coverage,
        evaluation_coverage=coverage,inconclusive_flags=flags,paired_win_uncertainty=intervals,
        limits=['One initialization; eight scenario clusters, with only four body/four meeting controlled cases.',
                'Actor/body pairs are not independent corpses or scenarios; compare per-game results as well.',
                'Controlled prefixes can introduce choreography cues; ordinary performance stays separate.',
                'Reporter-role comparison uses nonreporter actors and per-game macro weighting.',
                'The shared frozen belief predictor is not separately trained for each policy.'])
