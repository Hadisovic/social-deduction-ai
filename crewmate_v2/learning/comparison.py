"""Development-only aggregation, conditional rates and paired scenario uncertainty."""
from collections import Counter
import numpy as np


def calibration(rows,key):
    records=[r[key] for r in rows];n=sum(r['n'] for r in records)
    if not n:return dict(n=0)
    result={k:sum(r.get(k,0.)*r['n'] for r in records if r['n'])/n for k in ('top1','nll','brier')}
    bins=[];ece=0.
    for i in range(10):
        count=sum(r['bins'][i]['n'] for r in records if r['n'])
        confidence=sum(r['bins'][i]['confidence']*r['bins'][i]['n'] for r in records if r['n'])/count if count else None
        accuracy=sum(r['bins'][i]['correct']*r['bins'][i]['n'] for r in records if r['n'])/count if count else None
        if count:ece+=count/n*abs(confidence-accuracy)
        bins.append(dict(n=count,confidence=confidence,accuracy=accuracy))
    return dict(n=n,**result,ece=ece,bins=bins,high_confidence_wrong=sum(r.get('high_confidence_wrong',0) for r in records))


def summarize(rows):
    counts=Counter();behavior=Counter();heuristic=Counter()
    for row in rows:
        counts.update(row['reward_event_counts']);behavior.update(row['behavior']);heuristic.update(row['reporter_only_expected_correct'])
    denominator=sum(r['report_legal_unique'] for r in rows)
    cast=counts['correct_votes']+counts['incorrect_votes']
    n=sum(r['pre_vote_count'] for r in rows)
    strata={role:calibration([dict(metric=r['policy_vote_ranking_by_reporter_role'].get(role,dict(n=0))) for r in rows],'metric')
            for role in ('crewmate','impostor')}
    balanced=(strata['crewmate']['top1']+strata['impostor']['top1'])/2 if all(r['n'] for r in strata.values()) else None
    def rate(a,b):return a/b if b else None
    return dict(games=len(rows),distinct_scenarios=len({r['seed'] for r in rows}),
        crew_win_rate=rate(sum(r['crew_win'] for r in rows),len(rows)),reasons=dict(Counter(r['reason'] for r in rows)),
        event_counts=dict(counts),behavior=dict(behavior),report_opportunities=denominator,
        reporting_rate=rate(counts['reports'],denominator),legal_ballot_opportunities=n,
        voting_participation=rate(cast,n),skip_rate=rate(counts['skips']+counts['automatic_skips'],n),
        unsubmitted_ballots=n-cast-counts['skips']-counts['automatic_skips'],vote_accuracy=rate(counts['correct_votes'],cast),
        frozen_belief=calibration(rows,'frozen_belief'),policy_vote_ranking=calibration(rows,'policy_vote_ranking'),
        reporter_only_accuracy={k:rate(v,n) for k,v in heuristic.items()},
        policy_reporter_role_strata=strata,reporter_role_balanced_policy_top1=balanced,
        policy_means={key:sum(r[key]*r['pre_vote_count'] for r in rows if r['pre_vote_count'])/n if n else None
            for key in ('mean_vote_probability','mean_skip_probability','mean_self_vote_probability','mean_report_counter_channel_tv')},
        regimes={key:dict(games=sum(r['opponents']==key for r in rows),crew_win_rate=rate(sum(r['crew_win'] for r in rows if r['opponents']==key),sum(r['opponents']==key for r in rows))) for key in ('original','varied')})


def compare(records):
    arms={arm:[g for r in records if r['arm']==arm for g in r['games']] for arm in ('original','shaped')}
    summaries={k:summarize(v) for k,v in arms.items()};deltas=[]
    for seed in sorted({g['seed'] for g in arms['original']}):
        left=summarize([g for g in arms['original'] if g['seed']==seed]);right=summarize([g for g in arms['shaped'] if g['seed']==seed])
        deltas.append(dict(seed=seed,win_rate=right['crew_win_rate']-left['crew_win_rate'],
            reporting_rate=(right['reporting_rate']-left['reporting_rate']) if right['reporting_rate'] is not None and left['reporting_rate'] is not None else None))
    rng=np.random.default_rng(2730050);intervals={}
    for metric in ('win_rate','reporting_rate'):
        values=np.asarray([d[metric] for d in deltas if d[metric] is not None])
        samples=rng.choice(values,(5000,len(values)),replace=True).mean(1) if len(values) else []
        intervals[metric]=dict(paired_scenarios=len(values),mean=float(values.mean()) if len(values) else None,
            percentile_95=np.quantile(samples,[.025,.975]).tolist() if len(values) else None)
    return dict(status='exploratory_development_only_acceptance_pending',arms=summaries,paired_deltas=deltas,
        paired_bootstrap=intervals,uncertainty_note='Four scenario clusters; rotations/modes stay within each cluster. Initial stage has one initialization; not acceptance evidence.')
