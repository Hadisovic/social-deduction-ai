"""Whole-match summaries, paired uncertainty, and explicit validation selection."""
import hashlib
import json
from pathlib import Path
import numpy as np
from phase5_research_report import read_rows

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'docs/benchmarks/phase5/expanded'


def summarize(rows):
    keys=[(r['method'],r['family'],r['seed']) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate method/family/match seed')
    metrics={};paired={};rng=np.random.default_rng(505)
    for method in sorted({r['method'] for r in rows}):
        metrics[method]={};paired[method]={}
        for family in sorted({r['family'] for r in rows if r['method']==method}):
            subset=[r for r in rows if r['method']==method and r['family']==family]
            n=len(subset);p=np.mean([r['crew_win'] for r in subset]);z=1.96
            center=(p+z*z/(2*n))/(1+z*z/n)
            half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
            cast=sum(r['votes_cast'] for r in subset)
            voting_interval=None
            if cast:
                indices=rng.integers(0,n,size=(5000,n))
                vote_counts=np.asarray([r['votes_cast'] for r in subset])[indices].sum(1)
                correct_counts=np.asarray([r['votes_correct'] for r in subset])[indices].sum(1)
                ratios=correct_counts[vote_counts>0]/vote_counts[vote_counts>0]
                voting_interval=np.quantile(ratios,[.025,.975]).tolist()
            metrics[method][family]=dict(matches=n,crew_win_rate=float(p),
                win_rate_wilson95=[float(center-half),float(center+half)],
                mean_own_tasks=float(np.mean([r['own_tasks'] for r in subset])),
                mean_survival=float(np.mean([r['survival'] for r in subset])),
                voting_accuracy=sum(r['votes_correct'] for r in subset)/cast if cast else None,
                voting_match_bootstrap95=voting_interval,
                votes_cast=cast,votes_correct=sum(r['votes_correct'] for r in subset),
                skips=sum(r['skips'] for r in subset),
                illegal_actions=sum(r['illegal_actions'] for r in subset),
                navigation_failures=sum(r['navigation_failures'] for r in subset),
                timeouts=sum(r['reason']=='TIMEOUT' for r in subset),
                correct_ejections=sum(r['correct_ejections'] for r in subset),
                innocent_ejections=sum(r['innocent_ejections'] for r in subset))
            paired[method][family]={}
            for baseline in ('scripted','idle','random'):
                control={r['seed']:r for r in rows if r['method']==baseline and r['family']==family}
                shared=[r for r in subset if r['seed'] in control]
                if not shared or baseline==method:continue
                delta=np.array([float(r['crew_win'])-float(control[r['seed']]['crew_win']) for r in shared])
                boot=rng.choice(delta,size=(5000,len(delta)),replace=True).mean(1)
                paired[method][family][baseline]=dict(matches=len(shared),
                    win_rate_difference=float(delta.mean()),bootstrap95=np.quantile(boot,[.025,.975]).tolist())
    return dict(metrics=metrics,paired=paired,
                uncertainty='95% Wilson match intervals and paired whole-match bootstrap; not training-seed confidence intervals.')


def review_selection(lock=False):
    screening=json.loads((OUTPUT/'screening-selection.json').read_text())
    rows=[]
    names=['selection-baselines','selection-v1-17']+['selection-'+f['run'] for f in screening['finalists']]
    for name in names:
        record=json.loads((OUTPUT/name/'metrics.json').read_text())
        if record['status']!='validation' or record['matches_per_method']!=100:
            raise ValueError('Complete 100-match validation required before selection')
        rows+=read_rows(OUTPUT/name/'matches.jsonl')
    result=summarize(rows)
    candidates=[]
    for finalist in screening['finalists']:
        if 'ablation' in finalist['run']:continue
        method=finalist['checkpoint'];m=result['metrics'][method]['ID']
        if m['illegal_actions'] or m['navigation_failures']:continue
        candidates.append((m['crew_win_rate'],m['mean_own_tasks'],m['voting_accuracy'] or 0.,method))
    if not candidates:raise ValueError('No mechanically valid full-memory candidate')
    best=max(candidates)[-1]
    m=result['metrics'][best]['ID'];paired=result['paired'][best]['ID']
    useful=(m['mean_own_tasks']>.5 and all(paired[b]['bootstrap95'][0]>0 for b in ('idle','random')))
    result.update(status='validation_review',selected_checkpoint=best,
                  useful_vs_idle_and_random=useful,
                  notice='Selection and any budget extension use validation only. Phase 6 still requires independent final robustness.',
                  final_test_opened=False)
    (OUTPUT/'validation-review.json').write_text(json.dumps(result,indent=2)+'\n')
    if lock:
        destination=OUTPUT/'locked-selection.json'
        if destination.exists():raise FileExistsError('Selection is already locked; do not change it using test outcomes')
        methods=[f['checkpoint'] for f in screening['finalists']]
        methods.append(str(Path('artifacts/phase5/runs/full-17-approved/policy.pt')))
        from phase5_policy import StrategicPolicy
        _,metadata=StrategicPolicy.load(best)
        initial=Path(f"artifacts/phase5/runs/options-{metadata['seed']}-initial-control/policy.pt")
        if not initial.exists():raise FileNotFoundError('Reconstruct selected seed initial control first')
        methods.append(str(initial))
        locked=dict(status='locked_before_final_tests',selected_checkpoint=best,
                    checkpoint_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in methods},
                    baseline_methods=['random','scripted','idle'],matches_per_method=1000,
                    seed_partitions={'ID':[630000,630499],'patient':[640000,640499]},
                    validation_review_sha256=hashlib.sha256((OUTPUT/'validation-review.json').read_bytes()).hexdigest(),
                    notice='Freeze selection; final results are reporting only, not a signal to choose another checkpoint.')
        destination.write_text(json.dumps(locked,indent=2)+'\n')
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock',action='store_true')
    args=parser.parse_args()
    r=review_selection(args.lock)
    print(json.dumps({k:r[k] for k in ('selected_checkpoint','useful_vs_idle_and_random','metrics','paired')},indent=2))
