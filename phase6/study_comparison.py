"""Nine-model validation comparison with explicit match/initialization uncertainty."""
import json
from pathlib import Path
import numpy as np
from phase5_analysis import summarize
from .study_training import CONDITIONS
from .study_validation import CONTROLS


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def build_comparison(output,training,selections,pilot,seconds,screening_seconds,validation_seconds):
    from .study_continuation import PILOT
    output=Path(output);all_rows=[];details={};arrays={}
    for seed in (17,29,43):
        for condition in CONDITIONS:
            key=f'{condition}-{seed}'
            data=read_rows((PILOT/'validation'/condition if seed==17 else output/'validation'/key)/'matches.jsonl')
            data=sorted(data,key=lambda r:r['seed'])
            if len(data)!=200 or [r['seed'] for r in data]!=list(range(1220000,1220200)):
                raise ValueError('Comparison requires complete paired validation only')
            t=pilot['details'][condition]['training'] if seed==17 else training[key]
            cp=pilot['details'][condition]['selected_checkpoint'] if seed==17 else selections[key]
            arrays[key]=data
            cast=sum(r['votes_cast'] for r in data);correct=sum(r['votes_correct'] for r in data)
            details[key]=dict(training=t,selected_checkpoint=cp,initialization_seed=seed,
                votes_incorrect=cast-correct,voting_match_fraction=sum(r['votes_cast']>0 for r in data)/200,
                valid_reports=sum(r['valid_reports'] for r in data),
                action_counts={kind:sum(r['action_counts'].get(kind,0) for r in data)
                    for kind in sorted({k for r in data for k in r['action_counts']})})
            all_rows.extend(dict(r,method=key) for r in data)
    for control in CONTROLS:
        data=read_rows(PILOT/'validation'/control/'matches.jsonl')
        arrays[control]=sorted(data,key=lambda r:r['seed']);all_rows.extend(data)
    summary=summarize(sorted(all_rows,key=lambda r:(r['method'],r['seed'])))
    aggregate={}
    for c in CONDITIONS:
        metrics=[summary['metrics'][f'{c}-{s}']['ID'] for s in (17,29,43)]
        aggregate[c]={key:dict(mean=float(np.mean(vals)),sample_sd=float(np.std(vals,ddof=1)),
                              minimum=float(min(vals)),maximum=float(max(vals)))
                      for key in ('crew_win_rate','mean_own_tasks')
                      for vals in [[m[key] for m in metrics]]}
        cast=sum(m['votes_cast'] for m in metrics);correct=sum(m['votes_correct'] for m in metrics)
        aggregate[c].update(votes_cast=cast,votes_correct=correct,votes_incorrect=cast-correct,
            voting_accuracy=correct/cast if cast else None,skips=sum(m['skips'] for m in metrics),
            valid_reports=sum(details[f'{c}-{s}']['valid_reports'] for s in (17,29,43)))
    # Paired scenario bootstrap conditions on the three trained models, not 600 independent scenarios.
    rng=np.random.default_rng(606);paired={}
    for c in CONDITIONS:
        paired[c]={}
        for other in ('scripted','idle','random','phase5_release_transfer')+tuple(x for x in CONDITIONS if x!=c):
            comparisons={}
            for metric in ('crew_win','own_tasks'):
                a=np.array([[float(r[metric]) for r in arrays[f'{c}-{s}']] for s in (17,29,43)])
                b=np.array([[float(r[metric]) for r in arrays[f'{other}-{s}']] for s in (17,29,43)]) if other in CONDITIONS else np.tile([float(r[metric]) for r in arrays[other]],(3,1))
                delta=a-b;scenario_delta=delta.mean(0)
                draws=rng.choice(scenario_delta,size=(5000,200),replace=True).mean(1)
                comparisons[metric]=dict(mean_difference=float(delta.mean()),
                    paired_scenario_bootstrap95=np.quantile(draws,[.025,.975]).tolist(),
                    differences_by_initialization=delta.mean(1).tolist(),initialization_seeds=[17,29,43])
            paired[c][other]=comparisons
    result=dict(status='nine_models_validation_complete_final_approval_required',seconds=seconds,
        screening_seconds=screening_seconds,validation_seconds=validation_seconds,
        additional_training_seconds=sum(t['seconds'] for t in training.values()),
        additional_training_decisions=sum(t['decisions'] for t in training.values()),
        total_training_decisions=73728,additional_validation_executions=1680,total_validation_executions=3320,
        initialization_seeds=[17,29,43],details=details,aggregate=aggregate,
        paired_condition_comparisons=paired,**summary,final_test_opened=False,
        notice='Validation only. Three initializations, same 200 scenarios. Screening reuses first 20. Aggregate bootstrap conditions on these models; SD/range describe only three training seeds. No independent final robustness or Phase 6.1 acceptance.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for i,c in enumerate(CONDITIONS):
        vals=[summary['metrics'][f'{c}-{s}']['ID'] for s in (17,29,43)]
        for ax,key in zip(axes,('crew_win_rate','mean_own_tasks','votes_cast')):
            ax.plot([17,29,43],[v[key] for v in vals],marker='o',label=c)
            ax.set(xlabel='Initialization seed',ylabel=key);ax.set_xticks([17,29,43]);ax.grid(alpha=.25)
    axes[0].legend(fontsize=8);fig.suptitle('Nine models • 200 paired validation scenarios • no final tests')
    fig.tight_layout();fig.savefig(output/'nine_model_comparison.png',dpi=160);plt.close(fig)
    return result
