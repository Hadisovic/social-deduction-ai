"""Report a single initialization honestly; no final-test or seed-spread claims."""
import json
from pathlib import Path
import numpy as np
from phase5_analysis import summarize
from .study_training import CONDITIONS
from .study_support import write_json


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def build_report(output, training, selection, screening, validation, seconds, screening_seconds, validation_seconds):
    output = Path(output)
    summary = summarize(sorted(validation,key=lambda row:(row['method'],row['seed'])))
    details = {}
    for condition in CONDITIONS:
        episodes = read_rows(output/condition/'episodes.jsonl')
        rows = [r for r in validation if r['method']==condition]
        total_votes = sum(r['votes_cast'] for r in rows)
        total_skips = sum(r['skips'] for r in rows)
        details[condition] = dict(training=training[condition], selected_checkpoint=selection[condition],
            training_mean_own_tasks=float(np.mean([r['own_tasks'] for r in episodes])),
            training_votes_cast=sum(r['votes_cast'] for r in episodes),
            training_votes_correct=sum(r['votes_correct'] for r in episodes),
            training_skips=sum(r['skips'] for r in episodes),
            validation_valid_reports=sum(r['valid_reports'] for r in rows),
            validation_voting_match_fraction=sum(r['votes_cast']>0 for r in rows)/len(rows),
            validation_skip_fraction=total_skips/(total_votes+total_skips) if total_votes+total_skips else None,
            action_counts={kind:sum(r['action_counts'].get(kind,0) for r in rows)
                           for kind in sorted({k for r in rows for k in r['action_counts']})})
    # Worker throughput includes the actual complete-match workload, not CPU core estimates.
    staged_validation_matches=len(screening)+len(validation)
    observed_seconds_per_match=(screening_seconds+validation_seconds)/staged_validation_matches
    remaining_training=2*sum(r['seconds'] for r in training.values())
    remaining_validation=1680*observed_seconds_per_match
    estimate=remaining_training+remaining_validation
    result=dict(status='seed17_validation_complete_not_phase6_1_accepted', initialization_seeds=[17],
        seconds=seconds, screening_seconds=screening_seconds, validation_seconds=validation_seconds,
        training_decisions=sum(r['decisions'] for r in training.values()),
        validation_executions=staged_validation_matches, details=details, **summary,
        updated_remaining_six=dict(training_seconds_estimate=remaining_training,
            validation_executions=1680, validation_seconds_estimate=remaining_validation,
            total_seconds_estimate=estimate,provisional_range_seconds=[.8*estimate,1.5*estimate],
            controls_reused=True,workers=2,approved=False),
        final_test_opened=False, learned_impostor=False,
        notice='Validation only, one initialization per condition. Screening reuses the first 20 of 200 validation seeds. No final robustness, replicated memory benefit or training-seed confidence interval is established.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,3,figsize=(13,3.7))
    for condition in CONDITIONS:
        rows=training[condition]['updates']
        for ax,key,label in zip(axes,('loss','entropy','simulated_seconds'),('PPO loss','Policy entropy','Simulated seconds')):
            ax.plot([r['decisions'] for r in rows],[r[key] for r in rows],label=condition)
            ax.set(xlabel='Training decisions',ylabel=label);ax.grid(alpha=.25)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(output/'training_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    values=[summary['metrics'][c]['ID'] for c in CONDITIONS]
    axes[0].bar(CONDITIONS,[v['mean_own_tasks'] for v in values]);axes[0].set_ylabel('Own tasks / validation match')
    axes[1].bar(CONDITIONS,[v['votes_cast'] for v in values],label='Cast votes')
    axes[1].bar(CONDITIONS,[v['skips'] for v in values],bottom=[v['votes_cast'] for v in values],label='Skips')
    axes[1].legend();axes[1].set_ylabel('Validation vote/skip counts')
    for ax in axes:ax.tick_params(axis='x',labelrotation=12)
    fig.suptitle('Seed 17 only • 200 validation matches / condition • no final tests')
    fig.tight_layout();fig.savefig(output/'tasks_voting.png',dpi=160);plt.close(fig)
    return result
