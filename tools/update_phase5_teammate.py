"""Refresh a measured-results block in the owner's external teammate handoff."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from phase5_research_report import build,label
OUTPUT=ROOT/'docs/benchmarks/phase5/expanded'
START='<!-- PHASE5_MEASURED_START -->'
END='<!-- PHASE5_MEASURED_END -->'


def main(document):
    progress=build()
    lines=[START,'## Measured study snapshot','',
           'These numbers are generated from saved records. Training outcomes are not final-test results.','',
           '| Run | Seed | Decisions | Complete training matches | Own tasks | Finished |',
           '|---|---:|---:|---:|---:|---|']
    for r in progress['runs']:
        lines.append(f"| {r['name']} | {r['seed']} | {r['decisions']:,} | {r['episodes']} | {r['own_tasks']} | {'Yes' if r['complete'] else 'Running'} |")
    final=OUTPUT/'final-results.json'
    review=OUTPUT/'validation-review.json'
    if final.exists() or review.exists():
        record=json.loads((final if final.exists() else review).read_text())
        selected=record['selected_checkpoint']
        lines+=['',f"**{'Final test' if final.exists() else 'Validation'} assessment:** `{label(selected)}`.",'',
                '| Family | Matches | Crew wins | 95% match interval | Own tasks / match | Voting accuracy | Cast votes |',
                '|---|---:|---:|---|---:|---:|---:|']
        for family,r in record['metrics'][selected].items():
            lo,hi=r['win_rate_wilson95'];accuracy='N/A' if r['voting_accuracy'] is None else f"{r['voting_accuracy']:.1%}"
            lines.append(f"| {family} | {r['matches']} | {r['crew_win_rate']:.1%} | {lo:.1%}–{hi:.1%} | {r['mean_own_tasks']:.2f} | {accuracy} | {r['votes_cast']} |")
        lines+=['','See the evaluation report for all methods and paired comparisons.']
    else:
        screening=(OUTPUT/'screening-selection.json').exists()
        lines+=['',('Checkpoint screening is complete. The larger 100-match validation comparison is running.'
                    if screening else 'Checkpoint screening is still underway.'),
                'Final selection has not been locked.']
        available=[r for r in progress['validation'] if r['matches']==100]
        if available:
            lines+=['','Completed 100-match validation measurements (not final tests):','',
                    '| Model/control | Crew wins | Own tasks / match | Voting accuracy | Cast votes |',
                    '|---|---:|---:|---:|---:|']
            for r in available:
                accuracy='N/A' if r['voting_accuracy'] is None else f"{r['voting_accuracy']:.1%}"
                lines.append(f"| {r['label']} | {r['crew_win_rate']:.1%} | {r['mean_own_tasks']:.2f} | {accuracy} | {r['votes_cast']} |")
    lines+=['',('Final tests are complete.' if final.exists() else
                'Final tests are running; do not interpret partial results as the final result.' if progress['final_test_opened'] else
                'Final ID/patient test partitions remain unopened.'),
            f"{len(progress['figures'])} figures currently cover learning, optimization, validation and behavior.",
            'Three full-memory seeds are planned/measured separately. The current-only comparison has one seed.',
            'That comparison is exploratory; it does not establish a replicated memory benefit.',
            'Phase 6 remains a suggested future sequence, not an automatic next training stage.',END]
    source=document.read_text(encoding='utf-8')
    block='\n'.join(lines)
    if START in source:
        a=source.index(START);b=source.index(END,a)+len(END)
        source=source[:a]+block+source[b:]
    else:
        anchor='## What changed, and why'
        source=source.replace(anchor,block+'\n\n'+anchor,1)
    document.write_text(source,encoding='utf-8')
    print(str(document))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('document',type=Path,help='Existing external teammate Markdown file')
    args=parser.parse_args()
    main(args.document)
