"""Append a reproducible experiment ledger to the external teammate document."""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import platform
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase5_research_report import read_rows, label

OUTPUT = ROOT / 'docs/benchmarks/phase5/expanded'
START = '<!-- PHASE5_LEDGER_START -->'
END = '<!-- PHASE5_LEDGER_END -->'


def build(document):
    result = json.loads((OUTPUT / 'final-results.json').read_text())
    assert result['status'] == 'final_evaluation_complete'
    lock = json.loads((OUTPUT / 'locked-selection.json').read_text())
    progress = json.loads((OUTPUT / 'progress.json').read_text())
    local_zone = timezone(timedelta(hours=3))
    def stamp(seconds):
        return datetime.fromtimestamp(seconds, local_zone).isoformat(timespec='seconds')
    groups = {}
    for path in OUTPUT.glob('*/metrics.json'):
        record = json.loads(path.read_text())
        name = path.parent.name
        group = ('Final testing' if name.startswith('final-') else
                 '100-match validation' if name.startswith('selection-') else
                 'Untrained diagnostic controls' if name.startswith('initial-') else
                 '20-match diagnostic screening')
        groups.setdefault(group, []).append((path, record))
    timings = {}
    for group, records in groups.items():
        ends = [p.stat().st_mtime for p, _ in records]
        starts = [p.stat().st_mtime - r['seconds'] for p, r in records]
        timings[group] = dict(jobs=len(records),
            matches=sum(len(read_rows(p.parent/'matches.jsonl')) for p, _ in records),
            summed_job_seconds=sum(r['seconds'] for _, r in records),
            inferred_elapsed_seconds=max(ends)-min(starts),
            inferred_start=stamp(min(starts)), inferred_end=stamp(max(ends)))
    import torch, numpy
    metadata = dict(hardware={
        'processor': 'Intel Core i9-14900HX', 'physical_cores':24,
        'logical_processors':32, 'physical_memory_bytes':16869548032,
        'source':'Read-only Win32_Processor and Win32_ComputerSystem query on experiment host'},
        software={'python':platform.python_version(), 'torch':torch.__version__,
                  'numpy':numpy.__version__, 'platform':platform.platform()},
        execution={'device':'CPU', 'torch_threads_per_worker':1,
                   'final_process_workers':4, 'validation_process_workers':3,
                   'multi_device':False}, timings=timings,
        timing_caution='Job seconds are measured perf_counter durations. Stage spans are inferred from metrics file modification times minus measured job duration, include scheduling gaps, and are approximate. Summed job time is not CPU time or elapsed wall time.')
    earliest = min((ROOT/r['directory']/'run-config.json').stat().st_mtime for r in progress['runs'])
    latest = (OUTPUT/'final-results.json').stat().st_mtime
    metadata['substantive_study_span'] = dict(start=stamp(earliest), end=stamp(latest),
        hours=(latest-earliest)/3600,
        caution='Filesystem-based elapsed span from earliest substantive run config to final summary; includes reviews, pauses and overlapping jobs, excludes earlier development/smoke work.')
    (OUTPUT/'execution-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    lines = [START, '## Complete experiment ledger', '',
        'The simulator study is complete. The selected policy is useful against fixed opponents.',
        'Further training is not required to demonstrate this Phase 5 result. Mastery is not established.',
        'It does not clearly outperform scripted play, and real-game transfer has not been tested.',
        'Keep this release frozen. Future improvements need a new experiment and fresh held-out scenarios.', '',
        '### Hardware and execution', '',
        'Intel Core i9-14900HX; 24 physical cores; 32 logical processors; '
        f'{16869548032/2**30:.2f} GiB visible physical RAM. Windows; CPU execution.',
        f"Python {platform.python_version()}, PyTorch {torch.__version__}, NumPy {numpy.__version__}.",
        'Each training run used one local process and one Torch thread. Independent runs overlapped.',
        'Larger validation used three process workers. Final testing used four, with one Torch thread each.',
        'No GPU training or two-device distributed training was used.', '',
        '### Training runs and measured time', '',
        '| Run | Seed | Decisions | Episodes | Simulated episode hours | Measured minutes |',
        '|---|---:|---:|---:|---:|---:|']
    for r in progress['runs']:
        episodes = read_rows(ROOT/r['directory']/'episodes.jsonl')
        lines.append(f"| {r['name']} | {r['seed']} | {r['decisions']:,} | {r['episodes']} | {sum(e['time'] for e in episodes)/3600:.2f} | {r['training_seconds']/60:.2f} |")
    lines += ['', f"Summed training-run time: {sum(r['training_seconds'] for r in progress['runs'])/60:.2f} minutes. Runs overlapped, so this is not elapsed study time.",
        'Simulated episode hours sum complete episodes. Decision counts are not equal experience budgets across V1 and V2.', '',
        'All strategic networks started from random weights. Phase 4 belief weights stayed frozen.',
        'The policy has 47,938 parameters. PPO uses 256-decision rollouts, four epochs, minibatches of 64,',
        'Adam learning rate 0.0003, clip 0.2, value coefficient 0.5 and gradient clipping 0.5.',
        'Time-aware discounting uses gamma 0.99 and GAE lambda 0.95.',
        'V1 entropy coefficient decays 0.02 to 0.001; V2 uses 0.02 to 0.005.',
        'V1 resamples every 0.6 seconds. V2 holds a selected mechanical goal up to 24 seconds;',
        'follow/flee holds are capped at 3 seconds. Observable interruptions return control to the network.',
        'The physics step is 0.2 seconds. A* executes the chosen destination; it does not choose strategy.', '',
        'Rewards: crew win +10, impostor win -10, timeout -5, own elimination once -5,',
        'unique own task +1, and time cost -0.005 per 0.6 seconds. Reports/votes have no direct reward.', '',
        '### Validation, testing and selection process', '',
        '1. Training draws sequential match seeds from 600000–619999. Each run uses only its completed prefix.',
        '2. Screen each V2 checkpoint at 2048/4096/6144/8192 decisions on the same 20 validation seeds, 620000–620019.',
        '3. Validate the four selected V2 checkpoints, preserved V1 and three baselines on all 100 seeds, 620000–620099.',
        '4. Rank mechanically valid full-memory policies by crew win rate, own tasks and conditional vote accuracy.',
        '5. Freeze seed 17 / checkpoint 8192 and record all checkpoint SHA-256 digests before opening final tests.',
        '6. Evaluate nine methods on identical final seeds: 630000–630499 (ID) and 640000–640499 (patient).',
        '7. Aggregate all 90 completed chunks: 100 matches per chunk, 1,000 per method, 9,000 total.',
        '8. Report all methods, including failed V1 and the untrained control; do not replace selection using final results.', '',
        'Diagnostic subsets are reused inside validation; they are not independent replications.',
        'Three full-memory initializations are replicated. The current-only ablation has one seed only.',
        'The final seed-17 initialization control uses identical construction/RNG order without PPO updates.',
        'Greedy learned-policy evaluation differs from stochastic training. Other players remain scripted.', '',
        '### Evaluation timing', '',
        '| Stage | Jobs | Match executions | Sum of measured job minutes | Approximate elapsed minutes |',
        '|---|---:|---:|---:|---:|']
    for stage, t in timings.items():
        lines.append(f"| {stage} | {t['jobs']} | {t['matches']:,} | {t['summed_job_seconds']/60:.2f} | {t['inferred_elapsed_seconds']/60:.2f} |")
    span=metadata['substantive_study_span']
    lines += ['', metadata['timing_caution'],
        f"Substantive study elapsed span: approximately {span['hours']:.2f} hours, from {span['start']} to {span['end']} (Amman, UTC+03:00).",
        span['caution'], '', '| Stage | Approximate start (Amman) | Approximate finish (Amman) |', '|---|---|---|']
    for stage,t in timings.items():lines.append(f"| {stage} | {t['inferred_start']} | {t['inferred_end']} |")
    lines += ['', '### Metrics and interpretation', '',
        'Crew win rate measures team outcomes, not classification accuracy. Own tasks expose individual contribution.',
        'Voting accuracy is correct impostor votes divided by cast non-skip votes. Always inspect cast counts and skips.',
        'Survival is simulated seconds. Illegal actions, navigation failures, timeouts and ejections are also recorded.',
        'Win intervals use 95% Wilson intervals. Paired differences resample whole matched scenarios 5,000 times.',
        'Voting uncertainty also resamples whole matches. Repeated votes are not independent trials.',
        'These intervals measure match variation, not confidence across a population of training seeds.', '',
        '### Complete final result tables and figures', '']
    report = (OUTPUT/'README.md').read_text(encoding='utf-8')
    report = report[report.index('## Locked final results'):report.index('## Phase 6 gate')]
    report = report.replace('](figures/', '](Phase_5_Figures/')
    lines += [report, '', 'The external download bundle includes these twelve graphs and the measured metadata.',
        'The suggested Phase 6 sequence above remains gated; no self-play or learned opponent has been implemented.', END]
    source = document.read_text(encoding='utf-8')
    if START in source:source=source[:source.index(START)]+source[source.index(END)+len(END):]
    document.write_text(source.rstrip()+'\n\n'+'\n'.join(lines)+'\n',encoding='utf-8')
    figures = document.parent/'Phase_5_Figures'; figures.mkdir(exist_ok=True)
    for path in (OUTPUT/'figures').glob('*.png'):shutil.copy2(path, figures/path.name)
    bundle=document.with_name('Phase_5_Teammate_Package.zip')
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(document,document.name)
        archive.write(OUTPUT/'execution-metadata.json','execution-metadata.json')
        for path in figures.glob('*.png'):archive.write(path,'Phase_5_Figures/'+path.name)
    print(json.dumps({'document':str(document),'bundle':str(bundle),'timing':timings}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('document',type=Path)
    build(parser.parse_args().document)
