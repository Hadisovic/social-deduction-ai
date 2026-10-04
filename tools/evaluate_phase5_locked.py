"""Parallel final-test chunks, only after an explicit immutable selection lock."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from phase5_evaluation import evaluate
from phase5_options import PROTOCOL
from phase5_research_report import read_rows,build
from phase5_analysis import summarize
OUTPUT=ROOT/'docs/benchmarks/phase5/expanded'


def chunk(name,method,offset):
    destination=OUTPUT/f'final-{name}-{offset:03d}'
    if (destination/'metrics.json').exists():return str(destination)
    baseline=method in {'random','scripted','idle'}
    evaluate(destination,100,[] if baseline else [Path(method)],
             include_baselines=baseline,include_idle=baseline,
             baseline_methods=[method] if baseline else None,
             random_protocol=PROTOCOL,match_offset=offset)
    return str(destination)


def main():
    lock=json.loads((OUTPUT/'locked-selection.json').read_text())
    if lock['status']!='locked_before_final_tests':raise ValueError('Explicit selection lock required')
    for path,digest in lock['checkpoint_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Locked checkpoint changed: {path}')
    methods=list(lock['baseline_methods'])+list(lock['checkpoint_sha256'])
    completed=[]
    # Four workers bound memory use; each worker uses one Torch CPU thread.
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs={pool.submit(chunk,f'm{i}',method,offset):(method,offset)
              for offset in range(0,500,50) for i,method in enumerate(methods)}
        for future in as_completed(jobs):
            completed.append(Path(future.result()))
            print(f'Final chunks complete: {len(completed)}/{len(jobs)}',flush=True)
    rows=[row for directory in completed for row in read_rows(directory/'matches.jsonl')]
    result=summarize(rows)
    for method in methods:
        for family in ('ID','patient'):
            seeds={r['seed'] for r in rows if r['method']==method and r['family']==family}
            lo,hi=lock['seed_partitions'][family]
            if seeds!=set(range(lo,hi+1)):raise AssertionError('Incomplete final seed manifest')
    result.update(status='final_evaluation_complete',selected_checkpoint=lock['selected_checkpoint'],
                  matches_per_method=1000,method_count=len(methods),
                  selection_lock_sha256=hashlib.sha256((OUTPUT/'locked-selection.json').read_bytes()).hexdigest())
    (OUTPUT/'final-results.json').write_text(json.dumps(result,indent=2)+'\n')
    build()
    print(json.dumps({'status':result['status'],'matches':len(rows),'methods':len(methods)}),flush=True)


if __name__=='__main__':main()
