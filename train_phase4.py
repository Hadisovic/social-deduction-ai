"""Generate, train, select, calibrate and evaluate the complete Phase 4 experiment."""
import argparse
import json
from pathlib import Path
import shutil


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true',help='Separate smoke experiment; never replaces the release model')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--data-dir',type=Path)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    root = Path(__file__).resolve().parent
    base = root/'artifacts/phase4'
    if args.quick:
        base = base/'quick'
    data = args.data_dir or base/'data'
    runs = base/'runs/full'
    report = base/'reports' if args.quick else root/'docs/benchmarks/phase4'
    # Lazy torch import: spawned data workers must not initialize the training runtime.
    from phase4_data import generate_dataset
    manifest_path = data/'manifest.json'
    prior_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    manifest = generate_dataset(data,args.quick,args.workers)
    previous_seconds = (prior_manifest.get('generation_wall_seconds', prior_manifest['this_invocation_seconds'])
                        if prior_manifest else 0.)
    manifest['generation_wall_seconds'] = previous_seconds + manifest['this_invocation_seconds']
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    from phase4_training import train_experiment
    selection = train_experiment(data,runs,args.quick)
    for path in runs.glob('*.pt'):
        shutil.copy2(path,base/path.name)
    shutil.copy2(runs/'selection.json',base/'selection.json')
    report.mkdir(parents=True,exist_ok=True)
    (report/'dataset.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    from phase4_evaluation import evaluate
    evaluate(data,base,report)
    print(f'Experiment written to {report}. Model: {base / "belief.pt"}',flush=True)
    return 0


if __name__ == '__main__':
    from phase3_cli import use_project_environment
    use_project_environment()
    raise SystemExit(main())
