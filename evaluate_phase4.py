"""Evaluate committed Phase 4 models on the registered final ID/strategy splits."""
import argparse
from pathlib import Path


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--data-dir',type=Path)
    args = parser.parse_args(argv)
    base = root/'artifacts/phase4'
    if args.quick:
        base = base/'quick'
    data = args.data_dir or base/'data'
    reports = base/'reports' if args.quick else root/'docs/benchmarks/phase4'
    from phase4_evaluation import evaluate
    evaluate(data,base,reports)
    return 0


if __name__ == '__main__':
    from phase3_cli import use_project_environment
    use_project_environment()
    raise SystemExit(main())
