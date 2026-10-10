"""Phase 6 rules preview: one frozen crew policy, four scripted opponents."""
import argparse
from pathlib import Path
from phase6.config import ROOT, load_experiment, verify_frozen
from phase6.env import Phase6Env
from phase6.train import load_policy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--config', type=Path, default=ROOT / 'artifacts/phase6/configs/reference.json')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'artifacts/phase5/release/policy.pt')
    args, forwarded = parser.parse_known_args(argv)
    verify_frozen()
    experiment = load_experiment(args.config)
    release = args.checkpoint.resolve() == (ROOT / 'artifacts/phase5/release/policy.pt').resolve()
    if not release:
        _, trained, metadata = load_policy(args.checkpoint)
        if trained != experiment:
            raise ValueError('Viewer config differs from checkpoint experiment')
    def factory(metadata, **kwargs):
        kwargs.pop('ablation', None)
        return Phase6Env(experiment, **kwargs)
    from run_phase5 import main as viewer
    label = 'PHASE 6 RULES / ' + ('PHASE 5 CREW TRANSFER' if release else 'SMOKE CREW') + ' / SCRIPTED IMPOSTOR'
    return viewer(['--checkpoint', str(args.checkpoint), '--seed', '1250000', *forwarded],
                  environment_factory=factory, display_label=label)


if __name__ == '__main__':
    raise SystemExit(main())
