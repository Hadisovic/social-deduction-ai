"""Native spectator plus local, read-only original-door animation."""
import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--source-root',type=Path,required=True)
    args,remaining=parser.parse_known_args()
    from jane.impostor.v1.bootstrap import activate
    activate(args.source_root)
    from lizbon_viewer.native import install
    install()
    from spectator.v2.application import main as viewer
    return viewer(remaining)


if __name__=='__main__':raise SystemExit(main())
