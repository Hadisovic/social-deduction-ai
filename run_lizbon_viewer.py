"""Native spectator plus local door animation and configurable inference lobby."""
import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--lobby',action='store_true')
    args,remaining=parser.parse_known_args()
    from jane.impostor.v1.bootstrap import activate
    activate(args.source_root)
    from lizbon_viewer.native import install
    install()
    from lizbon_viewer.lobby_ui import install as install_lobby
    install_lobby(args.source_root,start_lobby=args.lobby)
    from spectator.v2.application import main as viewer
    return viewer(remaining)


if __name__=='__main__':raise SystemExit(main())
