"""Watch the explicitly locked V4.1 research checkpoint; never trains or falls back."""
import os
from pathlib import Path
import sys


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    os.chdir(root)
    from run_impostor_options_v41 import main
    raise SystemExit(main(['--checkpoint',str(root/'artifacts/impostor-options-v41/release/policy.pt'),*sys.argv[1:]]))
