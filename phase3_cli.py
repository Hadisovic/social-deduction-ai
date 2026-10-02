"""Use the existing project environment for the documented one-command launch."""
def use_project_environment():
    import os
    from pathlib import Path
    import sys
    import subprocess
    root = Path(__file__).resolve().parent
    executable = root / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if executable.is_file() and Path(sys.prefix).resolve() != (root / '.venv').resolve():
        raise SystemExit(subprocess.call([str(executable), str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]))
