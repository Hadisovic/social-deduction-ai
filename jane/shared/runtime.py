"""Common engine selection. Construct runtime classes inside each actor worker.

No policy, optimizer or collector is supplied by this package. Classic allocation
and historical native allocation are explicit, separate profiles.
"""
import hashlib,json
from pathlib import Path

PATRICK_PROFILE={'engine':'paired-map-planner-v1','allocation':'patrick-jane-native-v1'}


def engine_class(allocation):
    from jane.classic_v2.engine import game_class,native_game_class
    if allocation=='patrick-jane-native-v1':return native_game_class()
    if allocation=='classic-unique-family-v2':return game_class()
    raise ValueError('Unknown shared allocation profile: '+str(allocation))


def fingerprint():
    root=Path(__file__).resolve().parents[1]
    paths=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.py','.json') and
           (p.relative_to(root).parts[0] in ('shared','classic_v2') or p.relative_to(root).as_posix()=='__init__.py'
            or p.relative_to(root).as_posix().startswith('configs/classic-v2/'))]
    files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sorted(paths)}
    return dict(schema='jane-shared-runtime-v1',sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),files=files)
