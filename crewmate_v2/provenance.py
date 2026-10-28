"""A separate V2 compatibility contract. Historical loaders remain unmodified."""
import hashlib
from pathlib import Path
from phase6.config import source_hashes as historical_sources, verify_frozen
from . import SCHEMA

ROOT=Path(__file__).resolve().parents[1]


def sources():
    return {**historical_sources(),**{p.relative_to(ROOT).as_posix():hashlib.sha256(
        p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sorted((ROOT/'crewmate_v2').glob('*.py'))}}


def metadata(lobby):
    verify_frozen()
    return dict(environment_schema=SCHEMA,lobby=lobby.to_dict(),sources=sources())


def validate_metadata(record,lobby):
    verify_frozen()
    if record!=metadata(lobby):
        raise ValueError('V2 source/config/schema mismatch; historical checkpoints are not V2 policies')
    return True
