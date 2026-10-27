"""A new lock that includes learning files, without weakening historical loaders."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from crewmate_v2.provenance import metadata as canonical_metadata
from . import SCHEMA
from .config import pilot_plan

ROOT=Path(__file__).resolve().parents[2]


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def runtime_lock(lobby):
    package=Path(__file__).resolve().parent
    return dict(schema=SCHEMA,canonical=canonical_metadata(lobby),learning_sources={
        p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        for p in sorted(package.glob('*.py'))},evaluation_helpers={name:hashlib.sha256(
            (ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for name in ('tools/reporting_diversity_v2.py','tools/diagnose_crewmate_v2.py')})


def plan_hash():return digest(pilot_plan())


def validate_approval(record,lock):
    if (record.get('approved') is not True or record.get('plan_sha256')!=plan_hash()
        or record.get('runtime_sha256')!=digest(lock)):
        raise PermissionError('Full pilot requires separate owner approval binding the exact plan and runtime hashes')
