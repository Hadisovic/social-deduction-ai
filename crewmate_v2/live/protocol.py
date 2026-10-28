"""Additional source lock and owner/prerequisite gate for the approved live pilot."""
import hashlib
import importlib
from pathlib import Path
import sys
from crewmate_v2.collection.config import lock as collection_lock,proposal
from crewmate_v2.learning.provenance import digest
from . import SCHEMA


def spectator_import(source):
    source=Path(source).resolve()
    if str(source) not in sys.path:sys.path.append(str(source))
    module=importlib.import_module('spectator.recorder')
    if Path(module.__file__).resolve()!=source/'spectator'/'recorder.py':
        raise ValueError('Unexpected spectator source; do not shadow simulator imports')
    return module


def lock(source):
    source=Path(source).resolve();package=Path(__file__).parent
    return dict(schema=SCHEMA,collection=collection_lock(),live_sources={
        p.name:hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sorted(package.glob('*.py'))},
        spectator_sources={name:hashlib.sha256((source/'spectator'/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                           for name in ('recorder.py','replay_format.py')})


def plan():
    return dict(schema=SCHEMA,execution=proposal(4),record_streams=[0,1,2,3],
                playback='completed training matches, independent GUI, automatic queue',
                owner_request='d42ec8ae-3a1a-49b5-b136-bc91c368320e',
                rewards_unchanged=True,final_partitions=False,replication29=False)


def validate(record,source):
    if (record.get('approved') is not True or record.get('plan_sha256')!=digest(plan())
            or record.get('runtime_sha256')!=digest(lock(source))):
        raise PermissionError('Owner authorization must bind the exact live execution/runtime')
    prerequisites=record.get('prerequisites',{})
    for name in ('real_ppo_replay','noninterference','overhead_acceptable','tests_passed','desktop_gui_verified'):
        if prerequisites.get(name) is not True:raise PermissionError('Missing verified prerequisite: '+name)
