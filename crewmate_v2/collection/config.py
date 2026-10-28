"""Separate execution registration; no changes to the previous serial proposal."""
import hashlib
from pathlib import Path

from crewmate_v2.config import LobbySettings
from crewmate_v2.learning.config import pilot_plan
from crewmate_v2.learning.provenance import runtime_lock, digest
from . import SCHEMA

PARTITIONS = {'engineering': (2740000, 2749999), 'fixtures': (2750000, 2750999),
              'train': (2700000, 2709999)}


def context(partition, slot, episode, streams):
    if partition not in PARTITIONS or not 0 <= slot < streams or episode < 0:
        raise ValueError('Unregistered independent development stream')
    index = episode * streams + slot
    lo, hi = PARTITIONS[partition]
    if lo + index > hi:
        raise ValueError('Development seed partition exhausted')
    return dict(seed=lo + index, rotation=index % 3)


def lock():
    folder = Path(__file__).resolve().parent
    return dict(schema=SCHEMA, learning=runtime_lock(LobbySettings()), sources={
        p.name: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        for p in sorted(folder.glob('*.py'))})


def proposal(workers, streams=8):
    if workers not in (1, 2, 4, 8) or streams != 8:
        raise ValueError('Register worker/stream changes before a pilot')
    return dict(schema=SCHEMA, prior_scientific_plan=pilot_plan(), workers=workers,
                streams=streams, start_method='inline' if workers == 1 else 'spawn',
                batched_inference=True, sampler='independent torch Generator per stream',
                seed_assignment='train_start + episode_index * streams + slot; rotation=index%3',
                rollout=128, steps_per_stream_per_update=16, updates=4,
                decisions_per_model=512, initialization=17,
                selection='checkpoint-512 only', validation_games_per_model=20,
                total_wall_ceiling_seconds=10800,
                resume='exact replay of current episode action history; immutable checkpoint')


def validate_approval(record, workers):
    if (record.get('approved') is not True or record.get('execution_sha256') != digest(proposal(workers))
            or record.get('runtime_sha256') != digest(lock())):
        raise PermissionError('Separate owner approval must bind the new collection execution and runtime')
