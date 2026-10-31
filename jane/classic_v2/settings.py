"""Versioned host profiles reuse immutable shared value validation, not V1 locks."""
from dataclasses import dataclass,replace
import hashlib
import json
from pathlib import Path
from jane.shared.settings import GameSettings,_unique_object,bundle_receipt
from . import ENGINE_VERSION,GEOMETRY_REVISION


@dataclass(frozen=True)
class HostProfile:
    settings: GameSettings
    engine: str=ENGINE_VERSION
    allocation: str='unique-canonical-task-family-v2'

    def __post_init__(self):
        s=self.settings
        if self.engine!=ENGINE_VERSION or self.allocation!='unique-canonical-task-family-v2':
            raise ValueError('Unknown Classic engine/allocation version')
        if s.ruleset_id!='classic-skeld-v1-draft' or s.tasks.pool_id!='classic-skeld-v1-draft':
            raise ValueError('Use the versioned Classic profile; historical A2 profiles remain locked')
        if s.mechanics.enabled()!=frozenset(('lights','doors','vents')) or any(r.enabled for r in s.optional_roles):
            raise ValueError('Current Classic core enables lights/doors/vents and base roles only')
        if s.lobby.kill_distance!='medium':
            raise ValueError('Only calibrated native medium range is implemented; short/long need calibration')
        if s.meetings.voting_seconds==0:
            raise ValueError('Zero/infinite voting duration needs retail-build verification; use a positive limit')
        if s.simulation.cap_policy!='truncate' or s.simulation.stall_cap_seconds<=0:
            raise ValueError('Separate technical/stall cuts are required; cuts never become losses')

    def to_dict(self):
        return dict(engine=self.engine,allocation=self.allocation,settings=self.settings.to_dict())

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True,separators=(',',':')).encode()).hexdigest()

    @property
    def effective_counts(self):
        t=self.settings.tasks
        # Original ShipStatus.Begin falls back to one short task if all are zero.
        return (t.common,t.long,t.short if t.common+t.long+t.short else 1)

    @classmethod
    def from_dict(cls,value):
        if type(value) is not dict or set(value)!={'engine','allocation','settings'}:
            raise ValueError('Profile requires exactly engine, allocation and settings')
        return cls(GameSettings.from_dict(value['settings']),value['engine'],value['allocation'])


def load_profile(path):
    return HostProfile.from_dict(json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=_unique_object))


def save_profile(path,profile):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as handle:json.dump(profile.to_dict(),handle,indent=2);handle.write('\n')


def receipt(profile):
    folder=Path(__file__).resolve().parent
    sources={p.name:hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
             for p in sorted(folder.glob('*.py'))}
    value=dict(profile_sha256=profile.fingerprint(),geometry_revision=GEOMETRY_REVISION,classic_sources=sources,
               historical_shared_bundle=bundle_receipt(profile.settings)['bundle_sha256'])
    return {**value,'sha256':hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()}
