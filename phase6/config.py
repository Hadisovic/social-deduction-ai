"""Versioned experiments and fresh, disjoint partitions; never policy inputs."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from phase3_engine import GameConfig

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SHA = '6b78b08e41d0791e49ff486b996d7490666db970'
PROTOCOL = 'phase6-options-v1'
PARTITIONS = {
    'train': (1_200_000, 1_219_999),
    'validation': (1_220_000, 1_220_199),
    'final_id': (1_230_000, 1_230_499),
    'final_patient': (1_240_000, 1_240_499),
    'development': (1_250_000, 1_250_019),
    'smoke_train': (1_260_000, 1_260_099),
}
FROZEN = {
    'artifacts/phase4/belief.pt': 'e931b1fa8a421d9a329a29280d676bf52f793cc8f07c02b8da61f2b3e5e537d1',
    'artifacts/phase5/release/policy.pt': '1038054357cf1c2e0372de7e6c4c226aaa73162d667372e1cf876de82aa867b1',
}


@dataclass(frozen=True)
class Experiment:
    memory: str = 'full'
    distance: str = 'euclidean'
    max_option_seconds: int = 24
    learning_rate: float = .0003
    rollout: int = 256
    crewmate_sight_range: float = 4.5
    impostor_sight_range: float = 6.75
    schema: str = PROTOCOL

    def __post_init__(self):
        if self.memory not in ('full', 'history_no_belief', 'current_only'):
            raise ValueError('Unknown memory condition')
        if self.distance not in ('euclidean', 'route'):
            raise ValueError('Unknown distance condition')
        if self.max_option_seconds not in (12, 24) or self.rollout not in (256, 512):
            raise ValueError('Use registered option/rollout conditions')
        if self.learning_rate not in (.0001, .0003) or self.schema != PROTOCOL:
            raise ValueError('Unknown optimization condition or protocol')
        config = self.game_config()
        if config.crewmate_sight_range >= config.impostor_sight_range:
            raise ValueError('This Phase 6 study requires crew vision < impostor vision')

    def game_config(self):
        return GameConfig(rules_version=6, crewmate_sight_range=self.crewmate_sight_range,
                          impostor_sight_range=self.impostor_sight_range)

    def to_dict(self):
        return asdict(self)


def seeds(partition, count):
    if partition not in PARTITIONS or type(count) is not int or count <= 0:
        raise ValueError('Invalid seed partition/count')
    start, end = PARTITIONS[partition]
    if count > end - start + 1:
        raise ValueError('Seed partition exhausted')
    return tuple(range(start, start + count))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen():
    for path, expected in FROZEN.items():
        if digest(ROOT / path) != expected:
            raise ValueError(f'Frozen release changed: {path}')


def source_hashes():
    paths = list((ROOT / 'phase6').glob('*.py')) + [ROOT / name for name in (
        'phase3_engine.py', 'phase3_runner.py', 'phase5_env.py', 'phase5_features.py',
        'phase5_policy.py', 'phase5_options.py', 'phase5_options_training.py',
        'phase5_training.py', 'phase5_runner.py', 'phase4_scripts.py', 'phase3_bots.py',
        'social_deduction/phase3_api.py', 'social_deduction/phase3_observation.py',
        'among_us_map.py', 'navigation_service.py', 'run_phase6.py', 'assets/skeld/among_us_map.json')]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(
        p.read_bytes().replace(b'\r\n', b'\n') if p.suffix == '.py' else p.read_bytes()).hexdigest()
        for p in sorted(paths)}


def load_experiment(path):
    return Experiment(**json.loads(Path(path).read_text()))
