"""Small equivariant predictor. Input API accepts actor memory, never world truth."""
from dataclasses import dataclass
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .features import FEATURE_NAMES, FEATURE_SCHEMA, encode
from .memory import MEMORY_SCHEMA

CHECKPOINT_SCHEMA = 'belief-checkpoint-v1'
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / 'artifacts/phase4/belief.pt'


class CandidateNet(nn.Module):
    def __init__(self, architecture='set', feature_count=len(FEATURE_NAMES)):
        super().__init__()
        self.architecture = architecture
        self.register_buffer('mean', torch.zeros(feature_count))
        self.register_buffer('scale', torch.ones(feature_count))
        if architecture == 'linear':
            self.score = nn.Linear(feature_count, 1)
        elif architecture == 'set':
            self.encoder = nn.Sequential(nn.Linear(feature_count, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
            self.score = nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 1))
        else:
            raise ValueError('Unknown belief architecture')

    def forward(self, x):
        x = (x - self.mean) / self.scale
        if self.architecture == 'set':
            h = self.encoder(x)
            context = h.mean(dim=-2, keepdim=True).expand_as(h)
            x = torch.cat((h, context), dim=-1)
        return self.score(x).squeeze(-1)


@dataclass(frozen=True)
class Belief:
    player_ids: tuple[str, ...]
    probabilities: tuple[float, ...]
    logits: tuple[float, ...]
    entropy: float

    @property
    def by_player(self):
        return dict(zip(self.player_ids, self.probabilities))


class BeliefModel:
    def __init__(self, network, temperature=1., view='full'):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Temperature must be positive and finite')
        self.network = network.eval()
        self.temperature = float(temperature)
        self.view = view

    def predict(self, actor_memory, candidates=None):
        if self.view == 'current':
            raise ValueError('Current-only ablation requires a fresh observation window, not persistent memory')
        ids = tuple(candidates or actor_memory.candidates)
        x = encode(actor_memory, ids, self.view)
        with torch.inference_mode():
            logits = self.network(torch.from_numpy(x))
            p = torch.softmax(logits.double() / self.temperature, dim=-1).numpy()
        if not np.isfinite(p).all() or not np.isclose(p.sum(), 1.):
            raise ValueError('Invalid model probabilities')
        return Belief(ids, tuple(map(float, p)), tuple(map(float, logits.tolist())),
                      float(-(p * np.log(np.maximum(p, 1e-300))).sum()))

    def save(self, path, metadata=None):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'schema': CHECKPOINT_SCHEMA, 'feature_schema': FEATURE_SCHEMA,
                    'memory_schema': MEMORY_SCHEMA, 'feature_names': list(FEATURE_NAMES),
                    'architecture': self.network.architecture, 'temperature': self.temperature,
                    'view': self.view, 'state_dict': self.network.state_dict(),
                    'metadata': metadata or {}}, path)

    @classmethod
    def load(cls, path=DEFAULT_CHECKPOINT):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'Missing Phase 4 model: {path}. Run: python train_phase4.py')
        data = torch.load(path, map_location='cpu', weights_only=True)
        if (data.get('schema') != CHECKPOINT_SCHEMA or data.get('feature_schema') != FEATURE_SCHEMA
                or data.get('memory_schema') != MEMORY_SCHEMA or data.get('feature_names') != list(FEATURE_NAMES)):
            raise ValueError('Incompatible Phase 4 checkpoint schema/features; retrain explicitly')
        if data.get('view') not in ('full', 'current', 'no_claims', 'collapsed'):
            raise ValueError('Invalid checkpoint evidence view')
        network = CandidateNet(data['architecture'])
        network.load_state_dict(data['state_dict'], strict=True)
        if any(not torch.isfinite(p).all() for p in network.state_dict().values()) or (network.scale <= 0).any():
            raise ValueError('Invalid checkpoint weights/scaling')
        return cls(network, data['temperature'], data['view'])


def batch_logits(model, features):
    result = []
    with torch.inference_mode():
        for start in range(0, len(features), 4096):
            result.append(model.network(torch.from_numpy(features[start:start+4096])).numpy())
    return np.concatenate(result)
