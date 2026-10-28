"""Original strict loader plus publication-byte verification; no migration."""
from pathlib import Path
import hashlib
from crewmate_v2.fourcrew.policy import FourCrewPolicy

ROOT=Path(__file__).resolve().parents[1]
MODELS={
    'A':('artifacts/crew-handoff/original/checkpoint-batch-23-choices-2058.pt',
         '3b843996c5b9189da06888c89b42bc03c40479c629a01d060ad8ff4099162f7c'),
    'B':('artifacts/crew-handoff/shaped_v2/checkpoint-batch-23-choices-2092.pt',
         '88103ee9bab248c7be4bbc64d39173300ffc7c8cbbbc40c7e95182a2c38355aa')}


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model(name='B'):
    if name not in MODELS:raise ValueError('Choose frozen Model A or B')
    relative,expected=MODELS[name];path=ROOT/relative
    if sha256(path)!=expected:raise ValueError('Selected checkpoint bytes changed')
    policy,metadata=FourCrewPolicy.load(path)
    policy.requires_grad_(False)
    return policy,metadata
